# !/usr/bin/env python3

import math
import paddle
from functools import partial
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.distributed.fleet.utils import recompute

from modeling.backbones.vision_transformer import drop_path, to_2tuple, trunc_normal_, zeros_, ones_
from ppdet.modeling.shape_spec import ShapeSpec


def load_checkpoint(model, pretrained):
    print('----- LOAD -----', pretrained)
    state_dict = paddle.load(pretrained)

    if 'pos_embed' in state_dict:
        print('---- POS_EMBED -----')
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        print(orig_size, new_size)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = F.interpolate(pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
            new_pos_embed = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed

    model.set_state_dict(state_dict)
    print('----- LOAD END -----', pretrained)


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class LastLevelMaxPool(nn.Layer):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=1, stride=2, padding=0)


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lr=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, lr=1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=qkv_bias, weight_attr=ParamAttr(learning_rate=lr))
        self.window_size = window_size
        assert len(self.window_size) == 2, "window_size must include two-dimension information"
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1

        self.rel_pos_h = self.create_parameter(
            shape=(2 * window_size[0] - 1, head_dim), default_initializer=zeros_,  attr=ParamAttr(learning_rate=lr))
        self.rel_pos_w = self.create_parameter(
            shape=(2 * window_size[1] - 1, head_dim), default_initializer=zeros_,  attr=ParamAttr(learning_rate=lr))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape((B, N, 3, self.num_heads, -1)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make paddlescript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        # attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        _, _, q_length, _ = q.shape
        q_size = int(q_length ** 0.5)
        

        # if q_size == self.window_size[0]:
        if (H, W) == self.window_size:
            attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        else:
            rel_pos_h = F.interpolate(self.rel_pos_h.unsqueeze(axis=0), size=(2 * H - 1,), align_corners=False, mode='linear', data_format='NWC').squeeze(axis=0)
            rel_pos_w = F.interpolate(self.rel_pos_w.unsqueeze(axis=0), size=(2 * W - 1,), align_corners=False, mode='linear', data_format='NWC').squeeze(axis=0)
            attn = calc_rel_pos_spatial(attn, q, (H, W), (H, W), rel_pos_h, rel_pos_w)
        
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape((B, H // window_size, window_size, W // window_size, window_size, C))
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = x.transpose((0, 1, 3, 2, 4, 5)).reshape((-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, 
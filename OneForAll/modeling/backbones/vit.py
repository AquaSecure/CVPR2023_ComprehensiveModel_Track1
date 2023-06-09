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
        super().__init
"""Build Vision Transformer
"""
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from functools import partial
import copy
from collections import OrderedDict
import os

import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn.initializer import TruncatedNormal
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Normal
import numpy as np

logger = logging.getLogger(__name__)


def to_2tuple(x):
    """Convert x to [x, x]
    """
    return tuple([x] * 2)

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

BIAS_LR_FACTOR=2.0
WEIGHT_DECAY_NORM=0.0001 #TODO add additional weight decay for norm_type module

def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        """Init
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    """MLP Layer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        params:
        return:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    """Attention Layer
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Init
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, 
                    bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR) if qkv_bias else False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward
        """
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1
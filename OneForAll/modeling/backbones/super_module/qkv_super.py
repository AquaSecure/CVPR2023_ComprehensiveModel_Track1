"""super_module/qkv_super.py
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


xavier_uniform_ =  paddle.nn.initializer.XavierUniform()
constant_ = paddle.nn.initializer.Constant()

class QkvSuper(nn.Linear):
    """QkvSuper
    """
    def __init__(self, super_in_dim, super_out_dim,
                bias_attr=None, uniform_=None, non_linear='linear', scale=False, weight_attr=None):
        super().__init__(super_in_dim, super_out_dim, bias_attr=bias_attr, weight_attr=weight_attr)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear) #TODO add initialization for weights
        self.profiling = False

    def _reset_parameters(self, bias, uniform_, non_linear):
        xavier_uniform_(self.weight) if uniform_ is None else uniform_(self.weight) #TODO add non_linear
        if bias:
            constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        """set_sample_config
        """
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        """_sample_parameters
        """
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.
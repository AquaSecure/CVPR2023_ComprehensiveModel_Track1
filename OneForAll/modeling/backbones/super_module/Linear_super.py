"""super_module/Linear_super.py
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


xavier_uniform_ =  paddle.nn.initializer.XavierUniform()
constant_ = paddle.nn.initializer.Constant()

class LinearSuper(nn.Linear):
    """LinearSuper
    """
    def __init
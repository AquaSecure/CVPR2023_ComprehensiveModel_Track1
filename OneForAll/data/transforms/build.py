"""transforms/build.py
"""

import paddle.vision.transforms as T
import numpy as np
import random

from timm.data.random_erasing import RandomErasing
# from .random_erasing import RandomErasing
# from fastreid.data.transforms import *
from fastreid.data.transforms.autoaugment import AutoAugment

class RandomApply(object):
    """RandomApply
    """
    def __init__(self, prob=0.5, transform_function_class=None):
        self.prob = prob
        self.transform_function = transform_function_class()
    
    def __call__(self, x):
        if random.random() > self.prob:
            return self.transform_function(x)
        else:
            return x
        
# def build_transforms_lazy(is_train=True, **kwargs):
#     res = []

#     if is_train:
#         size_train = kwargs.get('size_train', [256, 128])

#         # crop
#         do_crop = kwargs.
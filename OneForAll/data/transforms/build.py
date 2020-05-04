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
#         do_crop = kwargs.get('do_crop', False)
#         crop_size = kwargs.get('crop_size', [224, 224])
#         crop_scale = kwargs.get('crop_scale', [0.16, 1])
#         crop_ratio = kwargs.get('crop_ratio', [3./4., 4./3.])

#         # augmix augmentation
#         do_augmix = kwargs.get('do_augmix', False)
#         augmix_prob = kwargs.get('augmix_prob', 0.0)

#         # auto augmentation
#         do_autoaug = kwargs.get('do_autoaug', False)
#         autoaug_prob = kwargs.get('autoaug_prob', 0.0)

#         # horizontal filp
#         do_flip = kwargs.get('do_flip', False)
#         flip_prob = kwargs.get('flip_prob', 0.5)

#         # padding
#         do_pad = kwargs.get('do_pad', False)
#         padding_size = kwargs.get('padding_size', 10)
#         padding_mode = kwargs.get('padding_mode', 'constant')

#         # color jitter
#         do_cj = kwargs.get('do_cj', False)
#         cj_prob = kwargs.get('cj_prob', 0.5)
#         cj_brightness = kwargs.get('cj_brightness', 0.15)
#         cj_contrast = kwargs.get('cj_contrast', 0.15)
#         cj_saturation = kwargs.get('cj_saturation', 0.1)
#         cj_hue = kwargs.get('cj_hue', 0.1)

#         # random affine
#         do_affine = kwargs.get('do_affine', False)

#         # random erasing
#         do_rea = kwargs.get('do_rea', False)
#         rea_prob = kwargs.get('rea_prob', 0.5)
#         rea_value = list(kwargs.get('rea_value', [0.485*255, 0.456*255, 0.406*255]))

#         # random patch
#         do_rpt = kwargs.get('do_rpt', False)
#         rpt_prob = kwargs.get('rpt_prob', 0.5)

#         # normalize
#         mean = kwargs.get('mean', [0.485*255, 0.456*255, 0.406*255])
#         std = kwargs.get('std', [0.229*255, 0.224*255, 0.225*255])

#         if do_autoaug:
#             res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))

#         if size_train[0] > 0:
#             res.append(T.Resize(size_train[0] if len(size_train) == 1 else size_train, interpolation=3))

#         if do_crop:
#             res.append(T.RandomResizedCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size,
#                                            interpolation=3,
#                                            scale=crop_scale, ratio=crop_ratio))
#         if do_pad:
#          
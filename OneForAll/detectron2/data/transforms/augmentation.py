# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import numpy as np
import pprint
from typing import Any, List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""


__all__ = [
    "Augmentation",
    "AugmentationList",
    "AugInput",
    "TransformGen",
    "apply_transform_gens",
    "StandardAugInput",
    "apply_augmentations",
]


def _check_img_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


def _get_aug_input_args(aug, aug_input) -> List[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    """
    if aug.input_args is None:
        # Decide what attributes are needed automatically
        prms = list(inspect.signature(aug.get_transform).parameters.items())
        # The default behavior is: if there is one parameter, then its "image"
        # (work automatically for majority of use cases, and also avoid BC breaking),
        # Otherwise, use the argument names.
        if len(prms) == 1:
            names = ("image",)
        else:
            names = []
            for 
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from typing import Tuple
import torch
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from PIL import Image

from .augmentation import Augmentation, _transform_to_aug
from .transform import ExtentTransform, ResizeTransform, RotationTransform

__all__ = [
    "FixedSizeCrop",
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeScale",
    "ResizeShortestEdge",
    "RandomCrop_CategoryAreaConstraint",
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentati
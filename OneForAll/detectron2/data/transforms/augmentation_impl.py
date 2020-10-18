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
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = _transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, "Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        do = self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                " Got {short_edge_length}!"
            )
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(
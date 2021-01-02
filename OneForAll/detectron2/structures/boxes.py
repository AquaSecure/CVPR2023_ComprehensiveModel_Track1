# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import math
import numpy as np
from enum import IntEnum, unique
from typing import List, Tuple, Union
import torch
from torch import device

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 
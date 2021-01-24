# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import typing
from typing import Any, List
import fvcore
from fvcore.nn import activation_count, flop_count, parameter_count, parameter_count_table
from torch import nn

from detectron2.export import TracingAdapter

__all__ = [
    "activation_count_operators",
    "flop_count_operators",
    "parameter_count_table",
    "parameter_count",
    "FlopCountAnalysis",
]

FLOPS_MODE = "flops"
ACTIVATIONS_MODE = "activations"


# Some extra ops to ignore from counting, including elementwise and reduction ops
_IGNORED_OPS = {
    "aten::add",
    "aten::add_",
    "aten::argmax",
    "
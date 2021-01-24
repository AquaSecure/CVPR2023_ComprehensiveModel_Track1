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
    "aten::argsort",
    "aten::batch_norm",
    "aten::constant_pad_nd",
    "aten::div",
    "aten::div_",
    "aten::exp",
    "aten::log2",
    "aten::max_pool2d",
    "aten::meshgrid",
    "aten::mul",
    "aten::mul_",
    "aten::neg",
    "aten::nonzero_numpy",
    "aten::reciprocal",
    "aten::rsub",
    "aten::sigmoid",
    "aten::sigmoid_",
    "aten::softmax",
    "aten::sort",
    "aten::sqrt",
    "aten::sub",
    "torchvision::nms",  # TODO estimate flop for nms
}


class FlopCountAnalysis(fvcore.nn.FlopCountAnalysis):
    """
    Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models.
    """

    def __init__(self, model, inputs):
        """
        Args:
            model (nn.Module):
            inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
        """
        wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
        super().__init__(wrapper, wrapper.flattened_inputs)
        self.set_op_handle(**{k: None for k in _IGNORED_OPS})


def flop_count_operators(model: nn.Module, inputs: list) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level flops counting using jit.
    This is a wrapper of :func:`fvcore.nn.flop_count` and adds supports for standard
    detection models in detectron2.
    Please use :class:`FlopCountAnalysis` for more advanced functionalities.

    Note:
        The function runs the input through the model to compute flops.
        The flops of a detection model is often input-dependent, for example,
        the flops of box & mask head depends on the number of proposals &
        the number of detected objects.
        Therefore, the flops counting using a single input may not accurately
        reflect the computation cost of a model. It's recommended to average
        across a number of inputs.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.
        supported_ops (dict[str, Handle]): see documentation of :func:`fvcore.nn.flop_count`

    Returns:
        Counter: Gflop count per operator
    """

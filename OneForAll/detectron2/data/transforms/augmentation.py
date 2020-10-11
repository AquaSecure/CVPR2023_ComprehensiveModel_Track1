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
            for name, prm in prms:
                if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError(
                        """ \
The default implementation of `{type(aug)}.__call__` does not allow \
`{type(aug)}.get_transform` to use variable-length arguments (*args, **kwargs)! \
If arguments are unknown, reimplement `__call__` instead. \
"""
                    )
                names.append(name)
        aug.input_args = tuple(names)

    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError as e:
            raise AttributeError(
                "{type(aug)}.get_transform needs input attribute '{f}', "
                "but it is not an attribute of {type(aug_input)}!"
            ) from e
    return args


class Augmentation:
    """
    Augmentation defines (often random) policies/strategies to generate :class:`Transform`
    from data. It is often used for pre-processing of input data.

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method. When called with the positional arguments,
    the :meth:`get_transform` method executes the policy.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to execute the actual transform operations to those data.
    Its :meth:`__call__` method will use :meth:`AugInput.transform` to execute the transform.

    The returned `Transform` object is meant to describe deterministic transformation, which means
    it can be re-applied on associated data, e.g. the geometry of an image and its segmentation
    masks need to be transformed together.
    (If such re-application is not needed, then determinism is not a crucial requirement.)
    """

    input_args: Optional[Tuple[str]] = None
    """
    Stores the attribute names needed by :meth:`get_transform`, e.g.  ``("image", "sem_seg")``.
    By default, it is just a tuple of argument names in :meth:`self.get_transform`, which often only
    contain "image". As long as the argument name convention is followed, there is no need for
    users to touch this attribute.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args) -> Transform:
        """
        Execute the policy based on input data, and decide what transform to apply to inputs.

        Args:
            args: Any fixed-length positional arguments. By default, the name of the arguments
                should exist in the :class:`AugInput` to be used.

        Returns:
            Transform: Returns the deterministic transform to apply to the input.

        Examples:
        ::
            class MyAug:
                # if a policy needs to know both image and semantic segmentation
                def get_transform(image, sem_seg) -> T.Transform:
                    pass
            tfm: Transform = MyAug().get_transform(image, sem_seg)
            new_image = tfm.apply_image(image)

        Notes:
            Users can freely use arbitrary new argument names in custom
            :meth:`get_transform` method, as long as they are available in the
            input data. In detectron2 we use the following convention:

            * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
              floating point in range [0, 1] or [0, 255].
            * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
              of N instances. Each is in XYXY format in unit of absolute coordinates.
            * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

            We do not specify convention for other types and do not include builtin
            :class:`Augmentation` that uses other types in detectron2.
        """
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        """
        Augment the given `aug_input` **in-place**, and return the transform that's used.

        This method will be called to apply the augmentation. In most 
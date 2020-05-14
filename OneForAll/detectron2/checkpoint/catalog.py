# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from detectron2.utils.file_io import PathHandler, PathManager


class ModelCatalog(object):
    """
    Store mappings from names to third-party models.
    """

    S3_C2_DETECTRON_PREFIX = "https://dl.fbaipublicfiles.com/detectron"

    # MSRA models have STRIDE_IN_1X1=True. False otherwise.
    # NOTE: all BN models here have fused BN into an affine layer.
    # As a result, you should only load them to a model with "FrozenBN".
    # Loading them to a model with regular BN or SyncBN is wrong.
    # Even when loaded to FrozenBN, it is still different from affine by an epsilon,
    # which should be negligible for training.
    # NOTE: all models here uses PIXEL_STD=[1,1,1]
    # NOTE: Most of the BN models here are no longer used. We use the
    # re-converted pre-trained models under detectron2 model zoo instead.
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "FAIR/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/X-101-64x
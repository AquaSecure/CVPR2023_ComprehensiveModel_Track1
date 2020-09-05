# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from fvcore.common.timer import Timer

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from .builtin_meta import _get_coco_instances_meta
from .lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES
from .lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_lvis_json", "register_lvis_instances", "get_lvis_instances_meta"]


def register_lvis_
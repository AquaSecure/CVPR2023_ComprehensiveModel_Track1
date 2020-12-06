# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from .coco_evaluation import instances_to_coco_json
from .evaluator import DatasetEvaluator


class LVISEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
        from lvis import LVIS

        self._logger = logging.getLogger(__name__)

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it fr
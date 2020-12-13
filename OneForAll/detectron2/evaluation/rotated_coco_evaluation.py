# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import numpy as np
import os
import torch
from pycocotools.cocoeval import COCOeval, maskUtils

from detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.file_io import PathManager

from .coco_evaluation import COCOEvaluator


class RotatedCOCOeval(COCOeval):
    @staticmethod
    def is_rotated(box_list):
        if t
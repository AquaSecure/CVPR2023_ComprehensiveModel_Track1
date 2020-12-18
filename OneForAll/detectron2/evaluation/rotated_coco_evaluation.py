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
        if type(box_list) == np.ndarray:
            return box_list.shape[1] == 5
        elif type(box_list) == list:
            if box_list == []:  # cannot decide the box_dim
                return False
            return np.all(
                np.array(
                    [
                        (len(obj) == 5) and ((type(obj) == list) or (type(obj) == np.ndarray))
                        for obj in box_list
                    ]
                )
            )
        return False

    @staticmethod
    def boxlist_to_tensor(boxlist, output_box_dim):
        if type(boxlist) == np.ndarray:
            box_tensor = torch.from_numpy(boxlist)
        elif type(boxlist) == list:
            if boxlist == []:
                return torch.zeros((0, output_box_dim), dtype=torch.float32)
            else:
                box_tensor = torch.FloatTensor(boxlist)
        else:
            raise Exception("Unrecognized boxlist type")

        input_box_dim = box_tensor.shape[1]
        if input_box_dim != output_box_dim:
            if input_box_dim == 4 and output_box_dim == 5:
                box_tensor = BoxMode.convert(box_tensor, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)
            else:
                raise Exception(
                    "Unable to convert from {}-dim box to {}-dim box".format(
                        input_box_dim, output_box_dim
                    )
                )
        return box_tensor

    def compute_iou_dt_gt(self, dt, gt, is_crowd):
        if self.is_rotated(dt) or self.is_rotated(gt):
            # TODO: take is_crowd into consideration
            assert all(c == 0 for c in is_crowd)
            dt = RotatedBoxes(self.boxlist_to_tensor(dt, output_box_dim=5))
            gt = RotatedBoxes(self.boxlist_to_tensor(gt, output_box_dim=5))
            return pairwise_iou_rotated(dt, gt)
        else:
            # This is the same as the classical COCO evaluation
            return maskUtils.iou(dt, gt, is_crowd)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId
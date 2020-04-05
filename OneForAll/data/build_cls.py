import os
import random
import logging
from collections.abc import Mapping

import numpy as np
import paddle

from utils import comm
from fastreid.data import samplers
from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
from tools import moe_group_utils
from paddle.io import Dataset
from data.build import fast_batch_collator

from data.samplers.clsaware_reader import VehicleMultiTaskClassAwareSampler
from .datasets.fgvc_dataset import *

_root = os.getenv("FASTREID_DATASETS", "datasets")


class HierachicalCommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=False, dataset_name=None, num_classes=1000, is_train=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.dataset_name = dataset_name
        self._num_classes = num_classes
        self.labels = []
        self.id2imgname = self.id2name(img_items)
        self.is_train = is_train

        cam_set = set()
        self.cams = sorted(list(cam_set))
        if relabel:
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def id2name(self, img_items):
        id2name = {}
        for i, item in enumerate(img_items):
             img_path = item[0]
             id2name[i] = img_path
        return id2name

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        n_retry = 10
        for _ in range(n_retry):
            try:
                img_item = self.img_items[index]
                img_path = img_item[0]
                pid = img_item[1]
                camid = img_item[2]
                im_id = img_item[3]
                img = read_image(img_path)
                ori_h, ori_w, _ = np.array(img).shape
                if self.transform is not None: img = self.transform(img)
                _, h, w = img.shape
                im_shape = np.array((h, w), 'float32')
                scale_factor = np.array((h / ori_h, w / ori_w), 'float32')
                break
            except Exception as e:
                index = (index + 1) % len(self.img_items)
                print(e)
        
        if self.relabel:
            camid = self.cam_dict[camid]
            
        if self.is_train:
            return {
                "image": img,
                "targets": pid,
                "camids": camid,
                "im_shape": im_shape,
                "scale_factor": scale_factor,
                "img_paths": img_path,
                "im_id": im_id,
                "id2imgname": self.id2imgname,
            
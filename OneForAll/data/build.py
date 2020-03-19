# !/usr/bin/env python3
import os
import random
import logging
from collections.abc import Mapping

import numpy as np
import paddle

from utils import comm
from fastreid.data import samplers
from fastreid.data import CommDataset
from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
from data.transforms import detection_ops
from data.transforms.detection_ops import Compose, BatchCompose


_root = os.getenv("FASTREID_DATASETS", "datasets")


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks.
    There is no need of transforming data to GPU in fast_batch_collator
    """
    elem = batched_inputs[0]
    if isinstance(elem, np.ndarray):
        # return paddle.to_tensor(np.concatenate([ np.expand_dims(elem, axis=0) for elem in batched_inputs], axis=0))
        return np.concatenate([np.expand_dims(elem, axis=0) for elem in batched_inputs], axis=0)

    elif isinstance(elem, Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}
    elif isinstance(elem, float):
        # return paddle.to_tensor(batched_inputs, dtype=paddle.float64)
        return np.array(batched_inputs, dtype=np.float64) 
    elif isinstance(elem, int):
        #return paddle.to_tensor(batched_inputs)
        return np.array(batched_inputs) 
    elif isinstance(elem, str):
        return batched_inputs


class MultiTaskDataLoader(object):
    """MultiTaskDataLoader
    """
    def __init__(self, task_loaders, cfg):
        super().__init__()
        self.task_loaders = task_loaders
        self.cfg = cfg

        self.task_iters = {}
        for name, loader in self.task_loaders.items():
            self.task_iters[name] = iter(loader)

    def __iter__(self):
        return self
        
    def __len__(self):
        # TODO: make it more general
        return len(list(self.task_iters.values())[0])

    def __next__(self):
        batch = {}

        if sel
# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]


def _shard_iterator_dataloader_worker(iterable):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        yield from itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)


class _MapIterableDataset(data.IterableDataset):
    """
    Map a function over elements in an IterableDataset.

    Similar to pytorch's MapIterDataPipe, but support filtering when map_func
    returns None.

    This class is not public-facing. Will be called by `MapDataset`.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for x in map(self._map_func, self._dataset):
            if x is not None:
                yield x


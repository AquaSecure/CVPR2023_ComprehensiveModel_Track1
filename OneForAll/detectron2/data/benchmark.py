# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from itertools import count
from typing import List, Tuple
import torch
import tqdm
from fvcore.common.timer import Timer

from detectron2.utils import comm

from .build import build_batch_data_loader
from .common import DatasetFromList, MapDataset
from .samplers import TrainingSampler

logger = logging.getLogger(__name__)


class _EmptyMapDataset(torch.utils.data.Dataset):
    """
    Map anything to emptiness.
    """

    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        _ = self.ds[idx]
        return [0]


def iter_benchmark(
    iterator, num_iter: int, warmup: int = 5, max_time_seconds: float = 60
) -> Tuple[float, List[float]]:
    """
    Benchmark an iterator/iterable for `num_iter` iterations with an extra
    `warmup` iterations of warmup.
    End early if `max_time_seconds` time is spent on iterations.

    Returns:
        float: average time (seconds) per iteration
        list[float]: time spent on each iteration. Sometimes useful for further analysis.
    """
    num_iter, warmup = int(num_iter), int(warmup)

    iterator = iter(iterator)
    for _ in range(warmup):
        next(iterator)
    timer = Timer()
    all_times = []
    for curr_iter in tqdm.trange(num_iter):
        start = timer.seconds()
        if start > max_time_seconds:
            num_iter = curr_iter
            break
        next(iterator)
        all_times.append(timer.seconds() - start)
    avg = timer.seconds() / num_iter
    return avg, all_times


class DataLoaderBenchmark:
    """
    Some common benchmarks that help understand perf bottleneck of a standard dataloader
    made of dataset, mapper and sampler.
    """

    def __init__(
        self,
        dataset,
        *,
        mapper,
        sampler=None,
        total_batch_size,
        num_workers=0,
        max_time_seconds: int = 90,
    ):
        """
        Args:
            max_time_seconds (int): maximum time to spent for each benchmark
            other args: same as in `build.py:build_detection_train_loader`
        """
        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False, serialize=True)
        if sampler is None:
            sampler = TrainingSampler(len(dataset))

        self.dataset = dataset
        self.mapper = mapper
        self.sampler = sampler
        self.total_batch_size = total_batch_size
        self.num_workers = num_workers
        self.per_gpu_batch_size = self.total_batch_size // comm.get_world_size()

        self.max_time_seconds = max_time_seconds

    def _be
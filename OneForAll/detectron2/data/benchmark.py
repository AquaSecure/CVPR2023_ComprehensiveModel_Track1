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

    def _benchmark(self, iterator, num_iter, warmup, msg=None):
        avg, all_times = iter_benchmark(iterator, num_iter, warmup, self.max_time_seconds)
        if msg is not None:
            self._log_time(msg, avg, all_times)
        return avg, all_times

    def _log_time(self, msg, avg, all_times, distributed=False):
        percentiles = [np.percentile(all_times, k, interpolation="nearest") for k in [1, 5, 95, 99]]
        if not distributed:
            logger.info(
                "{msg}: avg={1.0/avg:.1f} it/s, "
                "p1={percentiles[0]:.2g}s, p5={percentiles[1]:.2g}s, "
                "p95={percentiles[2]:.2g}s, p99={percentiles[3]:.2g}s."
            )
            return
        avg_per_gpu = comm.all_gather(avg)
        percentiles_per_gpu = comm.all_gather(percentiles)
        if comm.get_rank() > 0:
            return
        for idx, avg, percentiles in zip(count(), avg_per_gpu, percentiles_per_gpu):
            logger.info(
                "GPU{idx} {msg}: avg={1.0/avg:.1f} it/s, "
                "p1={percentiles[0]:.2g}s, p5={percentiles[1]:.2g}s, "
                "p95={percentiles[2]:.2g}s, p99={percentiles[3]:.2g}s."
            )

    def benchmark_dataset(self, num_iter, warmup=5):
        """
        Benchmark the speed of taking raw samples from the dataset.
        """

        def loader():
            while True:
                for k in self.sampler:
                    yield self.dataset[k]

        self._benchmark(loader(), num_iter, warmup, "Dataset Alone")

    def benchmark_mapper(self, num_iter, warmup=5):
        """
        Benchmark the speed of taking raw samples from the dataset and map
        them in a single process.
        """

        def loader():
            while True:
                for k in self.sampler:
                    yield self.mapper(self.dataset[k])

        self._benchmark(loader(), num_iter, warmup, "Single Process Mapper (sec/sample)")

    def benchmark_workers(self, num_iter, warmup=10):
        """
        Benchmark the dataloader by tuning num_workers to [0, 1, self.num_workers].
        """
        candidates = [0, 1]
        if self.num_workers not in candidates:
            candidates.append(self.num_workers)

        dataset = MapDataset(self.dataset, self.mapper)
        for n i
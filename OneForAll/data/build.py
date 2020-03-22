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

        if self.cfg.sample_mode == 'batch':
            for name, iter_ in self.task_iters.items():
                batch[name] = next(iter_)
        elif self.cfg.sample_mode == 'sample':
            name = random.choices(self.task_iters.keys(), self.cfg.sample_prob)[0]
            batch[name] = next(self.task_iters[name])
        else:
            raise NotImplementedError

        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        """shutdown
        """
        for name, loader in self.task_loaders:
            loader.shutdown()


class MOEplusplusMultiTaskDataLoader(MultiTaskDataLoader):
    """MOEplusplusMultiTaskDataLoader
    """
    def __init__(self, task_loaders, cfg):
        globalrank2taskid = cfg['globalrank2taskid']
        rank = comm.get_rank()
        taskid = globalrank2taskid[str(rank)]
        taskname = list(task_loaders.keys())[taskid]
        # task_loader = {taskname: task_loaders[taskname]}
        print('rank {} has task_loader of {}'.format(rank, taskname))
        # super().__init__(task_loader, cfg)

        self.task_loaders = task_loaders
        self.cfg = cfg

        self.task_iters = {taskname: iter(task_loaders[taskname])}
        # for name, loader in self.task_loaders.items():
        #     self.task_iters[name] = iter(loader)


def build_reid_test_loader_lazy(test_set, test_batch_size, num_workers, dp_degree=None, alive_rank_list=None):
    """
    build reid test_loader for tasks of Person, Veri and Sop
    this test_loader only supports single gpu
    """
    if dp_degree is not None:
        if (alive_rank_list is not None) and (comm.get_rank() not in alive_rank_list):
            return {}
        dp_group = None
        moe_group = None
    else:
        dp_group = None
        moe_group = None
    mini_batch_size = test_batch_size // comm.get_world_size(dp_group)
    data_sampler = samplers.InferenceSampler(test_set, dp_group=dp_group, moe_group=moe_group)
    batch_sampler = paddle.io.BatchSampler(sampler=data_sampler, batch_size=mini_batch_size)
    test_loader = paddle.io.DataLoader(
        dataset=test_set,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        num_workers=0,
        )
    return test_loader


def build_test_set(
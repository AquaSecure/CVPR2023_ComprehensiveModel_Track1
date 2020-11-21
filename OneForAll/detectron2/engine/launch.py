# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull
"""paddle authors
"""
import logging
from collections import Counter

from detectron2.engine.train_loop import HookBase


class LRScheduler(HookBase):
    """
    A hook which executes a builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (optim.Optimizer):
            scheduler (optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
       
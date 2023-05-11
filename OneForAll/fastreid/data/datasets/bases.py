# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import copy
import logging
import os
import paddle
from typing import Sequence

from tabulate import tabulate
from termcolor import colored
import numpy as np
import random

logger = logging.getLogger(__name__)


class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list or Callable): contains tuples of (img_path(s), pid, camid).
        query (list or Callable): contains tuples of (img_path(s), pid, camid).
        gallery (list or Callable): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self._train = train
        self._query = query
        self._gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

    @property
    def train(self):
        """train
        """
        if callable(self._train):
            self._train = self._train()
        return self._train

    @property
    def query(self):
        """query
        """
        if callable(self._query):
            self._query = self._query()
        return self._query

    @property
    def gallery(self):
        """gallery
        """
        if callable(self._gallery):
            self._gallery = self._gallery()
        return self._gallery

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):
        """Supports s
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

    
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

from data.samplers
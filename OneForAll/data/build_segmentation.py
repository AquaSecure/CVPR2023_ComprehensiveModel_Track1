# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import paddle
import paddleseg

from utils import comm
import numpy as np
from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from data.datasets.cityscapes_datasets import *
from data.datasets.bdd100k_datasets import *


def build_segmentation_dataset(dataset_name=None, transforms=[], dataset_root=None, 
        mode='train', **kwargs):
    """
    Build Citysca
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest import result
from PIL import Image
import numpy as np
import time
import paddle
import paddle.nn.functional as F
import json

from utils import comm
import collections.abc
import cv2
from tqdm import tqdm 

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer


def seg_inference_on_dataset(model,
             data_loader,
             evaluate,
             aug_eval=F
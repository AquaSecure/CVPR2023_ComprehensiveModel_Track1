# !/usr/bin/env python3
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
import logging
import numpy as np

from data.transforms.ops.fmix import sample_mask

logger = logging.getLogger(__name__)

class BatchOperator(object):
    """ BatchOperator """

    def __init__(self, *args, **kwargs):
        pass

    def _unpack(self, batch):
        """ _unpack """
        assert isinstance(batch, list), \
                'batch should be a list filled with tuples (img, label)'
        bs = len(batch)
        assert bs > 0, 'size of the batch data should > 0'
        #imgs, labels = list(zip(*batch))
        imgs = []
        labels = []
        for item in batch:
            imgs.append(item[0])
            labels.append(item[1])
        return np.array(imgs), np.array(labels), bs

    def _one_hot(self, targets):
        return np.eye(self.class_num, dtype="float32")[targets]

    def _mix_target(self, targets0, targets1, lam):
        one_hots0 = self._one_hot(targets0)
        one_hots1 = self._one_hot(targets1)
        return one_hots0 * lam + one_hots1 * (1 - lam)

    def __call__(self, batch):
        return batch


class MixupOperator(BatchOperator):
    """ Mixup operator 
    reference: https://arxiv.org/abs/1710.09412

    """

    def __init__(self
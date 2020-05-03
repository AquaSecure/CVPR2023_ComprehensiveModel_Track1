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

    def __init__(self, class_num, alpha: float=1.):
        """Build Mixup operator

        Args:
            alpha (float, optional): The parameter alpha of mixup. Defaults to 1..

        Raises:
            Exception: The value of parameter is illegal.
        """
        if alpha <= 0:
            raise Exception(
                f"Parameter \"alpha\" of Mixup should be greater than 0. \"alpha\": {alpha}."
            )
        if not class_num:
            msg = "Please set \"Arch.class_num\" in config if use \"MixupOperator\"."
            logger.error(Exception(msg))
            raise Exception(msg)

        self._alpha = alpha
        self.class_num = class_num

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class CutmixOperator(BatchOperator):
    """ Cutmix operator
    reference: https://arxiv.org/abs/1905.04899

    """

    def __init__(self, class_num, alpha=0.2):
        """Build Cutmix operator

        Args:
            alpha (float, optional): The parameter alpha of cutmix. Defaults to 0.2.

        Raises:
            Exception: The value of parameter is illegal.
        """
        if alpha <= 0:
            raise Exception(
                f"Parameter \"alpha\" of Cutmix should be greater than 0. \"alpha\": {alpha}."
            )
        if not class_num:
            msg = "Please set \"Arch.class_num\" in config if use \"CutmixOperator\"."
            logger.error(Exception(msg))
            raise Exception(msg)

        self._alpha = alpha
        self.class_num = class_num

    def _rand_bbox(self, size, lam):
        """ _rand_bbox """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1
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
import sys
if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np
from paddle.io import DistributedBatchSampler
    
from ppdet.utils.logger import setup_logger
from collections import Counter

logger = setup_logger('reader')
MAIN_PID = os.getpid()


class VehicleMultiTaskClassAwareSampler(DistributedBatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        super(VehicleMultiTaskClassAwareSampler, self).__init__(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size
        self.category_imgids = self._classaware_sampler(dataset.img_items)

        # counter = [0 for _ in range(len(self.category_imgids))]
        # for i in range(len(self.category_imgids)):
        #     counter += len(self.category_imgids[i])
        #
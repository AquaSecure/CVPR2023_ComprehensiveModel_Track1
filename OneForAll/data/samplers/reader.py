import os
import traceback
import six
import sys
if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np
from collections import defaultdict

from paddle.io import DataLoader, DistributedBatchSampler
from paddle.fluid.dataloader.collate import default_collate_fn

from ppdet.core.workspace import register
from .. import transforms as transform
    
from ppdet.utils.logger import setup_logger
from collections import Counter
logger = setup_logger('reader')

MAIN_PID = os.getpid()


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        self.transforms_det = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)
                if k not in ['RandomErasing', 'TimmAutoAugment']:
                    self.transforms_det.append(f)

    def __call__(self, data):
        if 'is_cls' in data and data['is_cls']:
            for f in self.transforms_cls:
                # skip TimmAutoAugment for vehicle color data
                if type(f).__name__ == "TimmAutoAugment" and data['cls_label_1'] == -1:
                    continue
                try:
                    data = f(data)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logger.warning("fail to map sample transform [{}] "
                                "with error: {} and stack:\n{}".format(
                                    f, e, str(stack_info)))
                    raise e
        else:
            for f in self.transforms_det:
                try:
                    data = f(data)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logger.warning("fail to map sample transform [{}] "
                                "with error: {} and stack:\n{}".format(
                                    f, e, str(stack_info)))
                    raise e
        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
    
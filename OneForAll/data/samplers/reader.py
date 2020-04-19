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

    def __c
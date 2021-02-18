"""evaluation.evaluator
"""
import logging
import time
import datetime
from contextlib import contextmanager

import paddle
from utils import comm
from utils.logger import log_every_n_seconds

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by 
# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import itertools
import logging
from collections import OrderedDict
import os

import paddle

from utils import comm
from evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)

class ClasEvaluatorSingleTask(DatasetEvaluator):
    """ClasEvaluatorSingleTask
    """
    def __init__(self, cfg, output_dir=None, num_valid_samples=None, **kwargs):
        self.cfg = cfg
        self._output_dir = output_dir

        self._predictions = []
        self.topk = (1,)
        self._num_valid_samples = num_valid_samples

    def reset(self):
        """reset
        """
        self._predictions = []

    def process(self, inputs, outputs):
        """process
        """
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'
        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        pred_logits = outputs
        labels = inputs["targets"]

        with paddle.no_grad():
            maxk = max(self.topk)
            batch_size = labels.shape[0]
            _, pred = pred_logits.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.equal(labels.reshape((1, -1)).expand_as(pred))
            
            k = 1 #to
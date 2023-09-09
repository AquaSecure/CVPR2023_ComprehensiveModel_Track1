# encoding: utf-8
import paddle
import paddle.nn.functional as F

from detectron2.utils.events import EventStorage, get_event_storage
from utils import comm


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.shape[0]
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.equal(gt_classes.reshape((1, -1)).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = paddle.cast(correct[:k].reshape((-1,)), 'float32').sum(axis=0, keepdim=True)
        ret.append(correct_k * (1. / bsz))

    if comm.is_main_process():
        storage = get_event_storage()
        storage.put_scalar("cls_accuracy", ret[0])
    return ret[0]


def cross_en
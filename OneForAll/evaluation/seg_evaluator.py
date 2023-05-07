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
             aug_eval=False,
             scales=1.0,
             flip_horizontal=False,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             precision='fp32',
             amp_level='O1',
             print_detail=True,
             auc_roc=False):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    
    if print_detail: #and hasattr(data_loader, 'dataset'):
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(list(data_loader.task_loaders.values())[0].dataset), len(data_loader)))

    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    intersect_area_all = paddle.zeros([1], dtype='int64')
    pred_area_all = paddle.zeros([1], dtype='int64')
    label_area_all = paddle.zeros([1], dtype='int64')
    logits_all = None
    label_all = None


    with paddle.no_grad():
        for iter, data in enumerate(data_loader):
            label = data['segmentation']['label'].astype('int64')
            trans_info = data['segmentation']['trans_info']
            if aug_eval:
                pred, logits = aug_inference(
                    model,
                    data,
                    trans_info=trans_info,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, logits = inference(
                    model,
                    data,
                    trans_info=trans_info,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            
            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                19,
                ignore_index=list(data_loader.task_loaders.values())[0].dataset.ignore_index)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)
                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(list(data_loader.task_loaders.values())[0].dataset):
                    valid = len(list(data_loader.task_loaders.values())[0].dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area

            if auc_roc:
                logits = F.softmax(logits, axis=1)
                if logits_all is None:
                    logits_all = logits.numpy()
                    label_all = label.numpy()
                else:
                    logits_all = np.concatenate(
                        [logits_all, logits.numpy()])  # (KN, C, H, W)
                    label_all = np.concatenate([label_all, label.numpy()])
            
            if comm.get_world_size() > 1:
                comm.synchronize()
            time.sleep(0.01)

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = metrics.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics.class_measurement(
        *metrics_input)
    kappa = metrics.kappa(*metrics_input)
    class_dice, mdice = metrics.dice(*metrics_input)
    
    model.train()
    
    if auc_roc:
        auc_roc = metrics.auc_roc(
            logits_all, label_all, num_classes=19)
        auc_infor = ' Auc_roc: {:.4f}'.format(auc_roc)

    if print_detail:
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
            len(list(data_loader.task_loaders.values())[0].dataset), miou, acc, kappa, mdice)
        infor = infor + auc_infor if auc_roc else infor
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Precision: \n" + str(
            np.round(class_precision, 4)))
        logger.info("[EVAL] Class Recall: \n" + str(np.round(class_recall, 4)))
    
    result = {}
    result['miou'] = miou

    return result

import copy
def mask2polygon(mask_image):

    """
    :param mask_image: 输入mask图片地址, 默认为gray, 且像素值为0或255
    :return: list, 每个item为一个labelme的points
    """
    cls_2_polygon = {}
    for i in range(19):
        mask = copy.deepcopy(mask_image)
        mask[mask != i] = 0
        mask[mask == i] = 1
        mask.astype('uint8')
 
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        results = [item.squeeze().tolist() for item in contours]
        cls_2_polygon[i] = results

    return cls_2_polygon  #results


def seg_inference_on_test_dataset(model,
             data_loader,
             evaluate,
             aug_eval=False,
             scales=1.0,
             flip_horizontal=False,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             precision='fp32',
             amp_level='O1',
             print_detail=True,
             auc_roc=False):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    
    if print_detail: #and hasattr(data_loader, 'dataset'):
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(list(data_loader.task_loaders.values())[0].dataset), len(data_loader)))

    model.eval()

    pred_res = []
    with paddle.no_grad():
        for iter, data in enumerate(tqdm(data_loader, mininterval=10)):
            trans_info = data['segmentation']['trans_info']
            img_path = data['segmentation']['im_path'][0]
            im_id = data['segmentation']['im_id'][0]
            id2path = data['segmentation']['id2path']
            # imgname = os.path.splitext(os.path.basename(img_path))[0] + '.png'
            if aug_eval:
                pred, _ = aug_inference(
                    model,
                    data,
                    trans_info=trans_info,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, _ = inference(
                    model,
                    data,
                    trans_info=trans_info,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)

            results = []
            results_id = []
            paddle.distributed.all_gather(results, pred)
            paddle.distributed.all_gather(results_id, im_id)
            if not comm.is_main_process():
                continue
            for k, result in enumerate(results):                               
            # pred_img = pred.numpy().squeeze(0).transpose(1,2,0).astype(np.uint8)
            # cv2.imwrite(save_path + '/' + imgname, pred_img) 
                res = mask2polygon(result.numpy().squeeze(0).squeeze(0).astype(np.uint8))
                tmp = dict()
                id = results_id[k].numpy()[0]
                imgname = os.path.splitext(os.path.basename(id2path[0][id][0]))[0] + '.png'
                tmp[imgname] = res
                pred_res.append(tmp)
    if not comm.is_main_process():
        return {}
    return {'seg': pred_res}


def inference(model,
              im,
              trans_info=None,
              is_slide=False,
              stride=None,
              crop_size=None):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Image shape informating changed process. Default: None.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im = im.transpose((0, 2, 3, 1))
    if not is_slide:
        logits = model(im)
        logits = list(logits.values())[0]
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
      
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
import numpy as np


def bbox_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)


def bbox_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * paddle.log(tgt_w / src_w)
    dh = wh * paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1)
    return deltas


def delta2bbox(deltas, boxes, weights):
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into paddle.exp()
    dw = paddle.clip(dw, max=clip_scale)
    dh = paddle.clip(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = paddle.exp(dw) * widths.unsqueeze(1)
    pred_h = paddle.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = paddle.stack(pred_boxes, axis=-1)

    return pred_boxes


def expand_bbox(bboxes, scale):
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    bboxes_exp = np.zeros(bboxes.shape, dtype=np.float32)
    bboxes_exp[:, 0] = x_c - w_half
    bboxes_exp[:, 2] = x_c + w_half
    bboxes_exp[:, 1] = y_c - h_half
    bboxes_exp[:, 3] = y_c + h_half

    return bboxes_exp


def clip_bbox(boxes, im_shape):
    h, w = im_shape[0], im_shape[1]
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    return paddle.stack([x1, y1, x2, y2], axis=1)


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = paddle.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    keep = paddle.nonzero(mask).flatten()
    return keep


def bbox_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def bbox_overlaps(boxes1, boxes2):
    """
    Calculate overlaps between boxes1 and boxes2

    Args:
        boxes1 (Tensor): boxes with shape [M, 4]
        boxes2 (Tensor): boxes with shape [N, 4]

    Return:
        overlaps (Tensor): overlaps between boxes1 and boxes2 with shape [M, N]
    """
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    if M * N == 0:
        return paddle.zeros([M, N], dtype='float32')
    area1 = bbox_area(boxes1)
    area2 = bbox_area(boxes2)

    xy_max = paddle.minimum(
        paddle.unsqueeze(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = paddle.maximum(
        paddle.unsqueeze(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(min=0)
    inter = width_height.prod(axis=2)

    overlaps = paddle.where(inter > 0, inter /
                            (paddle.unsqueeze(area1, 1) + area2 - inter),
                            paddle.zeros_like(inter))
    return overlaps


def batch_bbox_overlaps(bboxes1,
                        bboxes2,
                        mode='iou',
                        is_aligned=False,
                        eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ['iou', 'iof', 'giou'], 'Unsupported mode {}'.format(mode)
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2] if bboxes1.shape[0] > 0 else 0
    cols = bboxes2.shape[-2] if bboxes2.shape[0] > 0 else 0
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return paddle.full(batch_shape + (rows, ), 1)
        else:
            return paddle.full(batch_shape + (rows, cols), 1)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    if is_aligned:
        lt = paddle.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [B, rows, 2]
        rb = paddle.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [B, rows, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, 2]
        overlap = wh[:, 0] * wh[:, 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = paddle.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    else:
        lt = paddle.maximum(bboxes1[:, :2].reshape([rows, 1, 2]),
                            bboxes2[:, :2])  # [B, rows, cols, 2]
        rb = paddle.minimum(bboxes1[:, 2:].reshape([rows, 1, 2]),
                            bboxes2[:, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]

        if mode in ['iou', 'giou']:
            union = area1.reshape([rows,1]) \
                    + area2.reshape([1,cols]) - overlap
        else:
            union = area1[:, None]
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[:, :2].reshape([rows, 1, 2]),
                                         bboxes2[:, :2])
            enclosed_rb = paddle.maximum(bboxes1[:, 2:].reshape([rows, 1, 2]),
                                         bboxes2[:, 2:])

    eps = paddle.to_tensor([eps])
    union = paddle.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
    enclose_area = paddle.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return 1 - gious


def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return [x1, y1, x2, y2]


def make_grid(h, w, dtype):
    yv, xv = paddle.meshgrid([paddle.arange(h), paddle.arange(w)])
    return paddle.stack((xv, yv), 2).cast(dtype=dtype)


def decode_yolo(box, anchor, downsample_ratio):
    """decode yolo box

    Args:
        box (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        anchor (list): anchor with the shape [na, 2]
        downsample_ratio (int): downsample ratio, default 32
        scale (float): scale, default 1.

    Return:
        box (list): decoded box, [x, y, w, h], all have the shape [b, na, h, w, 1]
    """
    x, y, w, h = box
    na, grid_h, grid_w = x.shape[1:4]
    grid = make_grid(grid_h, grid_w, x.dtype).reshape((1, 1, grid_h, grid_w, 2))
    x1 = (x + grid[:, :, :, :, 0:1]) / grid_w
    y1 = (y + grid[:, :, :, :, 1:2]) / grid_h

    anchor = paddle.to_tensor(anchor)
    anchor = paddle.cast(anchor, x.dtype)
    anchor = anchor.reshape((1, na, 1, 1, 2))
    w1 = paddle.exp(w) * anchor[:, :, :, :, 0:1] / (downsample_ratio * grid_w)
    h1 = paddle.exp(h) * anchor[:, :, :, :, 1:2] / (downsample_ratio * grid_h)

    return [x1, y1, w1, h1]


def iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = paddle.maximum(px1y1, gx1y1)
    x2y2 = paddle.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2
    x1 = paddle.maximum(px1, gx1)
    y1 = paddle.maximum(py1, gy1)
    x2 = paddle.minimum(px2, gx2)
    y2 = paddle.minimum(py2, gy2)

    overlap = ((x2 - x1).clip(0)) * ((y2 - y1).clip(0))

    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clip(0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clip(0)

    union = area1 + area2 - overlap + eps
    iou = overlap / union

    if giou or ciou or diou:
        # convex w, h
        cw = paddle.maximum(px2, gx2) - paddle.minimum(px1, gx1)
        ch = paddle.maximum(py2, gy2) - paddle.minimum(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw**2 + ch**2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2)**2 + (py1 + py2 - gy1 - gy2)**2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = paddle.atan(w1 / h1) - paddle.atan(w2 / h2)
                v = (4 / math.pi**2) * paddle.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou


def rect2rbox(bboxes):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    bboxes = bboxes.reshape(-1, 4)
    num_boxes = bboxes.shape[0]

    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = np.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = np.abs(bboxes[:, 3] - bboxes[:, 1])
    angles = np.zeros([num_boxes], dtype=bboxes.dtype)

    inds = edges1 < edges2

    rboxes = np.stack((x_ctr, y_ctr, edges1, edges2, angles), axis=1)
    rboxes[inds, 2] = edges2[inds]
    rboxes[inds, 3] = edges1[inds]
    rboxes[inds, 4] = np.pi / 2.0
    return rboxes


def delta2rbox(rrois,
               deltas,
               means=[0, 0, 0, 0, 0],
               stds=[1, 1, 1, 1, 1],
               wh_ratio_clip=1e-6):
    """
    :param rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param wh_ratio_clip:
    :return:
    """
    means = paddle.to_tensor(means)
    stds = paddle.to_tensor(stds)
    deltas = paddle.reshape(deltas, [-1, deltas.shape[-1]])
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0]
    dy = denorm_deltas[:, 1]
    dw = denorm_deltas[:, 2]
    dh = denorm_deltas[:, 3]
    dangle = denorm_deltas[:, 4]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = paddle.clip(dw, min=-max_ratio, max=max_ratio)
    dh = paddle.clip(dh, min=-max_ratio, max=max_ratio)

    rroi_x = rrois[:, 0]
    rroi_y = rrois[:, 1]
    rroi_w = rrois[:, 2]
    rroi_h = rrois[:, 3]
    rroi_angle = rrois[:, 4]

    gx = dx * rroi_w * paddle.cos(rroi_angle) - dy * rroi_h * paddle.sin(
        rroi_angle) + rroi_x
    gy = dx * rroi_w * paddle.sin(rroi_angle) + dy * rroi_h * paddle.cos(
        rroi_angle) + rroi_y
    gw = rroi_w * dw.exp()
    gh = rroi_h * dh.exp()
    ga = np.pi * dangle + rroi_angle
    ga = (ga + np.pi / 4) % np.pi - np.pi / 4
    ga = paddle.to_tensor(ga)

    gw = paddle.to_tensor(gw, dtype='float32')
    gh = paddle.to_tensor(gh, dtype='float32')
    bboxes = paddle.stack([gx, gy, gw, gh, ga], axis=-1)
    return bboxes


def rbox2delta(proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """

    Args:
        proposals:
        gt:
        means: 1x5
        stds: 1x5

    Returns:

    """
    proposals = proposals.astype(np.float64)

    PI = np.pi

    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    dx = (np.cos(proposals[..., 4]) * coord[..., 0] + np.sin(proposals[..., 4])
          * coord[..., 1]) / proposals_widths
    dy = (-np.sin(proposals[..., 4]) * coord[..., 0] + np.cos(proposals[..., 4])
          * coord[..., 1]) / proposals_heights
    dw = np.log(gt_widths / proposals_widths)
    dh = np.log(gt_heights / proposals_heights)
    da = (gt_angle - proposals_angle)

    da = (da + PI / 4) % PI - PI / 4
    da /= PI

    deltas = np.stack([dx, dy, dw, dh, da], axis=-1)
    means = np.array(means, dtype=deltas.dtype)
    stds = np.array(stds, dtype=deltas.dtype)
    deltas = (deltas - means) / stds
    deltas = deltas.astype(np.float32)
    return deltas


def bbox_decode(bbox_preds,
                anchors,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1]):
    """decode bbox from deltas
    Args:
        bbox_preds: [N,H,W,5]
        anchors: [H*W,5]
    return:
        bboxes: [N,H,W,5]
    """
    means = paddle.to_tensor(means)
    stds = paddle.to_tensor(stds)
    num_imgs, H, W, _ = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred
        anchors = paddle.to_tensor(anchors)
        bboxes = delta2rbox(
            anchors, bbox_delta, means, stds, wh_ratio_clip=1e-6)
        bboxes = paddle.reshape(bboxes, [H, W, 5])
        bboxes_list.append(bboxes)
    return paddle.stack(bboxes_list, axis=0)


def poly2rbox(polys):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_boxes:[x_ctr,y_ctr,w,h,angle]
    """
    rotated_boxes = []
    for poly in polys:
        poly = np.array(poly[:8], dtype=np.float32)

        pt1 = (poly[0], poly[1])
        pt2 = (poly[2], poly[3])
        pt3 = (poly[4], poly[5])
        pt4 = (poly[6], poly[7])

        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[
            1]) * (pt1[1] - pt2[1]))
        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[
            1]) * (pt2[1] - pt3[1]))

        width = max(edge1, edge2)
        height = min(edge1, edge2)

        rbox_angle = 0
        if edge1 > edge2:
            rbox_angle = np.arctan2(
                float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
        elif edge2 >= edge1:
            rbox_angle = np.arctan2(
                float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))

        def norm_angle(angle, range=[-np.pi / 4, np.pi]):
            return (angle - range[0]) % range[1] + range[0]

        rbox_angle = norm_angle(rbox_angle)

        x_ctr = float(pt1[0] + pt3[0]) / 2
        y_ctr = float(pt1[1] + pt3[1]) / 2
        rotated_box = np.array([x_ctr, y_ctr, width, height, rbox_angle])
        rotated_boxes.append(rotated_box)
    ret_rotated_boxes = np.array(rotated_boxes)
    assert ret_rotated_boxes.shape[1] == 5
    return ret_rotated_boxes


def cal_line_length(point1, point2):
    import math
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
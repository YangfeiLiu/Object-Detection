import torch
import torch.nn as nn
import numpy as np
import math
from utils.util import bbox_iou


def get_target(target, scaled_anchors, out_w, out_h, num_classes, ignore_threshold):
    '''该函数已经查验，逻辑以及实现没有问题'''
    bs = target.size(0)
    num_anchors = len(scaled_anchors)
    '''将feature map划分为hxw个格子，每个格子预测self.num_anchors个框，如果格子中有目标中心点，则负责预测该目标，'''
    mask = torch.zeros(bs, num_anchors, out_h, out_w, requires_grad=False)
    noobj_mask = torch.ones(bs, num_anchors, out_h, out_w, requires_grad=False)

    tx = torch.zeros(bs, num_anchors, out_h, out_w, requires_grad=False)
    ty = torch.zeros(bs, num_anchors, out_h, out_w, requires_grad=False)
    tw = torch.zeros(bs, num_anchors, out_h, out_w, requires_grad=False)
    th = torch.zeros(bs, num_anchors, out_h, out_w, requires_grad=False)
    tconf = torch.zeros(bs, num_anchors, out_h, out_w, requires_grad=False)
    tconf_cls = torch.zeros(bs, num_anchors, out_h, out_w, num_classes, requires_grad=False)

    for b in range(bs):  # 对于每一张图像
        for t in range(target.shape[1]):  # 对于每一个目标
            if target[b, t].sum() == 0:  # 没有目标
                continue
            '''目标从相对位置转绝对位置'''
            gx = target[b, t, 1] * out_w
            gy = target[b, t, 2] * out_h
            gw = target[b, t, 3] * out_w
            gh = target[b, t, 4] * out_h
            '''目标中心点位置'''
            gi = int(gx)
            gj = int(gy)

            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)  # (1,4)
            anchor_box = torch.FloatTensor(np.concatenate((np.zeros((num_anchors, 2)),
                                                           np.array(scaled_anchors)), axis=1))  # (num_anchors,4)
            anchor_ious = bbox_iou(gt_box, anchor_box, xyxy=False)
            '''iou大于阈值，说明当前的格子有目标，因此noobj_mask置0'''
            noobj_mask[b, anchor_ious > ignore_threshold, gj, gi] = 0
            '''找到最适合当前标签的anchor'''
            best_anchor = np.argmax(anchor_ious)
            '''iou最大的anchor负责当前目标的边框回归，将当前格子mask置为1'''
            mask[b, best_anchor, gj, gi] = 1
            noobj_mask[b, best_anchor, gj, gi] = 0
            '''需要回归的中心点的偏移量'''
            tx[b, best_anchor, gj, gi] = gx - gi
            ty[b, best_anchor, gj, gi] = gy - gj
            '''需要回归的边框宽高的偏移量'''
            tw[b, best_anchor, gj, gi] = math.log(gw / scaled_anchors[best_anchor][0] + 1e-16)
            th[b, best_anchor, gj, gi] = math.log(gh / scaled_anchors[best_anchor][1] + 1e-16)
            ''''''
            tconf[b, best_anchor, gj, gi] = 1
            tconf_cls[b, best_anchor, gj, gi, int(target[b, t, 0])] = 1
    return mask, noobj_mask, tx, ty, tw, th, tconf, tconf_cls


def compute_loss(prediction, targets, anchors, img_size, lambda_coord=5., lambda_noobj=0.5, ignore_threshold=0.5):
    out_h, out_w = prediction.size(2), prediction.size(3)
    num_classes = prediction.size(-1) - 5
    stride_h, stride_w = img_size[1] / out_h, img_size[0] / out_w
    scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]
    mask, noobj_mask, tx, ty, tw, th, tconf, tconf_cls = get_target(targets, scaled_anchors, out_w, out_h,
                                                                    num_classes=num_classes,
                                                                    ignore_threshold=ignore_threshold)
    device = prediction.device
    mse_loss = nn.MSELoss().to(device)
    bce_loss = nn.BCELoss().to(device)

    '''获得预测结果'''
    x = prediction[..., 0]
    y = prediction[..., 1]
    w = prediction[..., 2]
    h = prediction[..., 3]
    conf = torch.sigmoid(prediction[..., 4])  # 有无目标的置信度
    conf_cls = torch.sigmoid(prediction[..., 5:])  # 目标类别置信度

    mask, noobj_mask = mask.to(device=device), noobj_mask.to(device)
    tx, ty, tw, th = tx.to(device), ty.to(device), tw.to(device), th.to(device)
    tconf, tconf_cls = tconf.to(device), tconf_cls.to(device)
    '''计算损失值'''
    loss_x = bce_loss(x.sigmoid() * mask, tx * mask)
    loss_y = bce_loss(y.sigmoid() * mask, ty * mask)
    loss_w = mse_loss(w * mask, tw * mask)
    loss_h = mse_loss(h * mask, th * mask)
    loss_coord = loss_x + loss_y + loss_w + loss_h

    pxy = prediction[..., :2].sigmoid()
    pwh = prediction[..., 2:4].sigmoid() * scaled_anchors
    pbox = torch.cat((pxy, pwh), dim=1)
    loss_conf = bce_loss(conf * mask, mask) + lambda_noobj * bce_loss(conf * noobj_mask, noobj_mask * 0.)
    loss_cls = bce_loss(conf_cls[mask == 1], tconf_cls[mask == 1])

    loss = loss_coord * lambda_coord + loss_conf + loss_cls
    return loss, loss_coord, loss_conf, loss_cls




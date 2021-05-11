import math
import torch
import numpy as np
from torchvision.ops import nms


def xywh2xyxy(bbox):
    box = bbox.new(bbox.shape)
    box[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    box[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    box[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    box[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return box


def bbox_iou(box1, box2, xyxy=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    """
    :param box1: anchor框（多个框）
    :param box2: 预测框（一个框）
    :param xyxy: 是否通过两个点坐标确定
    :return: 返回两个框的IOU
    """
    if not xyxy:  # 如果不是通过两点确定，那就是通过中心点和宽高确定，需要转化坐标
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    '''确定重叠框的两个坐标'''
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    '''计算重叠面积'''
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    '''计算总面积'''
    w1, h1 = b1_x2 - b1_x1 + 1, b1_y2 - b1_y1 + 1
    w2, h2 = b2_x2 - b2_x1 + 1, b2_y2 - b2_y1 + 1
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union = b1_area + b2_area - inter_area + eps
    '''计算IOU'''
    iou = inter_area / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
            if DIoU:
                diou = iou - rho2 / c2
                return diou
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                    ciou = iou - rho2 / c2 - v * alpha
                    return ciou
        else:
            c_area = cw * ch + eps
            giou = iou - (c_area - union) / c_area
            return giou
    else:
        return iou


def non_max_supperssion(prediction, conf_thres=0.8, nms_thres=0.3):
    '''
    :param prediction: 网络预测结果，Tensor，shape=(b,num_boxes,pre_each_box)  pre_each_box=(x,y,w,h,conf,conf_cls)
    :param conf_thres: 是否有目标的置信率,值越大，过滤的box越多，值越小，保留的box越多
    :param nms_thres: nms阈值，值越大，保留的box越多，值越小，过滤的box越多
    :return:
    '''
    box_corner = prediction.new(prediction.shape)
    '''将x, y, w, h转为x1, y1, x2, y2'''
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_idx, image_pre in enumerate(prediction):
        '''过滤出所有大于阈值的预测'''
        conf_mask = (image_pre[:, 4] >= conf_thres).squeeze()
        image_pre = image_pre[conf_mask]

        if not image_pre.size(0):
            continue
        '''获得分类置信度和类别'''
        class_conf, class_pred = torch.max(image_pre[:, 5:], dim=1, keepdim=True)
        '''获得预测结果'''
        detections = torch.cat((image_pre[:, :5], class_conf.float(), class_pred.float()), dim=1)
        '''获得所有预测的类别'''
        unique_label = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_label = unique_label.cuda()
        for c in unique_label:
            '''获得预测为特定类别的所有结果'''
            detection_class = detections[detections[:, -1] == c]
            '''依据有无目标的置信度降序排列'''
            _, conf_sort_index = torch.sort(detection_class[:, 4], descending=True)
            detection_class = detection_class[conf_sort_index]
            '''开始NMS'''
            max_detections = list()
            while detection_class.size(0):
                max_detections.append(detection_class[0].unsqueeze(0))
                if len(detection_class) == 1:
                    break
                '''计算当前置信率最高的框与其他框的iou'''
                ious = bbox_iou(max_detections[-1], detection_class[1:])
                '''iou大于阈值，说明两个框预测基本一致，选择置信率高的，因此留下小于阈值的框'''
                detection_class = detection_class[1:][ious < nms_thres]
            max_detections = torch.cat(max_detections).data
            output[image_idx] = max_detections if output[image_idx] is None else torch.cat([output[image_idx],
                                                                                            max_detections], dim=0)
    return output


def nms_simple(prediction, conf_thres=0.8, nms_thres=0.3):
    output = [None for _ in range(len(prediction))]
    for image_idx, image_pre in enumerate(prediction):
        '''过滤出所有大于阈值的预测'''
        conf_mask = (image_pre[:, 4] >= conf_thres).squeeze()
        image_pre = image_pre[conf_mask]

        if not image_pre.size(0):
            continue
        '''获得分类置信度和类别'''
        class_conf, class_pred = torch.max(image_pre[:, 5:], dim=1, keepdim=True)
        box = xywh2xyxy(image_pre[:, :4])
        conf = image_pre[:, 4].unsqueeze(-1)
        image_pre = torch.cat((box, conf, class_conf.float(), class_pred.float()), dim=1)
        _, conf_sort_index = torch.sort(image_pre[:, 4], descending=True)
        image_pre = image_pre[conf_sort_index]
        unique_label = class_pred.unique()
        for c in unique_label:
            detection = class_pred == c
            box_c = image_pre[image_pre[:, -1] == c]
            conf_c = conf[detection]
            '''获得预测结果'''
            res = nms(box_c, conf_c, iou_threshold=nms_thres)
            output[image_idx] = image_pre[res]
    return output


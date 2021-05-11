import numpy as np
from utils.util import non_max_supperssion, xywh2xyxy, bbox_iou
import torch
from tqdm import tqdm


def parse_prediction(prediction, img_size, anchors):
    bs, _, out_h, out_w, _ = prediction.size()
    stride_h, stride_w = img_size[1] / out_h, img_size[0] / out_w
    scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]
    num_anchors = len(scaled_anchors)
    num_classes = prediction.size(-1) - 5

    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    conf = torch.sigmoid(prediction[..., 4])  # 有无目标的置信度
    conf_cls = torch.sigmoid(prediction[..., 5:])  # 目标类别置信度

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

    grid_x = torch.linspace(0, out_w - 1, out_w).repeat(out_w, 1).repeat(
        bs * num_anchors, 1, 1).view(x.shape).type(FloatTensor)  # bs * self.num_anchors, out_w, out_w
    grid_y = torch.linspace(0, out_h - 1, out_h).repeat(out_h, 1).t().repeat(
        bs * num_anchors, 1, 1).view(y.shape).type(FloatTensor)
    anchor_w = FloatTensor(np.array(scaled_anchors)).index_select(1, LongTensor([0]))  # 沿列搜索，取出第0列
    anchor_h = FloatTensor(np.array(scaled_anchors)).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, out_h * out_w).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, out_h * out_w).view(h.shape)

    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    # pred_boxes[..., 2] = w.data ** 2 * anchor_w
    # pred_boxes[..., 3] = h.data ** 2 * anchor_h

    _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
    output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                        conf.view(bs, -1, 1),
                        conf_cls.view(bs, -1, num_classes)), dim=-1)
    return output.data


def evaluate(pre_out, targets, img_size, iou_thres):
    targets[..., 1:] = xywh2xyxy(targets[..., 1:])
    targets[..., 1:] *= img_size
    return get_batch_statistics(pre_out, targets, iou_thres)


def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for i, prediction in enumerate(outputs):
        if prediction is None:  # 没有预测出目标
            continue
        prediction = prediction.cpu()
        pre_boxes = prediction[:, :4]
        pre_scores = prediction[:, 4]
        pre_labels = prediction[:, -1]

        tp = np.zeros(pre_boxes.size(0))

        annotations = targets[i]
        '''过滤出填补的框'''
        obj = torch.sum(annotations, dim=1) != 0
        annotations = annotations[obj]
        gt_labels = annotations[:, 0]

        if torch.sum(annotations) != 0:
            detect_boxes = []
            gt_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_lab) in enumerate(zip(pre_boxes, pre_labels)):
                if len(detect_boxes) == len(annotations):
                    break

                if pred_lab not in gt_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), gt_boxes).max(0)
                if iou >= iou_threshold and box_index not in detect_boxes:
                    tp[pred_i] = 1
                    detect_boxes += [box_index]
        batch_metrics.append([tp, pre_scores, pre_labels])
    return batch_metrics


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)

    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            recall = tpc / (n_gt + 1e-16)
            r.append(recall[-1])

            precision = tpc / (tpc + fpc)
            p.append(precision[-1])

            ap.append(compute_ap(recall, precision))
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype("int32")

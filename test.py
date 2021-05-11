import torch
import numpy as np
import cv2
from net import yolov3
from data.yolo_data import TestSet
import torch.utils.data
from tqdm import tqdm
from utils.util import non_max_supperssion, nms_simple
from utils.load_config import load_config
from utils.evaluate import parse_prediction
import os
import torch.nn as nn


def test(common_config, train_config, test_config):
    classes = test_config['classes']
    colores = test_config['colors']
    anchors = common_config['anchors']
    font = cv2.FONT_HERSHEY_SIMPLEX
    root = common_config['img_path']
    result = os.path.join(root, test_config['save_result'])
    os.makedirs(result, exist_ok=True)
    img_size = tuple(test_config['img_size'])
    num_classes = common_config['num_classes']
    conf_thres = test_config['conf_thres']
    nms_thres = test_config['nms_thres']
    ignore_threshold = common_config['ignore_threshold']
    lambda_coord = common_config['lambda_coord']
    lambda_noobj = common_config['lambda_noobj']
    gpu_ids = train_config['gpu_ids']
    n_gpus = len(gpu_ids)
    device = torch.device('cuda:%d' % gpu_ids[0]) if torch.cuda.is_available() else torch.device('cpu')
    net = yolov3.YOLOv3(ins=common_config['img_channels'], num_classes=num_classes)
    if n_gpus > 1:
        net = nn.DataParallel(net, device_ids=gpu_ids)
    net = net.to(device)
    net.load_state_dict(torch.load(test_config['pre_train_model'], map_location=device))
    net.eval()
    dataloader = torch.utils.data.DataLoader(TestSet(root=os.path.join(root, 'test/image'), size=img_size),
                                             batch_size=test_config['batch_size'])

    tqdm_testloader = tqdm(enumerate(dataloader))
    for step, sample in tqdm_testloader:
        image = sample['image'].to(device)
        with torch.no_grad():
            outputs = net(image)
            pred_list = list()
            for i in range(3):
                pred_list.append(parse_prediction(outputs[i], img_size=img_size, anchors=anchors[i]))
            predictions = torch.cat(pred_list, dim=1)
            batch_pred = non_max_supperssion(predictions, conf_thres, nms_thres)
            # batch_pred = nms_simple(predictions, conf_thres, nms_thres)
        origin_imgs = (image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for idx, detection in enumerate(batch_pred):
            ori_h, ori_w = eval(sample['origin_size'][idx])
            origin_img = cv2.resize(origin_imgs[idx], (ori_w, ori_h)).copy()
            # origin_img = origin_imgs[idx]
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
            if detection is not None:
                for x1, y1, x2, y2, conf, conf_cls, cls_pred in detection.cpu().numpy():
                    color = colores[int(cls_pred)]
                    cls = classes[int(cls_pred)]
                    pre_h, pre_w = img_size[0], img_size[1]
                    y1 = int((y1 / pre_h) * ori_h)
                    y1 = 1 if y1 <= 0 else y1
                    x1 = int((x1 / pre_w) * ori_w)
                    x1 = 1 if x1 <= 0 else x1
                    y2 = int((y2 / pre_h) * ori_h)
                    y2 = ori_h - 1 if y2 >= ori_h else y2
                    x2 = int((x2 / pre_w) * ori_w)
                    x2 = ori_w - 1 if x2 >= ori_w else x2
                    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(origin_img, (x1, y1), (x2, y2), color)
                    cv2.putText(origin_img, cls, (x1, y1 - 2), font, 0.5, color, 1)
            cv2.imwrite(os.path.join(result, sample['img_name'][idx]), origin_img)


if __name__ == '__main__':
    config_path = './config/DIOR.yaml'
    config = load_config(config_path)
    test(config['common'], config['train'], config['test'])

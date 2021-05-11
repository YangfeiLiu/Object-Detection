import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.load_config import load_config
import torch.utils.data
from net import yolov3
from data import yolo_data
from tqdm import tqdm
from collections import OrderedDict
import warnings
from loguru import logger
from utils.evaluate import evaluate, ap_per_class, parse_prediction
from utils.util import non_max_supperssion
from utils.loss import compute_loss
import time
warnings.filterwarnings("ignore")


def train(common_config, train_config, test_config):
    root = common_config['img_path']
    max_objects = train_config['max_objects']
    anchors = common_config['anchors']
    num_classes = common_config['num_classes']
    img_size = train_config['img_size']
    class_first = common_config['class_first']
    xyxy = common_config['xyxy']
    label_start = common_config['label_start']
    batch_size = train_config['batch_size']
    gpu_ids = train_config['gpu_ids']
    n_gpus = len(gpu_ids)
    num_workers = train_config['num_workers'] * n_gpus
    epochs = train_config['epoches']
    lr = eval(train_config['lr'])
    evaluate_interval = common_config['evaluate_interval']
    weight_decay = eval(train_config['weight_decay'])
    ignore_threshold = common_config['ignore_threshold']
    conf_thres = test_config['conf_thres']
    nms_thres = test_config['nms_thres']
    lambda_coord = common_config['lambda_coord']
    lambda_noobj = common_config['lambda_noobj']
    save_model_path = train_config['save_model_path']
    save_metric = float("-inf")

    device = torch.device('cuda:%d' % gpu_ids[0]) if torch.cuda.is_available() else torch.device('cpu')
    net = yolov3.YOLOv3(ins=common_config['img_channels'], num_classes=num_classes)
    if n_gpus > 1:
        net = nn.DataParallel(net, device_ids=gpu_ids)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    if train_config['pre_train']:
        pre_train_state = torch.load(test_config['pre_train_model'], map_location=device)
        net.load_state_dict(pre_train_state['model'])
        optimizer.load_state_dict(pre_train_state['optim'])
        start_epoch = pre_train_state['epoch']
    else:
        start_epoch = 0
    train_dataset = yolo_data.DataSet(root=root, size=img_size, phase='train', max_objects=max_objects, xyxy=xyxy,
                                      class_first=class_first, label_start=label_start)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True)
    valid_dataset = yolo_data.DataSet(root=root, size=img_size, phase='valid', max_objects=max_objects, xyxy=xyxy,
                                      class_first=class_first, label_start=label_start)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True)
    losses_name = ['total_loss', 'coord', 'conf', 'conf_cls']
    for epoch in range(start_epoch, epochs):
        '''训练'''
        train_tqdm_dataloader = tqdm(enumerate(train_dataloader))
        lr = optimizer.param_groups[0]['lr']
        logger.info('Epoch:[%d]/[%d]' % (epoch, epochs))
        net.train()
        train_orderloss = OrderedDict()
        for i, k in enumerate(losses_name):
            train_orderloss[k] = 0.0
        for step, sample in train_tqdm_dataloader:
            train_tqdm_dataloader.set_description('train [%d] batches' % len(train_dataloader))
            images, labels = sample['image'], sample['label']
            label_sum = torch.sum(labels)
            if label_sum == 0:
                continue
            images = images.to(device)
            optimizer.zero_grad()

            outputs = net(images)
            '''计算损失函数'''
            losses = list()
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                loss_item = compute_loss(prediction=outputs[i], targets=labels, anchors=anchors[i], img_size=img_size,
                                         lambda_coord=lambda_coord, lambda_noobj=lambda_noobj,
                                         ignore_threshold=ignore_threshold)
                for k, v in enumerate(loss_item):
                    losses[k].append(v)
            losses = [sum(l) for l in losses]
            for i, k in enumerate(losses_name):
                train_orderloss[k] = (train_orderloss[k] * step + losses[i].item()) / (step + 1)
            train_tqdm_dataloader.set_postfix(train_orderloss)
            loss = losses[0]
            loss.backward()
            optimizer.step()
        logger.info("train loss=%s" % train_orderloss.items())

        if epoch % evaluate_interval == 0 or epoch == epochs - 1:
            '''验证'''
            valid_tqdm_dataloader = tqdm(enumerate(valid_dataloader))
            net.eval()
            valid_orderloss = OrderedDict()
            valid_orderloss['lr'] = lr
            for i, k in enumerate(losses_name):
                valid_orderloss[k] = 0.0
            with torch.no_grad():
                all_labels = []
                sample_metrics = []
                valid_losses = torch.zeros(size=[len(losses_name)])
                for step, sample in valid_tqdm_dataloader:
                    valid_tqdm_dataloader.set_description('evaluate [%d] batches' % len(valid_dataloader))
                    images, labels = sample['image'], sample['label']
                    label_sum = torch.sum(labels)
                    if label_sum == 0:
                        continue
                    obj = torch.sum(labels, dim=2) != 0
                    obj_labels = labels[obj]
                    all_labels += obj_labels[..., 0].tolist()
                    images = images.to(device)

                    outputs = net(images)
                    batch_losses = torch.zeros(size=[len(losses_name)])
                    for i in range(3):
                        loss_item = compute_loss(prediction=outputs[i], targets=labels, anchors=anchors[i],
                                                 img_size=img_size,
                                                 lambda_coord=lambda_coord, lambda_noobj=lambda_noobj,
                                                 ignore_threshold=ignore_threshold)
                        batch_losses += torch.tensor(loss_item)
                    valid_losses = (valid_losses * step + batch_losses) / (step + 1)
                    for i, k in enumerate(losses_name):
                        valid_orderloss[k] = valid_losses[i].item()
                    valid_tqdm_dataloader.set_postfix(valid_orderloss)
                    pred_out = list()
                    for i in range(3):
                        pred_out.append(parse_prediction(prediction=outputs[i], img_size=img_size, anchors=anchors[i]))
                    pred_out = torch.cat(pred_out, dim=1)
                    pred_out = non_max_supperssion(pred_out, conf_thres, nms_thres)

                    sample_metrics += evaluate(pred_out, labels, img_size[0], 0.5)
                if len(sample_metrics) == 0:
                    mean_precision = 0.
                    mean_recall = 0.
                    mean_ap = 0.
                    mean_f1 = 0.
                else:
                    tp, pred_scores, pred_labels = [np.concatenate(x, axis=0) for x in list(zip(*sample_metrics))]
                    precision, recall, ap, f1, ap_class = ap_per_class(tp, pred_scores, pred_labels, all_labels)
                    mean_precision = float(np.mean(precision))
                    mean_recall = float(np.mean(recall))
                    mean_ap = float(np.mean(ap))
                    mean_f1 = float(np.mean(f1))
                logger.info("valid loss=%s" % valid_orderloss.items())
                logger.info("mean_precision=%.6f\t mean_recall=%.6f\t mean_ap=%.6f\t mean_f1=%.6f"
                            % (mean_precision, mean_recall, mean_ap, mean_f1))
                valid_metric = mean_ap - valid_losses[0].item()
                if valid_metric > save_metric:
                    save_metric = valid_metric
                    state = {'model': net.state_dict(),
                             'optim': optimizer.state_dict(),
                             'epoch': epoch}
                    torch.save(state, os.path.join(save_model_path, 'yolov3.pth'))
                    logger.info("the %d epoch saved" % epoch)
        lr_scheduler.step(epoch)


if __name__ == '__main__':
    config_path = './config/DIOR.yaml'
    logger.info("config path: {}", config_path)
    config = load_config(config_path)
    log_path = config['common']['log_path']
    os.makedirs(log_path, exist_ok=True)
    logger.add(os.path.join(log_path, 'train_{time}.log'), format="{time}{level}{message}", level="INFO",
               encoding='utf-8')
    train(config['common'], config['train'], config['test'])

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from data import transforms


class DataSet(Dataset):
    def __init__(self, root, size=(416, 416), phase='train', max_objects=100, xyxy=True, class_first=True, label_start=1):
        self.root = root
        self.size = size
        self.phase = phase
        self.data_list = os.listdir(os.path.join(root, phase, 'image'))
        np.random.shuffle(self.data_list)
        self.max_objects = max_objects
        self.xyxy = xyxy
        self.label_start = label_start
        self.class_first = class_first
        if phase == 'train':
            self.transforms = transforms.Compose([])
            self.transforms.add(transforms.ImageBaseAug())
            self.transforms.add(transforms.ResizeImg(size))
            self.transforms.add(transforms.ToTensor(max_objects))
        if phase == 'valid':
            self.transforms = transforms.Compose([])
            self.transforms.add(transforms.ResizeImg(size))
            self.transforms.add(transforms.ToTensor(max_objects))

    def class_first_to_class_last(self, array):
        return array[:, [1, 2, 3, 4, 0]]

    def xyxy2xywh(self, array, size, class_first):
        '''如果顺序是c,xyxy，则转成xyxy,c'''
        if class_first:
            array = self.class_first_to_class_last(array)
        '''规范不合理的边框'''
        array[:, 0][array[:, 0] <= 0] = 1
        array[:, 1][array[:, 1] <= 0] = 1
        array[:, 2][array[:, 2] >= size[1]] = size[1]
        array[:, 3][array[:, 3] >= size[0]] = size[0]

        center_x = ((array[:, 0] + array[:, 2]) / 2 / size[1]).reshape(-1, 1)
        center_y = ((array[:, 1] + array[:, 3]) / 2 / size[0]).reshape(-1, 1)
        w = ((array[:, 2] - array[:, 0]) / size[1]).reshape(-1, 1)
        h = ((array[:, 3] - array[:, 1]) / size[0]).reshape(-1, 1)
        cls = (array[:, -1] - self.label_start).reshape(-1, 1)
        new_array = np.hstack((cls, center_x, center_y, w, h))
        return new_array

    def __getitem__(self, item):
        img_name = self.data_list[item]
        image_path = os.path.join(self.root, self.phase, 'image', img_name)
        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img.shape[:2]

        lab_name = img_name.replace('jpg', 'txt')
        label_path = os.path.join(self.root, self.phase, 'label', lab_name)
        if os.path.exists(label_path):
            lab = np.loadtxt(label_path).reshape(-1, 5)
            if self.xyxy:
                if len(lab):
                    lab = self.xyxy2xywh(lab, img.shape[:2], class_first=self.class_first)
                else:
                    lab = np.zeros((1, 5), np.float32)
        else:
            lab = np.zeros((1, 5), np.float32)
        sample = dict()
        sample['image'] = img
        sample['label'] = lab
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample['origin_size'] = str([ori_h, ori_w])
        return sample

    def __len__(self):
        return len(self.data_list)


class TestSet(Dataset):
    def __init__(self, root, size):
        self.root = root
        self.size = size
        self.data_list = os.listdir(root)

    def to_tensor(self, img):
        img = img.astype(np.float32)
        img /= 255.
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        return torch.from_numpy(img)

    def resize(self, img):
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, item):
        sample = dict()
        img_name = self.data_list[item]
        image_path = os.path.join(self.root, img_name)
        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sample['img_name'] = img_name
        ori_h, ori_w = img.shape[:2]
        img = self.resize(img)
        img = self.to_tensor(img)
        sample['image'] = img
        sample['origin_size'] = str([ori_h, ori_w])
        return sample

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    root = '/media/hb/1500/LEVIR'
    dataloader = torch.utils.data.DataLoader(DataSet(root), batch_size=2)
    for step, sample in enumerate(dataloader):
        for i, (img, lab) in enumerate(zip(sample['image'], sample['label'])):
            img = img.permute(1, 2, 0).numpy() * 255
            img = img.astype(np.uint8).copy()
            h, w = img.shape[:2]
            for l in lab:
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('/media/lyf/WORK/DATA/NWPU VHR-10 dataset/train/a/' + str(step) + '.jpg', img)


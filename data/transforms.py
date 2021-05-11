import numpy as np
import torch
import cv2
import imgaug as ia
from imgaug import augmenters as iaa


class Compose(object):
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects):
        self.max_objects = max_objects

    def __call__(self, sample):
        img, lab = sample['image'], sample['label']
        img = img.astype(np.float32)
        img /= 255.
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        filled_lab = np.zeros((self.max_objects, 5), np.float32)
        index = np.arange(len(lab))
        np.random.shuffle(index)
        filled_lab[range(len(lab))[:self.max_objects]] = lab[index[:self.max_objects]]
        return {'image': torch.from_numpy(img), 'label': torch.from_numpy(filled_lab)}


class ResizeImg(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, sample):
        img, lab = sample['image'], sample['label']
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        return {'image': img, 'label': lab}


class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([iaa.OneOf([iaa.GaussianBlur((0, 3.0)),
                                              iaa.AverageBlur(k=(2, 7)),
                                              iaa.MedianBlur(k=(3, 11))]),
                                   sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                                   sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
                                   sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                                   sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                                   sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
                                   ], random_order=True)

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['label']
        image = seq_det.augment_images([image])[0]
        return {'image': image, 'label': label}

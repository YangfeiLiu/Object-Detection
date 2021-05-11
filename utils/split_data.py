import os
from shutil import move
import random


def split_train_valid(train_path, valid_path):
    data_list = os.listdir(os.path.join(train_path, 'image'))
    random.shuffle(data_list)
    valid_data = data_list[::5]
    for data in valid_data:
        train_img = os.path.join(train_path, 'image', data)
        train_txt = os.path.join(train_path, 'label', data.replace('jpg', 'txt'))
        valid_img = os.path.join(valid_path, 'image', data)
        valid_txt = os.path.join(valid_path, 'label', data.replace('jpg', 'txt'))
        move(train_img, valid_img)
        move(train_txt, valid_txt)


def move_test(source_path, dest_path):
    labels = os.listdir(source_path)
    test_label = [x for x in labels if int(x.rstrip('.txt')) > 11725]
    for label in test_label:
        move(os.path.join(source_path, label), os.path.join(dest_path, label))


if __name__ == '__main__':
    train_path = '/workspace/lyf/detect/DIOR/train'
    valid_path = '/workspace/lyf/detect/DIOR/valid'
    # move_test('/workspace/lyf/detect/DIOR/train/label/', '/workspace/lyf/detect/DIOR/test/label/')
    split_train_valid(train_path, valid_path)

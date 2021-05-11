import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


def cal_iou(box, cluster):
    '''计算每个box与聚类中心的iou'''
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)
    return iou


def class_first_to_class_last(array):
    return array[:, [1, 2, 3, 4, 0]]


def get_box(path, size=(800, 800)):
    boxes = list()
    for txt in os.listdir(path):
        data = np.loadtxt(os.path.join(path, txt))
        if len(data):
            data = data.reshape(-1, 5)
            data = class_first_to_class_last(data)
            data[:, 0][data[:, 0] <= 0] = 1
            data[:, 1][data[:, 1] <= 0] = 1
            data[:, 2][data[:, 2] >= size[1]] = size[1]
            data[:, 3][data[:, 3] >= size[0]] = size[0]
        else:
            continue
        w = data[:, 2] - data[:, 0]
        h = data[:, 3] - data[:, 1]
        for i in range(data.shape[0]):
            boxes.append([w[i], h[i]])
    return boxes


def kmeans(box, k):
    row = box.shape[0]
    # 初始化距离
    distance = np.zeros(shape=(row, k))
    # 初始化每个box所属anchor
    last_clu = np.zeros(shape=(row, ))
    # 初始化聚类中心
    np.random.seed(9)
    cluster = box[np.random.choice(row, k, replace=False)]
    iter = 1
    while True:
        # 计算每个box与cluster的距离
        for i in range(row):
            distance[i] = 1 - cal_iou(box[i], cluster)
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break
        for j in range(k):
            cluster[j] = np.mean(box[near == j], axis=0)
        last_clu = near
        print("iter:%d, cluster:%s" % (iter, cluster))
        iter += 1
    return cluster


def get_cluster_area(cluster):
    area = [cluster[i, 0] * cluster[i, 1] for i in range(cluster.shape[0])]
    return np.argsort(np.array(area))


if __name__ == '__main__':
    boxes = get_box('/workspace/lyf/detect/DIOR/all_labels/')
    k = 9
    cluster = kmeans(np.array(boxes), k)
    dims = get_cluster_area(cluster)
    cluster_sort = cluster[dims].astype(np.uint8)
    print(cluster_sort)

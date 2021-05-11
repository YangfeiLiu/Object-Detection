import os
from shutil import copy


if __name__ == '__main__':
    root = '/media/hb/1500/NWPU VHR-10 dataset/negative image set/'
    for j, i in enumerate(os.listdir(root)):
        copy(os.path.join(root, i), '/media/hb/1500/NWPU VHR-10 dataset/train/image/%03d.jpg' % (651 + j))
    # for txt in os.listdir(root):
    #     with open(os.path.join(root, txt), 'r') as file:
    #         lines = []
    #         for line in file.readlines():
    #             line = line.rstrip('\n').replace('(', ',').replace(')', ',').replace(',', ' ').strip(' ')
    #             line = line.replace('  ', ' ').replace('  ', ' ')
    #             lines.append(line)
    #         with open(os.path.join(root, txt), 'w') as f:
    #             for line in lines:
    #                 f.write(line + '\n')


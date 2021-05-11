import torch
import torch.nn as nn
from net.backbone.darknet import Darknet
from net.neck.yolov3_neck import Neck
from net.head.yolov3_head import Head


class YOLOv3(nn.Module):
    def __init__(self, ins=3, num_classes=10, blocks=[1, 2, 8, 8, 4], head_ins=[1024, 512, 256]):
        super(YOLOv3, self).__init__()
        self.backbone = Darknet(ins, blocks)
        outs = (5 + num_classes) * 3
        self.neck = Neck(head_ins, outs)
        self.head = list()
        for _ in range(3):
            self.head.append(Head(3, num_classes))

    def forward(self, x):
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        head_out = list()
        for i, x in enumerate(neck_out):
            head_out.append(self.head[i](x))
        return head_out


if __name__ == '__main__':
    x = torch.rand(size=(1, 3, 416, 416))
    net = YOLOv3()
    net.eval()
    y = net(x)
    for yy in y:
        print(yy.size())


import torch
import torch.nn as nn
import math


class Conv2d(nn.Module):
    def __init__(self, ins, outs, ksize, stride):
        super(Conv2d, self).__init__()
        padding = 1 if ksize == 3 else 0
        self.ops = nn.Sequential(nn.Conv2d(ins, outs, kernel_size=ksize, stride=stride, padding=padding),
                                 nn.BatchNorm2d(outs),
                                 nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.ops(x)


class BasicBlock(nn.Module):
    def __init__(self, ins, outs):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(ins, outs, 1, 1)
        self.conv2 = Conv2d(outs, ins, 3, 1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        out = x + y
        return out


class Darknet(nn.Module):
    def __init__(self, ins, blocks):
        super(Darknet, self).__init__()
        self.features = 32
        self.conv1 = Conv2d(ins, self.features, 3, 1)
        self.conv2 = Conv2d(self.features, self.features * 2, 3, 2)

        self.layer1 = self._make_layer_(self.features * 2, blocks[0])
        self.conv3 = Conv2d(self.features * 2, self.features * 4, 3, 2)
        self.layer2 = self._make_layer_(self.features * 4, blocks[1])
        self.conv4 = Conv2d(self.features * 4, self.features * 8, 3, 2)
        self.layer3 = self._make_layer_(self.features * 8, blocks[2])
        self.conv5 = Conv2d(self.features * 8, self.features * 16, 3, 2)
        self.layer4 = self._make_layer_(self.features * 16, blocks[3])
        self.conv6 = Conv2d(self.features * 16, self.features * 32, 3, 2)
        self.layer5 = self._make_layer_(self.features * 32, blocks[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_(self, ins, block):
        layers = list()
        for i in range(block):
            layers.append(BasicBlock(ins, ins // 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.layer1(x)
        x = self.conv3(x)

        x = self.layer2(x)
        x = self.conv4(x)

        out_l = self.layer3(x)
        out_m = self.layer4(self.conv5(out_l))
        out_s = self.layer5(self.conv6(out_m))
        return [out_l, out_m, out_s]

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.darknet import Conv2d


class UpSample(nn.Module):
    def __init__(self, factor):
        super(UpSample, self).__init__()
        self.factor = factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=True)
        return x


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        return torch.cat((x2, x1), dim=1)


class Neck(nn.Module):
    def __init__(self, features_in, features_out):
        super(Neck, self).__init__()
        self.conv_set_s = nn.Sequential(Conv2d(features_in[0], 512, 1, 1),
                                        Conv2d(512, 1024, 3, 1),
                                        Conv2d(1024, 512, 1, 1),
                                        Conv2d(512, 1024, 3, 1),
                                        Conv2d(1024, 512, 1, 1))
        self.conv0_0 = Conv2d(512, 1024, 3, 1)
        self.conv0_1 = Conv2d(1024, features_out, 1, 1)

        self.conv0 = Conv2d(512, 256, 1, 1)
        self.up = UpSample(factor=2)
        self.route = Route()

        self.conv_set_m = nn.Sequential(Conv2d(features_in[1] + 256, 256, 1, 1),
                                        Conv2d(256, 512, 3, 1),
                                        Conv2d(512, 256, 1, 1),
                                        Conv2d(256, 512, 3, 1),
                                        Conv2d(512, 256, 1, 1))
        self.conv1_0 = Conv2d(256, 512, 3, 1)
        self.conv1_1 = Conv2d(512, features_out, 1, 1)
        self.conv1 = Conv2d(256, 128, 1, 1)

        self.conv_set_l = nn.Sequential(Conv2d(features_in[2] + 128, 128, 1, 1),
                                        Conv2d(128, 256, 3, 1),
                                        Conv2d(256, 128, 1, 1),
                                        Conv2d(128, 256, 3, 1),
                                        Conv2d(256, 128, 1, 1))
        self.conv2_0 = Conv2d(128, 256, 3, 1)
        self.conv2_1 = Conv2d(256, features_out, 1, 1)

    def forward(self, backbone_out):
        x_l, x_m, x_s = backbone_out
        x_s = self.conv_set_s(x_s)
        out_s = self.conv0_1(self.conv0_0(x_s))

        x_sm = self.conv0(x_s)
        x_sm = self.up(x_sm)
        x_m = self.route(x_sm, x_m)

        x_m = self.conv_set_m(x_m)
        out_m = self.conv1_1(self.conv1_0(x_m))

        x_ml = self.conv1(x_m)
        x_ml = self.up(x_ml)
        x_l = self.route(x_ml, x_l)

        x_l = self.conv_set_l(x_l)
        out_l = self.conv2_1(self.conv2_0(x_l))
        return out_l, out_m, out_s

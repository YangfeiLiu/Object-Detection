import torch.nn as nn


class Head(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(Head, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # x, y, w, h, conf, conf_cls

    def forward(self, net_out):
        bs, _, out_h, out_w = net_out.size()
        '''(bs, 255, h, w)--->(bs, 3, h, w, 85)'''
        prediction = net_out.view(bs, self.num_anchors, self.bbox_attrs, out_h, out_w).permute(0, 1, 3, 4, 2).contiguous()
        return prediction

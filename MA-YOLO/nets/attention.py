import torch
import torch.nn as nn


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # avg_pool_x -> n,c,1,h
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        # avg_pool_y -> n,c,1,w
        x_w = self.avg_pool_y(x)
        # cat -> n,c,2,w/h
        conv = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        conv_h, conv_w = conv.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(conv_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(conv_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

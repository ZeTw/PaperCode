from collections import OrderedDict

import torch
import torch.nn as nn

from nets.marrNet import marrNet
from nets import attention
from utils.utils import get_classes, get_anchors


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU(inplace=True),
    )


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),

        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained=False, phi=2):
        super(YoloBody, self).__init__()
        self.phi = phi
        self.backbone = marrNet(pretrained=pretrained, phi=self.phi)
        in_filters = [40, 112, 160]

        self.conv1 = make_three_conv([512, 1024], in_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

        if self.phi == 1 or self.phi == 2:
            self.feat1 = attention.CA_Block(160, 16, 16)
            self.feat2 = attention.CA_Block(112, 32, 32)
            self.feat3 = attention.CA_Block(40, 64, 64)
            self.featP5Up = attention.CA_Block(256, 32, 32)
            self.featP4Up = attention.CA_Block(128, 64, 64)

            # self.feat1 = attention.CA_Block(160, 25, 25)
            # self.feat2 = attention.CA_Block(112, 50, 50)
            # self.feat3 = attention.CA_Block(40, 100, 100)
            # self.featP5Up = attention.CA_Block(256, 50,50)
            # self.featP4Up = attention.CA_Block(128, 100, 100)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        if self.phi == 1 or self.phi == 2:
            x0 = self.feat1(x0)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        if self.phi == 1 or self.phi == 2:
            P5_upsample = self.featP5Up(P5_upsample)
        if self.phi == 1 or self.phi == 2:
            x1 = self.feat2(x1)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        if self.phi == 1 or self.phi == 2:
            P4_upsample = self.featP4Up(P4_upsample)
        if self.phi == 1 or self.phi == 2:
            x2 = self.feat3(x2)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], axis=1)
        P5 = self.make_five_conv4(P5)
        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)
        return out0, out1, out2


if __name__ == "__main__":
    from torchstat import stat

    classes_path = '../model_data/voc_classes.txt'

    anchors_path = '../model_data/yolo_anchors.txt'
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YoloBody(anchors_mask, num_classes, pretrained=False)
    # stat(model, (3, 800, 800))
    # stat(model, (3, 512,512))
    # test_tensor = torch.randn((1, 3, 512, 512))
    # out0, out1, out2 = model(test_tensor)
    # print(out2.shape)
    # print(out1.shape)
    # print(out0.shape)

import math

import torch
import torch.nn as nn
from nets import attention


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(

            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # dw
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
            #           bias=False),
            # 将 mobilenetv3 中的dw替换为标准卷积 并且只使用
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            attention.se_block(hidden_dim),

            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


class MarrNet(nn.Module):
    def __init__(self, width_mult=1., phi=0):
        super(MarrNet, self).__init__()
        self.cfgs = [
            # `   k,   t,   c,s
            [3, 1, 16, 1],  # 1
            [3, 4, 24, 2],  # 2
            [3, 3, 24, 1],  # 3
            [5, 3, 40, 2],  # 4
            [5, 3, 40, 1],  # 5
            [5, 3, 40, 1],  # 6
            [3, 6, 80, 2],  # 7
            [3, 2.5, 80, 1],  # 8
            [3, 2.3, 80, 1],  # 9
            [3, 2.3, 80, 1],  # 10
            [3, 6, 112, 1],  # 11
            [3, 6, 112, 1],  # 12
            [5, 6, 160, 2],  # 13
            [5, 6, 160, 1],  # 14
            [5, 6, 160, 1]  # 15
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for k, t, c, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        # self._initialize_weights()
        self.phi = phi
        if self.phi == 2:
            self.feat1 = attention.CA_Block(24, 128, 128)
            self.feat2 = attention.CA_Block(40, 64, 64)
            self.feat3 = attention.CA_Block(80, 32, 32)
            self.feat4 = attention.CA_Block(160, 16, 16)
            # self.feat1 = attention.CA_Block(24, 200, 200)
            # self.feat2 = attention.CA_Block(40, 100, 100)
            # self.feat3 = attention.CA_Block(80, 50, 50)
            # self.feat4 = attention.CA_Block(160, 25, 25)

    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        if self.phi == 2:
            z = self.feat1(x)
        x = self.features[3](x)
        if self.phi == 2:
            x = x + z
        x = self.features[4](x)
        if self.phi == 2:
            z = self.feat2(x)
        x = self.features[5](x)
        if self.phi == 2:
            x = x + z
        out3 = self.features[6](x)
        x = self.features[7](out3)
        if self.phi == 2:
            z = self.feat3(x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        if self.phi == 2:
            x = x + z
        x = self.features[11](x)
        out4 = self.features[12](x)
        x = self.features[13](out4)
        if self.phi == 2:
            z = self.feat4(x)
        x = self.features[14](x)
        if self.phi == 2:
            x = x + z
        out5 = self.features[15](x)
        return out3, out4, out5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def marrNet(pretrained=False, **kwargs):
    model = MarrNet(**kwargs)
    if pretrained:
        state_dict = torch.load('./log/marrNet.pth')
        model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":

    import time

    marrnet = MarrNet(phi=2).cuda()
    with open("marrnet.txt", 'w') as f:
        f.write(str(marrnet))
    exit()

    test_tensor = torch.randn((1, 3, 800, 800)).cuda()
    out = marrnet(test_tensor)
    # print(str(marrnet))
    marrnet(test_tensor)
    # # for i in out:
    # #     print(i.shape)
    start = time.time()
    for i in range(100):
        marrnet(test_tensor)
    end = time.time()
    print((end - start) / 100)

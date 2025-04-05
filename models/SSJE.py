import torch
import torch.nn as nn

from models.MobileNetV2 import ConvBNReLU
from models.MobileNetV2 import mobilenet_v2

FEATS = [16, 24, 32, 96, 320]


class Spectral_Attention_Block(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_dim, stride):
        super(Spectral_Attention_Block, self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.pool_branch = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, stride=stride, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        conv_out = self.conv_branch(x)  # [B, C, H, W]
        pool_out = self.pool_branch(x)  # [B, C, 1, 1]
        pool_out = torch.softmax(pool_out, dim=1)  # [B, C, 1, 1]
        out = conv_out * pool_out
        return self.out_conv(out)


class InvertedResidual_Spec(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_Spec, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = Spectral_Attention_Block(inp, oup, hidden_dim, stride)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_Spec(nn.Module):
    def __init__(self, width_mult=1.0, in_channels=3):
        super(MobileNetV2_Spec, self).__init__()
        block = InvertedResidual_Spec
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # conv1 112*112*16
            [6, 24, 2, 2],  # conv2 56*56*24
            [6, 32, 3, 2],  # conv3 28*28*32
            [6, 64, 4, 2],
            [6, 96, 3, 1],  # conv4 14*14*96
            [6, 160, 3, 2],
            [6, 320, 1, 1],  # conv5 7*7*320
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        features = [ConvBNReLU(in_channels, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        res = []
        for idx, m in enumerate(self.features):
            x = m(x)
            if idx in [1, 3, 6, 13, 17]:
                res.append(x)
        return res


class FusionModule(nn.Module):
    def __init__(self, in_ch):
        super(FusionModule, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU6(inplace=True),
        )
        self.spectral_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x_spa, x_spec):
        spatial_weight = self.spatial_conv(x_spa)
        spectral_weight = torch.softmax(self.spectral_weight(x_spec), dim=1)
        out = spatial_weight * x_spec + spectral_weight * x_spa
        return out


class SSJE(nn.Module):  # Spectral-Spatial Joint Extractor
    def __init__(self, in_channels=32):
        super(SSJE, self).__init__()
        self.spatial_branch = mobilenet_v2(in_channels=in_channels)
        self.spectral_branch = MobileNetV2_Spec(in_channels=in_channels)
        for i in range(len(FEATS)):
            setattr(self, 'fusion_{}'.format(i + 1), FusionModule(FEATS[i]))

    def forward(self, x):
        x_spa = self.spatial_branch(x)
        x_spec = self.spectral_branch(x)
        outs = []
        for i in range(x_spec.__len__()):
            outs.append(getattr(self, 'fusion_{}'.format(i + 1))(x_spa[i], x_spec[i]))
        return outs

import os
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


# 1.使用 nn.AvgPool2d 对输入特征图进行平滑操作，提取其低频信息。
# 2.将原始输入特征图与平滑后的特征图进行相减，得到增强的边缘信息（高频信息）。
# 3.用卷积操作进一步处理增强的边缘信息。
# 4.将处理后的边缘信息与原始输入特征图相加，以形成增强后的输出。
bins = [3,6,9,12]

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class MultiScaleEdgeConv(nn.Module):
    def __init__(self, inc, bins = [3,6,9,12]):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))
    
    
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, padding_mode='zeros', act=True):
        super().__init__()
        self.in_channels = in_channels

        # 如果未指定 out_channels，则保持总通道数不变
        if out_channels is None:
            out_channels = in_channels

        # 检查通道数是否能被3整除，否则动态调整分组策略
        assert in_channels >= 3, f"Input channels must >= 3, got {in_channels}"
        self.split_sizes = self._calculate_split(in_channels)

        # 初始化三个不同尺度的卷积
        self.conv3x3 = nn.Conv2d(
            self.split_sizes[0], self.split_sizes[0],
            kernel_size=3, stride=stride, padding=1,
            padding_mode=padding_mode,
            groups=self.split_sizes[0]
        )
        self.conv5x5 = nn.Conv2d(
            self.split_sizes[1], self.split_sizes[1],
            kernel_size=5, stride=stride, padding=2,
            padding_mode=padding_mode,
            groups=self.split_sizes[1]
        )
        self.conv7x7 = nn.Conv2d(
            self.split_sizes[2], self.split_sizes[2],
            kernel_size=7, stride=stride, padding=3,
            padding_mode=padding_mode,
            groups=self.split_sizes[2]
        )

        # 1x1卷积调整通道数（可选）
        if out_channels != in_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        # 定义BatchNorm层
        self.bn = nn.BatchNorm2d(out_channels)

        # 定义激活函数（默认SiLU，可配置）
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def _calculate_split(self, channels):
        """动态计算分组大小（尽量均分）"""
        base = channels // 3
        remainder = channels % 3
        return [base + (1 if i < remainder else 0) for i in range(3)]

    def forward(self, x):
        # 按通道数分割张量
        x1, x2, x3 = torch.split(x, self.split_sizes, dim=1)

        # 分别通过不同卷积核
        x1 = self.conv3x3(x1)
        x2 = self.conv5x5(x2)
        x3 = self.conv7x7(x3)

        # 拼接、通道投影、归一化与激活
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.proj(out)
        out = self.bn(out)
        out = self.act(out)
        return out

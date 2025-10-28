import torch
import torch.nn as nn

class MultiScaleConv(nn.Module):
    """
    多尺度、Depthwise分支 + 通道拼接 + 1×1通道融合的结构
    Input X (C)
    └─> Channel Split into 3 parts → [C1, C2, C3]
        ├──> Depthwise Conv 3x3 → x1
        ├──> Depthwise Conv 5x5 → x2
        └──> Depthwise Conv 7x7 → x3
    └─> Concat(x1, x2, x3) → Projection (1x1 conv if needed)
    └─> BatchNorm → Activation → Output (out_channels)
    同时提取 不同感受野尺度的特征
    使用了 Depthwise卷积
    支持自定义激活、通道数对齐
    Args:
        nn (_type_): _description_
    """
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
    
if __name__ == '__main__':
    import torch

    model = MultiScaleConv(in_channels=96, out_channels=128).cuda()
    x = torch.randn(1, 96, 256, 256).cuda()
    with torch.no_grad():
        y = model(x)
    print("输出 shape：", y.shape)

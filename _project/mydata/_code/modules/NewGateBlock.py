"""
AdaptiveScaleBlock: 自适应尺度块，专为目标检测优化
核心创新：
1. 自适应多尺度卷积：根据输入特征动态选择最优卷积核组合
2. 多尺度特征融合：多分支并行处理，增强特征表达能力
3. 轻量级通道-空间注意力：智能关注重要特征区域
4. 高效特征选择机制：通过门控单元实现自适应特征筛选
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import trunc_normal_, DropPath


class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class AdaptiveMultiScaleConv(nn.Module):
    """
    自适应多尺度卷积模块
    核心创新：根据输入特征自动选择最优的卷积核组合
    """
    def __init__(self, dim, kernel_sizes=[3, 5, 7], reduction=4):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        
        # 多个不同尺度的深度卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim) 
            for k in kernel_sizes
        ])
        
        # 自适应权重生成网络
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, self.num_kernels, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        # 生成每个卷积核的自适应权重
        weights = self.attention(x)  # [B, num_kernels, 1, 1]
        
        # 多尺度卷积特征提取
        multi_scale_features = []
        for i, conv in enumerate(self.convs):
            feat = conv(x)
            weighted_feat = weights[:, i:i+1] * feat
            multi_scale_features.append(weighted_feat)
        
        # 特征融合
        fused = sum(multi_scale_features)
        output = self.fusion(fused)
        
        return output


class ChannelSpatialAttention(nn.Module):
    """
    轻量级通道-空间注意力模块
    专为检测任务优化，关注重要区域和通道
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),  # 2个通道：max和mean
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        x = x * sa_weight
        
        return x


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块
    集成不同感受野的特征信息
    """
    def __init__(self, dim):
        super().__init__()
        
        # 不同尺度的特征提取分支
        self.branch_1x1 = nn.Conv2d(dim, dim // 4, 1)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.Conv2d(dim // 4, dim // 4, 5, padding=2, groups=dim // 4)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(dim, dim // 4, 1)
        )
        
        # 特征融合和增强
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 多分支特征提取
        branch1 = self.branch_1x1(x)
        branch2 = self.branch_3x3(x)
        branch3 = self.branch_5x5(x)
        branch4 = self.branch_pool(x)
        
        # 特征拼接和融合
        concat_feat = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        output = self.fusion(concat_feat)
        
        return output


class AdaptiveScaleBlock(nn.Module):
    """
    自适应尺度块 (Adaptive Scale Block, ASB)
    专为目标检测任务设计的高效特征提取模块
    
    核心创新点：
    1. 自适应多尺度卷积：根据输入特征动态选择最优卷积核组合
    2. 多尺度特征融合：类似Inception的多分支并行处理结构  
    3. 轻量级通道-空间注意力：同时关注重要通道和空间区域
    4. 智能特征选择机制：通过门控单元实现特征自适应筛选
    """
    def __init__(self, 
                 dim, 
                 expansion_ratio=8/3, 
                 kernel_sizes=[3, 5, 7],
                 conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral, eps=1e-6, normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 use_attention=True,
                 use_multiscale_fusion=True,
                 **kwargs):
        super().__init__()
        
        self.dim = dim
        self.use_attention = use_attention
        self.use_multiscale_fusion = use_multiscale_fusion
        
        # 输入归一化
        self.norm = norm_layer((dim, 1, 1))
        
        # 特征扩展
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        
        # 卷积通道分配
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        
        # 核心创新1: 自适应多尺度卷积
        self.adaptive_conv = AdaptiveMultiScaleConv(conv_channels, kernel_sizes)
        
        # 核心创新2: 轻量级注意力机制
        if use_attention:
            self.attention = ChannelSpatialAttention(hidden)
        
        # 特征压缩
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        
        # 核心创新3: 多尺度特征融合
        if use_multiscale_fusion:
            self.multiscale_fusion = MultiScaleFeatureFusion(dim)
        else:
            self.multiscale_fusion = nn.Identity()
        
        # 路径丢弃
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        shortcut = x
        
        # 层归一化
        x = self.norm(x)
        
        # 特征扩展并分割
        expanded = self.fc1(x)
        g, i, c = torch.split(expanded, self.split_indices, dim=1)
        
        # 自适应多尺度卷积处理
        c = self.adaptive_conv(c)
        
        # 特征重组
        combined = torch.cat((i, c), dim=1)
        
        # 门控机制
        gated = self.act(g) * combined
        
        # 注意力增强
        if self.use_attention:
            gated = self.attention(gated)
        
        # 特征压缩
        output = self.fc2(gated)
        
        # 多尺度特征融合
        output = self.multiscale_fusion(output)
        
        # 加权残差连接
        output = self.drop_path(output) + self.residual_weight * shortcut
        
        return output


class DetectionHead(nn.Module):
    """
    使用AdaptiveScaleBlock的目标检测头
    """
    def __init__(self, 
                 in_channels=256, 
                 num_classes=80, 
                 num_blocks=3,
                 expansion_ratio=8/3,
                 kernel_sizes=[3, 5, 7],
                 drop_path_rate=0.1):
        super().__init__()
        
        # 渐进式drop_path率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        # 多个AdaptiveScaleBlock堆叠
        self.blocks = nn.ModuleList([
            AdaptiveScaleBlock(
                dim=in_channels,
                expansion_ratio=expansion_ratio,
                kernel_sizes=kernel_sizes,
                drop_path=dpr[i],
                use_attention=True,
                use_multiscale_fusion=True
            ) for i in range(num_blocks)
        ])
        
        # 检测头
        self.cls_head = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness_head = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 通过多个EnhancedGatedBlock
        for block in self.blocks:
            x = block(x)
        
        # 多任务输出
        cls_score = self.cls_head(x)
        bbox_pred = self.reg_head(x)
        centerness = self.centerness_head(x)
        
        return {
            'cls_score': cls_score,
            'bbox_pred': bbox_pred,
            'centerness': centerness
        }


# 使用示例和性能测试
if __name__ == "__main__":
    import time
    
    # 创建模型
    model = DetectionHead(
        in_channels=256, 
        num_classes=80, 
        num_blocks=3,
        kernel_sizes=[3, 5, 7]
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 测试输入
    x = torch.randn(2, 256, 64, 64)  # [B, C, H, W]
    
    # 性能测试
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(x)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            outputs = model(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Average inference time: {avg_time*1000:.2f} ms")
    
    # 输出形状
    print(f"Classification output: {outputs['cls_score'].shape}")
    print(f"Bbox regression output: {outputs['bbox_pred'].shape}")
    print(f"Centerness output: {outputs['centerness'].shape}")
    
    # 单独测试AdaptiveScaleBlock
    block = AdaptiveScaleBlock(dim=256, kernel_sizes=[3, 5, 7])
    block_output = block(x)
    print(f"Block output shape: {block_output.shape}")
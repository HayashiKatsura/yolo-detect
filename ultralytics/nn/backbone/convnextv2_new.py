# Copyright (c) Meta Platforms, Inc. and affiliates.
# Enhanced ConvNeXtV2 with Local Complexity Adaptive Multi-Scale Fusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath

__all__ = ['convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico', 'convnextv2_nano', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge','convnextv2_atto_half']


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LocalComplexity(nn.Module):
    """
    Local Complexity Assessment Module
    计算输入特征的局部复杂度，用于自适应权重分配
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        
        # 用于计算复杂度的卷积层
        self.complexity_conv = nn.Conv2d(dim, dim // reduction, kernel_size=3, padding=1, groups=dim // reduction)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 复杂度到权重的映射
        self.fc1 = nn.Linear(dim // reduction, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, 3)  # 3个分支的权重
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 计算局部方差作为复杂度指标
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean) ** 2, kernel_size=3, stride=1, padding=1)
        
        # 通过卷积提取复杂度特征
        complexity_features = self.complexity_conv(local_var)
        complexity_global = self.global_pool(complexity_features).flatten(1)
        
        # 生成自适应权重
        weights = self.fc1(complexity_global)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = F.softmax(weights, dim=1)  # [B, 3]
        
        return weights


class MultiScaleBranch(nn.Module):
    """
    多尺度卷积分支
    包含三个并行分支：细节分支、中等分支、全局分支
    """
    def __init__(self, dim):
        super().__init__()
        
        # 细节分支 - 小卷积核捕获细节特征
        self.detail_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # 中等分支 - 中等卷积核捕获中层特征  
        self.medium_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # 全局分支 - 大卷积核捕获全局特征
        self.global_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
    def forward(self, x, weights):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            weights: 自适应权重 [B, 3]
        """
        batch_size = x.size(0)
        
        # 三个分支并行处理
        detail_out = self.detail_branch(x)      # 细节特征
        medium_out = self.medium_branch(x)      # 中层特征
        global_out = self.global_branch(x)      # 全局特征
        
        # 权重重塑以匹配特征维度
        weights = weights.view(batch_size, 3, 1, 1, 1)  # [B, 3, 1, 1, 1]
        
        # 堆叠三个分支的输出
        multi_scale_features = torch.stack([detail_out, medium_out, global_out], dim=1)  # [B, 3, C, H, W]
        
        # 加权融合
        fused_features = (multi_scale_features * weights).sum(dim=1)  # [B, C, H, W]
        
        return fused_features


class EnhancedBlock(nn.Module):
    """ 
    Enhanced ConvNeXtV2 Block with Local Complexity Adaptive Multi-Scale Fusion
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        use_multi_scale (bool): Whether to use multi-scale fusion. Default: True
    """
    def __init__(self, dim, drop_path=0., use_multi_scale=True):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        
        if use_multi_scale:
            # 局部复杂度评估模块
            self.local_complexity = LocalComplexity(dim)
            # 多尺度分支
            self.multi_scale_branch = MultiScaleBranch(dim)
        else:
            # 原始的depthwise conv
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 后续处理保持不变
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        
        if self.use_multi_scale:
            # 计算局部复杂度权重
            complexity_weights = self.local_complexity(x)
            # 多尺度特征提取
            x = self.multi_scale_branch(x, complexity_weights)
        else:
            # 原始的depthwise convolution
            x = self.dwconv(x)
        
        # 维度变换：(N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)  # GRN特征校准
        x = self.pwconv2(x)
        # 维度变换：(N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ 
    Enhanced ConvNeXt V2 with Local Complexity Adaptive Multi-Scale Fusion
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        multi_scale_stages (list): Which stages to apply multi-scale fusion. Default: [1, 2, 3]
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,
                 multi_scale_stages=[1, 2, 3]  # 在哪些阶段使用多尺度融合
                 ):
        super().__init__()
        self.depths = depths
        self.multi_scale_stages = multi_scale_stages
        
        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            use_multi_scale = i in multi_scale_stages
            stage = nn.Sequential(
                *[EnhancedBlock(dim=dims[i], 
                               drop_path=dp_rates[cur + j],
                               use_multi_scale=use_multi_scale) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            res.append(x)
        return res


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


def convnextv2_atto(weights='', **kwargs):
        model = ConvNeXtV2(
            depths=[2, 2, 6, 2], 
            dims=[40, 80, 160, 320], 
            multi_scale_stages=[1, 2, 3],  # 在阶段1,2,3使用多尺度融合
            **kwargs
        )
        if weights:
            # 注意：由于模型结构有变化，可能需要手动处理权重加载
            model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['model']))
        return model

def convnextv2_atto_half(weights='', **kwargs):
    pass

def convnextv2_femto(weights='', **kwargs):
    pass

def convnextv2_pico(weights='', **kwargs):
    pass

def convnextv2_nano(weights='', **kwargs):
   pass

def convnextv2_tiny(weights='', **kwargs):
    pass

def convnextv2_base(weights='', **kwargs):
    pass

def convnextv2_large(weights='', **kwargs):
    pass

def convnextv2_huge(weights='', **kwargs):
    pass


if __name__ == '__main__':
    
    def enhanced_convnextv2_case(weights='', **kwargs):
        model = ConvNeXtV2(
            depths=[2, 2, 6, 2], 
            dims=[40, 80, 160, 320], 
            multi_scale_stages=[1, 2, 3],  # 在阶段1,2,3使用多尺度融合
            **kwargs
        )
        if weights:
            # 注意：由于模型结构有变化，可能需要手动处理权重加载
            model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['model']))
        return model
    
    # 创建增强版模型
    model = enhanced_convnextv2_case()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(model)
  
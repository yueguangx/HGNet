import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import time
import pdb


__all__ = (
    "EMA","CAA","HyperConv", "PWconv", "DWconv", 
)
# GitHub 地址：https://github.com/YOLOonMe/EMA-attention-module
# 论文地址：https://arxiv.org/abs/2305.13563v2

class PWconv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        初始化逐点卷积层
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
            bias (bool): 是否添加偏置项，默认为 False
        """
        super(PWconv, self).__init__()
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 输入特征图, 形状为 (B, C_in, H, W)
        Returns:
            Tensor: 输出特征图, 形状为 (B, C_out, H, W)
        """
        return self.pw_conv(x)

#深度可分离卷积
class DWconv(nn.Module):
    def __init__(self,in_channel,out_channel):
 
        #这一行千万不要忘记
        super(DWconv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        #逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    


class EMA(nn.Module):
    def __init__(self, channels, factor=8,HH=1):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        
        # 定义模块层
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w) # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)    
  


# 定义卷积模块类，来自mmcv.cnn
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,         # 输入通道数
        out_channels: int,        # 输出通道数
        kernel_size: int,         # 卷积核大小
        stride: int = 1,          # 步长
        padding: int = 0,         # 填充
        groups: int = 1,          # 组卷积数
        norm_cfg: Optional[dict] = None,   # 归一化配置
        act_cfg: Optional[dict] = None      # 激活函数配置
    ):
        super().__init__()

        layers = []
        # 卷积层
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=(norm_cfg is None)  # 如果有归一化层，则不使用偏置
            )
        )

        # 归一化层
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)

        # 激活层
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)

        # 将所有层组合为一个序列层
        self.block = nn.Sequential(*layers)

    def _get_norm_layer(self, out_channels: int, norm_cfg: dict):
        # 根据 norm_cfg 获取对应的归一化层
        # 示例：norm_cfg = {'type': 'BN', 'momentum': 0.1}
        norm_type = norm_cfg['type']
        if norm_type == 'BN':  # 批归一化
            return nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.1))
        elif norm_type == 'LN':  # 层归一化
            return nn.LayerNorm([out_channels, 1, 1])  # 假设是2D数据
        # 可以根据需要添加其他归一化类型
        raise ValueError(f"Unsupported norm type: {norm_type}")

    def _get_act_layer(self, act_cfg: dict):
        # 根据 act_cfg 获取对应的激活函数
        act_type = act_cfg['type']
        if act_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_type == 'LeakyReLU':
            return nn.LeakyReLU(negative_slope=act_cfg.get('negative_slope', 0.01), inplace=True)
        elif act_type == 'PReLU':
            return nn.PReLU()
        # 可以根据需要添加其他激活函数类型
        raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x):
        return self.block(x)
    
    # 定义获取归一化层的辅助函数
    def _get_norm_layer(self, num_features, norm_cfg):
        # 根据传入的 norm_cfg 配置选择归一化层类型
        if norm_cfg['type'] == 'BN':
            # 批归一化（Batch Normalization）
            return nn.BatchNorm2d(
                num_features,
                momentum=norm_cfg.get('momentum', 0.1),  # 默认动量值
                eps=norm_cfg.get('eps', 1e-5)  # 默认 epsilon 值
            )
        
        # 这里可以继续扩展其他归一化类型，例如：
        # if norm_cfg['type'] == 'LN':  # 层归一化（Layer Normalization）
        #     return nn.LayerNorm(...)

        # 若未实现的归一化类型，抛出异常
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    # 定义获取激活层的辅助函数
    def _get_act_layer(self, act_cfg):
        # 根据传入的 act_cfg 配置选择激活函数类型
        if act_cfg['type'] == 'ReLU':
            # ReLU 激活函数
            return nn.ReLU(inplace=True)
        
        if act_cfg['type'] == 'SiLU':
            # SiLU 激活函数（Sigmoid Linear Unit）
            return nn.SiLU(inplace=True)

        # 这里可以继续扩展其他激活类型，例如：
        # if act_cfg['type'] == 'LeakyReLU':  
        #     return nn.LeakyReLU(...)

        # 若未实现的激活类型，抛出异常
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

# 定义上下文锚点注意力 (Context Anchor Attention) 模块
class CAA(nn.Module):
    """上下文锚点注意力模块"""
    
    def __init__(
        self,
        channels: int,                # 输入通道数
        CH:1,                                        
        h_kernel_size: int = 11,           # 水平卷积核大小
        v_kernel_size: int = 11,           # 垂直卷积核大小
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置
        act_cfg: Optional[dict] = dict(type='SiLU')                         # 激活函数配置
    ):
        super().__init__()

        # 平均池化层
        self.avg_pool = nn.AvgPool2d(7, 1, 3)

        # 1x1卷积模块，用于调整通道数
        self.conv1 = ConvModule(
            channels, channels, 1, 1, 0, 
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        # 水平卷积模块，使用1xh_kernel_size的卷积核，仅在水平方向上进行卷积
        self.h_conv = ConvModule(
            channels, channels, (1, h_kernel_size), 1, 
            (0, h_kernel_size // 2), groups=channels,
            norm_cfg=None, act_cfg=None
        )

        # 垂直卷积模块，使用v_kernel_sizex1的卷积核，仅在垂直方向上进行卷积
        self.v_conv = ConvModule(
            channels, channels, (v_kernel_size, 1), 1, 
            (v_kernel_size // 2, 0), groups=channels,
            norm_cfg=None, act_cfg=None
        )

        # 1x1卷积模块，用于进一步调整通道数
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 使用Sigmoid激活函数
        self.act = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 通过平均池化、卷积和激活函数计算注意力系数
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        #调整 attn_factor 的尺寸，使其与 x 的尺寸匹配
        attn_factor = F.interpolate(attn_factor, size=x.shape[2:], mode='bilinear', align_corners=False)
        # x与生成的注意力系数相乘，生成增强后特征图
        return x * attn_factor





#默认使用 1 范数（L1-norm），即 p_norm=1
#如果你想用 2 范数（L2-norm），修改 model.p_norm = 2
#如果你想用无穷范数（L∞-norm），修改 model.p_norm = float('inf')


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyperGraphConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperConv(nn.Module):
    def __init__(self, c1, c2, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyperGraphConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.c1 = c1
        self.c2 = c2

    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        # 记录原始数据类型
        original_dtype = feature.dtype
        # 转换为 torch.float32 进行距离计算
        feature = feature.to(torch.float32)
        distance = torch.cdist(feature, feature, p=1)           #1范数度量
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(original_dtype)
        out = self.hgconv(x, hg).to(x.device).to(original_dtype)

        x = out
        x = x.transpose(1, 2).contiguous().view(b, self.c2, h, w)
        x = self.act(self.bn(x))

        return x







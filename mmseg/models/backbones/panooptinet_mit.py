# Copyright (c) OpenMMLab. All rights reserved.

from ..builder import BACKBONES
from ..utils import nlc_to_nchw
import os

from mmseg.models.backbones.mit import MixVisionTransformer

@BACKBONES.register_module()
class PanoOptiNetMixVisionTransformer(MixVisionTransformer):
    """PanoOptiNet 的主干网络。

    该主干网络实现了 `PanoOptiNet`，基于 SegFormer 的设计，并增加了滑动窗口的重叠处理。
    该类继承自 MixVisionTransformer，并添加了处理重叠区域的功能。

    参数:
        in_channels (int): 输入通道数。默认值：3。
        embed_dims (int): 嵌入维度。默认值：64。
        num_stages (int): 阶段数量。默认值：4。
        num_layers (Sequence[int]): 每个 Transformer 编码层的层数。
            默认值：[3, 4, 6, 3]。
        num_heads (Sequence[int]): 每个 Transformer 编码层的注意力头数。
            默认值：[1, 2, 4, 8]。
        patch_sizes (Sequence[int]): 每个重叠补丁嵌入的补丁大小。
            默认值：[7, 3, 3, 3]。
        strides (Sequence[int]): 每个重叠补丁嵌入的步长。
            默认值：[4, 2, 2, 2]。
        sr_ratios (Sequence[int]): 每个 Transformer 编码层的空间降维比率。
            默认值：[8, 4, 2, 1]。
        out_indices (Sequence[int] | int): 指定输出哪些阶段的特征。
            默认值：(0, 1, 2, 3)。
        mlp_ratio (int): MLP 隐藏层维度与嵌入维度的比例。
            默认值：4。
        qkv_bias (bool): 是否为 QKV 启用偏置。默认值：True。
        drop_rate (float): 元素被置零的概率。默认值：0.0。
        attn_drop_rate (float): 注意力层的 dropout 率。默认值：0.0。
        drop_path_rate (float): 随机深度率。默认值：0.0。
        norm_cfg (dict): 归一化层的配置字典。
            默认值：dict(type='LN')。
        act_cfg (dict): FFN 的激活函数配置。
            默认值：dict(type='GELU')。
        pretrained (str, optional): 预训练模型路径。默认值：None。
        init_cfg (dict or list[dict], optional): 初始化配置字典。
            默认值：None。
        with_cp (bool): 是否使用检查点。使用检查点可以节省部分显存，
            但会降低训练速度。默认值：False。
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN", eps=1e-6),
        pretrained=None,
        init_cfg=None,
        with_cp=False,
    ):
        # 调用父类的初始化方法
        super(PanoOptiNetMixVisionTransformer, self).__init__(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_stages=num_stages,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            strides=strides,
            sr_ratios=sr_ratios,
            out_indices=out_indices,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            with_cp=with_cp
        )
        
        # 添加PanoOptiNet特有的属性
        # last_dx/last_dy 已废弃：方向信息由文件名中的 recv_side 直接携带，无需运行时状态

    # our PanoOptiNet code: 根据 recv_side 提供的边缘标识将重叠特征粘贴到对应位置
    def featureTemplatePaste(self, x, overlap_data, recv_side, out_hw_shape):
        """
        根据 recv_side 直接将前一帧的前沿特征（overlap）注入当前 patch 的后沿（FTP）。

        参数:
            x (Tensor): 输入特征图，形状为 (batch, channels, H, W)。
            overlap_data (Tensor): 前一帧 FTC 裁剪的前沿特征，形状与目标区域一致。
            recv_side (int): FTP 接收边编码（由文件名直接携带，SAS 阶段已算好）
                0=底边  1=顶边  2=右边  3=左边
                4=右下角  5=左下角  6=右上角  7=左上角
            out_hw_shape (tuple): 输出特征图的空间形状 (H, W)。
        返回:
            Tensor: 注入后的特征图，形状为 (batch, channels, H, W)。
        """
        sh = int(out_hw_shape[0] / 4)  # 边带高度
        sw = int(out_hw_shape[1] / 4)  # 边带宽度
        rh = int(out_hw_shape[0])      # 全高
        rw = int(out_hw_shape[1])      # 全宽

        # recv_side → (行切片, 列切片)
        # 语义：将 overlap 粘贴到当前 patch 中朝向前一个 patch 的那条边（后沿）
        # recv_side 恒等于前一帧 ftc_side 的对边，由 SAS 直接写入文件名，无需运行时推算。
        side_slices = {
            0: (slice(rh - sh, rh), slice(0, rw)),       # 底边（前一帧向下，当前帧从底部接收）
            1: (slice(0, sh),       slice(0, rw)),        # 顶边（前一帧向上，当前帧从顶部接收）
            2: (slice(0, rh),       slice(rw - sw, rw)),  # 右边（前一帧向右，当前帧从右侧接收）
            3: (slice(0, rh),       slice(0, sw)),        # 左边（前一帧向左，当前帧从左侧接收）
            4: (slice(rh - sh, rh), slice(rw - sw, rw)), # 右下角
            5: (slice(rh - sh, rh), slice(0, sw)),       # 左下角
            6: (slice(0, sh),       slice(rw - sw, rw)), # 右上角
            7: (slice(0, sh),       slice(0, sw)),       # 左上角
        }

        if recv_side not in side_slices:
            raise ValueError(f"不支持的 recv_side={recv_side}，合法值为 0~7")

        slice_h, slice_w = side_slices[recv_side]
        target = x[:, :, slice_h, slice_w]
        if overlap_data.shape != target.shape:
            raise ValueError(
                f"overlap_data.shape={overlap_data.shape} 与目标区域 shape={target.shape} 不匹配")
        x[:, :, slice_h, slice_w] = overlap_data
        return x

    def forward(self, x, img_metas, overlap):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入特征，形状为 (batch, channels, H, W)。
            img_metas (list): 图像的元信息，包含文件名等信息。
            overlap (Tensor): 前一帧 FTC 裁剪的前沿特征，用于 FTP 注入当前帧后沿。
        返回:
            list: 输出特征列表，每个元素的形状为 (batch, channels, H, W)。
        """
        # 如果文件名未指定，则使用父类的标准处理流程
        if img_metas[0]["filename"] is None:
            return super(PanoOptiNetMixVisionTransformer, self).forward(x)
        # 如果文件名已指定，则走 PanoOptiNet 的自定义处理流程
        else:
            filename = img_metas[0]["filename"]
            filename = os.path.splitext(os.path.basename(filename))[0]
            parts = filename.split("_")
            # 文件名格式：<name>_<id>_<ftc_side>_<recv_side>
            patch_id   = int(parts[-3])
            # ftc_side 属于本帧（由解码头使用），此处只需 recv_side
            recv_side  = int(parts[-1])

            outs = []
            for i, layer in enumerate(self.layers):

                x, hw_shape = layer[0](x)  # hw_shape=(128,128) (64,64) (32,32)

                for block in layer[1]:
                    x = block(x, hw_shape)

                x = layer[2](x)
                x = nlc_to_nchw(x, hw_shape)

                # our PanoOptiNet code: 第一层（i==0）且有有效 recv_side 且非第一帧时，执行 FTP 注入
                if i == 0 and recv_side != -1 and patch_id != 1 and overlap is not None:
                    x = self.featureTemplatePaste(x, overlap, recv_side, hw_shape)

                if i in self.out_indices:
                    outs.append(x)

            return outs

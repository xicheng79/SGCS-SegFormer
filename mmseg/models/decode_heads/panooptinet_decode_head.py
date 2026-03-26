# 修改自
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
#
# 本作品遵循NVIDIA源代码许可证
#
# 版权所有 (c) 2021, NVIDIA Corporation。保留所有权利。
# NVIDIA StyleGAN2 with Adaptive Discriminator Augmentation (ADA)源代码许可证
#
#  1. 定义
#  "许可方"指分发其作品的任何个人或实体。
#  "软件"指根据本许可证提供的原创作品。
#  "作品"指软件及其根据本许可证提供的任何补充或衍生作品。
#  术语"复制"、"复制品"、"衍生作品"和"分发"具有美国版权法规定的含义；
#  但就本许可证而言，衍生作品不应包括与作品保持可分离或仅链接（或通过名称绑定）到作品接口的作品。
#  作品（包括软件）通过以下方式"提供"本许可证：(a)在作品中包含引用本许可证适用性的版权声明，或(b)附上本许可证副本。
#
#  2. 许可授予
#      2.1 版权授予。根据本许可证的条款和条件，每个许可方授予您永久、全球性、非排他性、免版税的版权许可，
#      以复制、准备衍生作品、公开展示、公开表演、分许可和分发其作品及任何由此产生的衍生作品。
#
#  3. 限制
#      3.1 再分发。您只能在以下条件下复制或分发作品：(a)根据本许可证，(b)在分发中包含完整的本许可证副本，
#      以及(c)保留作品中现有的任何版权、专利、商标或归属声明。
#      3.2 衍生作品。您可以指定适用于您作品衍生作品的使用、复制和分发的附加或不同条款（"您的条款"），
#      前提是(a)您的条款规定第3.3节的使用限制适用于您的衍生作品，且(b)您明确标识受您的条款约束的特定衍生作品。
#      尽管有您的条款，本许可证（包括第3.1节的再分发要求）将继续适用于作品本身。
#      3.3 使用限制。作品及其任何衍生作品只能用于或旨在非商业用途。尽管有上述规定，
#      NVIDIA及其关联公司可以商业用途使用作品及其任何衍生作品。本文中"非商业用途"仅指用于研究或评估目的。
#      3.4 专利主张。如果您对任何许可方提出或威胁提出专利主张（包括诉讼中的任何主张、交叉主张或反诉），
#      以强制执行您声称被任何作品侵犯的专利，则您从该许可方获得的本许可证下的权利（包括第2.1节的授予）将立即终止。
#      3.5 商标。本许可证不授予使用任何许可方或其关联公司名称、徽标或商标的权利，
#      除非为复制本许可证所述通知所必需。
#      3.6 终止。如果您违反本许可证的任何条款，则您在本许可证下的权利（包括第2.1节的授予）将立即终止。
#
#  4. 免责声明
#  作品按"原样"提供，不附带任何明示或默示的保证或条件，包括但不限于适销性、特定用途适用性、所有权和非侵权性的保证或条件。
#  您需自行承担根据本许可证开展任何活动的风险。
#
#  5. 责任限制
#  除非适用法律禁止，在任何情况下，根据任何法律理论（包括侵权（包括过失）、合同或其他），
#  任何许可方均不对您承担任何损害赔偿责任，包括任何直接、间接、特殊、附带或后果性损害，
#  包括但不限于商誉损失、业务中断、利润或数据丢失、计算机故障或失灵，或其他商业损害或损失，
#  即使许可方已被告知此类损害的可能性。

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.ops import resize

from mmseg.models.decode_heads.segformer_head import SegformerHead

import os

@HEADS.register_module()
class PanoOptiNetHead(SegformerHead):

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        # 调用SegformerHead的初始化方法
        super().__init__(interpolate_mode=interpolate_mode, **kwargs)
        
        # 添加PanoOptiNetHead特有的组件
        num_inputs = len(self.in_channels)
            
        self.convs_feature_template_copy = nn.ModuleList()
        for i in range(num_inputs):
            self.convs_feature_template_copy.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=64,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        
        self.fusion_conv_overlap = ConvModule(
            in_channels=64 * num_inputs,
            out_channels=64,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    # ---------- Bug 4 修复：文件名解析封装在子类，不再污染 BaseDecodeHead ----------
    # 注意（Bug 5 - 架构级风险）：
    #   self.overlap / self.last_dx / self.last_dy 等内部状态隐式依赖 patch 的处理顺序。
    #   以下条件成立时，跨样本状态传递将出错：
    #     - DataLoader 启用 shuffle=True；
    #     - batch_size > 1（当前代码仅正确处理 img_metas[0]）；
    #     - 多 GPU / 多进程并行推理（每个进程维护独立状态）；
    #     - 推理时 patch 顺序与 SAS 路径不一致。
    #   在上述场景下，请确保数据集按 patch ID 顺序排列，且 batch_size=1，才能得到正确结果。
    @staticmethod
    def _parse_dx_dy_from_meta(img_metas):
        """从 img_metas 中解析滑动方向 (dx, dy)。
        
        文件命名约定：<prefix>_<id>_<dx>_<dy>.<ext>
        dx: 行方向偏移（-1/0/1），dy: 列方向偏移（-1/0/1）
        """
        filename = img_metas[0]['filename']
        if filename is None:
            return 0, 0
        filename = os.path.splitext(os.path.basename(filename))[0]
        parts = filename.split('_')
        dx = int(parts[-2])
        dy = int(parts[-1])
        return dx, dy

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """PanoOptiNet 训练阶段前向函数，支持返回 overlap 特征。"""
        dx, dy = self._parse_dx_dy_from_meta(img_metas)
        seg_logits, overlap = self(inputs, dx, dy)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, overlap

    def forward_test(self, inputs, img_metas, test_cfg):
        """PanoOptiNet 测试阶段前向函数，支持返回 overlap 特征。"""
        dx, dy = self._parse_dx_dy_from_meta(img_metas)
        seg_logits, overlap = self.forward(inputs, dx, dy)
        return seg_logits, overlap

    # our PanoOptiNet code: 根据dx, dy提供的方向信息找到滑动路径上的重叠的部分
    def featureTemplateCopy(self, x, dx, dy, out_hw_shape):
        """
        根据滑动方向提取重叠区域特征。

        参数:
            x: 输入特征图，形状为 (B, C, H, W)
            dx: x轴滑动方向 (-1, 0, 1)
            dy: y轴滑动方向 (-1, 0, 1)
            out_hw_shape: 输出特征图的 (height, width)

        返回:
            提取的重叠区域特征，形状为 (B, C, H/4, W/4)
        """
        # 检查输入参数合法性
        if not isinstance(out_hw_shape, (tuple, list)) or len(out_hw_shape) != 2:
            raise ValueError("out_hw_shape 应为包含两个整数的元组或列表 (height, width)")
        
        if not (dx in [-1, 0, 1] and dy in [-1, 0, 1]) or (dx == 0 and dy == 0):
            raise ValueError("dx 和 dy 必须为 -1、0 或 1，且不能同时为 0")
        
        if x.ndim != 4:
            raise ValueError("输入特征图 x 应为 4 维张量 (B, C, H, W)")

        B, C, H, W = x.shape
        out_h, out_w = out_hw_shape
        shift_h = out_h // 4
        shift_w = out_w // 4

        region_h = out_h
        region_w = out_w

        # 映射方向到切片范围
        # 坐标约定：dx 对应行（height），dy 对应列（width）
        #   dx = -1 → 行索引减小 → 向上移动 → 裁剪顶部边缘
        #   dx = +1 → 行索引增大 → 向下移动 → 裁剪底部边缘
        #   dy = -1 → 列索引减小 → 向左移动 → 裁剪左侧边缘
        #   dy = +1 → 列索引增大 → 向右移动 → 裁剪右侧边缘
        direction_slices = {
            (-1, 0):  (slice(0, shift_h), slice(0, region_w)),                           # 向上：裁剪顶部行带
            (-1, 1):  (slice(0, shift_h), slice(region_w - shift_w, region_w)),          # 向上+向右：裁剪右上角
            (0, 1):   (slice(0, region_h), slice(region_w - shift_w, region_w)),         # 向右：裁剪右侧列带
            (1, 1):   (slice(region_h - shift_h, region_h), slice(region_w - shift_w, region_w)),  # 向下+向右：裁剪右下角
            (1, 0):   (slice(region_h - shift_h, region_h), slice(0, region_w)),         # 向下：裁剪底部行带
            (1, -1):  (slice(region_h - shift_h, region_h), slice(0, shift_w)),          # 向下+向左：裁剪左下角
            (0, -1):  (slice(0, region_h), slice(0, shift_w)),                            # 向左：裁剪左侧列带
            (-1, -1): (slice(0, shift_h), slice(0, shift_w)),                             # 向上+向左：裁剪左上角
        }

        slice_h, slice_w = direction_slices.get((dx, dy), (None, None))
        if slice_h is None or slice_w is None:
            raise ValueError(f"不支持的滑动方向 dx={dx}, dy={dy}")

        # 安全裁剪
        try:
            overlap_data = x[:, :, slice_h, slice_w]
        except Exception as e:
            raise RuntimeError(f"在裁剪重叠区域时发生错误: {e}")

        return overlap_data

    
    def forward(self, inputs, dx=0, dy=0):
        """前向传播函数
        
        功能：
        1. 处理多尺度输入特征（利用父类SegformerHead的功能）
        2. 生成主分割结果
        3. 提取滑动窗口重叠区域特征
        
        参数：
            inputs: 多尺度输入特征列表
            dx: x轴滑动方向，默认为0
            dy: y轴滑动方向，默认为0
            
        返回：
            tuple: (主分割结果, 重叠区域特征)
        """
        # 处理输入特征
        inputs = self._transform_inputs(inputs)
        
        # 处理重叠区域特征提取
        outs_overlap = []

        overlap = None

        for idx in range(len(inputs)):
            x_overlap = inputs[idx]
            conv_overlap = self.convs_feature_template_copy[idx]
            outs_overlap.append(
                resize(
                    input=conv_overlap(x_overlap),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        
        # 使用父类SegformerHead的处理逻辑生成分割结果
        # 但不调用父类的forward方法，因为我们需要自定义返回值
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out_overlap = self.fusion_conv_overlap(torch.cat(outs_overlap, dim=1))
        
        # 当有滑动时(dx或dy不为0)，提取重叠区域
        if (dx!=0 or dy!=0) :
            hw_shape = (out_overlap.shape[2], out_overlap.shape[3])
            overlap = self.featureTemplateCopy(out_overlap, dx, dy, hw_shape) # hw_shape=(128,128)
            overlap = overlap.detach()
            
        # 生成最终分割结果
        out = self.cls_seg(out)   #torch.Size([1, 2, 128, 128])
        
        return out, overlap

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """BaseDecodeHead的基类。

    参数:
        in_channels (int|Sequence[int]): 输入通道数。
        channels (int): 模块后的通道数，在conv_seg之前。
        num_classes (int): 类别数。
        out_channels (int): conv_seg的输出通道数。
        threshold (float): 二分类分割的阈值，当`out_channels==1`时使用。默认: None。
        dropout_ratio (float): dropout层的比例。默认: 0.1。
        conv_cfg (dict|None): 卷积层的配置。默认: None。
        norm_cfg (dict|None): 归一化层的配置。默认: None。
        act_cfg (dict): 激活层的配置。默认: dict(type='ReLU')
        in_index (int|Sequence[int]): 输入特征索引。默认: -1
        input_transform (str|None): 输入特征的变换类型。
            选项: 'resize_concat', 'multiple_select', None。
            'resize_concat': 多个特征图将被调整到与第一个特征图相同的大小，然后拼接在一起。
                通常用于HRNet的FCN头。
            'multiple_select': 多个特征图将被打包成一个列表并传递到解码头。
            None: 只允许选择一个特征图。
            默认: None。
        loss_decode (dict | Sequence[dict]): 解码损失的配置。
            `loss_name`是相应损失函数的属性，可以显示在训练日志中。如果希望将此损失项包含在反向传播图中，`loss_`必须是名称的前缀。默认: 'loss_ce'。
             例如: dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            默认: dict(type='CrossEntropyLoss')。
        ignore_index (int | None): 要忽略的标签索引。使用掩码BCE损失时，ignore_index应设置为None。默认: 255。
        sampler (dict|None): 分割图采样器的配置。默认: None。
        align_corners (bool): F.interpolate的align_corners参数。默认: False。
        downsample_label_ratio (int): 在损失中下采样seg_label的比例。downsample_label_ratio > 1将减少内存使用。如果downsample_label_ratio = 0，则禁用。
            默认: 0。
        init_cfg (dict or list[dict], optional): 初始化配置字典。
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 downsample_label_ratio=0,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.downsample_label_ratio = downsample_label_ratio  
        if not isinstance(self.downsample_label_ratio, int) or \
           self.downsample_label_ratio < 0:
            warnings.warn('downsample_label_ratio should '
                          'be set as an integer equal or larger than 0.')

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """检查并初始化输入变换。

        in_channels、in_index和input_transform必须匹配。
        具体来说，当input_transform为None时，只会选择单个特征图。
        因此，in_channels和in_index必须是int类型。
        当input_transform

        参数:
            in_channels (int|Sequence[int]): 输入通道数。
            in_index (int|Sequence[int]): 输入特征索引。
            input_transform (str|None): 输入特征的变换类型。
            选项: 'resize_concat', 'multiple_select', None。
            'resize_concat': 多个特征图将被调整到与第一个特征图相同的大小，然后拼接在一起。
                通常用于HRNet的FCN头。
            'multiple_select': 多个特征图将被打包成一个列表并传递到解码头。
            None: 只允许选择一个特征图。
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """训练时的前向函数。
        参数:
            inputs (list[Tensor]): 多级图像特征的列表。
            img_metas (list[dict]): 图像信息字典的列表，每个字典包含:
            'img_shape', 'scale_factor', 'flip'，还可能包含
            'filename', 'ori_shape', 'pad_shape' 和 'img_norm_cfg'。
            这些键的值的详细信息请参见
            `mmseg/datasets/pipelines/formatting.py:Collect`。
            gt_semantic_seg (Tensor): 语义分割掩码
            如果架构支持语义分割任务，则使用。
            train_cfg (dict): 训练配置。

        返回:
            dict[str, Tensor]: 损失组件的字典
        """
        seg_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """测试时的前向函数。

        参数:
            inputs (list[Tensor]): 多级图像特征的列表。
            img_metas (list[dict]): 图像信息字典的列表，每个字典包含:
            'img_shape', 'scale_factor', 'flip'，还可能包含
            'filename', 'ori_shape', 'pad_shape' 和 'img_norm_cfg'。
            这些键的值的详细信息请参见
            `mmseg/datasets/pipelines/formatting.py:Collect`。
            test_cfg (dict): 测试配置。

        返回:
            Tensor: 输出分割图。
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        if self.downsample_label_ratio > 0:
            seg_label = seg_label.float()
            target_size = (seg_label.shape[2] // self.downsample_label_ratio,
                           seg_label.shape[3] // self.downsample_label_ratio)
            seg_label = resize(
                input=seg_label, size=target_size, mode='nearest')
            seg_label = seg_label.long()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

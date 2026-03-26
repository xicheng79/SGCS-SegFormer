# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """用于分割任务的图像预处理模块。

    与 :class:`mmengine.ImgDataPreprocessor` 相比：

    1. 如果未指定 ``mean``，则不会进行归一化操作。
    2. 在堆叠批次后进行归一化和颜色空间转换。
    3. 支持 Mixup 和 Cutmix 等批量增强操作。

    它提供了以下数据预处理功能：

    - 整理数据并将其移动到目标设备。
    - 使用定义的 ``pad_val`` 将输入填充到输入大小，并使用定义的 ``seg_pad_val`` 填充分割图。
    - 将输入堆叠为批量输入。
    - 如果输入形状为 (3, H, W)，则将输入从 BGR 转换为 RGB。
    - 使用定义的标准差和均值对图像进行归一化。
    - 在训练期间进行 Mixup 和 Cutmix 等批量增强操作。

    参数:
        mean (Sequence[Number], 可选): R、G、B 通道的像素均值。
            默认值: None。
        std (Sequence[Number], 可选): R、G、B 通道的像素标准差。
            默认值: None。
        size (tuple, 可选): 固定的填充大小。
        size_divisor (int, 可选): 填充后大小的除数。
        pad_val (float, 可选): 填充值。默认值: 0。
        seg_pad_val (float, 可选): 分割图的填充值。
            默认值: 255。
        padding_mode (str): 填充类型。默认值: constant。
            - constant: 使用常量值进行填充，该值由 pad_val 指定。
        bgr_to_rgb (bool): 是否将图像从 BGR 转换为 RGB。
            默认值: False。
        rgb_to_bgr (bool): 是否将图像从 RGB 转换为 RGB。
            默认值: False。
        batch_augments (list[dict], 可选): 批量级别的增强操作
        test_cfg (dict, 可选): 测试时的填充大小配置，如果未指定，将使用 `size` 和 `size_divisor` 参数作为默认值。
            默认值: None，仅支持键 `size` 或 `size_divisor`。
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """基于 ``BaseDataPreprocessor`` 执行归一化、填充和 BGR 转 RGB 转换。

        参数:
            data (dict): 从数据加载器中采样的数据。
            training (bool): 是否启用训练时的数据增强。

        返回:
            Dict: 与模型输入格式相同的数据。
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            assert len(inputs) == 1, (
                'Batch inference is not support currently, '
                'as the image size might be different in a batch')
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """多尺度和翻转的测试时数据增强。

    以下是一个配置示例：

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    使用上述配置进行 MultiScaleFLipAug 后，结果将被包装成相同长度的列表，如下所示：

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    参数:
        transforms (list[dict]): 每次增强时要应用的变换。
        img_scale (None | tuple | list[tuple]): 用于调整大小的图像尺度。
        img_ratios (float | list[float]): 用于调整大小的图像比例。
        flip (bool): 是否应用翻转增强。默认值: False。
        flip_direction (str | list[str]): 翻转增强的方向，
            选项为 "horizontal" 和 "vertical"。如果 flip_direction 是列表，
            则会应用多次翻转增强。
            当 flip == False 时无效。默认值: "horizontal"。
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        if flip:
            trans_index = {
                key['type']: index
                for index, key in enumerate(transforms)
            }
            if 'RandomFlip' in trans_index and 'Pad' in trans_index:
                assert trans_index['RandomFlip'] < trans_index['Pad'], \
                    'Pad must be executed after RandomFlip when flip is True'
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """调用函数以对结果应用测试时的数据增强变换。

        参数:
            results (dict): 包含待变换数据的结果字典。

        返回:
           dict[str: list]: 增强后的数据，其中每个值都被包装在一个列表中。
        """

        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str

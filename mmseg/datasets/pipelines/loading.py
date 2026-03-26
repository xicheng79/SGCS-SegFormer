# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES
from osgeo import gdal
gdal.PushErrorHandler("CPLQuietErrorHandler") # 忽略GDAL的一些警告                      

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """从文件中加载图像。

    必需的键有 "img_prefix" 和 "img_info"（这是一个字典，必须包含键 "filename"）。
    添加或更新的键有 "filename"、"img"、"img_shape"、"ori_shape"（与 `img_shape` 相同）、
    "pad_shape"（与 `img_shape` 相同）、"scale_factor"（1.0）和 "img_norm_cfg"（均值为 0，标准差为 1）。

    参数:
        to_float32 (bool): 是否将加载的图像转换为 float32 类型的 numpy 数组。
            如果设置为 False，加载的图像将是 uint8 类型的数组。默认为 False。
        color_type (str): 传递给 :func:`mmcv.imfrombytes` 的标志参数。
            默认为 'color'。
        file_client_args (dict): 用于实例化 FileClient 的参数。
            有关详细信息，请参阅 :class:`mmcv.fileio.FileClient`。
            默认为 ``dict(backend='disk')``。
        imdecode_backend (str): 用于 :func:`mmcv.imdecode` 的后端。默认值:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """调用函数以加载图像并获取图像元信息。

        参数:
            results (dict): 来自 :obj:`mmseg.CustomDataset` 的结果字典。

        返回:
            dict: 包含加载的图像和元信息的字典。
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend) 
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadTiffImageFromFile(object):
    """从文件中加载图像。

    必需的键有 "img_prefix" 和 "img_info"（这是一个字典，必须包含键 "filename"）。
    添加或更新的键有 "filename"、"img"、"img_shape"、"ori_shape"（与 `img_shape` 相同）、
    "pad_shape"（与 `img_shape` 相同）、"scale_factor"（1.0）和 "img_norm_cfg"（均值为 0，标准差为 1）。

    参数:
        to_float32 (bool): 是否将加载的图像转换为 float32 类型的 numpy 数组。
            如果设置为 False，加载的图像将是 uint8 类型的数组。默认为 False。
        color_type (str): 传递给 :func:`mmcv.imfrombytes` 的标志参数。
            默认为 'color'。
        file_client_args (dict): 用于实例化 FileClient 的参数。
            有关详细信息，请参阅 :class:`mmcv.fileio.FileClient`。
            默认为 ``dict(backend='disk')``。
        imdecode_backend (str): 用于 :func:`mmcv.imdecode` 的后端。默认值:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """调用函数以加载图像并获取图像元信息。

        参数:
            results (dict): 来自 :obj:`mmseg.CustomDataset` 的结果字典。

        返回:
            dict: 包含加载的图像和元信息的字典。
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # 原来的数据加载语句
        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(
        #     img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        #现在的语句
        dataset = gdal.Open(filename)
        img = dataset.ReadAsArray()
        img = img.transpose((1,2,0)) #GDAL读取文件是（C,W,H）,而numpy是(H,W,C),需要转换一下
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotations(object):
    """加载语义分割的标注信息。

        参数:
            reduce_zero_label (bool): 是否将所有标签值减 1。
                通常用于将 0 作为背景标签的数据集。
                默认值: False。
            file_client_args (dict): 用于实例化 FileClient 的参数。
                有关详细信息，请参阅 :class:`mmcv.fileio.FileClient`。
                默认值为 ``dict(backend='disk')``。
            imdecode_backend (str): 用于 :func:`mmcv.imdecode` 的后端。默认值:
                'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """调用函数以加载多种类型的标注信息。

        参数:
            results (dict): 来自 :obj:`mmseg.CustomDataset` 的结果字典。

        返回:
            dict: 包含已加载的语义分割标注信息的字典。
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

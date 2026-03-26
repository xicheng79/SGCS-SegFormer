# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS


@DATASETS.register_module()
class BaseSegDataset(BaseDataset):
    """用于语义分割的自定义数据集。文件结构示例如下。

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    BaseSegDataset 的图像/真实语义分割图对除后缀外应相同。有效的图像/真实语义分割图文件名对应类似于
    ``xxx{img_suffix}`` 和 ``xxx{seg_map_suffix}``（后缀中包含文件扩展名）。如果提供了分割文件，则 ``xxx`` 在 txt 文件中指定。
    否则，将加载 ``img_dir/`` 和 ``ann_dir`` 中的所有文件。
    更多详细信息请参考 ``docs/en/tutorials/new_dataset.md``。

    参数:
        ann_file (str): 注释文件路径。默认为 ''。
        metainfo (dict, 可选): 数据集的元信息，例如指定要加载的类别。默认为 None。
        data_root (str, 可选): ``data_prefix`` 和 ``ann_file`` 的根目录。默认为 None。
        data_prefix (dict, 可选): 训练数据的前缀。默认为 dict(img_path=None, seg_map_path=None)。
        img_suffix (str): 图像的后缀。默认值: '.jpg'
        seg_map_suffix (str): 分割图的后缀。默认值: '.png'
        filter_cfg (dict, 可选): 过滤数据的配置。默认为 None。
        indices (int 或 Sequence[int], 可选): 支持使用注释文件中的前几个数据，以便在较小的数据集上进行训练/测试。默认为 None，表示使用所有 ``data_infos``。
        serialize_data (bool, 可选): 是否使用序列化对象来占用内存。启用时，数据加载器工作进程可以使用主进程的共享内存，而不是创建副本。默认为 True。
        pipeline (list, 可选): 处理流水线。默认为 []。
        test_mode (bool, 可选): ``test_mode=True`` 表示处于测试阶段。默认为 False。
        lazy_init (bool, 可选): 是否在实例化时加载注释。在某些情况下，例如可视化，只需要数据集的元信息，此时不需要加载注释文件。通过设置 ``lazy_init=True``，``Basedataset`` 可以跳过加载注释以节省时间。默认为 False。
        max_refetch (int, 可选): 如果 ``Basedataset.prepare_data`` 获取到一个 None 图像，获取有效图像的最大额外循环次数。默认为 1000。
        ignore_index (int): 要忽略的标签索引。默认值: 255
        reduce_zero_label (bool): 是否将标签 0 标记为忽略。默认为 False。
        backend_args (dict, 可选): 实例化文件后端的参数。有关详细信息，请参阅 https://mmengine.readthedocs.io/en/latest/api/fileio.htm。默认为 None。
            注意: 需要 mmcv>=2.0.0rc4，mmengine>=0.2.0。
    """
    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """需要标签映射。

        ``label_map`` 是一个字典，其键为旧的标签 ID，值为新的标签 ID，用于在 ``load_annotations`` 中更改像素标签。当且仅当 ``cls.METAINFO`` 中的旧类别与 ``self._metainfo`` 中的新类别不相等，且两者都不为 None 时，`label_map` 才不为 None。

        参数:
            new_classes (list, tuple, 可选): 从元信息中获取的新类别名称。默认为 None。

        返回:
            dict, 可选: 从 ``cls.METAINFO`` 中的旧类别到 ``self._metainfo`` 中的新类别的映射。
        """
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

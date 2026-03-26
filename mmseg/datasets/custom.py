# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations

@DATASETS.register_module()
class CustomDataset(Dataset):
    """用于语义分割的自定义数据集。以下是一个文件结构示例。

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

    CustomDataset 的图像/地面真实语义分割对除后缀外应相同。有效的图像/地面真实语义分割文件名对应类似于 ``xxx{img_suffix}`` 和 ``xxx{seg_map_suffix}``（后缀中也包含扩展名）。如果提供了分割文件，则 ``xxx`` 在文本文件中指定。否则，将加载 ``img_dir/`` 和 ``ann_dir`` 中的所有文件。
    有关更多详细信息，请参阅 ``docs/en/tutorials/new_dataset.md``。

    参数:
        pipeline (list[dict]): 处理管道
        img_dir (str): 图像目录的路径
        img_suffix (str): 图像的后缀。默认值: '.jpg'
        ann_dir (str, 可选): 注释目录的路径。默认值: None
        seg_map_suffix (str): 分割图的后缀。默认值: '.png'
        split (str, 可选): 分割文本文件。如果指定了分割文件，则仅加载分割文件中后缀匹配的文件。否则，将加载 img_dir/ann_dir 中的所有图像。默认值: None
        data_root (str, 可选): 图像目录/注释目录的数据根目录。默认值: None。
        test_mode (bool): 如果 test_mode=True，则不会加载地面真实数据。
        ignore_index (int): 要忽略的标签索引。默认值: 255
        reduce_zero_label (bool): 是否将标签 0 标记为要忽略的标签。默认值: False
        classes (str | Sequence[str], 可选): 指定要加载的类别。如果为 None，则将使用 ``cls.CLASSES``。默认值: None。
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            分割图的调色板。如果未提供，且 self.PALETTE 为 None，则将生成随机调色板。默认值: None
        gt_seg_map_loader_cfg (dict): 构建 LoadAnnotations 以加载用于评估的地面真实数据，默认从磁盘加载。默认值: ``dict()``。
        file_client_args (dict): 实例化 FileClient 的参数。
            有关详细信息，请参阅 :class:`mmcv.fileio.FileClient`。
            默认值为 ``dict(backend='disk')``。
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=dict(),
                 file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
            reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir): #用于判断是否是绝对路径
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations 在这个部分修改数据加载方式
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """从目录中加载注释。

        参数:
            img_dir (str): 图像目录的路径
            img_suffix (str): 图像的后缀。
            ann_dir (str|None): 注释目录的路径。
            seg_map_suffix (str|None): 分割图的后缀。
            split (str|None): 分割文本文件。如果指定了分割文件，则仅加载分割文件中后缀匹配的文件。
                否则，将加载 img_dir/ann_dir 中的所有图像。默认值: None

        返回:
            list[dict]: 数据集中所有图像的信息。
        """
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        return img_infos

    def get_ann_info(self, idx):
        """通过索引获取注释信息。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 指定索引的注释信息。
        """
        return self.img_infos[idx]['annotation']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """在数据处理管道之后获取训练/测试数据。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 训练/测试数据（如果 `test_mode` 设置为 False，则包含注释信息）。
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """在数据处理管道之后获取训练数据和注释。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 经过数据处理管道后的训练数据和注释，包含管道引入的新键。
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """在数据处理管道之后获取测试数据。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 经过数据处理管道后的测试数据，包含管道引入的新键。
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """用于将结果格式化为数据集特定输出的占位符。"""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """获取一张用于评估的真实语义分割图。"""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        '''获取用于评估的真实语义分割图。'''
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval(self, preds, indices):
        """从每次迭代中收集评估结果。

        参数:
            preds (list[torch.Tensor] | torch.Tensor): 经过 argmax 操作后的分割逻辑值，形状为 (N, H, W)。
            indices (list[int] | int): 预测结果对应的真实标签索引。

        返回:
            list[torch.Tensor]: (交集面积, 并集面积, 预测面积, 真实标签面积)。
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the label map has already been applied and zero label
                    # has already been reduced by get_gt_seg_map_by_idx() i.e.
                    # LoadAnnotations.__call__(), these operations should not
                    # be duplicated. See the following issues/PRs:
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # https://github.com/open-mmlab/mmsegmentation/pull/1417
                    # https://github.com/open-mmlab/mmsegmentation/pull/2504
                    # for more details
                    label_map=dict(),
                    reduce_zero_label=False))

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """获取当前数据集的类别名称。

        参数:
            classes (Sequence[str] | str | None): 如果 classes 为 None，则使用内置数据集定义的默认 CLASSES。
                如果 classes 是一个字符串，则将其视为文件名。该文件包含类别名称，每行一个类别名。
                如果 classes 是一个元组或列表，则覆盖数据集定义的 CLASSES。
            palette (Sequence[Sequence[int]] | np.ndarray | None):
                分割图的调色板。如果为 None，则将生成随机调色板。默认值: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # 字典，其键为旧的标签 ID，值为新的标签 ID。
            # 用于在 load_annotations 函数中更改像素标签。
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = 255
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # 在设置随机种子之前获取随机状态，并在之后恢复随机状态。
                # 这将防止随机性丢失，因为如果不指定，每次迭代中的调色板可能会不同。
                # 请参阅：https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """评估数据集。

        参数:
            results (list[tuple[torch.Tensor]] | list[str]): 每张图像的预评估结果或用于计算评估指标的预测分割图。
            metric (str | list[str]): 要评估的指标。支持 'mIoU'、'mDice' 和 'mFscore'。
            logger (logging.Logger | None | str): 评估期间用于打印相关信息的日志记录器。默认值: None。
            gt_seg_maps (generator[ndarray]): 自定义的真实分割图作为输入，用于 ConcatDataset

        返回:
            dict[str, float]: 默认指标。
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=False)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results

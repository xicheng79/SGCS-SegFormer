# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy
from itertools import chain

import mmcv
import numpy as np
from mmcv.utils import build_from_cfg, print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, PIPELINES

@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """一个连接数据集的包装器。

    与 :obj:`torch.utils.data.dataset.ConcatDataset` 相同，但支持评估和格式化结果。

    参数:
        datasets (list[:obj:`Dataset`]): 数据集列表。
        separate_eval (bool): 是否分别评估连接的数据集结果，默认为 True。
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE
        self.separate_eval = separate_eval
        assert separate_eval in [True, False], \
            f'separate_eval can only be True or False,' \
            f'but get {separate_eval}'
        if any([isinstance(ds, CityscapesDataset) for ds in datasets]):
            raise NotImplementedError(
                'Evaluating ConcatDataset containing CityscapesDataset'
                'is not supported!')

    def evaluate(self, results, logger=None, **kwargs):
        """评估结果。

        参数:
            results (list[tuple[torch.Tensor]] | list[str]]): 每张图像的
            预评估结果或预测分割图，用于计算评估指标。
            logger (logging.Logger | str | None): 用于在评估期间打印相关信息的日志记录器。默认值: None。

        返回:
            dict[str: float]: 总数据集的评估结果或每个单独数据集的评估结果
            如果 `self.separate_eval=True`。
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # 检查所有数据集是否支持评估
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.img_dir} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results

        if len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types when '
                'self.separate_eval=False')
        else:
            if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                    results, str):
                # merge the generators of gt_seg_maps
                gt_seg_maps = chain(
                    *[dataset.get_gt_seg_maps() for dataset in self.datasets])
            else:
                # if the results are `pre_eval` results,
                # we do not need gt_seg_maps to evaluate
                gt_seg_maps = None
            eval_results = self.datasets[0].evaluate(
                results, gt_seg_maps=gt_seg_maps, logger=logger, **kwargs)
            return eval_results

    def get_dataset_idx_and_sample_idx(self, indice):
        """给定 ConcatDataset 的索引时，返回数据集和样本索引。

        参数:
            indice (int): ConcatDataset 中样本的索引

        返回:
            int: 样本所属子数据集的索引
            int: 样本在其对应子集中的索引
        """
        if indice < 0:
            if -indice > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            indice = len(self) + indice
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, indice)
        if dataset_idx == 0:
            sample_idx = indice
        else:
            sample_idx = indice - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """format result for every sample of ConcatDataset."""
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].format_results(
                [results[i]],
                imgfile_prefix + f'/{dataset_idx}',
                indices=[sample_idx],
                **kwargs)
            ret_res.append(res)
        return sum(ret_res, [])

    def pre_eval(self, preds, indices):
        """do pre eval for every sample of ConcatDataset."""
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].pre_eval(preds[i], sample_idx)
            ret_res.append(res)
        return sum(ret_res, [])


@DATASETS.register_module()
class RepeatDataset(object):
    """重复数据集的包装器。

    重复数据集的长度将是原始数据集的 `times` 倍。当数据加载时间较长但数据集较小时，这很有用。
    使用 RepeatDataset 可以减少 epoch 之间的数据加载时间。

    参数:
        dataset (:obj:`Dataset`): 要重复的数据集。
        times (int): 重复次数。
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len


@DATASETS.register_module()
class MultiImageMixDataset:
    """多个图像混合数据集的包装器。

    适用于在多个图像混合数据增强（如 mosaic 和 mixup）上进行训练。
    对于混合图像数据的增强管道，需要提供 `get_indexes` 方法以获取图像索引，
    并且可以设置 `skip_flags` 来更改管道运行过程。

    参数:
        dataset (:obj:`CustomDataset`): 要混合的数据集。
        pipeline (Sequence[dict]): 要组成的变换对象或配置字典序列。
        skip_type_keys (list[str], optional): 要跳过管道的类型字符串序列。默认为 None。
    """

    def __init__(self, dataset, pipeline, skip_type_keys=None):
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indexes
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """更新 skip_type_keys。

        它由外部钩子调用。

        参数:
            skip_type_keys (list[str], optional): 要跳过管道的类型字符串序列。
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

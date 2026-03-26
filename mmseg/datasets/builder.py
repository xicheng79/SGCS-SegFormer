# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader, IterableDataset

from .samplers import DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .dataset_wrappers import ConcatDataset
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)
    # pop 'separate_eval' since it is not a valid key for common datasets.
    separate_eval = cfg.pop('separate_eval', True)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                                   RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=False,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """构建 PyTorch 数据加载器。

    在分布式训练中，每个 GPU/进程都有一个数据加载器。
    在非分布式训练中，所有 GPU 仅使用一个数据加载器。

    参数:
        dataset (Dataset): 一个 PyTorch 数据集。
        samples_per_gpu (int): 每个 GPU 上的训练样本数量，即每个 GPU 的批量大小。
        workers_per_gpu (int): 每个 GPU 用于数据加载的子进程数量。
        num_gpus (int): GPU 的数量。仅在非分布式训练中使用。
        dist (bool): 是否进行分布式训练/测试。默认值: True。
        shuffle (bool): 是否在每个训练周期打乱数据。
            默认值: True。
        seed (int | None): 要使用的随机种子。默认值: None。
        drop_last (bool): 是否丢弃每个周期中最后一个不完整的批次。
            默认值: False
        pin_memory (bool): 是否在数据加载器中使用锁页内存。
            默认值: True
        persistent_workers (bool): 如果为 True，数据加载器在数据集被使用一次后不会关闭工作进程。
            这允许保持工作进程的数据集实例处于活动状态。
            该参数在 PyTorch >= 1.7.0 版本中也有效。
            默认值: True
        kwargs: 用于初始化数据加载器的任何关键字参数

    返回:
        DataLoader: 一个 PyTorch 数据加载器。
    """
    rank, world_size = get_dist_info()
    if dist and not isinstance(dataset, IterableDataset):
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, seed=seed)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    elif dist:
        sampler = None
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

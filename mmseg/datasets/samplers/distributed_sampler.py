# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmseg.core.utils import sync_random_seed
from mmseg.utils import get_device


class DistributedSampler(_DistributedSampler):
    """继承自 `torch.utils.data.DistributedSampler` 的 `DistributedSampler`。

    参数:
        datasets (Dataset): 将要加载的数据集。
        num_replicas (int, 可选): 参与分布式训练的进程数量。默认情况下，会从当前分布式组中获取世界大小（world_size）。
        rank (int, 可选): 当前进程在 `num_replicas` 中的排名。默认情况下，会从当前分布式组中获取排名。
        shuffle (bool): 如果为 True（默认值），采样器将对索引进行打乱。
        seed (int): 当 `shuffle=True` 时，用于打乱采样器的随机种子。该数字在分布式组的所有进程中应保持一致。默认值: ``0``。
    """

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = False,
                 seed=0) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        device = get_device()
        self.seed = sync_random_seed(seed, device)

    def __iter__(self) -> Iterator:
        """
         Yields:
            Iterator: iterator of indices for rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

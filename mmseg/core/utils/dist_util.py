# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def check_dist_init():
    return dist.is_available() and dist.is_initialized()


def sync_random_seed(seed=None, device='cuda'):
    """确保不同的进程排名（rank）共享相同的随机种子。所有工作进程都必须调用此函数，否则会发生死锁。此方法通常用于 `DistributedSampler` 中，因为在分布式组中的所有进程的随机种子都应该相同。

    在分布式采样中，不同的进程排名应该从数据集中采样不重叠的数据。因此，此函数用于确保每个进程排名基于相同的随机种子以相同的顺序对数据索引进行洗牌。然后不同的进程排名可以使用不同的索引从相同的数据列表中选择不重叠的数据。

    参数:
        seed (int, 可选): 随机种子。默认为 None。
        device (str): 随机种子将放置的设备。
            默认为 'cuda'。
    返回:
        int: 要使用的随机种子。
    """

    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

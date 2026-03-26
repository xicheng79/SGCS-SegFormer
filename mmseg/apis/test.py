# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """将 ndarray 保存到本地的 numpy 文件中。

    参数:
        array (ndarray): 要保存的 ndarray。
        temp_file_name (str): numpy 文件的名称。如果 'temp_file_name=None'，此函数将使用 tempfile.NamedTemporaryFile 生成一个文件名来保存 ndarray。默认值: None。
        tmpdir (str): 保存 ndarray 文件的临时目录。默认值: None。
    返回:
        str: numpy 文件的名称。
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """通过渐进模式使用单 GPU 进行测试。

    参数:
        model (nn.Module): 待测试的模型。
        data_loader (utils.data.Dataloader): Pytorch 数据加载器。
        show (bool): 推理过程中是否显示结果。默认值: False。
        out_dir (str, 可选): 如果指定，结果将被保存到该目录中。
        efficient_test (bool): 评估期间是否将结果保存为本地 numpy 文件以节省 CPU 内存。与 pre_eval 和 format_results 互斥。默认值: False。
        opacity (float): 绘制的分割图的不透明度。
            默认值为 0.5。
            必须在 (0, 1] 范围内。
        pre_eval (bool): 使用 dataset.pre_eval() 函数生成用于指标评估的预结果。与 efficient_test 和 format_results 互斥。默认值: False。
        format_only (bool): 仅格式化结果以提交。
            与 pre_eval 和 efficient_test 互斥。
            默认值: False。
        format_args (dict): 用于 format_results 的参数。默认值: {}。
    返回:
        list: 评估预结果列表或保存文件名列表。
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # dataset = data_loader['dataset']
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0] #只保
            #只保留img_metas的第一个元素
            img_metas = [img_metas[0]]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """通过渐进模式使用多个 GPU 测试模型。

    此方法使用多个 GPU 测试模型，并在两种不同模式下收集结果：GPU 模式和 CPU 模式。通过设置 'gpu_collect=True'，
    它会将结果编码为 GPU 张量，并使用 GPU 通信来收集结果。在 CPU 模式下，它会将不同 GPU 上的结果保存到 'tmpdir' 中，
    并由排名为 0 的工作进程收集这些结果。

    参数:
        model (nn.Module): 待测试的模型。
        data_loader (utils.data.Dataloader): Pytorch 数据加载器。
        tmpdir (str): 在 CPU 模式下，用于保存不同 GPU 临时结果的目录路径。相同的路径也用于高效测试。默认值: None。
        gpu_collect (bool): 选择使用 GPU 还是 CPU 来收集结果。默认值: False。
        efficient_test (bool): 在评估期间是否将结果保存为本地 numpy 文件以节省 CPU 内存。与 pre_eval 和 format_results 互斥。默认值: False。
        pre_eval (bool): 使用 dataset.pre_eval() 函数生成用于指标评估的预结果。与 efficient_test 和 format_results 互斥。默认值: False。
        format_only (bool): 仅格式化结果以提交。与 pre_eval 和 efficient_test 互斥。默认值: False。
        format_args (dict): 用于 format_results 的参数。默认值: {}。

    返回:
        list: 评估预结果列表或保存文件名列表。
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

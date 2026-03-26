# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.datasets.pipelines.formatting import ToDataContainer
from mmseg.models import build_segmentor
import numpy as np

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """从配置文件初始化一个分割器。

    参数:
        config (str 或 :obj:`mmcv.Config`): 配置文件路径或配置对象。
        checkpoint (str, 可选): 检查点路径。如果留为 None，模型将不加载任何权重。
        device (str, 可选): CPU/CUDA 设备选项。默认为 'cuda:0'。
            使用 'cpu' 可在 CPU 上加载模型。
    返回:
        nn.Module: 构建好的分割器。
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """调用函数将图像加载到 results 中。

        参数:
            results (dict): 一个包含待读取图像文件名的结果字典。

        返回:
            dict: 返回包含已加载图像的 ``results``。
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img.astype(np.int32)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, imgs, img_filname=None):
    """使用分割器对图像进行推理。

    参数:
        model (nn.Module): 加载好的分割器。
        imgs (str/ndarray 或 list[str/ndarray]): 可以是图像文件路径，也可以是已加载的图像。

    返回:
        (list[Tensor]): 分割结果。
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        # img_meatas = dict(filename=None)
        # img_data2 = dict(img_meatas=img_meatas)
        img_data = dict(img=img)
        # img_data['image_metas'] = None
        img_data = test_pipeline(img_data)
        img_data['img_metas'][0].data['filename'] = img_filname
        data.append(img_data)
        
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True,
                       out_file=None):
    """在图像上可视化分割结果。

    参数:
        model (nn.Module): 加载好的分割器。
        img (str 或 np.ndarray): 图像文件名或已加载的图像。
        result (list): 分割结果。
        palette (list[list[int]]] | None): 分割图的调色板。如果为 None，则将生成随机调色板。
            默认值: None
        fig_size (tuple): pyplot 图形的大小。
        opacity(float): 绘制的分割图的不透明度。
            默认值为 0.5。
            必须在 (0, 1] 范围内。
        title (str): pyplot 图形的标题。
            默认值为空字符串。
        block (bool): 是否阻塞 pyplot 图形。
            默认值为 True。
        out_file (str 或 None): 保存图像的路径。
            默认值: None。
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

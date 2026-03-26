# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from PIL import Image

from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[SegDataSample, SampleList]


class MMSegInferencer(BaseInferencer):
    """语义分割推理器，提供推理和可视化接口。注意：需要 MMEngine >= 0.5.0。

    参数:
        model (str, 可选): 配置文件的路径或元文件中定义的模型名称。
            以 `mmseg 元文件 <https://github.com/open-mmlab/mmsegmentation/blob/main/configs/fcn/metafile.yaml>`_ 为例，
            `model` 可以是 "fcn_r50-d8_4xb2-40k_cityscapes-512x1024"，模型权重将自动下载。
            如果使用配置文件，如 "configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py"，则需要指定 `weights`。
        weights (str, 可选): 检查点的路径。如果未指定且 `model` 是元文件中的模型名称，权重将从元文件中加载。默认为 None。
        classes (list, 可选): 用于结果渲染的输入类别。由于分割模型的预测结果是一个带有标签索引的分割图，
            `classes` 是一个包含与标签索引对应的类别的列表。如果未定义 `classes`，可视化器将默认使用 `cityscapes` 数据集的类别。默认为 None。
        palette (list, 可选): 用于结果渲染的输入调色板，它是一个与类别对应的颜色调色板列表。
            如果未定义 `palette`，可视化器将默认使用 `cityscapes` 数据集的调色板。默认为 None。
        dataset_name (str, 可选): `数据集名称或别名 <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_，
            可视化器将使用数据集的元信息，即类别和调色板，但 `classes` 和 `palette` 具有更高的优先级。默认为 None。
        device (str, 可选): 运行推理的设备。如果为 None，将自动使用可用的设备。默认为 None。
        scope (str, 可选): 模型的作用域。默认为 'mmseg'。
    """ # noqa

    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode', 'out_dir'}
    visualize_kwargs: set = {'show', 'wait_time', 'img_out_dir', 'opacity'}
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 classes: Optional[Union[str, List]] = None,
                 palette: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmseg') -> None:
        # A global counter tracking the number of images processes, for
        # naming of the output images
        self.num_visualized_imgs = 0
        self.num_pred_imgs = 0
        init_default_scope(scope if scope else 'mmseg')
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        if device == 'cpu' or not torch.cuda.is_available():
            self.model = revert_sync_batchnorm(self.model)

        assert isinstance(self.visualizer, SegLocalVisualizer)
        self.visualizer.set_dataset_meta(palette, classes, dataset_name)

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """从配置文件（cfg）和检查点（checkpoint）中加载模型权重和元信息。

        子类可以重写此方法，以从 ``checkpoint`` 和 ``cfg`` 中加载额外的元信息到模型中。

        参数:
            model (nn.Module): 用于加载权重和元信息的模型。
            checkpoint (dict, 可选): 已加载的检查点。
            cfg (Config 或 ConfigDict, 可选): 已加载的配置文件。
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmsegmentation 1.x
                model.dataset_meta = {
                    'classes': checkpoint_meta['dataset_meta'].get('classes'),
                    'palette': checkpoint_meta['dataset_meta'].get('palette')
                }
            elif 'CLASSES' in checkpoint_meta:
                # mmsegmentation 0.x
                classes = checkpoint_meta['CLASSES']
                palette = checkpoint_meta.get('PALETTE', None)
                model.dataset_meta = {'classes': classes, 'palette': palette}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use classes of Cityscapes by '
                    'default.')
                model.dataset_meta = {
                    'classes': get_classes('cityscapes'),
                    'palette': get_palette('cityscapes')
                }
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn(
                'weights is None, use cityscapes classes by default.')
            model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 show: bool = False,
                 wait_time: int = 0,
                 out_dir: str = '',
                 img_out_dir: str = 'vis',
                 pred_out_dir: str = 'pred',
                 **kwargs) -> dict:
        """调用推理器。

        参数:
            inputs (Union[list, str, np.ndarray]): 推理器的输入。
            return_datasamples (bool): 是否将结果作为 :obj:`SegDataSample` 返回。默认为 False。
            batch_size (int): 批次大小。默认为 1。
            show (bool): 是否在弹出窗口中显示渲染的彩色分割掩码。默认为 False。
            wait_time (float): 显示间隔时间（秒）。默认为 0。
            out_dir (str): 推理结果的输出目录。默认为 ''。
            img_out_dir (str): `out_dir` 的子目录，用于保存渲染的彩色分割掩码，因此如果您想保存预测掩码，则必须定义 `out_dir`。默认为 'vis'。
            pred_out_dir (str): `out_dir` 的子目录，用于保存预测的掩码文件，因此如果您想保存预测掩码，则必须定义 `out_dir`。默认为 'pred'。

            **kwargs: 传递给 :meth:`preprocess`、:meth:`forward`、:meth:`visualize` 和 :meth:`postprocess` 的其他关键字参数。
                kwargs 中的每个键都应在 ``preprocess_kwargs``、``forward_kwargs``、``visualize_kwargs`` 和 ``postprocess_kwargs`` 对应的集合中。

        返回:
            dict: 推理和可视化结果。
        """

        if out_dir != '':
            pred_out_dir = osp.join(out_dir, pred_out_dir)
            img_out_dir = osp.join(out_dir, img_out_dir)
        else:
            pred_out_dir = ''
            img_out_dir = ''

        return super().__call__(
            inputs=inputs,
            return_datasamples=return_datasamples,
            batch_size=batch_size,
            show=show,
            wait_time=wait_time,
            img_out_dir=img_out_dir,
            pred_out_dir=pred_out_dir,
            **kwargs)

    def visualize(self,
                  inputs: list,
                  preds: List[dict],
                  show: bool = False,
                  wait_time: int = 0,
                  img_out_dir: str = '',
                  opacity: float = 0.8) -> List[np.ndarray]:
        """可视化预测结果。

        参数:
            inputs (list): 经过 :meth:`_inputs_to_list` 预处理后的输入。
            preds (Any): 模型的预测结果。
            show (bool): 是否在弹出窗口中显示图像。
                默认值为 False。
            wait_time (float): 显示间隔时间（秒）。默认值为 0。
            img_out_dir (str): 渲染预测结果（即彩色分割掩码）的输出目录。
                默认值: ''
            opacity (int, float): 分割掩码的透明度。
                默认值为 0.8。

        返回:
            List[np.ndarray]: 可视化结果。
        """
        if self.visualizer is None or (not show and img_out_dir == ''):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None')
        self.visualizer.set_dataset_meta(**self.model.dataset_meta)
        self.visualizer.alpha = opacity

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8) + '_vis'
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type:'
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=True,
                out_file=out_file)
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results

    def postprocess(self,
                    preds: PredType,
                    visualization: List[np.ndarray],
                    return_datasample: bool = False,
                    pred_out_dir: str = '') -> dict:
        """处理 ``forward`` 和 ``visualize`` 方法输出的预测结果和可视化结果。

        此方法应负责以下任务：

        1. 打包预测结果和可视化结果并返回。
        2. 若有需要，保存预测结果。

        参数:
            preds (List[Dict]): 模型的预测结果。
            visualization (List[np.ndarray]): 渲染后的彩色分割掩码列表。
            return_datasample (bool): 是否以数据样本的形式返回结果。
                默认值为 False。
            pred_out_dir: 用于保存无可视化效果的推理结果的文件路径。如果留空，则不会保存任何文件。
                默认值为 ''。

        返回:
            dict: 包含键 ``predictions`` 和 ``visualization`` 的推理和可视化结果字典。

            - ``visualization (Any)``: 由 :meth:`visualize` 方法返回。
            - ``predictions`` (List[np.ndarray], np.ndarray): 由 :meth:`forward` 方法返回，并在 :meth:`postprocess` 方法中处理。
              如果 ``return_datasample=False``，它将是带有标签索引的分割掩码。
        """
        if return_datasample:
            if len(preds) == 1:
                return preds[0]
            else:
                return preds

        results_dict = {}

        results_dict['predictions'] = []
        results_dict['visualization'] = []

        for i, pred in enumerate(preds):
            pred_data = pred.pred_sem_seg.numpy().data[0]
            results_dict['predictions'].append(pred_data)
            if visualization is not None:
                vis = visualization[i]
                results_dict['visualization'].append(vis)
            if pred_out_dir != '':
                mmengine.mkdir_or_exist(pred_out_dir)
                img_name = str(self.num_pred_imgs).zfill(8) + '_pred.png'
                img_path = osp.join(pred_out_dir, img_name)
                output = Image.fromarray(pred_data.astype(np.uint8))
                output.save(img_path)
            self.num_pred_imgs += 1

        if len(results_dict['predictions']) == 1:
            results_dict['predictions'] = results_dict['predictions'][0]
            if visualization is not None:
                results_dict['visualization'] = \
                    results_dict['visualization'][0]
        return results_dict

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """初始化测试流水线。

        返回一个用于处理各种输入数据（如 ``str``、``np.ndarray``）的流水线。这是 BaseInferencer 中的一个抽象方法，需要在子类中实现。

        返回的流水线将用于处理单个数据。它将在 :meth:`preprocess` 方法中按如下方式使用：

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]
        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')

        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'InferencerLoader'
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

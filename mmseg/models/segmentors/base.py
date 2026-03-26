# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """分割器的基类。"""

    def __init__(self, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """布尔值：分割器是否有颈部结构"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """布尔值：分割器是否有辅助头"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """布尔值：分割器是否有解码头"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """从图像中提取特征的占位符。"""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """使用主干网络编码图像并解码为与输入相同大小的语义分割图的占位符。"""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """训练时前向函数的占位符。"""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """单图像测试的占位符。"""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """增强测试的占位符。"""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        参数：
            imgs (List[Tensor])：外层列表表示测试时间的增强，
                内层张量应具有形状NxCxHxW，包含批次中的所有图像。
            img_metas (List[List[dict]])：外层列表表示测试时间的增强
                （多尺度、翻转等），内层列表表示批次中的图像。
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """根据``return_loss``是否为``True``调用:func:`forward_train`或:func:`forward_test`。

        注意，此设置将更改预期的输入。当``return_loss=True``时，
        img和img_meta是单层嵌套的（即Tensor和List[dict]），
        当``return_loss=False``时，img和img_meta应该是双层嵌套的
        （即List[Tensor]，List[List[dict]]），外层列表表示测试时间增强。
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
            # return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
            # return self.forward_test(img, **kwargs)
    def train_step(self, data_batch, optimizer, **kwargs):
        """训练期间的迭代步骤。

        此方法定义了训练期间的迭代步骤，除了反向传播和优化器更新外，
        这些在优化器钩子中完成。请注意，在某些复杂的情况或模型中，
        包括反向传播和优化器更新在内的整个过程也在此方法中定义，例如GAN。

        参数：
            data (dict)：数据加载器的输出。
            optimizer (:obj:`torch.optim.Optimizer` | dict)：运行器的优化器
                传递给``train_step()``。此参数未使用且保留。

        返回：
            dict：它应该至少包含3个键：``loss``、``log_vars``、
                ``num_samples``。
                ``loss``是用于反向传播的张量，可以是多个损失的加权和。
                ``log_vars``包含要发送到记录器的所有变量。
                ``num_samples``表示批次大小（当模型是DDP时，
                它表示每个GPU上的批次大小），用于平均日志。
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """验证期间的迭代步骤。

        此方法与:func:`train_step`具有相同的签名，但在验证周期中使用。
        请注意，训练周期后的评估不是通过此方法实现的，而是通过评估钩子实现的。
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """解析网络的原始输出（损失）。

        参数：
            losses (dict)：网络的原始输出，通常包含损失和其他必要信息。

        返回：
            tuple[Tensor, dict]：(loss, log_vars)，loss是损失张量，
                可能是所有损失的加权和，log_vars包含要发送到记录器的所有变量。
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """在`img`上绘制`result`。

        参数：
            img (str or Tensor)：要显示的图像。
            result (Tensor)：要在`img`上绘制的语义分割结果。
            palette (list[list[int]]] | np.ndarray | None)：分割图的调色板。
                如果给定None，将生成随机调色板。默认值：None
            win_name (str)：窗口名称。
            wait_time (int)：waitKey参数的值。
                默认值：0。
            show (bool)：是否显示图像。
                默认值：False。
            out_file (str or None)：写入图像的文件名。
                默认值：None。
            opacity(float)：绘制的分割图的不透明度。
                默认值0.5。
                必须在(0, 1]范围内。
        返回：
            img (Tensor)：仅当不是`show`或`out_file`时
        """
        img = mmcv.imread(img)
        img = img.copy()
        #更改过
        img = img[..., :3]
        #
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        # color_seg = np.zeros((seg.shape[0], seg.shape[1], 4), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        # if show:
        #     mmcv.imshow(img, win_name, wait_time)
        # if out_file is not None:
        #     mmcv.imwrite(img, out_file)
        if out_file is not None:
            seg = np.array(seg)
            seg[seg>0] = 255
            mmcv.imwrite(seg, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

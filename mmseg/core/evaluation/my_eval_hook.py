import os.path as osp
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv import print_log
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        runner.model.eval()

        results = []
        for batch_indices, data in zip(self.dataloader.batch_sampler, self.dataloader):
            # 滑动窗口计算每张图片的预测结果
            label = self.dataloader.dataset.get_gt_seg_map_by_idx(batch_indices[0])
            with torch.no_grad():
                img_scale = tuple(runner.data_loader._dataloader.dataset[0]['img'].data.shape[1:])
                result = np.zeros_like(label)
                for i, l, c in self.sliding_window(data, label, img_scale):
                    result[c[0]:c[1], c[2]:c[3]] = runner.model(return_loss=False, **i)[0]
            C = self.dataloader.dataset.confusion_matrix(result, label)
            results.append(C)
        C = np.sum(results, 0)

        # 计算其他指标
        oa = self.dataloader.dataset.c2oa(C)
        pac = self.dataloader.dataset.c2pac(C)
        uac = self.dataloader.dataset.c2uac(C)
        kappa = self.dataloader.dataset.c2kappa(C)
        iou = self.dataloader.dataset.c2iou(C)
        miou = np.mean(iou)

        s = '\n'
        s += '-'*40 + '\n'
        s += 'Evaluation:\n'
        s += '-'*40 + '\n'
        s += 'OA   :\t{:.4f}\n'.format(oa)
        s += 'Kappa:\t{:.4f}\n'.format(kappa)
        s += 'mIoU :\t{:.4f}\n'.format(miou)
        s += '-'*40 + '\n'
        s += 'Producer_acc & User_acc & IoU:\n'
        for i, class_name in enumerate(self.dataloader.dataset.CLASSES):
            s += '{:^20}|{:^6.4f}|{:^6.4f}|{:^6.4f}\n'.format(class_name, pac[i], uac[i], iou[i])
        s += '-'*40 + '\n'
        print_log(s, logger=runner.logger)

        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        # 保存最优结果
        if self.save_best:
            self._save_ckpt(runner, oa)

    def sliding_window(self, data, label, img_scale):
        ori_shape = data['img_metas'][0].data[0][0]['img_shape']
        channel_num = data['img_metas'][0].data[0][0]['img_shape'][-1]
        meta_shape = (*img_scale, channel_num)
        data['img_metas'][0].data[0][0]['img_shape'] = meta_shape
        data['img_metas'][0].data[0][0]['pad_shape'] = meta_shape
        data['img_metas'][0].data[0][0]['ori_shape'] = meta_shape
        img = data['img'][0].clone()
        lab = label.copy()
        for y in range(0, ori_shape[0], img_scale[0]):
            if y + img_scale[0] > ori_shape[0]:
                y = ori_shape[0] - img_scale[0]
            for x in range(0, ori_shape[1], img_scale[1]):
                if x + img_scale[1] > ori_shape[1]:
                    x = ori_shape[1] - img_scale[1]
                c = [y, y+img_scale[0], x, x+img_scale[1]]
                data['img'][0] = img[:, :, c[0]:c[1], c[2]:c[3]]
                label = lab[c[0]:c[1], c[2]:c[3]]
                yield data, label, c


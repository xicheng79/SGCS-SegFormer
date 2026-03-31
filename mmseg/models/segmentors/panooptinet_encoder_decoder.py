# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmseg.ops import resize
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from ..builder import SEGMENTORS

@SEGMENTORS.register_module()
class PanoOptiNet_EncoderDecoder(EncoderDecoder):
    """基于继承的PanoOptiNet滑窗机制分割器，支持传递overlap信息。"""

    def __init__(self, *args, inference_tbti=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap = None
        self.inference_tbti = inference_tbti  # 推理时是否启用 TBTI（默认关闭）

    def extract_feat(self, img, img_metas, overlap):
        """从图像中提取特征，增加overlap参数用于特殊处理。"""
        x = self.backbone(img, img_metas, overlap)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, overlap):
        """编码解码流程，加入overlap参数用于引导窗口。"""
        x = self.extract_feat(img, img_metas, overlap)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """训练时的解码头前向传播，支持返回overlap。"""
        losses = dict()
        loss_decode, overlap = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        self.overlap = overlap
        losses.update({f'decode.{k}': v for k, v in loss_decode.items()})
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """测试时的解码头前向传播。

        当 inference_tbti=False（默认）时，不更新 self.overlap，
        FTP 注入自动失效，避免推理时跨图像传递错误特征。
        """
        seg_logits, overlap = self.decode_head.forward_test(
            x, img_metas, self.test_cfg)
        if self.inference_tbti:
            self.overlap = overlap
        # inference_tbti=False 时，self.overlap 保持 None，FTP 不会被激活
        return seg_logits

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """训练阶段的前向传播函数，增加overlap信息。"""
        x = self.extract_feat(img, img_metas, self.overlap)
        losses = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def whole_inference(self, img, img_meta, rescale):
        """使用完整图像进行推理，增加overlap控制。"""
        seg_logit = self.encode_decode(img, img_meta, self.overlap)
        if rescale:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit

    def slide_inference(self, img, img_meta, rescale):
        """重写滑窗推理，解决接口不兼容问题。
        
        父类 slide_inference 内部调用 self.encode_decode(crop_img, img_meta)，
        而本类重写的 encode_decode 需要额外的 overlap 参数，直接调用父类会导致
        参数数量不匹配的运行时错误。
        
        注意：此模式下 self.overlap 按顺序在每个 crop 之间传递，若 crop 顺序
        与 SAS 路径不一致则语义连续性可能不准确（架构级风险，详见 Bug 3 说明）。
        """
        import torch.nn.functional as F_pad
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                # 关键修复：调用子类 encode_decode，传入 self.overlap
                crop_seg_logit = self.encode_decode(crop_img, img_meta, self.overlap)
                preds += F_pad.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2),
                     int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def inference(self, img, img_meta, rescale):
        """推理接口，自动调用滑窗或整图模式，处理overlap。"""
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = torch.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        return output

    def simple_test(self, img, img_meta, rescale=True):
        """简单测试入口，对结果进行后处理。"""
        seg_logit = self.inference(img, img_meta, rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred.unsqueeze(0)
        return list(seg_pred.cpu().numpy())

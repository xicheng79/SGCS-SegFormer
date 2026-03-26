# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """级联编码器解码器分割器。

    CascadeEncoderDecoder与EncoderDecoder几乎相同，但CascadeEncoderDecoder的解码器是级联的。
    前一个decoder_head的输出将作为下一个decoder_head的输入。
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """初始化``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes
        self.out_channels = self.decode_head[-1].out_channels

    def encode_decode(self, img, img_metas):
        """使用主干网络编码图像并解码生成与输入相同尺寸的语义分割图。"""
        x = self.extract_feat(img)
        out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out, img_metas,
                                                   self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """在训练中运行前向函数并计算解码头的损失。"""
        losses = dict()

        loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            # 再次进行前向测试，对大多数方法可能不必要。
            if i == 1:
                prev_outputs = self.decode_head[0].forward_test(
                    x, img_metas, self.test_cfg)
            else:
                prev_outputs = self.decode_head[i - 1].forward_test(
                    x, prev_outputs, img_metas, self.test_cfg)
            loss_decode = self.decode_head[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses

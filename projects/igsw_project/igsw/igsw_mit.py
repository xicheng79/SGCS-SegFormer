# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.models import BACKBONES
from mmseg.models.backbones import ResNetV1c


@BACKBONES.register_module()
class DummyResNet(ResNetV1c):
    """实现一个用于演示目的的虚拟 ResNet 包装器。
    参数:
        **kwargs: 所有参数都将传递给父类。
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
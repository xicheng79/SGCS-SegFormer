# 虚拟 ResNet 包装器

这是社区 `projects/` 的一个示例 README 文件。我们以 HTML 注释的形式为每个字段提供了详细说明，当你查看此 README 文件的源代码时可以看到这些注释。如果你希望将你的项目提交到我们的主仓库，那么此 README 中的所有字段都是必需的，以便其他人了解你在此实现中所取得的成果。有关更多详细信息，请阅读我们的 [贡献指南](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/CONTRIBUTING.md) 或在 [讨论区](https://github.com/open-mmlab/mmsegmentation/discussions) 与我们交流。

## 描述

<!-- 分享任何你希望其他人了解的信息。例如：

作者：@xxx。

这是 [XXX] 的一个实现。 -->

本项目实现了一个虚拟的 ResNet 包装器，实际上它并没有做任何新的事情，只是在初始化时打印 “hello world”。

## 使用方法

<!-- 对于一个典型的模型，此部分应包含训练和测试的命令。建议你通过 `conda env export > env.yml` 将环境配置导出到 env.yml 文件中。 -->

### 先决条件

- Python 3.7
- PyTorch 1.6 或更高版本
- [MIM](https://github.com/open-mmlab/mim) v0.33 或更高版本
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v0.29.1 或更高版本

以下所有命令都依赖于 `PYTHONPATH` 的正确配置，`PYTHONPATH` 应指向项目目录，以便 Python 能够找到模块文件。在 `example_project/` 根目录下，运行以下命令将当前目录添加到 `PYTHONPATH` 中：

### Training commands

```shell
mim train mmsegmentation configs/fcn_dummy-r50-d8_4xb2-40k_cityscapes-512x1024.py --work-dir work_dirs/dummy_resnet
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmsegmentation configs/fcn_dummy-r50-d8_4xb2-40k_cityscapes-512x1024.py --work-dir work_dirs/dummy_resnet --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmsegmentation configs/fcn_dummy-r50-d8_4xb2-40k_cityscapes-512x1024.py --work-dir work_dirs/dummy_resnet --checkpoint ${CHECKPOINT_PATH} --eval mIoU
```

<!-- 像其他模型的 README 那样列出结果。[示例](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn#results-and-models)

你应该说明这是基于从官方发布版本转换而来的预训练权重；还是在本项目中重新训练模型得到的复现结果。 -->

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                             | download                                                                                                                                                                                                                                                                                                                           |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN    | R-50-D8  | 512x1024  |   40000 | 5.7      | 4.17           | 72.25 |         73.36 | [config](configs/fcn_dummy-r50-d8_4xb2-40k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x1024_40k_cityscapes/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x1024_40k_cityscapes/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608.log.json) |

## Citation

<!-- You may remove this section if not applicable. -->

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## 检查清单

以下是一个清单，展示了一个成功项目的常规开发工作流程，同时也概述了本项目的进展情况。

<!-- 本项目的负责人（PIC）或贡献者应勾选他们认为已完成的所有事项，这些事项将由代码库维护者通过拉取请求（PR）进一步验证。

OpenMMLab 的维护者将审查代码以确保项目质量。达到第一个里程碑意味着该项目满足合并到 `projects/` 目录的最低要求。但只有达到最后一个里程碑，该项目才有资格成为核心包的一部分。

请注意，及时更新此部分内容不仅对本项目的开发人员至关重要，对整个社区也很重要，因为可能会有其他贡献者加入本项目，并从这个清单中确定他们的起点。如果需要，它还能帮助维护者准确估计进一步代码优化所需的时间和精力。

一个项目不一定非要在一个拉取请求中完成，但项目至少要在第一个拉取请求中达到第一个里程碑。 -->

- [ ] 里程碑 1：准备好拉取请求，并且可以被纳入 `projects/` 目录。

  - [ ] 完成代码

  <!-- 代码设计应遵循现有的接口和约定。例如，每个模型组件都应注册到 `mmseg.registry.MODELS` 中，并可通过配置文件进行配置。 -->

  - [ ] 基本的文档字符串和正确的引用

  <!-- 每个主要对象都应包含一个文档字符串，描述其功能和参数。如果你从其他开源项目改编了代码，不要忘记在文档字符串中引用源项目，并确保你的行为不违反其许可协议。通常，我们不接受任何 GPL 许可下的代码片段。[开源许可简短指南](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] 测试时的正确性

  <!-- 如果你正在复现一篇论文中的结果，请确保你的模型在推理时的性能与原论文中的一致。权重通常可以通过简单地重命名官方预训练权重中的键来获得。不过，如果你能够证明训练时的正确性并勾选第二个里程碑，则可以跳过此测试。 -->

  - [ ] 完整的 README 文件

  <!-- 就像这个模板一样。 -->

- [ ] 里程碑 2：表明模型实现成功。

  - [ ] 训练时的正确性

  <!-- 如果你正在复现一篇论文中的结果，勾选此事项意味着你应该根据原论文的规范从头开始训练模型，并验证最终结果在较小的误差范围内与报告一致。 -->

- [ ] 里程碑 3：适合成为我们核心包的一部分！

  - [ ] 类型提示和文档字符串

  <!-- 理想情况下，*所有* 方法都应该有 [类型提示](https://www.pythontutorial.net/python-basics/python-type-hints/) 和 [文档字符串](https://google.github.io/styleguide/pyguide.html#381-docstrings)。[示例](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/utils/misc.py#L7) -->

  - [ ] 单元测试

  <!-- 每个模块都需要进行单元测试。[示例](https://github.com/open-mmlab/mmsegmentation/blob/master/tests/test_utils/test_misc.py) -->

  - [ ] 代码优化

  <!-- 根据评审人员的意见重构你的代码。 -->

  - [ ] Metafile.yml 文件

  <!-- 它将由 MIM 和 Inferencer 解析。[示例](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fcn/fcn.yml) -->

- [ ] 根据代码库的文件层次结构将你的模块移动到核心包中。

  <!-- 特别是，你可能需要将此 README 文件重构为标准文件。[示例](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fcn/README.md) -->

- [ ] 根据代码库的文件层次结构将你的模块重构到核心包中。

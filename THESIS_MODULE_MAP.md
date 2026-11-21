# 本毕设代码模块说明

## 先看整体

本项目不是“把 RGB 和红外图片合成一张图片”，而是一个双模态目标检测程序：

```text
RGB 图像 ──> RGB 特征分支 ──┐
                            ├─> 跨分辨率适配 ─> RGB/IR 融合 ─> YOLO 检测头
红外图像 ─> IR 特征分支 ────┘
                                      └─> 火焰框、人员框
```

红外 1:1 和 2:1 的区别只在于：RGB 保持输入尺寸，2:1 时红外先缩小到宽高各一半；
模型仍然会把对应特征对齐后再融合。

## 你的毕设真正需要关注的文件

| 文件 | 作用 | 是否是毕设核心 |
|---|---|---|
| `configs/rgbt3m.yaml` | RGBT-3M 主实验配置 | 是 |
| `configs/rgbt3m_ratio2.yaml` | 红外 2:1 实验配置 | 是 |
| `configs/hierafire.yaml` | 原始 HieraFire 双流网络配置 | 是，基线 |
| `configs/hierafire_gated.yaml` | 特征适配 + 门控融合配置 | 是，改进模型 |
| `configs/yolov8n_dual_concat.yaml` | YOLOv8n 风格双流直接拼接基线 | 是，基础对照 |
| `ultralytics/data/dataset.py` | 配对读取 RGB、IR、标签，控制红外尺寸 | 是，数据核心 |
| `ultralytics/data/build.py` | 根据配置创建数据集和 DataLoader | 是，数据入口辅助文件 |
| `ultralytics/nn/tasks.py` | 创建双流网络、跨尺度对齐、连接检测头 | 是，模型核心 |
| `ultralytics/nn/modules/cross_resolution.py` | 特征适配器和 RGB/IR 门控 | 是，你的创新模块 |
| `ultralytics/nn/modules/conv.py` | YOLO 基础卷积 | 是，原框架基础，不建议重写 |
| `ultralytics/nn/modules/block.py` | YOLO Backbone/Neck 基础块及原有融合块 | 是，主要复用 |
| `ultralytics/nn/modules/head.py` | YOLO 检测头 | 是，主要复用 |
| `ultralytics/nn/modules/transformer.py` | Transformer 类通用模块 | 暂时不是你的创新模块 |
| `train.py` | 启动训练 | 是 |
| `val.py` | 验证并输出指标 | 是 |
| `predict.py` | 单图/样本预测 | 是，展示结果时用 |
| `profile_flops.py` | 参数量、FLOPs、延迟测试辅助 | 是，效率实验用 |
| `tools/convert_wildfire_masks_to_yolo.py` | 其他野火数据格式转换 | 仅使用 Wildfire 数据时需要 |
| `tests/` | 自动检查模型和数据是否正常 | 建议保留 |

## 为什么 `build.py` 会用 NumPy

这不表示你的毕设变成了 NumPy 项目。数据流是：

```text
磁盘图片 -> OpenCV -> NumPy 数组 -> 同步增强/拼接 -> PyTorch Tensor -> 模型
```

因此 `numpy` 是数据处理的基础依赖，和 PyTorch 并不冲突。`build.py` 中的 `numpy`
还用于随机种子、识别内存中的图片等公共功能。它不是一个需要你单独运行的脚本，
训练时会被 `dataset.py` 和训练器自动调用。

## 暂时不用逐个学习的目录

项目保留了 Ultralytics 原框架的通用能力，例如分割、姿态、跟踪、导出、RT-DETR、
视频流和通用解决方案。这些文件是框架自带的兼容功能，不代表你的论文必须实现这些
任务。当前论文主线只需要：检测、RGB/IR 配对数据、1:1/2:1 分辨率、特征适配、
模态门控、验证指标和效率统计。

## 推荐学习和运行顺序

1. `train.py`：知道程序从哪里启动。
2. `ultralytics/data/build.py`：知道训练器如何拿到数据集。
3. `ultralytics/data/dataset.py`：看 RGB、IR 和标签如何配对、如何做 2:1。
4. `ultralytics/nn/tasks.py`：看双流 Backbone、融合和检测头如何连接。
5. `ultralytics/nn/modules/cross_resolution.py`：看你的创新模块。
6. `ultralytics/nn/modules/head.py`：最后理解检测框是怎样输出的。
7. `val.py`、`profile_flops.py`：完成论文指标和效率实验。

## 当前结论

项目已经包含你毕设的主要代码骨架：RGB 单模态、IR 单模态、双流检测、红外 1:1/2:1、
简单融合、跨分辨率特征适配和门控融合。现在缺的不是再复制一套目录，而是按照论文
实验表逐项训练、验证和记录结果。

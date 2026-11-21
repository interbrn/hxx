# 配置说明

配置文件分为两类：

- 数据配置：`rgbt3m_*.yaml`，描述数据集路径、类别、划分文件和分辨率比例。
- 模型配置：`yolov8n_dual_*.yaml`，描述 YOLOv8n 双流结构和融合方式。

## 数据配置约定

正式数据配置使用视频分组划分：

```text
train: splits/train_grouped.txt
val: splits/val_grouped.txt
test: splits/test_grouped.txt
```

本地 CPU 配置使用 `*_micro.yaml`，对应 `splits/*_grouped_micro.txt`；该配置只用于流程验证。

所有正式配置遵循以下约定：

| 项目 | 设置 |
|---|---|
| 类别 | `fire`、`person` |
| RGB | 480×640 |
| 1∶1 IR | 480×640 |
| 2∶1 IR | 240×320 |
| 输入通道 | 6 通道双模态输入 |
| 标签格式 | YOLO 检测格式 |

`train.py` 会根据 `--data-root` 覆盖 YAML 中的 `path`，因此配置文件不包含个人电脑路径。

## 模型配置

| 模型文件 | 用途 |
|---|---|
| `yolov8n_dual_concat.yaml` | 1∶1 双流直接拼接 |
| `yolov8n_dual_concat_ratio2.yaml` | 2∶1 双流直接拼接 |
| `yolov8n_dual_simple_ratio2.yaml` | 2∶1 固定等权融合 |
| `yolov8n_dual_adapter_ratio2.yaml` | 2∶1 多感受野特征适配 |
| `yolov8n_dual_gated_ratio2.yaml` | 2∶1 通道与空间门控 |

扩展模型 `hierafire*.yaml` 不属于任务书核心矩阵，正式使用时应单独记录其结构和消融条件。

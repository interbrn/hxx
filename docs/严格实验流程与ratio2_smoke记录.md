# 严格实验流程与 smoke 记录

## 1. 固定流程

所有模型使用相同的数据划分和测试流程：

1. RGB 和 IR 按相同文件名配对读取。
2. 按视频分组划分 train、val 和 test。
3. 每个 epoch 使用 val 集评估。
4. 保存 val 指标最好的 `best.pt`。
5. 训练结束后只使用 `best.pt` 在独立 test 集评估。
6. 将 Precision、Recall、mAP50 和 mAP50-95 写入 `test_metrics.json`。

## 2. 2∶1 输入规则

| 模态 | 输入尺寸 |
|---|---:|
| RGB | 480×640 |
| IR | 240×320 |

IR 分支先使用面积插值缩小，再经过独立 Backbone。IR 特征在 P2、P3、P4 和 P5 层对齐到 RGB 特征尺寸后再融合。

## 3. 历史 smoke 数据规模

| 划分 | 图片数量 |
|---|---:|
| 训练集 | 1387 |
| 验证集 | 618 |
| 测试集 | 237 |

训练参数为 2 epoch、batch=1、nbs=1、AdamW、CPU、无增强、关闭辅助损失。结果只能用于流程验证。

上述 1387/618/237 是早期较大 smoke 运行，仍作为历史记录保留。当前 `local_cpu` profile 默认使用更小的 micro 划分，以便在无 NVIDIA GPU 的电脑上完成全实验矩阵的流程回归。

## 4. 当前 micro smoke 重跑

本轮运行标签为 `rebuild_micro_20261121`，使用相同的视频分组原则和统一的独立 test 流程：

| 划分 | 图片数量 | 视频组 | 用途 |
|---|---:|---|---|
| `train_grouped_micro.txt` | 128 | video1 | 2 epoch CPU 流程验证 |
| `val_grouped_micro.txt` | 64 | video4 | best 模型选择 |
| `test_grouped_micro.txt` | 64 | video10 | 独立 test |

micro 划分在每个原始分组内确定性地抽取有标注帧和背景帧，避免验证集或测试集全部为空标注；三组视频互不重叠。完整产物、SHA-256、manifest 和指标见 [`docs/本机micro-smoke审计.md`](本机micro-smoke审计.md)。本轮所有目标目录均包含 `best.pt`、`last.pt`、`results.csv`、`results.png`、`test_metrics.json` 和 `experiment_manifest.json`。

micro 结果只用于检查配对读取、输入尺寸、特征对齐、融合模块、训练、验证、独立 test 和输出归档，不能作为论文最终精度或模型优劣结论。

在同一组 micro 划分和同一独立 test 流程下，运行族 `rebuild_5epoch_20261121` 已完成 RGB、IR、1∶1 直接拼接以及 2∶1 直接拼接、等权融合、特征适配和门控融合共 7 个配置的 5 epoch 流程验证。完整产物和指标见 [`docs/本机5epoch-smoke审计.md`](本机5epoch-smoke审计.md)；该运行族仍属于 CPU smoke 结果，不用于论文最终结论。

## 5. 历史 2 epoch 结果

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 直接拼接 | 0.11504 | 0.27483 | 0.12449 | 0.02933 |
| 固定等权融合 | 0.00041 | 0.07989 | 0.00023 | 0.00005 |
| 特征适配 | 0.04932 | 0.32693 | 0.06904 | 0.01729 |
| 通道和空间门控 | 0.15729 | 0.42123 | 0.31494 | 0.08987 |

对应输出目录：

- `runs/detect/ratio2_grouped_smoke_epoch2/`
- `runs/detect/ratio2_grouped_smoke_simple_epoch2_test/`
- `runs/detect/ratio2_grouped_smoke_adapter_epoch2/`
- `runs/detect/ratio2_grouped_smoke_gated_epoch2_test_v6/`

## 6. 历史 RGB 和 IR 单模态 smoke 基线

| 模态 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| RGB | 0.00063 | 0.01370 | 0.00032 | 0.00021 |
| IR | 0.09253 | 0.07384 | 0.04205 | 0.00874 |

对应结果文件：

- `runs/detect/rgb_grouped_smoke_epoch2/test_metrics.json`
- `runs/detect/ir_grouped_smoke_epoch2/test_metrics.json`

## 7. 历史 1∶1 双流直接拼接记录

1∶1 实验保持 RGB 和 IR 均为 480×640。5 epoch test 结果如下：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.07255 | 0.16872 | 0.03564 | 0.00866 |

结果文件为 `runs/detect/ratio1_grouped_smoke_epoch5/test_metrics.json`。

为了与 2 epoch 的 2∶1 直接拼接公平比较，又使用 1∶1 的第二个 epoch 权重进行 test，结果如下：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.24598 | 0.05247 | 0.01986 | 0.00299 |

结果文件为 `runs/detect/ratio1_grouped_smoke_epoch2_fair_test/test_metrics.json`。

## 8. 历史 2∶1 门控 5 epoch 记录

在同一 smoke 数据划分下，另完成了 5 epoch 的门控流程验证：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.29458 | 0.23575 | 0.26083 | 0.06239 |

结果文件为 `runs/detect/ratio2_grouped_smoke_gated_epoch5_protocol/test_metrics.json`。该结果不能与上面的 2 epoch 结果直接横向比较，只用于检查较长 smoke 训练和独立 test 链路。

## 9. 结论限制

以上全部是 smoke 结果，不能作为论文最终精度，也不能证明某个融合模块在正式实验中一定更优。正式实验必须使用完整分组数据、统一 100 epoch、统一参数和独立 test 集。早期中断的 `rgb_local_cpu_rebuild_20261121/` 和 `rgb_local_cpu_rebuild_202611212/` 目录保留作过程记录，不纳入有效结果。

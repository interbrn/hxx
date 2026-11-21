# 严格实验流程与 2∶1 smoke 记录

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

## 3. Smoke 数据规模

| 划分 | 图片数量 |
|---|---:|
| 训练集 | 1387 |
| 验证集 | 618 |
| 测试集 | 237 |

训练参数为 2 epoch、batch=1、nbs=1、AdamW、CPU、无增强、关闭辅助损失。结果只能用于流程验证。

## 4. 已完成的 2 epoch 结果

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

## 5. RGB 和 IR 单模态 smoke 基线

| 模态 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| RGB | 0.00063 | 0.01370 | 0.00032 | 0.00021 |
| IR | 0.09253 | 0.07384 | 0.04205 | 0.00874 |

对应结果文件：

- `runs/detect/rgb_grouped_smoke_epoch2/test_metrics.json`
- `runs/detect/ir_grouped_smoke_epoch2/test_metrics.json`

## 6. 1∶1 双流直接拼接记录

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

## 7. 结论限制

以上全部是 smoke 结果，不能作为论文最终精度，也不能证明某个融合模块在正式实验中一定更优。正式实验必须使用完整分组数据、统一 100 epoch、统一参数和独立 test 集。

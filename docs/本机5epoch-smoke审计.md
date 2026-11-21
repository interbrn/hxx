# 本机 micro smoke 实验审计（2026-11-21）

运行标签：`rebuild_5epoch_20261121`。本报告只审计已有输出，不启动训练、不覆盖权重。micro 数据仅用于 CPU 流程验证，不能作为论文最终精度结论。

## 审计结论

- 发现运行目录：7 个；manifest 完整：7 个。
- 所有 manifest 的数据根目录和 train/val/test 路径一致：`是`；分辨率比例按实验设计分别记录。
- 训练 epoch 配置：`5`；设备配置：`cpu`。
- train/val/test 视频组无交集：`是`。
- 所有实验应使用同一组 `train_grouped_micro.txt`、`val_grouped_micro.txt` 和 `test_grouped_micro.txt`；独立 test 指标来自各自目录的 `test_metrics.json`。

## 数据划分

| 文件 | 样本数 | 视频组 | SHA-256 |
|---|---:|---|---|
| `train_grouped_micro.txt` | 128 | video1 | `300878b51810db91c12bd69dc953a0ff597c3ac7f887799d86fd29b972c5c31a` |
| `val_grouped_micro.txt` | 64 | video4 | `58e6bf2b7f07e8914b80382f05ac9ca4024ee44746374873a7ec584f61b2dc69` |
| `test_grouped_micro.txt` | 64 | video10 | `cba88df1af5eb7aac34166a13dc55944aa85d055de1d96cff6820e8d2e1e00f7` |

## 运行产物与独立 test 指标

| 实验 | 必需产物 | Precision | Recall | mAP50 | mAP50-95 | profile | 分辨率比例 |
|---|---|---:|---:|---:|---:|---|---:|
| `ir_local_cpu` | 完整 | 0.00110 | 0.00877 | 0.00029 | 0.00005 | `local_cpu` | `1` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。
| `ratio1_concat_local_cpu` | 完整 | 0.20951 | 0.29825 | 0.25279 | 0.05475 | `local_cpu` | `1` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。
| `ratio2_adapter_local_cpu` | 完整 | 0.10274 | 0.40351 | 0.15588 | 0.03714 | `local_cpu` | `2` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。
| `ratio2_concat_local_cpu` | 完整 | 0.04009 | 0.46491 | 0.14102 | 0.02882 | `local_cpu` | `2` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。
| `ratio2_gated_local_cpu` | 完整 | 0.26941 | 0.37719 | 0.25099 | 0.05111 | `local_cpu` | `2` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。
| `ratio2_simple_local_cpu` | 完整 | 0.08757 | 0.33333 | 0.12655 | 0.02779 | `local_cpu` | `2` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。
| `rgb_local_cpu` | 完整 | 0.00564 | 0.45614 | 0.01064 | 0.00218 | `local_cpu` | `1` |
  - 参数：`epochs=5`、`device=cpu`、`batch=1`。

## 解释边界

1. 本轮使用 `local_cpu`、epoch 配置为 `5`、batch=1 和 micro 划分，目的为验证 RGB/IR 配对、2∶1 尺寸变换、模型前向、训练、验证、独立 test 和结果归档。
2. micro 划分固定为 train 128、val 64、test 64，不能代表完整 RGBT-3M 的样本分布，也不能用于论文最终结论。
3. 服务器正式实验仍使用 `server_full`、完整视频分组、100 epoch 和统一的正式测试流程。

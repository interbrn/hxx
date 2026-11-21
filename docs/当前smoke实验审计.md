# 当前 smoke 实验审计（2026-11-21）

本报告由 `tools/audit_experiments.py` 生成，不启动训练，不修改权重。所有数值均为 smoke/短 epoch 流程验证结果，不能作为论文最终结论。

## 结论

- 已审计核心实验：7 个；共同 smoke 划分配置一致：`是`；视频组无交集：`是`。
- 统一数据契约为按视频分组的 `train_grouped_smoke`、`val_grouped_smoke`、`test_grouped_smoke`；独立 test 指标从已有 `test_metrics.json` 读取。
- 历史输出目录均未发现 `experiment_manifest.json`，因此只能依据现有配置、目录命名和指标文件核对，不能把历史运行声明为完全可追溯。后续训练由当前 `train.py` 生成 manifest。
- `ratio1_gated` 与 `hierafire` 是注册的扩展模式，不是任务书 A–G 核心矩阵的缺口；本次不为它们重复训练。

## 共同 smoke 划分

| 文件 | 样本数 | SHA-256 |
|---|---:|---|
| `train_grouped_smoke.txt` | 1387 | `226de8731c75a9bced6d11a147358f5da544d25b92be31a6382234f44e6e7116` |
| `val_grouped_smoke.txt` | 618 | `3b912c99e8193f98abe0c409aa9c09e34b180769a727c720e2296096b3bf6953` |
| `test_grouped_smoke.txt` | 237 | `61981627440e9540993485d3d75d606c43995e1498cb783d0b10617962357651` |

## 实验状态与指标

`metric_dir` 与 `train_dir` 不同的项目，是历史上将独立 test 结果写到 sibling 目录的情况；这不是重新训练。

| 实验 | 训练目录产物 | 主要 test 指标目录 | Precision | Recall | mAP50 | mAP50-95 | 公平 test 目录 | manifest |
|---|---|---|---:|---:|---:|---:|---|---|
| `rgb` | 缺 experiment_manifest.json（`rgb_grouped_smoke_epoch2`） | `rgb_grouped_smoke_epoch2` | 0.00063 | 0.01370 | 0.00032 | 0.00021 | — | 无 |
| `ir` | 缺 experiment_manifest.json（`ir_grouped_smoke_epoch2`） | `ir_grouped_smoke_epoch2` | 0.09253 | 0.07384 | 0.04205 | 0.00874 | — | 无 |
| `ratio1_concat` | 缺 experiment_manifest.json（`ratio1_grouped_smoke_epoch5`） | `ratio1_grouped_smoke_epoch5` | 0.07255 | 0.16872 | 0.03564 | 0.00866 | sibling test 目录 | 无 |
| `ratio2_concat` | 缺 experiment_manifest.json（`ratio2_grouped_smoke_epoch2`） | `ratio2_grouped_smoke_epoch2` | 0.11504 | 0.27483 | 0.12449 | 0.02933 | — | 无 |
| `ratio2_simple` | 缺 results.png, test_metrics.json, experiment_manifest.json（`ratio2_grouped_smoke_simple_epoch2`） | `ratio2_grouped_smoke_simple_epoch2_test` | 0.00041 | 0.07989 | 0.00023 | 0.00005 | — | 无 |
| `ratio2_adapter` | 缺 experiment_manifest.json（`ratio2_grouped_smoke_adapter_epoch2`） | `ratio2_grouped_smoke_adapter_epoch2` | 0.04932 | 0.32693 | 0.06904 | 0.01729 | — | 无 |
| `ratio2_gated` | 缺 experiment_manifest.json（`ratio2_grouped_smoke_gated_epoch5_protocol`） | `ratio2_grouped_smoke_gated_epoch5_protocol` | 0.29458 | 0.23575 | 0.26083 | 0.06239 | sibling test 目录 | 无 |

## 阶段式消融记录

在统一 2∶1、2 epoch smoke 协议下，已有结果可作为流程级阶段式消融。若某实验还有 5 epoch 结果，5 epoch 结果只作为单独记录，不与 2 epoch 数值横向比较：

| 条件 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `直接拼接`（ratio2_concat） | 0.11504 | 0.27483 | 0.12449 | 0.02933 |
| `固定等权`（ratio2_simple） | 0.00041 | 0.07989 | 0.00023 | 0.00005 |
| `特征适配`（ratio2_adapter） | 0.04932 | 0.32693 | 0.06904 | 0.01729 |
| `特征适配 + 通道/空间门控`（ratio2_gated） | 0.15729 | 0.42123 | 0.31494 | 0.08987 |

上述表格优先使用目录名明确标注的 2 epoch 公平 test 指标；`ratio2_gated` 的 5 epoch 结果仍保留在主要指标目录中，但不混入这张对比表。

阶段式定义：

1. `ratio2_concat`：直接拼接，不使用适配器和门控。
2. `ratio2_adapter`：加入多感受野特征适配。
3. `ratio2_gated`：在适配基础上加入通道 + 空间门控。
4. `ratio2_simple`：固定 0.5/0.5 等权融合，用于与可学习融合区分。

正式论文消融仍应在完整分组数据、100 epoch、同一 seed/增强/优化器和独立 test 上重跑；至少保留 `concat`、`adapter`、`adapter+channel gate`、`adapter+channel+spatial gate` 和 `aux loss on/off` 五个条件。

## 下一步

- 服务器先执行所有模式的 `server_full --dry-run`，确认数据根目录、GPU、batch 和输出命名。
- 正式启动前逐个检查 `last.pt`、`best.pt`、`results.csv`、`results.png`、`test_metrics.json` 和 `experiment_manifest.json`；若有 `last.pt` 则续训，有有效 test 指标则直接登记，不覆盖历史结果。
- 本机继续只做模型/数据/测试流程检查，不新增重复 smoke 训练。

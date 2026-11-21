# RGBT-3M 可见光—红外跨分辨率特征融合

本项目用于毕业设计“面向火情目标检测的可见光—红外跨分辨率特征融合方法研究”。项目基于 YOLOv8n 构建 RGB/IR 双流目标检测器，在 RGBT-3M 数据集上检测 `fire` 和 `person`，研究 1∶1 与 2∶1 跨分辨率条件下的特征融合方法。

## 研究约束

| 项目 | 约定 |
|---|---|
| 数据集 | RGBT-3M |
| 类别 | `fire`、`person` |
| RGB | 480×640，保持原始分辨率 |
| 1∶1 IR | 480×640 |
| 2∶1 IR | 240×320，宽和高分别缩小一半 |
| 任务 | 目标检测，不进行图像融合或超分辨率重建 |
| 评价指标 | Precision、Recall、mAP50、mAP50-95 |
| 正式训练 | 完整视频分组、100 epoch、独立 test |

## 运行模式

| profile | 数据 | 设备 | 默认 epoch | 用途 |
|---|---|---|---:|---|
| `local_cpu` | micro 视频分组 | CPU | 2 | 本地流程验证 |
| `server_full` | 完整视频分组 | CUDA | 100 | 正式实验 |

本地可通过 `--epochs 5` 进行较长 smoke 验证，但任何 micro 或短 epoch 结果都不能作为论文最终结论。

## 数据集准备

数据集不随仓库发布。数据集根目录应包含 `RGB/`、`IR/` 和 `labels_fire_person/`。RGB、IR 和标签使用相同的相对文件名。正式实验使用按视频分组的 `train_grouped.txt`、`val_grouped.txt` 和 `test_grouped.txt`，防止相邻视频帧泄漏。配置文件不保存个人电脑路径，运行时通过 `--data-root` 指定数据集根目录。

## 环境安装

建议使用 Python 3.9 或更高版本。先按照 PyTorch 官方安装说明安装与设备匹配的 PyTorch 和 torchvision，再安装项目依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

服务器环境应先按照 CUDA 版本安装匹配的 PyTorch 和 torchvision，再执行 `python -m pip install -r requirements.txt`。项目直接使用仓库内的 `ultralytics/`，不需要额外安装同名外部包。

## 运行前检查

```powershell
python tools/validate_project.py `
  --data-root D:\path\to\RGBT-3M `
  --profile local_cpu `
  --experiments core
```

该命令不启动训练，检查数据目录、类别、划分文件、代表性样本、RGB/IR 尺寸和模型配置。

## 本地 CPU 流程

```powershell
python tools/run_experiments.py `
  --profile local_cpu `
  --data-root D:\path\to\RGBT-3M `
  --experiments core `
  --epochs 5 `
  --run-tag local_cpu_5epoch `
  --dry-run
```

确认命令后删除 `--dry-run` 启动训练。核心矩阵包含 RGB 单模态、IR 单模态、1∶1 直接拼接、2∶1 直接拼接、2∶1 等权融合、2∶1 特征适配和 2∶1 门控融合。

单独运行一个实验：

```powershell
python train.py `
  --profile local_cpu `
  --experiment ratio2_gated `
  --data-root D:\path\to\RGBT-3M `
  --epochs 5
```

## 服务器完整数据流程

```bash
python tools/validate_project.py \
  --data-root /data/RGBT-3M \
  --profile server_full \
  --experiments core

python tools/run_experiments.py \
  --profile server_full \
  --data-root /data/RGBT-3M \
  --experiments core \
  --device 0 \
  --dry-run
```

确认无误后启动正式实验：

```bash
python tools/run_experiments.py \
  --profile server_full \
  --data-root /data/RGBT-3M \
  --experiments core \
  --device 0 \
  --run-tag server_full_100epoch
```

`server_full` 使用完整视频分组、100 epoch、AdamW、`nbs=64`、余弦学习率和 nonmosaic 增强。显存不足时可以降低 `--batch`，但应记录实际 batch 和梯度累积参数。`--experiments all` 会额外运行 `ratio1_gated` 和 `hierafire`，扩展实验应单独记录。

## 结果保护和断点续训

批量入口默认保护已有实验目录：

1. 存在有效 `test_metrics.json` 时跳过训练；
2. 只有 `weights/last.pt` 时自动续训；
3. 只有 `weights/best.pt` 时保留目录，不自动覆盖；
4. 只有显式使用 `--force` 才允许重新训练。

每个完整运行目录应包含 `weights/best.pt`、`weights/last.pt`、`results.csv`、`results.png`、`test_metrics.json` 和 `experiment_manifest.json`。训练、测试、缓存和临时文件默认写入项目目录，不应提交到 GitHub。

## 独立测试和审计

```bash
python tools/test_checkpoint.py \
  --weights runs/detect/<RUN_NAME>/weights/best.pt \
  --data configs/rgbt3m_ratio2_grouped.yaml \
  --data-root /data/RGBT-3M \
  --device 0 \
  --name <RUN_NAME>_test

python tools/audit_run_tag.py \
  --run-tag server_full_100epoch \
  --output docs/server_full_100epoch_audit.md
```

## 代码检查

```bash
python tools/check_model_variants.py
python -m pytest -q
git diff --check
```

## 文档导航

- [完整运行指南](docs/运行指南.md)
- [配置说明](configs/README.md)
- [正式训练前检查清单](docs/正式训练前检查清单.md)
- [消融实验方案](docs/消融实验方案.md)
- [本机 5 epoch 审计](docs/本机5epoch-smoke审计.md)
- [复杂度与 CPU 前向记录](docs/复杂度与CPU前向记录.md)
- [GitHub 上传检查清单](docs/GitHub上传检查清单.md)

CPU micro、smoke 和短 epoch 结果只用于验证代码链路、参数设置和输出归档，不能用于论文主表或最终模型优劣结论。论文正式结果必须来自完整视频分组、统一 100 epoch、相同训练参数和相同独立 test 流程。

项目包含基于 Ultralytics 的本地修改代码。发布或再分发前，应同时遵守 Ultralytics 及 RGBT-3M 数据集提供方的许可和引用要求。

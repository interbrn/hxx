# RGBT-3M 跨分辨率火情目标检测

面向毕业设计“面向火情目标检测的可见光—红外跨分辨率特征融合方法研究”的实验代码。
项目基于 YOLOv8n 风格检测器，使用 RGBT-3M 数据集检测 `fire` 和 `person`，重点研究 RGB/IR 双流检测以及 1∶1、2∶1 跨分辨率特征融合。

## 项目范围

- RGB 保持 480×640；2∶1 实验将 IR 缩小为 240×320。
- 支持 RGB 单模态、IR 单模态、双流直接拼接、等权融合、特征适配和通道/空间门控。
- 按视频分组划分 train/val/test，避免相邻帧泄漏。
- 训练完成后使用独立 test 集评估，不把 smoke 结果当作论文最终精度。
- 训练结果、数据集、缓存和权重默认不提交到 Git；详见 `.gitignore`。

## 目录结构

```text
configs/       数据、模型和统一实验参数配置
demo/          单样本读取、张量和前向传播示例
docs/          任务书、实验流程、学习路线和检查清单
splits/        可复现实验划分文件
tests/         模型、数据集和检查点测试
tools/         数据划分、模型检查和转换工具
ultralytics/   本项目使用的本地 Ultralytics 修改版
train.py       训练入口
val.py         验证入口
predict.py     RGB/IR 成对图片预测入口
profile_flops.py  参数量和 FLOPs 统计
```

## 环境安装

建议使用 Python 3.9 或更高版本，并先按照 PyTorch 官方文档安装与 CPU/GPU 匹配的 PyTorch 和 torchvision，再安装项目依赖：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

本项目使用本地 `ultralytics/` 代码，不需要另外安装同名 Ultralytics 包。运行入口时请在项目根目录执行，或使用脚本中的项目根目录解析逻辑。

## 数据集准备

数据集不随代码仓库发布。请准备 RGBT-3M，并保证目录大致如下：

```text
<DATASET_ROOT>/
├── RGB/train/、RGB/val/、RGB/test/
├── IR/train/、IR/val/、IR/test/
└── labels_fire_person/
```

划分文件位于 `splits/`。配置文件中的 `train`、`val`、`test` 使用项目相对路径，`train.py` 和 `val.py` 会自动解析到项目根目录；`--data-root` 只需要指向数据集根目录。

## 快速检查

```bash
python tools/check_model_variants.py
pytest -q
```

上面命令只检查模型构建、前向传播、数据读取和小规模单元测试，不会启动训练。

## CPU smoke 训练

以下命令使用 1∶1 双流直接拼接配置。`<DATASET_ROOT>` 请替换成数据集实际路径：

```bash
python train.py \
  --data configs/rgbt3m_grouped_ratio1_smoke.yaml \
  --model configs/yolov8n_dual_concat.yaml \
  --data-root <DATASET_ROOT> \
  --imgsz 480 640 --epochs 2 --batch 1 --nbs 1 \
  --workers 0 --device cpu --no-amp --augmentation none \
  --optimizer AdamW --seed 0 --no-aux \
  --project runs/detect --name ratio1_grouped_smoke_epoch2 \
  --test-after-train
```

2∶1 门控模型只需替换 `--data` 和 `--model`：

```text
configs/rgbt3m_ratio2_grouped_smoke.yaml
configs/yolov8n_dual_gated_ratio2.yaml
```

推荐统一参数见 [`configs/experiment_protocol.yaml`](configs/experiment_protocol.yaml)。正式实验计划为 100 epoch，应在有 NVIDIA GPU 的环境中执行。

## 两种运行模式

训练入口已经把本地 CPU 和服务器正式实验分成两个 profile。`--experiment` 选择任务书中的模式，`--profile` 选择运行环境；未显式填写的路径、epoch、batch、增强和测试参数会自动补齐。

本地电脑只跑 smoke：

```bash
python train.py --profile local_cpu --experiment ratio2_gated \
  --data-root <DATASET_ROOT>
```

服务器跑完整分组数据和 100 epoch：

```bash
python train.py --profile server_full --experiment ratio2_gated \
  --data-root <DATASET_ROOT> --device 0 --batch 16 --workers 4
```

可选 `--experiment`：

```text
rgb             RGB 单模态
ir              IR 单模态
ratio1_concat   1∶1 双流直接拼接
ratio1_gated    1∶1 门控融合
ratio2_concat   2∶1 双流直接拼接
ratio2_simple   2∶1 固定等权融合
ratio2_adapter  2∶1 特征适配
ratio2_gated    2∶1 通道+空间门控
hierafire       HieraFire 扩展模型
```

先检查选择是否正确而不启动训练：

```bash
python train.py --profile server_full --experiment ratio2_adapter \
  --data-root <DATASET_ROOT> --dry-run
```

每次训练完成后，运行目录还会保存 `experiment_manifest.json`，记录 profile、实验模式、数据配置、模型配置和实际训练参数，方便服务器结果回溯。

如果服务器需要依次跑完全部模式，可以使用批量入口：

```bash
python tools/run_experiments.py --profile server_full \
  --data-root <DATASET_ROOT> --experiments all --device 0
```

本地只验证部分模式时可以缩小范围：

```bash
python tools/run_experiments.py --profile local_cpu \
  --data-root <DATASET_ROOT> \
  --experiments ratio1_concat ratio2_concat ratio2_adapter ratio2_gated
```

批量启动前建议先加 `--dry-run` 检查命令列表。

## 独立测试

如果训练已经完成，可使用 `tools/test_checkpoint.py`，不重新训练：

```bash
python tools/test_checkpoint.py \
  --weights runs/detect/<RUN_NAME>/weights/best.pt \
  --data configs/rgbt3m_ratio2_grouped_smoke.yaml \
  --data-root <DATASET_ROOT> \
  --device cpu --project runs/detect --name <RUN_NAME>_test
```

主要输出包括 `weights/best.pt`、`weights/last.pt`、`results.csv`、`results.png` 和 `test_metrics.json`。

## 当前实验说明

本仓库中的短 epoch 和 smoke 结果只用于流程验证、参数调试和消融流程检查。它们不能替代完整数据、统一 100 epoch 和独立 test 集上的正式论文结果。

## 许可证与上游代码

项目包含基于 Ultralytics 的本地修改代码。相关文件保留原有 AGPL-3.0 许可声明；发布或再分发前请同时遵守 Ultralytics 及数据集提供方的许可和引用要求。

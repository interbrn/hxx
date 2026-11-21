# GitHub 上传检查清单

## 必须保留

- `train.py`、`val.py`、`predict.py`
- `tools/` 下的运行、预检查、审计和独立测试工具
- `configs/` 下的数据配置、模型配置和配置说明
- `splits/` 下的可复现实验划分文件
- `tests/` 下的测试文件
- `README.md`、`requirements.txt`、`requirements-cpu.txt`
- `docs/` 下的运行指南、实验协议和消融方案
- `LICENSE` 或项目所依据的许可说明

## 不应上传

- 数据集图像和标签原始文件
- `runs/` 下的训练输出和权重
- `.dataset_cache/`、`.yolo_config/`、`.matplotlib/`
- `.venv/`、`__pycache__/`、`.pytest_cache/`
- 含有本地绝对路径、账号、密钥或服务器地址的私密配置
- 未经确认可以公开的模型权重和实验数据

## 上传前命令

```bash
python tools/check_model_variants.py
python -m pytest -q
git diff --check
git status --short
```

## 结果说明

只在文档中记录实验配置、划分哈希、指标和结果目录名称。正式论文结果应保留对应的 `experiment_manifest.json`、独立 test 指标和运行日志，但不要求将大文件权重提交到代码仓库。

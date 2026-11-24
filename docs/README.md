# 文档导航

| 文档 | 用途 |
|---|---|
| [协议A探路实验报告与主划分方案.md](协议A探路实验报告与主划分方案.md) | RGBT-3M 协议 A 主划分方案、四组 45-epoch 探路结果、细粒度拆解与模块设计指导 |
| [任务书完成情况与本地实验方案.md](任务书完成情况与本地实验方案.md) | 任务书要求、实验矩阵、当前结果和本地限制 |
| [运行指南.md](运行指南.md) | CPU 与 CUDA 两种 profile 的安装、检查、训练和审计流程 |
| [GitHub上传检查清单.md](GitHub上传检查清单.md) | 代码、数据、权重和隐私文件的提交边界 |
| [严格实验流程与ratio2_smoke记录.md](严格实验流程与ratio2_smoke记录.md) | 数据划分、独立测试和 2∶1 smoke 记录 |
| [当前smoke实验审计.md](当前smoke实验审计.md) | 当前输出目录、指标、划分一致性和缺口审计 |
| [本机micro-smoke审计.md](本机micro-smoke审计.md) | 本轮 CPU micro 全矩阵输出、manifest 和独立 test 指标 |
| [本机5epoch-smoke审计.md](本机5epoch-smoke审计.md) | 5 epoch CPU micro 全矩阵输出、manifest 和独立 test 指标 |
| [消融实验方案.md](消融实验方案.md) | 正式服务器消融矩阵与判读规则 |
| [复杂度与CPU前向记录.md](复杂度与CPU前向记录.md) | 参数量、FLOPs 和本机网络前向耗时记录 |
| [正式训练前检查清单.md](正式训练前检查清单.md) | 正式训练前必须确认的项目 |
| [学习路线.md](学习路线.md) | 从数据读取到双流融合的代码学习顺序 |

本地 CPU micro 划分由 `tools/make_micro_splits.py` 生成，运行族审计由 `tools/audit_run_tag.py` 生成。

实验输出默认位于 `runs/detect/`，该目录被 Git 忽略；提交代码时只保留配置、划分文件和可复现实验说明。

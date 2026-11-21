# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
训练入口：在 RGBT-3M 上训练 RGB/红外火情目标检测模型。

本文件负责读取数据配置、创建 YOLO/HieraFire 模型、设置输入分辨率比例、
启动训练并保存权重。真正的数据读取在 ultralytics/data/dataset.py，
真正的双流网络在 ultralytics/nn/tasks.py。
"""
import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
from tools.experiment_registry import EXPERIMENT_REGISTRY as EXPERIMENTS


PROFILES = {
    "local_cpu": {
        "epochs": 2,
        "batch": 1,
        "nbs": 1,
        "workers": 0,
        "device": "cpu",
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "augmentation": "none",
        "no_amp": True,
        "no_aux": True,
        "test_after_train": True,
    },
    "server_full": {
        "epochs": 100,
        "batch": 16,
        "nbs": 64,
        "workers": 4,
        "device": "0",
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": True,
        "augmentation": "nonmosaic",
        "no_amp": False,
        "no_aux": True,
        "test_after_train": True,
    },
}

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Ultralytics 的运行时设置和持久化缓存也固定在 D 盘项目目录。
os.environ["YOLO_CONFIG_DIR"] = str((PROJECT_ROOT / ".yolo_config").resolve())
os.environ["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".matplotlib").resolve())

import cv2
from ultralytics import YOLO
from ultralytics.utils import yaml_load


def resolve_project_path(value):
    """Resolve a project-relative path without depending on the launch directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_split_paths(data, data_root):
    """Make train/val/test split paths portable while preserving absolute paths."""
    for key in ("train", "val", "test"):
        value = data.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            data[key] = str(path.resolve())
            continue
        project_path = (PROJECT_ROOT / path).resolve()
        dataset_path = (data_root / path).resolve()
        data[key] = str(project_path if project_path.exists() or not dataset_path.exists() else dataset_path)


def configure_auxiliary_losses(trainer, use_aux=False, aux_weight=0.25):
    # 设置辅助损失权重
    trainer.model.aux_loss_weight = aux_weight if use_aux else 0.0

    # 控制是否启用 RGB 和 IR 辅助检测头
    trainer.model.use_aux_head = use_aux
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "aux_rgb", "aux_ir")
    keys = trainer.validator.results_csv_keys() + trainer.label_loss_items(prefix="val")
    trainer.metrics = dict.fromkeys(keys, 0)


def apply_profile(args):
    """Fill unset CLI values from a reproducible local or server profile."""
    experiment = EXPERIMENTS.get(args.experiment)
    if args.profile == "custom":
        defaults = {
            "data": "configs/rgbt3m.yaml",
            "model": "configs/hierafire.yaml",
            "epochs": 200,
            "batch": 16,
            "nbs": 64,
            "workers": 8,
            "device": "0",
            "optimizer": "SGD",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.0,
            "cos_lr": False,
            "augmentation": "hsv-flip",
            "no_amp": False,
            "no_aux": False,
            "test_after_train": False,
        }
        tag = "custom"
    else:
        if experiment is None:
            raise ValueError(f"未知实验模式: {args.experiment}")
        defaults = dict(PROFILES[args.profile])
        split_key = "local_data" if args.profile == "local_cpu" else "full_data"
        defaults["data"] = experiment[split_key]
        defaults["model"] = experiment["model"]
        tag = experiment["tag"]

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.name is None:
        args.name = f"{tag}_{args.profile}"
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["local_cpu", "server_full", "custom"], default="custom",
                        help="预设训练环境：local_cpu 或 server_full；custom 保留手动参数模式")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="ratio2_gated",
                        help="任务书实验模式；profile 会自动选择对应数据和模型配置")
    parser.add_argument("--data", default=None, help="手动覆盖数据配置")
    parser.add_argument("--model", default=None, help="手动覆盖模型配置")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--nbs", type=int, default=None,
                        help="梯度累积使用的名义 batch；小样本调试时应设为实际 batch")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cls", type=float, default=0.1)
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", default=None,
                        help="训练结果保存的根目录；默认使用项目根目录下的 runs/detect")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed; default 0 keeps repeated experiments comparable")
    parser.add_argument("--no-amp", action="store_true", default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--augmentation", choices=["hsv-flip", "nonmosaic", "none"], default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None,
                        help="final learning-rate fraction relative to lr0")
    parser.add_argument("--optimizer", choices=["SGD", "AdamW"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--cos-lr", action="store_true", default=None,
                        help="use cosine learning-rate decay")
    parser.add_argument("--no-aux", action="store_true", default=None)
    parser.add_argument("--aux-weight", type=float, default=0.25,
                        help="RGB/IR auxiliary loss weight when auxiliary heads are enabled")
    parser.add_argument("--test-after-train", action="store_true", default=None,
                        help="训练结束后使用验证集最佳模型在独立 test 集上测试")
    parser.add_argument("--test-split", default=None,
                        help="独立测试集划分文件；未指定时读取数据 YAML 的 test 字段")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 profile/experiment 解析结果，不启动训练")
    args = apply_profile(parser.parse_args())
    if args.profile == "custom" and (args.data is None or args.model is None):
        raise ValueError("custom 模式必须提供 --data 和 --model")
    if args.dry_run:
        print(json.dumps({
            "profile": args.profile,
            "experiment": args.experiment,
            "data": str(resolve_project_path(args.data)),
            "model": str(resolve_project_path(args.model)),
            "data_root": str(resolve_project_path(args.data_root)),
            "name": args.name,
            "epochs": args.epochs,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "device": args.device,
            "optimizer": args.optimizer,
            "augmentation": args.augmentation,
            "cos_lr": args.cos_lr,
            "no_aux": args.no_aux,
            "test_after_train": args.test_after_train,
        }, ensure_ascii=False, indent=2))
        return
    # CPU 没有异步显存拷贝，关闭 pin_memory 可避免无效的 pinned-memory 警告；
    # server_full 使用 CUDA 时保留 Ultralytics 的 pin_memory 优化。
    from ultralytics.data import build as data_build
    data_build.PIN_MEMORY = str(args.device).lower() not in {"cpu", "-1"}
    cv2.setNumThreads(0)
    data_root = resolve_project_path(args.data_root)
    data_config = resolve_project_path(args.data)
    model_config = resolve_project_path(args.model)
    data = yaml_load(str(data_config))
    data["path"] = str(data_root)
    resolve_split_paths(data, data_root)
    # 训练/验证/测试扫描缓存放在当前 D 盘项目目录，不修改原始数据集目录。
    data["cache_dir"] = str((PROJECT_ROOT / ".dataset_cache").resolve())
    data["img_size"] = args.imgsz
    data["rgb_size"] = list(args.imgsz)
    ratio = int(data.get("resolution_ratio", 1))
    data["ir_size"] = [max(1, args.imgsz[0] // ratio), max(1, args.imgsz[1] // ratio)]
    resume_path = resolve_project_path(args.resume) if args.resume else None
    model = YOLO(str(resume_path or model_config))
    if hasattr(model.model, "resolution_ratio"):
        model.model.resolution_ratio = ratio
        model.model.yaml["resolution_ratio"] = ratio
    # 训练开始时，根据命令行参数决定是否启用辅助分支。
    model.add_callback(
        "on_train_start",
        lambda trainer: configure_auxiliary_losses(
            trainer, use_aux=not args.no_aux, aux_weight=args.aux_weight
        ),
    )
    augmentation = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
                        translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                        flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
    if args.augmentation == "none":
        augmentation.update(fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    elif args.augmentation == "nonmosaic":
        augmentation.update(hsv_h=0.005, hsv_s=0.2, hsv_v=0.2, degrees=8.0,
                            translate=0.05, scale=0.15, shear=2.0,
                            perspective=0.0003, flipud=0.1)
    project_dir = resolve_project_path(args.project) if args.project else (PROJECT_ROOT / "runs" / "detect")
    model.train(data=data, resume=bool(args.resume), epochs=args.epochs, batch=args.batch,
                imgsz=args.imgsz, device=args.device, workers=args.workers,
                seed=args.seed, deterministic=args.deterministic, optimizer=args.optimizer, lr0=args.lr0,
                lrf=args.lrf, momentum=0.937, weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs, warmup_momentum=args.warmup_momentum,
                warmup_bias_lr=args.warmup_bias_lr,
                nbs=args.nbs,
                cls=args.cls, dfl=1.5,
                cos_lr=args.cos_lr, pretrained=False, amp=not args.no_amp, val_period=1,
                patience=0, save_best=True, save_period=1,
                project=str(project_dir), name=args.name, **augmentation)

    save_dir = Path(getattr(model.trainer, "save_dir", project_dir / args.name))
    manifest = {
        "profile": args.profile,
        "experiment": args.experiment,
        "data_config": str(data_config),
        "model_config": str(model_config),
        "data_root": str(data_root),
        "resolved_data": data,
        "training": {
            key: value for key, value in vars(args).items()
            if key not in {"data", "model", "data_root", "project", "resume"}
        },
    }
    (save_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if args.test_after_train:
        # model.train() 结束后会优先加载验证集最佳权重 best.pt。
        test_split = args.test_split or data.get("test")
        if not test_split:
            raise ValueError("--test-after-train 需要 --test-split 或数据 YAML 中的 test 字段")
        test_path = Path(test_split)
        if not test_path.is_absolute():
            test_path = resolve_project_path(test_path)
        if not test_path.exists():
            raise FileNotFoundError(f"测试集划分文件不存在: {test_path}")

        test_data = dict(data)
        test_data["test"] = str(test_path)
        test_metrics = model.val(
            data=test_data,
            split="test",
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            workers=0,
            plots=True,
            project=str(save_dir),
            name="test",
        )
        results = getattr(test_metrics, "results_dict", {})
        serializable = {key: float(value) for key, value in results.items()}
        (save_dir / "test_metrics.json").write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"独立测试完成，指标已保存到: {save_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()

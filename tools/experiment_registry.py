"""Central registry for the reproducible experiment matrix."""

CORE_EXPERIMENTS = (
    "rgb",
    "ir",
    "ratio1_concat",
    "ratio2_concat",
    "ratio2_simple",
    "ratio2_adapter",
    "ratio2_gated",
)

OPTIONAL_EXPERIMENTS = (
    "ratio1_gated",
    "hierafire",
)

ALL_EXPERIMENTS = CORE_EXPERIMENTS + OPTIONAL_EXPERIMENTS

EXPERIMENT_REGISTRY = {
    "rgb": {
        "tag": "rgb",
        "local_data": "configs/rgbt3m_grouped_rgb_micro.yaml",
        "smoke_data": "configs/rgbt3m_grouped_rgb_smoke.yaml",
        "full_data": "configs/rgbt3m_grouped_rgb.yaml",
        "model": "configs/yolov8n_dual_concat.yaml",
    },
    "ir": {
        "tag": "ir",
        "local_data": "configs/rgbt3m_grouped_ir_micro.yaml",
        "smoke_data": "configs/rgbt3m_grouped_ir_smoke.yaml",
        "full_data": "configs/rgbt3m_grouped_ir.yaml",
        "model": "configs/yolov8n_dual_concat.yaml",
    },
    "ratio1_concat": {
        "tag": "ratio1_concat",
        "local_data": "configs/rgbt3m_grouped_ratio1_micro.yaml",
        "smoke_data": "configs/rgbt3m_grouped_ratio1_smoke.yaml",
        "full_data": "configs/rgbt3m_grouped_ratio1.yaml",
        "model": "configs/yolov8n_dual_concat.yaml",
    },
    "ratio1_gated": {
        "tag": "ratio1_gated",
        "local_data": "configs/rgbt3m_grouped_ratio1_micro.yaml",
        "smoke_data": "configs/rgbt3m_grouped_ratio1_smoke.yaml",
        "full_data": "configs/rgbt3m_grouped_ratio1.yaml",
        "model": "configs/hierafire_gated.yaml",
    },
    "ratio2_concat": {
        "tag": "ratio2_concat",
        "local_data": "configs/rgbt3m_ratio2_grouped_micro.yaml",
        "smoke_data": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
        "full_data": "configs/rgbt3m_ratio2_grouped.yaml",
        "model": "configs/yolov8n_dual_concat_ratio2.yaml",
    },
    "ratio2_simple": {
        "tag": "ratio2_simple",
        "local_data": "configs/rgbt3m_ratio2_grouped_micro.yaml",
        "smoke_data": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
        "full_data": "configs/rgbt3m_ratio2_grouped.yaml",
        "model": "configs/yolov8n_dual_simple_ratio2.yaml",
    },
    "ratio2_adapter": {
        "tag": "ratio2_adapter",
        "local_data": "configs/rgbt3m_ratio2_grouped_micro.yaml",
        "smoke_data": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
        "full_data": "configs/rgbt3m_ratio2_grouped.yaml",
        "model": "configs/yolov8n_dual_adapter_ratio2.yaml",
    },
    "ratio2_gated": {
        "tag": "ratio2_gated",
        "local_data": "configs/rgbt3m_ratio2_grouped_micro.yaml",
        "smoke_data": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
        "full_data": "configs/rgbt3m_ratio2_grouped.yaml",
        "model": "configs/yolov8n_dual_gated_ratio2.yaml",
    },
    "hierafire": {
        "tag": "hierafire",
        "local_data": "configs/rgbt3m_grouped_ratio1_micro.yaml",
        "smoke_data": "configs/rgbt3m_grouped_ratio1_smoke.yaml",
        "full_data": "configs/rgbt3m_grouped_ratio1.yaml",
        "model": "configs/hierafire.yaml",
    },
}

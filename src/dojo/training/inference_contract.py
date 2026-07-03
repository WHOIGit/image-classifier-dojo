"""Portable inference contract construction for supervised artifacts."""

from __future__ import annotations

from typing import Any

from dojo.config_schemas import RootConfig, compatibility_hashes, config_hash


def objective_summary(cfg: RootConfig) -> dict[str, Any]:
    objectives: dict[str, Any] = {}
    for name, objective in cfg.objectives.items():
        if not objective.enabled:
            continue
        head_name = objective.head or name
        head = cfg.model.heads[head_name]
        loss_type = objective.loss if isinstance(objective.loss, str) else objective.loss.type
        loss_params = {} if isinstance(objective.loss, str) else dict(objective.loss.params)
        objectives[name] = {
            "head": head_name,
            "target": head.target,
            "weight": objective.weight,
            "loss": {"type": loss_type, "params": loss_params},
            "metrics": [
                {"name": metric, "params": {}, "output": metric}
                for metric in objective.metrics
            ],
        }
    return {
        "schema_version": 1,
        "total_loss": {"reduction": "weighted_sum"} if objectives else None,
        "objectives": objectives,
    }


def build_inference_contract(
    cfg: RootConfig,
    *,
    class_mapping: dict[int, str],
) -> dict[str, Any]:
    compatibility = compatibility_hashes(cfg, class_mapping=class_mapping)
    class_maps = {
        head_name: [
            {"index": index, "label": class_mapping.get(index, str(index))}
            for index in range(head.num_classes)
        ]
        for head_name, head in sorted(cfg.model.heads.items())
        if head.type == "multiclass_classification"
    }
    return {
        "schema_version": "1.0.0",
        "model_config": compatibility["model_config_source"]["model"],
        "inference_pipeline": [
            step.model_dump(mode="json", exclude_none=True)
            for step in (cfg.transforms.inference_pipeline or [])
        ],
        "preprocessing_stats": compatibility["preprocessing_source"],
        "class_maps": class_maps,
        "target_schema": compatibility["target_schema_source"],
        "objective_summary": objective_summary(cfg),
        "compatibility": compatibility,
        "provenance": {
            "config_hash": config_hash(cfg),
            "source_run_id": cfg.runtime.run_id,
            "dojo_version": "0.3.0.dev0",
        },
    }

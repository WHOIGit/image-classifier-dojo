"""Execution helpers for `dojo infer` and `dojo eval holdout`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from dojo.config_schemas import RootConfig, config_hash, head_hash
from dojo.config_schemas.root import (
    BackboneConfig,
    BackboneWeightsConfig,
    FreezeConfig,
    ModelConfig,
)
from dojo.data import DataBundle, build_dataloader, build_datasets
from dojo.model import build_supervised_model
from dojo.results import (
    Provenance,
    ResultWriter,
    classification_output_record,
    embedding_record,
    sample_metadata_record,
)
from dojo.results.schemas import STAGE_HOLDOUT_EVAL, STAGE_INFER
from dojo.storage import get_storage
from dojo.training.checkpoint import checkpoint_hash
from dojo.training.metrics import build_metric_modules, iter_metric_logs


@dataclass(frozen=True)
class InferenceResult:
    output_dir: Path
    results_dir: Path
    manifest_path: Path
    checkpoint_hash: str
    result_record_count: int
    metric_summary: dict[str, Any] | None = None


def _load_checkpoint_contract(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("dojo_inference_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            f"checkpoint does not contain dojo_inference_contract: {checkpoint_path}"
        )
    return checkpoint, contract


def _model_config_from_contract(contract: dict[str, Any]) -> ModelConfig:
    model = contract["model_config"]
    backbone_arch = model["image_input"]["backbone"]["architecture"]
    params = dict(backbone_arch.get("params") or {})
    heads = {
        name: {
            "type": head["type"],
            "target": head["target"],
            "network": head["network"],
            "num_classes": head["num_classes"],
        }
        for name, head in model["heads"].items()
    }
    return ModelConfig.model_validate(
        {
            "image_input": {
                "name": "image",
                "backbone": BackboneConfig(
                    architecture={
                        "source": backbone_arch["source"],
                        "name": backbone_arch["name"],
                        "output_dim": backbone_arch["output_dim"],
                        "input_channels": params.get("input_channels", 3),
                    },
                    weights=BackboneWeightsConfig(source="none"),
                ).model_dump(mode="json"),
            },
            "tabular_input": model.get("tabular_input", {"enabled": False}),
            "embedding_adapter": model.get("embedding_adapter", {"enabled": False}),
            "heads": heads,
        }
    )


def _load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint, contract = _load_checkpoint_contract(checkpoint_path)
    model = build_supervised_model(
        _model_config_from_contract(contract),
        freeze_cfg=FreezeConfig(),
    )
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, contract


def _labels_by_head(contract: dict[str, Any]) -> dict[str, list[str]]:
    return {
        head_name: [str(item["label"]) for item in entries]
        for head_name, entries in contract.get("class_maps", {}).items()
    }


def _class_mapping_for_head(contract: dict[str, Any], head_name: str) -> dict[int, str]:
    return {
        int(item["index"]): str(item["label"])
        for item in contract.get("class_maps", {}).get(head_name, [])
    }


def _write_manifest(
    path: Path,
    *,
    cfg: RootConfig,
    bundle: DataBundle,
    checkpoint_path: Path,
    ckpt_hash: str,
    contract: dict[str, Any],
    record_count: int,
    metric_summary: dict[str, Any] | None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_hash": ckpt_hash,
                "run_id": cfg.runtime.run_id,
                "dataset_hash": bundle.dataset_hash,
                "record_count": record_count,
                "metric_summary": metric_summary,
                "compatibility": contract.get("compatibility", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _metric_modules_from_contract(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objectives = contract.get("objective_summary", {}).get("objectives", {})
    metrics_by_objective: dict[str, dict[str, Any]] = {}
    class_maps = contract.get("class_maps", {})
    for objective_name, objective in objectives.items():
        head_name = objective["head"]
        num_classes = len(class_maps.get(head_name, []))
        metric_names = [metric["name"] for metric in objective.get("metrics", [])]
        metrics_by_objective[objective_name] = {
            "head": head_name,
            "metrics": build_metric_modules(metric_names, num_classes),
        }
    return metrics_by_objective


def _compute_metric_summary(metrics_by_objective: dict[str, dict[str, Any]]) -> dict[str, Any]:
    objectives: dict[str, dict[str, float]] = {}
    for objective_name, spec in metrics_by_objective.items():
        values: dict[str, float] = {}
        for metric_name, metric in spec["metrics"].items():
            for output_name, scalar in iter_metric_logs(metric_name, metric.compute()):
                values[output_name] = float(scalar)
        objectives[objective_name] = values
    return {"objectives": objectives}


def _write_outputs(
    *,
    cfg: RootConfig,
    bundle: DataBundle,
    model: torch.nn.Module,
    contract: dict[str, Any],
    ckpt_hash: str,
    stage: Literal["infer", "holdout_eval"],
    output_kind: Literal["predictions", "embeddings"],
) -> tuple[int, dict[str, Any] | None]:
    provenance = Provenance(
        run_id=cfg.runtime.run_id,
        config_hash=config_hash(cfg),
        dataset_hash=bundle.dataset_hash,
        stage=stage,
    )
    writer = ResultWriter(
        cfg.eval_outputs.results.dir,
        partition_by=cfg.eval_outputs.results.partition_by or ("record_type",),
    )
    labels_by_head = _labels_by_head(contract)
    metrics_by_objective = (
        _metric_modules_from_contract(contract)
        if stage == STAGE_HOLDOUT_EVAL and output_kind == "predictions"
        else {}
    )
    record_count = 0
    for split, dataset in bundle.datasets.items():
        loader = build_dataloader(
            dataset,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.runtime.num_workers,
            batch_aspect_buckets=dataset.has_aspect_buckets(),
            seed=cfg.runtime.seed,
        )
        for batch in loader:
            records: list[dict[str, Any]] = []
            images = batch["image"]
            with torch.no_grad():
                embeddings = model.forward_features(images)
                logits_by_head = {
                    name: head(embeddings) for name, head in model.heads.items()
                }
            for spec in metrics_by_objective.values():
                for metric in spec["metrics"].values():
                    metric.update(logits_by_head[spec["head"]], batch["target"])

            for index, sample_id in enumerate(batch["sample_id"]):
                records.append(
                    sample_metadata_record(
                        provenance,
                        sample_id=sample_id,
                        split=split,
                        uri=batch["uri"][index],
                        native_width_px=batch["native_width_px"][index],
                        native_height_px=batch["native_height_px"][index],
                        resize_width_px=batch["resize_width_px"][index],
                        resize_height_px=batch["resize_height_px"][index],
                        aspect_bucket=batch["aspect_bucket"][index],
                        source_extra=batch["source_extra"][index],
                    )
                )
                if output_kind == "embeddings":
                    records.append(
                        embedding_record(
                            provenance,
                            sample_id=sample_id,
                            split=split,
                            uri=batch["uri"][index],
                            embedding_kind="head_input_embedding",
                            embedding=embeddings[index].tolist(),
                            checkpoint_hash=ckpt_hash,
                        )
                    )

            if output_kind == "predictions":
                for head_name, head_logits in logits_by_head.items():
                    probs = torch.softmax(head_logits, dim=1)
                    confidence, prediction = probs.max(dim=1)
                    labels = labels_by_head[head_name]
                    class_mapping = _class_mapping_for_head(contract, head_name)
                    current_head_hash = head_hash(
                        cfg,
                        head_name=head_name,
                        class_mapping=class_mapping,
                    )
                    for index, sample_id in enumerate(batch["sample_id"]):
                        pred_index = int(prediction[index])
                        target_index = int(batch["target"][index])
                        records.append(
                            classification_output_record(
                                provenance,
                                sample_id=sample_id,
                                split=split,
                                uri=batch["uri"][index],
                                head_name=head_name,
                                head_hash=current_head_hash,
                                target_index=target_index,
                                target_name=labels[target_index]
                                if 0 <= target_index < len(labels)
                                else str(target_index),
                                prediction_index=pred_index,
                                prediction_label=labels[pred_index],
                                prediction_confidence=float(confidence[index]),
                                logits=head_logits[index].tolist(),
                                probabilities=probs[index].tolist(),
                                checkpoint_hash=ckpt_hash,
                            )
                        )
            writer.write_records(records)
            record_count += len(records)
    metric_summary = (
        _compute_metric_summary(metrics_by_objective)
        if metrics_by_objective
        else None
    )
    return record_count, metric_summary


def _execute(
    cfg: RootConfig,
    *,
    checkpoint_path: Path,
    output_kind: Literal["predictions", "embeddings"],
    stage: Literal["infer", "holdout_eval"],
) -> InferenceResult:
    storage = get_storage(cfg.storage)
    bundle = build_datasets(cfg, storage)
    ckpt_hash = checkpoint_hash(checkpoint_path)
    model, contract = _load_model(checkpoint_path)
    record_count, metric_summary = _write_outputs(
        cfg=cfg,
        bundle=bundle,
        model=model,
        contract=contract,
        ckpt_hash=ckpt_hash,
        stage=stage,
        output_kind=output_kind,
    )
    manifest_path = _write_manifest(
        Path(cfg.eval_outputs.dir) / "eval_manifest.json",
        cfg=cfg,
        bundle=bundle,
        checkpoint_path=checkpoint_path,
        ckpt_hash=ckpt_hash,
        contract=contract,
        record_count=record_count,
        metric_summary=metric_summary,
    )
    return InferenceResult(
        output_dir=Path(cfg.eval_outputs.dir),
        results_dir=Path(cfg.eval_outputs.results.dir),
        manifest_path=manifest_path,
        checkpoint_hash=ckpt_hash,
        result_record_count=record_count,
        metric_summary=metric_summary,
    )


def execute_infer(
    cfg: RootConfig,
    *,
    checkpoint_path: Path,
    output_kind: Literal["predictions", "embeddings"],
) -> InferenceResult:
    return _execute(
        cfg,
        checkpoint_path=checkpoint_path,
        output_kind=output_kind,
        stage=STAGE_INFER,
    )


def execute_holdout_eval(
    cfg: RootConfig,
    *,
    checkpoint_path: Path,
) -> InferenceResult:
    return _execute(
        cfg,
        checkpoint_path=checkpoint_path,
        output_kind="predictions",
        stage=STAGE_HOLDOUT_EVAL,
    )

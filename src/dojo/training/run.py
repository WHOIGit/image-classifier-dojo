"""The P1 end-to-end supervised training run (framework-free).

:func:`execute_train` takes an already-resolved :class:`RootConfig` and ties
every architectural boundary together once: build data -> build
model/task -> fit -> hash the best checkpoint -> write canonical tall-Parquet
results + the ``_metadata.json`` sidecar, materializing a run directory with
``config/``, ``checkpoints/``, ``results/``, and ``metrics/``.

No CLI / Typer dependency lives here on purpose: this is the reusable entry
point that the ``dojo train`` command, tests, and any external orchestrator
(e.g. a Prefect flow) call. The CLI is one thin wrapper around it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
from omegaconf import OmegaConf

from dojo.config_schemas import RootConfig, config_hash, head_hash
from dojo.data import (
    DataBundle,
    build_dataloader,
    build_datasets,
    run_dataset_preflight,
)
from dojo.results import (
    ClassificationHeadMeta,
    ClassificationOutputMeta,
    CompatibilityMeta,
    JsonColumnMeta,
    Provenance,
    RecordTypesMeta,
    ResultsMetadata,
    ResultWriter,
    SampleMetadataMeta,
    classification_output_record,
    sample_metadata_record,
    write_metadata_json,
)
from dojo.storage import get_storage
from dojo.training.checkpoint import checkpoint_hash
from dojo.training.figures import merge_metrics_csv, write_training_figures
from dojo.training.inference_contract import build_inference_contract
from dojo.training.task import SupervisedTaskModule
from dojo.training.trainer import build_callbacks, build_logger, build_trainer

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    config_dir: Path
    checkpoint_dir: Path
    results_dir: Path | None
    metrics_dir: Path
    figures_dir: Path | None
    figure_paths: tuple[Path, ...]
    best_checkpoint: Path
    checkpoint_hash: str
    config_hash: str
    dataset_hash: str
    result_record_count: int


def _write_resolved_config(cfg: RootConfig, config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / "resolved.yaml"
    container = OmegaConf.create(cfg.model_dump(mode="json", exclude_none=True))
    out.write_text(OmegaConf.to_yaml(container), encoding="utf-8")
    return out


def _evaluation_splits(bundle: DataBundle) -> list[str]:
    """Splits to score for results: every non-train split, else all present."""

    non_train = [split for split in bundle.datasets if split != "train"]
    return non_train or list(bundle.datasets)


def _write_results(
    *,
    cfg: RootConfig,
    bundle: DataBundle,
    module: SupervisedTaskModule,
    results_dir: Path,
    provenance: Provenance,
    ckpt_hash: str,
    epoch: int,
    global_step: int,
    inference_contract: dict[str, Any],
    status_callback: StatusCallback | None = None,
) -> int:
    """Score the evaluation splits and write canonical result rows + sidecar."""

    device = next(module.parameters()).device
    module.eval()

    heads = cfg.model.heads
    # Names come from the resolved dataset class mapping (data name column, else
    # index strings); indexed by the head's class positions.
    labels = {
        name: [bundle.class_mapping.get(index, str(index)) for index in range(head.num_classes)]
        for name, head in heads.items()
    }
    head_hashes = {
        name: head_hash(cfg, head_name=name, class_mapping=bundle.class_mapping)
        for name in heads
    }

    writer = ResultWriter(
        results_dir,
        partition_by=cfg.training_outputs.results.partition_by or ("record_type",),
    )

    record_count = 0
    for split in _evaluation_splits(bundle):
        dataset = bundle.datasets[split]
        sample_count = len(dataset)
        if status_callback is not None:
            status_callback(
                f"Scoring split '{split}' for result rows ({sample_count} samples)"
            )
        loader = build_dataloader(
            dataset,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.runtime.num_workers,
            batch_aspect_buckets=dataset.has_aspect_buckets(),
            seed=cfg.runtime.seed,
        )
        for batch in loader:
            records: list[dict[str, Any]] = []
            images = batch["image"].to(device)
            with torch.no_grad():
                logits_by_head = module.model(images)

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

            for head_name in heads:
                head_logits = logits_by_head[head_name]
                probs = torch.softmax(head_logits, dim=1)
                confidence, prediction = probs.max(dim=1)
                head_labels = labels[head_name]
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
                            head_hash=head_hashes[head_name],
                            target_index=target_index,
                            target_name=head_labels[target_index]
                            if 0 <= target_index < len(head_labels)
                            else str(target_index),
                            prediction_index=pred_index,
                            prediction_label=head_labels[pred_index],
                            prediction_confidence=float(confidence[index]),
                            logits=head_logits[index].tolist(),
                            probabilities=probs[index].tolist(),
                            epoch=epoch,
                            global_step=global_step,
                            checkpoint_hash=ckpt_hash,
                        )
                    )
            writer.write_records(records)
            record_count += len(records)

    if status_callback is not None:
        status_callback(f"Wrote {record_count} result rows to {results_dir}")
    if status_callback is not None:
        status_callback("Writing result metadata")

    head_meta = {
        name: ClassificationHeadMeta(
            target=head.target,
            head_hash=head_hashes[name],
            classes=labels[name],
            class_mapping={
                str(index): label for index, label in enumerate(labels[name])
            },
        )
        for name, head in heads.items()
    }
    source_extra_meta = (
        JsonColumnMeta(columns=list(cfg.data.source_extra_columns))
        if cfg.data.source_extra_columns
        else None
    )
    metadata = ResultsMetadata(
        run_id=provenance.run_id,
        compatibility=CompatibilityMeta.model_validate(
            inference_contract["compatibility"]
        ),
        record_types=RecordTypesMeta(
            sample_metadata=SampleMetadataMeta(source_extra_json=source_extra_meta),
            classification_output=ClassificationOutputMeta(heads=head_meta),
        ),
    )
    write_metadata_json(results_dir / "_metadata.json", metadata)
    return record_count


def _status(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def execute_train(
    cfg: RootConfig,
    *,
    status_callback: StatusCallback | None = None,
    **trainer_overrides: Any,
) -> TrainResult:
    """Run one resolved supervised training job and write its run directory."""

    _status(status_callback, "Seeding runtime")
    L.seed_everything(cfg.runtime.seed, workers=True)
    storage = get_storage(cfg.storage)

    _status(status_callback, "Building datasets")
    bundle = build_datasets(cfg, storage)
    _status(status_callback, "Running dataset preflight checks")
    preflight_issues = run_dataset_preflight(cfg, bundle)
    errors = [issue for issue in preflight_issues if issue.severity == "error"]
    if errors:
        detail = "; ".join(f"{issue.check}: {issue.message}" for issue in errors)
        raise ValueError(f"dataset preflight failed: {detail}")
    for issue in preflight_issues:
        if issue.severity == "warn":
            warnings.warn(
                f"dataset preflight warning ({issue.check}): {issue.message}",
                stacklevel=2,
            )
    if "train" not in bundle.datasets:
        raise ValueError("training requires a 'train' split in the dataset")

    run_dir = Path(cfg.training_outputs.dir)
    config_dir = run_dir / "config"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    _status(status_callback, f"Writing resolved config to {config_dir}")
    _write_resolved_config(cfg, config_dir)

    _status(status_callback, "Building model and training task")
    inference_contract = build_inference_contract(
        cfg,
        class_mapping=bundle.class_mapping,
    )
    module = SupervisedTaskModule(
        model_config=cfg.model,
        training_config=cfg.training,
        objectives_config=cfg.objectives,
        optimizer_config=cfg.optimizer,
        class_counts_by_head={
            head_name: bundle.class_counts.get("train", {})
            for head_name in cfg.model.heads
        },
        inference_contract=inference_contract,
    )

    callbacks = build_callbacks(cfg.checkpointing, cfg.training, checkpoint_dir)
    logger = build_logger(run_dir)
    trainer = build_trainer(
        cfg, callbacks=callbacks, logger=logger, **trainer_overrides
    )

    _status(status_callback, "Building training dataloader")
    train_loader = build_dataloader(
        bundle.datasets["train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.runtime.num_workers,
        drop_last=True,
        seed=cfg.runtime.seed,
        sampler_type=cfg.training.sampler.type,
        class_counts=bundle.class_counts.get("train", {}),
        class_weight_scheme=cfg.training.sampler.class_weight_scheme,
        class_weight_beta=cfg.training.sampler.beta,
    )
    _status(status_callback, "Building validation dataloader")
    val_loader = (
        build_dataloader(
            bundle.datasets["val"],
            batch_size=cfg.training.batch_size,
            num_workers=cfg.runtime.num_workers,
            batch_aspect_buckets=bundle.datasets["val"].has_aspect_buckets(),
            seed=cfg.runtime.seed,
        )
        if "val" in bundle.datasets
        else None
    )
    _status(status_callback, "Starting Lightning trainer.fit")
    trainer.fit(module, train_loader, val_loader)
    _status(status_callback, "Training finished; merging metrics CSV")
    merge_metrics_csv(run_dir / "metrics" / "metrics.csv")

    _status(status_callback, "Selecting best checkpoint")
    checkpoint_cb = next(
        cb for cb in callbacks if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint)
    )
    best_path = Path(checkpoint_cb.best_model_path or checkpoint_cb.last_model_path)
    _status(status_callback, f"Hashing checkpoint {best_path}")
    ckpt_hash = checkpoint_hash(best_path)

    # Score the best checkpoint, not the final in-memory weights.
    _status(status_callback, "Loading best checkpoint for result scoring")
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"])

    results_dir: Path | None = None
    record_count = 0
    if cfg.training_outputs.results.enabled:
        results_dir = Path(cfg.training_outputs.results.dir)
        splits = ", ".join(_evaluation_splits(bundle))
        _status(status_callback, f"Writing results for split(s): {splits}")
        provenance = Provenance(
            run_id=cfg.runtime.run_id,
            config_hash=config_hash(cfg),
            dataset_hash=bundle.dataset_hash,
        )
        record_count = _write_results(
            cfg=cfg,
            bundle=bundle,
            module=module,
            results_dir=results_dir,
            provenance=provenance,
            ckpt_hash=ckpt_hash,
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            inference_contract=inference_contract,
            status_callback=status_callback,
        )

    figures_dir: Path | None = None
    figure_paths: tuple[Path, ...] = ()
    if cfg.training_outputs.figures.enabled:
        figures_dir = Path(cfg.training_outputs.figures.dir)
        _status(status_callback, f"Writing training figures to {figures_dir}")
        figure_paths = tuple(
            write_training_figures(
                metrics_csv=run_dir / "metrics" / "metrics.csv",
                figures_dir=figures_dir,
                results_dir=results_dir,
            )
        )

    _status(status_callback, "Finalizing run summary")
    return TrainResult(
        run_dir=run_dir,
        config_dir=config_dir,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        metrics_dir=run_dir / "metrics",
        figures_dir=figures_dir,
        figure_paths=figure_paths,
        best_checkpoint=best_path,
        checkpoint_hash=ckpt_hash,
        config_hash=config_hash(cfg),
        dataset_hash=bundle.dataset_hash,
        result_record_count=record_count,
    )

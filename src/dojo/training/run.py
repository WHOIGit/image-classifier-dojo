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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import OmegaConf

from dojo.config_schemas import RootConfig, config_hash
from dojo.data import DataBundle, build_dataloader, build_datasets
from dojo.results import (
    ClassificationHeadMeta,
    ClassificationOutputMeta,
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
from dojo.training.task import SupervisedTaskModule
from dojo.training.trainer import build_callbacks, build_logger, build_trainer


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    config_dir: Path
    checkpoint_dir: Path
    results_dir: Path | None
    metrics_dir: Path
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

    writer = ResultWriter(
        results_dir,
        partition_by=cfg.training_outputs.results.partition_by or ("record_type",),
    )

    records: list[dict[str, Any]] = []
    for split in _evaluation_splits(bundle):
        loader = build_dataloader(
            bundle.datasets[split],
            batch_size=cfg.training.batch_size,
            num_workers=cfg.runtime.num_workers,
        )
        for batch in loader:
            images = batch["image"].to(device)
            with torch.no_grad():
                logits_by_head = module.model(images)

            for index, sample_id in enumerate(batch["sample_id"]):
                records.append(
                    sample_metadata_record(
                        provenance,
                        sample_id=sample_id,
                        split=split,
                        native_width_px=batch["native_width_px"][index],
                        native_height_px=batch["native_height_px"][index],
                        resize_width_px=batch["resize_width_px"][index],
                        resize_height_px=batch["resize_height_px"][index],
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
                    records.append(
                        classification_output_record(
                            provenance,
                            sample_id=sample_id,
                            split=split,
                            head_name=head_name,
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

    head_meta = {
        name: ClassificationHeadMeta(
            target=head.target,
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
        compatibility=None,
        record_types=RecordTypesMeta(
            sample_metadata=SampleMetadataMeta(source_extra_json=source_extra_meta),
            classification_output=ClassificationOutputMeta(heads=head_meta),
        ),
    )
    write_metadata_json(results_dir / "_metadata.json", metadata)
    return len(records)


def execute_train(cfg: RootConfig, **trainer_overrides: Any) -> TrainResult:
    """Run one resolved supervised training job and write its run directory."""

    L.seed_everything(cfg.runtime.seed, workers=True)
    storage = get_storage(cfg.storage)

    bundle = build_datasets(cfg, storage)
    if "train" not in bundle.datasets:
        raise ValueError("training requires a 'train' split in the dataset")

    run_dir = Path(cfg.training_outputs.dir)
    config_dir = run_dir / "config"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(cfg, config_dir)

    module = SupervisedTaskModule(
        model_config=cfg.model,
        training_config=cfg.training,
        objectives_config=cfg.objectives,
        optimizer_config=cfg.optimizer,
    )

    callbacks = build_callbacks(cfg.checkpointing, cfg.training, checkpoint_dir)
    logger = build_logger(run_dir)
    trainer = build_trainer(
        cfg, callbacks=callbacks, logger=logger, **trainer_overrides
    )

    train_loader = build_dataloader(
        bundle.datasets["train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.runtime.num_workers,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            bundle.datasets["val"],
            batch_size=cfg.training.batch_size,
            num_workers=cfg.runtime.num_workers,
        )
        if "val" in bundle.datasets
        else None
    )
    trainer.fit(module, train_loader, val_loader)

    checkpoint_cb = next(
        cb for cb in callbacks if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint)
    )
    best_path = Path(checkpoint_cb.best_model_path or checkpoint_cb.last_model_path)
    ckpt_hash = checkpoint_hash(best_path)

    # Score the best checkpoint, not the final in-memory weights.
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"])

    results_dir: Path | None = None
    record_count = 0
    if cfg.training_outputs.results.enabled:
        results_dir = Path(cfg.training_outputs.results.dir)
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
        )

    return TrainResult(
        run_dir=run_dir,
        config_dir=config_dir,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        metrics_dir=run_dir / "metrics",
        best_checkpoint=best_path,
        checkpoint_hash=ckpt_hash,
        config_hash=config_hash(cfg),
        dataset_hash=bundle.dataset_hash,
        result_record_count=record_count,
    )

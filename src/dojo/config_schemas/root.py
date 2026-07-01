"""P1 root configuration schema.

The schema here intentionally covers only the P1 supervised thin slice.
Deferred and later-priority values are absent rather than stubbed.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    name: str


class TaskConfig(StrictModel):
    type: Literal["supervised"]


class AutobatchConfig(StrictModel):
    enabled: bool = False
    mode: Literal["binsearch", "power"] = "binsearch"


Severity = Literal["error", "warn", "ignore"]


class ThresholdCheckConfig(StrictModel):
    severity: Severity = "warn"
    threshold: float


class PreflightChecksConfig(StrictModel):
    empty_train_classes: Severity = "error"
    empty_eval_classes: Severity = "warn"
    missing_required_targets: Severity = "error"
    no_remaining_valid_target_labels: Severity = "error"
    non_contiguous_class_indices: Severity = "error"
    imbalance_ratio_gt: ThresholdCheckConfig = Field(
        default_factory=lambda: ThresholdCheckConfig(threshold=20.0)
    )


class PreflightConfig(StrictModel):
    enabled: bool = True
    checks: PreflightChecksConfig = Field(default_factory=PreflightChecksConfig)


class RuntimeConfig(StrictModel):
    seed: int = 123
    run_id: str = "{coolname:noseed}"
    sweep_id: str | None = None
    precision: str = "bf16-mixed"
    float32_matmul_precision: Literal["highest", "high", "medium"] = "medium"
    num_workers: int = 0
    progress_bar: bool = True
    fast_dev_run: bool = False
    autobatch: AutobatchConfig = Field(default_factory=AutobatchConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)


class StorageConfig(StrictModel):
    local_cache_dir: str = "./.cache/dojo"


SplitName = Literal["train", "val", "test", "unlabeled", "holdout"]


class ParquetImageColumnConfig(StrictModel):
    column: str
    bytes_field: str = "bytes"
    path_field: str | None = "path"


class TargetConfig(StrictModel):
    type: Literal["multiclass_classification"]
    # A target's per-sample label comes from an integer index column
    # (``label_index_column``) and/or a string name column (``label_name_column``).
    # At least one is required; when only names are given, class indices are
    # assigned to the distinct names alphabetically.
    label_index_column: str | None = None
    label_name_column: str | None = None
    missing_policy: Literal["error", "drop_sample", "mask_objective"] = "error"

    @model_validator(mode="after")
    def validate_label_source(self) -> "TargetConfig":
        if self.label_index_column is None and self.label_name_column is None:
            raise ValueError(
                "a data target requires label_index_column and/or label_name_column"
            )
        return self


class DataConfig(StrictModel):
    backend: Literal["parquet_images", "parquet_manifest"]
    manifest_uri: str
    file_pattern: str | None = None
    stats_cache_uri: str | None = None
    sample_id_column: str
    split_column: str | None = None
    split_from_filename: dict[SplitName, str] | None = None
    image_uri_column: str | None = None
    images: ParquetImageColumnConfig | None = None
    source_extra_columns: list[str] = Field(default_factory=list)
    tabular_feature_columns: list[str] | dict[str, Any] | None = None
    targets: dict[str, TargetConfig]

    @model_validator(mode="after")
    def validate_backend_fields(self) -> "DataConfig":
        if self.backend == "parquet_images" and self.images is None:
            raise ValueError("data.images is required when data.backend is parquet_images")
        if self.backend == "parquet_manifest" and self.image_uri_column is None:
            raise ValueError(
                "data.image_uri_column is required when data.backend is parquet_manifest"
            )
        split_sources = [
            self.split_column is not None,
            self.split_from_filename is not None,
        ]
        if sum(split_sources) != 1:
            raise ValueError(
                "exactly one of data.split_column or data.split_from_filename is required"
            )
        if self.split_from_filename is not None and not self.split_from_filename:
            raise ValueError(
                "data.split_from_filename must map at least one split to a filename pattern"
            )
        if not self.targets:
            raise ValueError("data.targets must contain at least one target")
        return self


class LetterboxStep(StrictModel):
    name: Literal["letterbox"]
    enabled: bool = True
    train_only: bool = False
    canvas_size: tuple[int, int]


class NormalizeStep(StrictModel):
    name: Literal["normalize"]
    enabled: bool = True
    train_only: bool = False
    mode: Literal["fixed"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class RotateStep(StrictModel):
    name: Literal["rotate"]
    enabled: bool = True
    train_only: bool = False
    mode: Literal["multiples_of_90"] = "multiples_of_90"
    p: float = Field(default=0.5, ge=0.0, le=1.0)


class HorizontalFlipStep(StrictModel):
    name: Literal["horizontal_flip"]
    enabled: bool = True
    train_only: bool = False
    p: float = Field(default=0.5, ge=0.0, le=1.0)


class VerticalFlipStep(StrictModel):
    name: Literal["vertical_flip"]
    enabled: bool = True
    train_only: bool = False
    p: float = Field(default=0.5, ge=0.0, le=1.0)


TransformStep = Annotated[
    LetterboxStep | NormalizeStep | RotateStep | HorizontalFlipStep | VerticalFlipStep,
    Field(discriminator="name"),
]


class TransformsConfig(StrictModel):
    image_mode: Literal["rgb", "grayscale", "grayscale_repeat3"]
    input_bit_depth: Literal["auto", 8, 12, 16] = "auto"
    pipeline: list[TransformStep] = Field(min_length=1)
    inference_pipeline: list[TransformStep] | None = None


class BackboneArchitectureConfig(StrictModel):
    source: Literal["torchvision"]
    name: Literal["efficientnet_b0"]
    output_dim: int | Literal["auto"] = "auto"
    input_channels: int = 3


class BackboneWeightsConfig(StrictModel):
    source: Literal["library", "none"]
    name: str | None = None


class BackboneConfig(StrictModel):
    architecture: BackboneArchitectureConfig
    weights: BackboneWeightsConfig


class ImageInputConfig(StrictModel):
    name: str = "image"
    backbone: BackboneConfig


class DisabledTabularInputConfig(StrictModel):
    enabled: Literal[False] = False
    name: str = "tabular"


class DisabledEmbeddingAdapterConfig(StrictModel):
    enabled: Literal[False] = False


class LinearNetworkConfig(StrictModel):
    type: Literal["linear"] = "linear"


class MlpNetworkConfig(StrictModel):
    type: Literal["mlp"]
    hidden_dims: tuple[int, ...] = Field(min_length=1)
    activation: Literal["gelu", "relu"] = "gelu"
    dropout: float = Field(default=0.0, ge=0.0)


NetworkConfig = Annotated[LinearNetworkConfig | MlpNetworkConfig, Field(discriminator="type")]


class HeadConfig(StrictModel):
    type: Literal["multiclass_classification"]
    target: str
    num_classes: int = Field(gt=1)
    network: NetworkConfig = Field(default_factory=LinearNetworkConfig)


class ModelConfig(StrictModel):
    image_input: ImageInputConfig
    tabular_input: DisabledTabularInputConfig = Field(
        default_factory=DisabledTabularInputConfig
    )
    embedding_adapter: DisabledEmbeddingAdapterConfig = Field(
        default_factory=DisabledEmbeddingAdapterConfig
    )
    heads: dict[str, HeadConfig]

    @model_validator(mode="after")
    def validate_heads(self) -> "ModelConfig":
        if not self.heads:
            raise ValueError("model.heads must contain at least one head")
        return self


MetricName = Literal[
    "accuracy",
    "f1_macro",
    "f1_micro",
    "f1_per_class",
    "precision_macro",
    "precision_micro",
    "recall_macro",
    "recall_micro",
]


class CrossEntropyLossConfig(StrictModel):
    type: Literal["cross_entropy"]
    params: dict[str, Any] = Field(default_factory=dict)


class ObjectiveConfig(StrictModel):
    head: str | None = None
    loss: Literal["cross_entropy"] | CrossEntropyLossConfig = "cross_entropy"
    metrics: list[MetricName] = Field(default_factory=lambda: ["accuracy"])
    weight: float = Field(default=1.0, ge=0.0)
    enabled: bool = True


class BackboneFreezeConfig(StrictModel):
    policy: Literal["none"]


class FreezeConfig(StrictModel):
    backbone: BackboneFreezeConfig = Field(
        default_factory=lambda: BackboneFreezeConfig(policy="none")
    )


class EarlyStoppingConfig(StrictModel):
    enabled: bool = False
    monitor: str | None = None
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=10, ge=1)


class TrainingConfig(StrictModel):
    max_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    freeze: FreezeConfig = Field(default_factory=FreezeConfig)
    early_stopping: EarlyStoppingConfig | None = None


class OptimizerConfig(StrictModel):
    name: Literal["adamw"]
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)


class CheckpointingConfig(StrictModel):
    monitor: str
    mode: Literal["min", "max"]
    save_top_k: int = Field(ge=-1)
    save_last: bool = True


class LocalLoggerSinkConfig(StrictModel):
    type: Literal["local"]


class LoggingConfig(StrictModel):
    sinks: list[LocalLoggerSinkConfig] = Field(
        default_factory=lambda: [LocalLoggerSinkConfig(type="local")]
    )


PartitionKey = Literal["record_type", "stage", "epoch", "sweep_id"]


class ResultsOutputConfig(StrictModel):
    enabled: bool = True
    backend: Literal["amplify_db_utils"] = "amplify_db_utils"
    dir: str = "results"
    format: Literal["parquet"] = "parquet"
    partition_by: list[PartitionKey] = Field(default_factory=lambda: ["record_type"])
    write_metadata_json: bool = True


class MetricsOutputConfig(StrictModel):
    enabled: bool = True
    dir: str = "metrics"


class FiguresOutputConfig(StrictModel):
    enabled: bool = False
    dir: str = "figures"


class ExportOutputConfig(StrictModel):
    enabled: bool = False
    dir: str = "exports"


class TrainingOutputsConfig(StrictModel):
    dir: str | None = None
    dir_template: str | None = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    results: ResultsOutputConfig = Field(default_factory=ResultsOutputConfig)
    metrics: MetricsOutputConfig = Field(default_factory=MetricsOutputConfig)
    figures: FiguresOutputConfig = Field(default_factory=FiguresOutputConfig)
    export: ExportOutputConfig = Field(default_factory=ExportOutputConfig)

    @model_validator(mode="after")
    def validate_dir_source(self) -> "TrainingOutputsConfig":
        if self.dir is None and self.dir_template is None:
            raise ValueError("training_outputs.dir or dir_template is required")
        return self


class RootConfig(StrictModel):
    experiment: ExperimentConfig
    task: TaskConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    data: DataConfig
    transforms: TransformsConfig
    model: ModelConfig
    objectives: dict[str, ObjectiveConfig]
    training: TrainingConfig
    optimizer: OptimizerConfig
    checkpointing: CheckpointingConfig
    output_root: str = "./runs"
    training_outputs: TrainingOutputsConfig

    @model_validator(mode="after")
    def validate_references(self) -> "RootConfig":
        if not self.objectives:
            raise ValueError("objectives must contain at least one objective")

        for head_name, head in self.model.heads.items():
            if head.target not in self.data.targets:
                raise ValueError(
                    f"model.heads.{head_name}.target references missing data target "
                    f"{head.target!r}"
                )
            target = self.data.targets[head.target]
            if target.type != head.type:
                raise ValueError(
                    f"model.heads.{head_name}.type {head.type!r} is incompatible "
                    f"with data.targets.{head.target}.type {target.type!r}"
                )

        for objective_name, objective in self.objectives.items():
            head_name = objective.head or objective_name
            if head_name not in self.model.heads:
                raise ValueError(
                    f"objectives.{objective_name}.head references missing head "
                    f"{head_name!r}"
                )
            if not objective.enabled:
                continue
            if objective.weight == 0:
                continue
            loss_type = (
                objective.loss
                if isinstance(objective.loss, str)
                else objective.loss.type
            )
            if loss_type != "cross_entropy":
                raise ValueError(
                    f"objectives.{objective_name}.loss {loss_type!r} is not "
                    "implemented for P1"
                )

        if not any(obj.enabled and obj.weight > 0 for obj in self.objectives.values()):
            raise ValueError("at least one enabled objective with positive weight is required")
        return self

    def supervised_record_types(self) -> list[str]:
        record_types = ["sample_metadata"]
        if any(head.type == "multiclass_classification" for head in self.model.heads.values()):
            record_types.append("classification_output")
        return record_types

    def result_stages(self) -> list[str]:
        return ["train_validation"]

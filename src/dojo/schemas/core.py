import os
from typing import List, Optional, Union, Literal, Annotated, Dict, Iterable, Tuple, Sequence
import random

import coolname
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import CliImplicitFlag


# =========================
# TRAINING AUGMENTATION
# =========================

class ImageNormalizationConfig(BaseModel):
    mean: Union[float,Tuple[float,float,float]] = Field((0.485, 0.456, 0.406),
        description="Mean values for each channel used in normalization. Defaults to ImageNet standards")
    std: Union[float,Tuple[float,float,float]] = Field((0.229, 0.224, 0.225),
        description="Standard deviation values for each channel used in normalization. Defaults to ImageNet standards")

    @field_validator('mean','std', mode="after")
    @classmethod
    def ensure_list(cls, val: BaseModel):
        # If it's a single object, wrap it in a list
        if isinstance(val, float):
            return val,val,val
        return val


class TrainingAugmentationConfig(BaseModel):
    # Accepts 'x', 'y', 'xy'
    flip: Optional[Literal["x", "y", "xy"]] = Field(None,
        description=(
            "Training images have 50% chance of being flipped along the designated axis: "
            "(x) vertically, (y) horizontally, (xy) either/both."
        ))


# =========================
# DATASET
# =========================
class ListfileDatasetConfig(BaseModel):
    classlist: str = Field(...,
        description="A text file, each line is a class label (the label order is significant)")
    trainlist: str = Field(...,
        description="A text file, one sample per line, each sample has a class-index and image path")
    vallist: str = Field(...,
        description="Like trainlist, but for validation metrics and early-stopping/overfit-prevention")
    testlist: Optional[str] = Field(None,
        description="Like trainlist, but for final test metrics. Optional")
    # sampler: Optional[str] = None  # e.g., "module.Class"
    # is_batch_sampler: bool = False
    # feature_store: Optional[str] = None
    # dataset: Optional[str] = None


class DatasetRuntimeConfig(BaseModel):
    dataset: ListfileDatasetConfig
    img_norm: Optional[ImageNormalizationConfig] = None
    #base_transforms: BaseTransformConfig
    training_transforms: Optional[TrainingAugmentationConfig] = None


# =========================
# TRACKING (AimLogger)
# =========================

class S3Config(BaseModel):
    bucket: str = Field(..., description='S3 Bucket Name')
    prefix: Optional[str] = Field('', description='Optional Prefix/Folder in the Bucket')
    endpoint: str = Field(..., description='S3 Endpoint for ')
    accesskey: str = Field(..., description='Access Key')
    secretkey: str = Field(..., description='Secret Key')

    @property
    def uri(self):
        if self.prefix:
            return f's3://{self.bucket}/{self.prefix}'
        return f's3://{self.bucket}'

class AimLoggerConfig(BaseModel):
    repo: str = Field(...,
        description="Aim repo path.")
    experiment: Optional[str] = Field(None,
         description="The broader category/grouping this RUN belongs to")
    run: Optional[str] = Field(None, #default_factory=lambda: coolname.generate_slug(2),
        description="The name of this run. May get overwritten by RuntimeConfig.run_name")
    note: Optional[str] = Field(None,
        description='Add any kind of note or description to the trained model.')
    artifacts_location: Optional[Union[str,S3Config]] = Field(None,
        description="Aim Artifacts location. Either a local path or an S3 location")
    context_prefixes: Optional[Dict[str, Dict[str, str]]] = Field(None, description=
        "Sets a metric's context based on the logged metric's name's prefix",
        examples=[dict(subset = dict(train='train_', val='val_', test='test_'))])
    context_postfixes: Optional[Dict[str, Dict[str, str]]] = Field(None, description=
        "Sets a metric's context based on the logged metric's name's postfix",
        examples=[dict( averaging={'macro': '_macro', 'weighted': '_weighted',
                                   'micro': '_micro', 'none': '_perclass'})])
    # todo RUN HASH
    # todo plot / callback / metric

    @field_validator("artifacts_location", mode="after")
    @classmethod
    def file_location_prefix(cls, val):
        # If it's a single object, wrap it in a list
        if isinstance(val, str):
            if val.startswith('file://'): return val
            return f'file://{os.path.abspath(val)}'
        return val

# =========================
# MODEL CONFIG
# =========================
class ModelBackboneConfig(BaseModel):
    # model is required
    model_name: str = Field(..., description="Model Class/Module Name")

    pretrained_weights: Union[Literal['DEFAULT',None],str] = Field(None, examples=["DEFAULT"],
        description='''Specify a model's downloadable pretrained weights. 
Either "DEFAULT", some specific identifier, or "None" for no-pretrained-weights''')


class MulticlassHeadConfig(BaseModel):
    name: Optional[str] = Field('head', description='An optional name for this head, if multiple heads are used')
    num_classes: Union[None,int] = Field(None, description='The number of classes to predict')

class EmbeddingsHeadConfig(BaseModel):
    name: Optional[str] = Field('head', description='An optional name for this head, if multiple heads are used')
    embedding_size: list[int] = Field(..., description='The size of the output embedding vector')

class RegressionHeadConfig(BaseModel):
    name: Optional[str] = Field('head', description='An optional name for this head, if multiple heads are used')
    range_min: float = Field(0.0, description='The minimum possible value of the output')
    range_max: float = Field(1.0, description='The maximum possible value of the output')

class ModelConfig(BaseModel):
    backbone: ModelBackboneConfig = Field(..., description='The model backbone configuration')
    head: Union[MulticlassHeadConfig, RegressionHeadConfig] = Field(..., description='The model head (output) configuration')
    weights: Optional[str] = Field(None, description='Path to model weights/checkpoint to load')

class MultiheadModelConfig(BaseModel):
    backbone: ModelBackboneConfig = Field(..., description='The model backbone configuration')
    heads: List[Union[MulticlassHeadConfig, RegressionHeadConfig]] = Field(..., description='A list of model heads (outputs). At least one head is required.')

# =========================
# TRAINING CONFIG
# =========================

## Loss Functions ##

class MulticlassLossFunctionConfig(BaseModel):
    # ignore_index: Optional[int] = Field(-100,
    #     description='Specifies a target value that is ignored and does not contribute to the loss / input gradient')
    reduction: Literal['sum','mean','none'] = Field('sum',
        description='How loss for each input gets aggregated.')

class CrossEntropyLossConfig(MulticlassLossFunctionConfig):
    label_smoothing: float = Field(0.0, examples=[0.1, 0.2],
        description='Label Smoothing Regularization arg. Range is 0-1. Default is 0')
    weight: Optional[Union[Literal['normalize'], list[float]]] = Field(None,
        description='If "normalize", rare class instances will be boosted. '
                    'Otherwise accepts a list of per-class weights. Length must match number of classes')

class FocalLossConfig(MulticlassLossFunctionConfig):
    gamma: float = Field(1.0, examples=[1.0,2.0,5.0],
        description='Rate at which easy examples are down-weighted')
    alpha: Optional[list[float]] = Field(None,
        description='A list of per-class weights. Length must match number of classes')

## Optimizers ##

class AdamConfig(BaseModel):
    lr: float = Field(0.001, description="Initial Learning Rate. Default is 0.001")
    amsgrad: bool = Field(False, description="Use the AMSGrad variant of Adam")
    #betas: tuple[float, float] = Field((0.9, 0.999), description="Adam beta coefficients (beta1, beta2)")
    #eps: float = Field(1e-8, description="Term added to the denominator for numerical stability")
    #weight_decay: float = Field(0.0, description="L2 penalty (weight decay)")
    # foreach: Optional[bool] = None,
    # maximize: bool = False,
    # capturable: bool = False,
    # differentiable: bool = False,
    # fused: Optional[bool] = None,
    # decoupled_weight_decay: bool = False,

class AdamWConfig(BaseModel):
    lr: float = Field(0.001, description="Initial Learning Rate. Default is 0.001")
    amsgrad: bool = Field(False, description="Use the AMSGrad variant of AdamW")
    weight_decay: float = Field(0.01, description="L2 penalty (weight decay)")

class SGDConfig(BaseModel):
    lr: float = Field(0.001, description="Initial Learning Rate. Default is 0.001")
    momentum: float = Field(0.0, description="SGD momentum factor")
    dampening: float = Field(0.0, description="SGD dampening for momentum")
    weight_decay: float = Field(0.0, description="L2 penalty (weight decay)")
    nesterov: bool = Field(False, description="Enable Nesterov momentum")

class ModelTrainingOptimizationConfig(BaseModel):
    loss_config: Union[CrossEntropyLossConfig, FocalLossConfig] = Field(..., description="Loss Function.")

    optimizer_config: Union[AdamConfig, AdamWConfig, SGDConfig] = Field(...,
        description="Optimizer configuration. Defaults to Adam with lr=0.001")
    # todo learning rate scheduler config

    freeze: Union[None,int,float] = Field(None,
        description=
            "Freezes a models leading feature layers. "
            "Positive int freezes the first N layers/features/blocks. "
            "A negative int like '-1' freezes all but the last N feature/layer/block. "
            "A positive float like '0.8' freezes the leading 80% of features/layers/blocks. "
            "fc or final_classifier layers are never frozen."
        ) # todo everything prior-to a NAMED layer

## SWA ##
# import lightning.pytorch.callbacks.stochastic_weight_avg
# https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
# see https://chatgpt.com/c/68f138d0-7fec-8327-8124-f30c9abbc297
#     Option A — Lightning-native or Option B — Post-hoc SWA with PyTorch utilities
class SWACallbackConfig(BaseModel):
    swa_lr: float = Field(0.0003, description="Stochastic Weight Averaging learning rate. Should be 0.1 to 0.3 of the global learning rate (typically 0.001)")
    swa_epoch_start: Union[int, float] = Field(0,
        description = "If provided as int, the procedure will start from the ``swa_epoch_start``-th epoch."
                      "If provided as float between 0 and 1, the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch")
    annealing_epochs: int = Field(5, description="Reduces the learning rate to swa_lr over these many epochs")
    annealing_strategy: Literal["cos", "linear"] = Field("cos", description="Stochastic Weight Averaging annealing strategy")

class SWAPolishConfig(BaseModel):
    swa_lr: float = Field(0.0003, description="Stochastic Weight Averaging learning rate during polishing. ")
    epochs: int = Field(5, description='Number of epochs to run SWA polishing for at the end of training. 2-5 epochs should be fine', examples=[2,5])
    sgd_momentum: float = Field(0.9, description="SGD momentum factor for SWA polishing")


## Epochs ##

class EpochConfig(BaseModel):
    max_epochs: int = Field(100, description="Maximum number of training epochs. Default is 100")
    min_epochs: Optional[int] = Field(None, description="Minimum number of training epochs. Default is 10")
    patience: Optional[int] = Field(None,
        description="Early Stopping: Number of epochs following a best-epoch after-which to stop training")
    # Todo EarlyStoppingCallbackConfig

## TRAINER CONFIG ##
_PRECISION_INPUT_STR = Literal[
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
]
class TrainingConfig(BaseModel):
    epochs: EpochConfig = Field(default_factory=EpochConfig, description='Epoch configuration')
    batch_size: int = Field(256, description="Number of images per batch. Defaults is 256")

    model_optims: ModelTrainingOptimizationConfig = Field(default_factory=ModelTrainingOptimizationConfig, description='loss, optimizer, and freeze configs')

    precision: _PRECISION_INPUT_STR = Field("16-mixed",
        description='dtype Precision')

    swa: Union[None,SWACallbackConfig, SWAPolishConfig] = Field(None,
        description="Stochastic Weight Averaging (SWA) configuration. If provided, enables SWA training or SWA best-epoch polishing.")

    # ensemble: Optional[str] = Field(None, description="Model Ensembling mode")

    @model_validator(mode='after')
    def validate_swa_and_early_stopping(self):
        if isinstance(self.swa, SWACallbackConfig) and self.epochs.patience:
            raise ValueError("SWACallbackConfig swa_config cannot be used with Early Stopping from epoch_config.patience"
                             "Either renounce Early Stopping, or switch to SWAPolishConfig")
        return self

# =========================
# UTILITIES / MISC
# =========================

class AutoBatchConfig(BaseModel):
    mode: Literal["power", "binsearch"] = Field('power', description="Power-of-2 scaling or binary search")
    max_size: int = Field(1024, description="batch size limit")
    min_size: int = Field(32, description="batch size minimum. Error is raised if autobatch tries to go lower.")


class ModelCheckpointConfig(BaseModel):
    dirpath: str = './experiments/{EXPERIMENT}/{RUN}'
    filename: str = 'loss-{val_loss:3.3f}_ep-{epoch:03.0f}'
    monitor: str = 'val_loss'
    save_last: str = 'link'
    save_top_k: int = 3
    auto_insert_metric_name: bool = False

class OnnxCheckpointConfig(BaseModel):
    filename: str = Field("{CHECKPOINT}{HALF}{BATCH}{DEVICE}",
                          description='Recognizes same formatting rules as ModelCheckpoint.filename, plus {CHECKPOINT} {HALF} {BATCH} {DEVICE}.'
                                      "CHECKPOINT inherits from run's ModelCheckpointConfig.filename")
    monitor: Optional[str] = Field(None, description='If None, will use ModelCheckpointConfig.monitor')
    half: bool = Field(False, description="Export the model with half-precision (float16)")
    batch_size: Optional[int] = Field(None, description="Batch size to use for the export. If not set, 'dynamic' batch sizing is used")
    device: Literal["cuda", "cpu"] = Field("cuda", description="Device to use for the export")
    opset: Optional[int] = Field(None, description="Force ONNX opset version to export the model with.", examples=[20,22])
    input_names: Sequence[str] = Field(["input"], description="Names to assign to the input nodes of the graph.")
    output_names: Sequence[str] = Field(["output"], description="Names to assign to the output nodes of the graph.")
    dirpath: Optional[str] = Field(None, description='If None, will use ModelCheckpointConfig.dirpath')

    @model_validator(mode='after')
    def format_filename(self):
        sep = '.'
        kwargs = dict(
            CHECKPOINT = '{CHECKPOINT}',
            HALF = f'{sep}fp16' if self.half else '',
            BATCH = f'{sep}b{self.batch_size}' if self.batch_size else '',
            DEVICE = f'{sep}{self.device}'
        )
        self.filename = self.filename.format(**kwargs)
        return self

class RuntimeConfig(BaseModel):
    experiment: Optional[str] = Field(None,
        description="The broader category/grouping this RUN belongs to")
    run_name: Optional[str] = Field(default_factory=lambda: coolname.generate_slug(2),
        description="The name of this run. By Default, a run name is automatically generated")
    checkpoint_callback_config: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    seed: Optional[int] = Field(default_factory=lambda: random.randint(0, 2**32 - 1),
        description="Set a specific seed for deterministic output")
    num_workers: int = Field(4, description="Number of data-loading threads. 4 per GPU is typical")
    autobatch: Optional[AutoBatchConfig] = Field(None, description="Auto-Tunes batch_size prior to training/inference.")
    fast_dev_run: CliImplicitFlag[bool] = Field(False, description="Runs a single batch of train, val, and test to find any bugs. Default is False")
    # device: Union[Literal['cpu','cuda','CUDA_VISIBLE_DEVICES'], List[int]] = Field('cuda',
    #     description='Which GPU(s) to use, by ID. Default is "cuda" (equivalent to "CUDA_VISIBLE_DEVICES"). Use "cpu" for CPU only.')
    onnx_callback_configs: Optional[List[OnnxCheckpointConfig]] = Field([], description="If provided, exports the trained model to ONNX")

    @model_validator(mode='after')
    def format_checkpoint_dirpath(self):
        self.checkpoint_callback_config.dirpath = \
            self.checkpoint_callback_config.dirpath.format(
                EXPERIMENT=self.experiment or '', RUN=self.run_name).replace(os.sep+os.sep,os.sep)

        # onnx callback inherits properties from checkpoint_callback
        for onnx_callback_config in self.onnx_callback_configs:
            if 'CHECKPOINT' in onnx_callback_config.filename:
                onnx_callback_config.filename = onnx_callback_config.filename.format(
                    CHECKPOINT=self.checkpoint_callback_config.filename)
            if onnx_callback_config.dirpath is None:
                onnx_callback_config.dirpath = self.checkpoint_callback_config.dirpath
            if onnx_callback_config.monitor is None:
                onnx_callback_config.monitor = self.checkpoint_callback_config.monitor

        return self

# =========================
# COMPOSED TASK CONFIG (handy for end-to-end configs)
# =========================
class TrainingRunConfig(BaseModel):
    logger: AimLoggerConfig
    dataset_config: DatasetRuntimeConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    # plotting callbacks


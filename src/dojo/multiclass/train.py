import os
import argparse
import random
import warnings
from typing import List

import coolname
from dotenv import load_dotenv

from torchvision.transforms import v2
import torch
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from aim.pytorch_lightning import AimLogger
from aim.storage.artifacts.s3_storage import S3ArtifactStorage_clientconfig

from dojo.schemas import *

from dojo.patches.model_summary_patch import ModelSummaryWithGradCallback
from dojo.utils.onnx_checkpoint import OnnxCheckpoint

from dojo.multiclass.callbacks import BarPlotMetricAim, PlotConfusionMetricAim, PlotPerclassDropdownAim, \
    LogNormalizedLoss
from dojo.multiclass.datasets import ImageListsWithLabelIndex
from dojo.multiclass.models import MulticlassClassifier, get_model_base_transforms, check_model_name


def parse_training_transforms(cfg: TrainingAugmentationConfig):
    training_transforms = []
    if cfg.flip:
        flip_tforms = []
        if 'x' in cfg.flip:
            flip_tforms.append(v2.RandomVerticalFlip(p=0.5))
        if 'y' in cfg.flip:
            flip_tforms.append(v2.RandomHorizontalFlip(p=0.5))
        training_transforms.extend(flip_tforms)
    return training_transforms


def setup_model_and_datamodule(cfg: TrainingRunConfig):
    # Training Augmentation Setup
    training_transforms = parse_training_transforms(cfg.dataset_config.training_transforms)

    # Model and Datamodule
    model_name = cfg.model.backbone.model_name = check_model_name(cfg.model.backbone.model_name)

    model_base_transforms = get_model_base_transforms(model_name, cfg.dataset_config.img_norm)
    dataset = cfg.dataset_config.dataset
    datamodule = ImageListsWithLabelIndex(
        dataset.trainlist,
        dataset.vallist,
        dataset.classlist,
        base_transforms=model_base_transforms,
        training_transforms=training_transforms,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.runtime.num_workers)
    cfg.model.head.num_classes = len(datamodule.classes)

    if isinstance(lc:=cfg.training.model_optims.loss_config, CrossEntropyLossConfig):
        if lc.weight == 'normalize':
            datamodule.setup('fit')
            class_counts = torch.bincount(torch.IntTensor(datamodule.training_dataset.targets + datamodule.validation_dataset.targets))
            class_weights = 1.0 / class_counts.float()
            lc.weight = class_weights / class_weights.mean()
        # elif os.path.isfile(loss_weights):
        #     with open(loss_weight) as f:
        #         loss_weights_tensor = torch.Tensor([float(line) for line in f.read().splitlines()])

    lightning_module = MulticlassClassifier(
        cfg.model, cfg.training.model_optims
    )
    return lightning_module, datamodule


def setup_aimlogger(cfg: AimLoggerConfig):

    if isinstance(cfg.artifacts_location,S3Config):
        s3cfg = cfg.artifacts_location
        S3ArtifactStorage_clientconfig(endpoint_url=s3cfg.endpoint,
                                  aws_access_key_id=s3cfg.accesskey,
                              aws_secret_access_key=s3cfg.secretkey)
    logger = AimLogger(
        repo=cfg.repo,
        run_name=cfg.run,
        experiment=cfg.experiment,
        context_prefixes=cfg.context_prefixes,
        context_postfixes=cfg.context_postfixes,
    )

    if cfg.artifacts_location:
        if isinstance(cfg.artifacts_location, S3Config):
            logger.experiment.set_artifacts_uri(cfg.artifacts_location.uri)
        else:
            logger.experiment.set_artifacts_uri(cfg.artifacts_location)
    if cfg.note:
        logger.experiment.props.description = cfg.note

    return logger


def main(cfg: TrainingRunConfig):
    torch.set_float32_matmul_precision('medium')
    # print(torch.__version__, torch.version.cuda)
    # print("is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # if torch.cuda.is_available():
    #     print(torch.cuda.get_device_name(0))
    #     print("capability:", torch.cuda.get_device_capability(0))
    pl.seed_everything(cfg.runtime.seed)

    ## Setup Model & Data Module ##
    model, datamodule = setup_model_and_datamodule(cfg)

    ## Setup Epoch Logger ##
    if isinstance(cfg.logger, AimLoggerConfig):
        logger = setup_aimlogger(cfg.logger)
    else:
        raise ValueError("Unsupported logger configuration")

    ## Setup Callbacks ##
    callbacks=[]

    validation_results_callbacks = [
        LogNormalizedLoss(),
    ]
    callbacks.extend(validation_results_callbacks)

    plotting_callbacks = [
        #BarPlotMetricAim('F1Score_perclass', order_reverse=True),
        BarPlotMetricAim('MulticlassF1Score-perclass', order_by='MulticlassF1Score-perclass'),
        #BarPlotMetricAim('F1Score_perclass', title='{METRIC} by {ORDER} (ep{EPOCH})', order_by='class-counts'),

        #BarPlotMetricAim('Recall_perclass', order_reverse=True),
        #BarPlotMetricAim('Recall_perclass', order_by='Recall_perclass'),
        #BarPlotMetricAim('Recall_perclass', title='{METRIC} by {ORDER} (ep{EPOCH})', order_by='class-counts'),

        #BarPlotMetricAim('Precision_perclass', order_reverse=True),
        #BarPlotMetricAim('Precision_perclass', order_by='Precision_perclass'),
        #BarPlotMetricAim('Precision_perclass', title='{METRIC} by {ORDER} (ep{EPOCH})', order_by='class-counts'),

        #PlotConfusionMetricAim(order_by='classes'),
        PlotConfusionMetricAim(order_by='classes', normalize=True),
        PlotConfusionMetricAim(order_by='MulticlassF1Score-perclass', normalize=True),
        #PlotConfusionMetricAim(order_by='Recall_perclass', normalize=True),

        PlotPerclassDropdownAim(),
    ]
    callbacks.extend(plotting_callbacks)

    if patience:=cfg.training.epochs.patience:  # Early Stopping
        callbacks.append( EarlyStopping('val_loss', mode='min', patience=patience))

    if cfg.training.model_optims.freeze:  # custom show-grad model summary callback, overwrites default
        callbacks.append( ModelSummaryWithGradCallback(max_depth=2) )

    # Checkpointing
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_advanced.html
    #hashid = logger.experiment.hash if isinstance(logger,AimLogger) else logger[0].experiment.hash
    ckpt_kwargs = cfg.runtime.checkpoint_callback_config.model_dump()
    ckpt_callback = ModelCheckpoint(**ckpt_kwargs)
    callbacks.append(ckpt_callback)

    # ONNX checkpointing
    if cfg.runtime.onnx_callback_configs:
        ckpt_kwargs_onnx = ckpt_kwargs.copy()
        ckpt_kwargs_onnx['save_top_k'] = 1
        ckpt_kwargs_onnx['save_last'] = None
        for onnx_callback_config in cfg.runtime.onnx_callback_configs:
            ckpt_kwargs_onnx.update(onnx_callback_config.model_dump())
            onnx_callback = OnnxCheckpoint(**ckpt_kwargs_onnx)
            callbacks.append(onnx_callback)


    if isinstance(cfg.training.swa, SWACallbackConfig):
        swa_callback_kwargs = cfg.training_params.swa.model_dump()
        callbacks.append(StochasticWeightAveraging(**swa_callback_kwargs))

    ## Setup Trainer  ##
    epochs_cfg = cfg.training.epochs
    trainer = pl.Trainer(num_sanity_val_steps = 0,
                         deterministic = True,
                         accelerator = 'auto', devices = 'auto', num_nodes = 1,
                         max_epochs = epochs_cfg.max_epochs,
                         min_epochs = epochs_cfg.min_epochs,
                         precision = cfg.training.precision,
                         logger = logger,
                         log_every_n_steps=-1,
                         callbacks = callbacks,
                         fast_dev_run = cfg.runtime.fast_dev_run,
                         default_root_dir = '/tmp/classifier',
                         )

    # auto-tune batch-size
    if cfg.runtime.autobatch:
        autobatch = cfg.runtime.autobatch
        tuner = Tuner(trainer)
        found_batch_size = tuner.scale_batch_size(model, datamodule=datamodule,
                                                  mode=autobatch.mode, method='fit', max_trials=10, init_val=cfg.training.batch_size)
        batch_size_init, batch_size = cfg.training.batch_size, min([found_batch_size, autobatch.max_size or float('inf')])
        model.save_hyperparameters()  # TODO Confirm this works?

    # save training artifacts
    if trainer.logger.experiment.artifacts_uri:
        dataset = cfg.dataset_config.dataset
        if os.path.isfile(dataset.classlist):
            trainer.logger.experiment.log_artifact(dataset.classlist,
                name=os.path.basename(dataset.classlist), block=True)
        if os.path.isfile(dataset.vallist):
            trainer.logger.experiment.log_artifact(dataset.trainlist,
                name=os.path.basename(dataset.vallist), block=True)
        if os.path.isfile(dataset.trainlist):
            trainer.logger.experiment.log_artifact(dataset.trainlist,
                name=os.path.basename(dataset.trainlist), block=True)

    # Do Training
    trainer.fit(model, datamodule=datamodule)

    # Do Testing
    if cfg.dataset_config.dataset.testlist:
        trainer.test(model, datamodule=datamodule)

    # Submit best model
    # TODO do this as a callback
    if trainer.logger.experiment.artifacts_uri:
        model_path = trainer.checkpoint_callback.best_model_path
        trainer.logger.experiment.log_artifact(model_path, name=os.path.basename(model_path), block=True)
        onnx_files = [c.best_model_path for c in trainer.checkpoint_callbacks if isinstance(c, OnnxCheckpoint)]
        for onnx_file in onnx_files:
            trainer.logger.experiment.log_artifact(onnx_file, name=os.path.basename(onnx_file), block=True)

    # TODO DO SWA Polish
    # see https://chatgpt.com/c/68f138d0-7fec-8327-8124-f30c9abbc297
    if isinstance(cfg.training.swa, SWAPolishConfig):
        raise NotImplementedError("SWAPolish is not yet implemented")

    print('DONE!')


if __name__ == '__main__':
    ...



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
    model_name = cfg.classifier_config.backbone.model_name = check_model_name(cfg.classifier_config.backbone.model_name)

    model_base_transforms = get_model_base_transforms(model_name, cfg.dataset_config.img_norm)
    dataset = cfg.dataset_config.dataset
    datamodule = ImageListsWithLabelIndex(
        dataset.trainlist,
        dataset.vallist,
        dataset.classlist,
        base_transforms=model_base_transforms,
        training_transforms=training_transforms,
        batch_size=cfg.training_config.batch_size,
        num_workers=cfg.runtime_config.num_workers)
    cfg.classifier_config.head.num_classes = len(datamodule.classes)

    if isinstance(lc:=cfg.training_config.training_optim.loss_config, CrossEntropyLossConfig):
        if lc.weight == 'normalize':
            datamodule.setup('fit')
            class_counts = torch.bincount(torch.IntTensor(datamodule.training_dataset.targets + datamodule.validation_dataset.targets))
            class_weights = 1.0 / class_counts.float()
            lc.weight = class_weights / class_weights.mean()
        # elif os.path.isfile(loss_weights):
        #     with open(loss_weight) as f:
        #         loss_weights_tensor = torch.Tensor([float(line) for line in f.read().splitlines()])

    lightning_module = MulticlassClassifier(
        cfg.classifier_config, cfg.training_config.training_optim
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
        logger.experiment.set_artifacts_uri(cfg.artifacts_location)
    if cfg.note:
        logger.experiment.props.description = cfg.note

    return logger


def main(cfg: TrainingRunConfig):

    # print(torch.__version__, torch.version.cuda)
    # print("is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # if torch.cuda.is_available():
    #     print(torch.cuda.get_device_name(0))
    #     print("capability:", torch.cuda.get_device_capability(0))

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.runtime_config.seed)

    ## Setup Model & Data Module ##
    model, datamodule = setup_model_and_datamodule(cfg)

    ## Setup Epoch Logger(s) ##
    loggers = []
    for logger_cfg in cfg.logger_configs:
        if isinstance(logger_cfg, AimLoggerConfig):
            logger = setup_aimlogger(logger_cfg)
        loggers.append(logger)
    assert loggers is not None, 'Must have at least one logger defined'

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

    if patience:=cfg.training_config.epochs_config.patience:  # Early Stopping
        callbacks.append( EarlyStopping('val_loss', mode='min', patience=patience))

    if cfg.training_config.training_optim.freeze:  # custom show-grad model summary callback, overwrites default
        callbacks.append( ModelSummaryWithGradCallback(max_depth=2) )

    # Checkpointing
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_advanced.html
    #hashid = logger.experiment.hash if isinstance(logger,AimLogger) else logger[0].experiment.hash
    ckpt_kwargs = cfg.runtime_config.checkpoint_callback_config.model_dump()
    ckpt_callback = ModelCheckpoint(**ckpt_kwargs)
    callbacks.append(ckpt_callback)

    # ONNX checkpointing
    if cfg.runtime_config.onnx_callback_configs:
        ckpt_kwargs_onnx = ckpt_kwargs.copy()
        ckpt_kwargs_onnx['save_top_k'] = 1
        ckpt_kwargs_onnx['save_last'] = None
        for onnx_callback_config in cfg.runtime_config.onnx_callback_configs:
            ckpt_kwargs_onnx.update(onnx_callback_config.model_dump())
            onnx_callback = OnnxCheckpoint(**ckpt_kwargs_onnx)
            callbacks.append(onnx_callback)


    if isinstance(cfg.training_config.swa_config,SWACallbackConfig):
        swa_callback_kwargs = cfg.training_params.swa_config.model_dump()
        callbacks.append(StochasticWeightAveraging(**swa_callback_kwargs))

    ## Setup Trainer  ##
    epochs_cfg = cfg.training_config.epochs_config
    trainer = pl.Trainer(num_sanity_val_steps = 0,
                         deterministic = True,
                         accelerator = 'auto', devices = 'auto', num_nodes = 1,
                         max_epochs = epochs_cfg.max_epochs,
                         min_epochs = epochs_cfg.min_epochs,
                         precision = cfg.training_config.precision,
                         logger = logger,
                         log_every_n_steps=-1,
                         callbacks = callbacks,
                         fast_dev_run = cfg.runtime_config.fast_dev_run,
                         default_root_dir = '/tmp/classifier',
                        )

    # auto-tune batch-size
    if cfg.runtime_config.autobatch:
        autobatch = cfg.runtime_config.autobatch
        tuner = Tuner(trainer)
        found_batch_size = tuner.scale_batch_size(model, datamodule=datamodule,
            mode=autobatch.mode, method='fit', max_trials=10, init_val=cfg.training_config.batch_size)
        batch_size_init, batch_size = cfg.training_config.batch_size, min([found_batch_size, autobatch.max_size or float('inf')])
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
    if isinstance(cfg.training_config.swa_config,SWAPolishConfig):
        ...

    print('DONE!')


if __name__ == '__main__':
    # do hydra stuff
    from hydra_zen import make_config, builds, zen, instantiate, just, store, launch
    from hydra_zen.third_party.pydantic import pydantic_parser

    CfgNode = builds(
        TrainingRunConfig,
        logger_configs = [builds(AimLoggerConfig,
            repo = '.aim',
            artifacts_location = '.aim/artifacts',
            context_postfixes = dict(
                averaging={'macro': '_macro', 'weighted': '_weighted',
                           'micro': '_micro', 'none': '_perclass'})  # lossfunction loss_rebalanced option?,
            )],
        dataset_config = builds(DatasetRuntimeConfig,
            dataset = builds(ListfileDatasetConfig,
                # classlist = ...,
                # trainlist = ...,
                # vallist = ...
                             ),
            training_transforms = builds(TrainingAugmentationConfig,
                flip='xy')),
        classifier_config = builds(ModelConfig,
            backbone = builds(ModelBackboneConfig,
                model_name = "efficientnet_b0",
                pretrained_weights = "DEFAULT"),
            head = builds(MulticlassHeadConfig),
            weights = None),
        training_config = builds(TrainingConfig,
            epochs_config = builds(EpochConfig), # max_epochs, min_epochs, patience
            batch_size = 256,
            training_optim = builds(TrainingOptimizationConfig,
                loss_config = builds(CrossEntropyLossConfig,
                    label_smoothing = 0.1),
                optimizer_config = builds(AdamConfig),
                freeze = None)),
        runtime_config=builds(RuntimeConfig,
            onnx_callback_configs = [builds(OnnxCheckpointConfig, monitor='val_loss')]
            ),
        populate_full_signature=True,  # exposes all fields for CLI overrides
    )

    AppConfig = make_config(cfg=CfgNode)
    store(AppConfig, name="main")
    store.add_to_hydra_store()

    zen(main, instantiation_wrapper=pydantic_parser).hydra_main(
        config_name="main",
        config_path=None,
        version_base=None,
    )

    # usage
    # python src/dojo/multiclass/train.py +cfg.dataset_config.dataset.classlist=datasets/miniset_labels.list +cfg.dataset_config.dataset.trainlist=datasets/miniset_training.list +cfg.dataset_config.dataset.vallist=datasets/miniset_validation.list +cfg.runtime_config.fast_dev_run=True

    # # uncomment for debug or cli run
    # overrides = ['+cfg.dataset_config.dataset.classlist=datasets/miniset_labels.list',
    #              '+cfg.dataset_config.dataset.trainlist=datasets/miniset_training.list',
    #              '+cfg.dataset_config.dataset.vallist=datasets/miniset_validation.list']
    #
    # os.chdir('/home/sbatchelder/Projects/ifcbNN')
    #
    # job = launch(
    #     AppConfig,
    #     zen(main, instantiation_wrapper=pydantic_parser),
    #     overrides=overrides,  # <- programmatic overrides go here
    #     version_base=None,
    # )
    # print(job)



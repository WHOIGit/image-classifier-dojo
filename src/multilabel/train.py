import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# if file is called directly, must set import paths to project root
if __name__ == '__main__':
    import sys, pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))

from src.multiclass.train import argparse_init, argparse_runtime_args, parse_training_transforms, setup_aimlogger, \
    make_onnx_callbacks
from src.multiclass.models import get_model_base_transforms, check_model_name
from src.multilabel.models import MultilabelClassifier
from src.multilabel.datasets import MultilabelDataModule
from src.multiclass.callbacks import LogNormalizedLoss
from src.patches.model_summary_patch import ModelSummaryWithGradCallback


def setup_model_and_datamodule(args):
    # Training Augmentation Setup
    training_transforms = parse_training_transforms(args)

    # Model and Datamodule
    args.model = check_model_name(args.model)
    model_base_transforms = get_model_base_transforms(args.model)
    datamodule = MultilabelDataModule(args.trainlist, args.vallist, args.classlist,
        base_transforms=model_base_transforms, training_transforms=training_transforms,
        batch_size=args.batch_size, num_workers=args.num_workers)
    args.num_classes = [len(labels) for labels in datamodule.classes]

    if 'loss_weights' in args:
        if args.loss_weights == 'normalize':
            args.loss_weights_tensor = []
            datamodule.setup('fit')
            targets_as_records = datamodule.training_dataset.targets + datamodule.validation_dataset.targets
            targets_as_columns = zip(*targets_as_records)
            for i, targets_for_given_label in targets_as_columns:
                class_counts = torch.bincount(torch.IntTensor(targets_for_given_label))
                class_weights = 1.0 / class_counts.float()
                args.loss_weights_tensor.append( class_weights / class_weights.sum() )
        elif os.path.isfile(args.loss_weights):
            raise NotImplementedError
            with open(args.loss_weight) as f:
                args.loss_weights_tensor = torch.Tensor([float(line) for line in f.read().splitlines()])

    loss_kwargs = []
    for i in range(len(args.num_classes)):
        kwargs = {}
        if args.loss_function == 'CrossEntropyLoss':
             kwargs['label_smoothing'] = args.loss_smoothing
             if args.loss_weights_tensor:
                 kwargs['weights'] = args.loss_weights_tensor
        elif args.loss_function == 'FocalLoss':
            kwargs['gamma'] = args.loss_gamma
            if args.loss_weights_tensor is not None:
                kwargs['alpha'] = args.loss_weights_tensor
        loss_kwargs.append(kwargs)

    optimizer_kwargs = dict(lr = args.lr)

    lightning_module = MultilabelClassifier(
                model_name = args.model,
                num_classes = args.num_classes,
                model_weights = args.weights,
                model_freeze = args.freeze,
                loss_function = args.loss_function,
                loss_kwargs = loss_kwargs,
                optimizer = args.optimizer,
                optimizer_kwargs = optimizer_kwargs,
    )
    return lightning_module, datamodule




def main(args):
    torch.set_float32_matmul_precision('medium')

    ## Setup Model & Data Module ##
    model, datamodule = setup_model_and_datamodule(args)
#    datamodule.setup('fit')


    labels = datamodule.labels
    ## Setup Epoch Logger ##
    contexts = dict(heads = {label:f'_{label}' for label in labels},
                    #averaging = {'macro': '_macro', 'micro': '_micro', 'weighted': '_weighted',
                    #             'none': '_perclass'},  # f1, precision, recall
                    normalized = {'no': '_summed', 'yes': '_normalized'},  # confusion matrix, loss?
                    rebalanced = {'yes': '_rebalanced', 'no': '_unbalanced'})  # lossfunction loss_rebalanced option?
    # val_/train_ already handled by default
    logger = setup_aimlogger(args, context_postfixes=contexts)
    assert logger is not None, 'Aim logger is None. Did you forget to set --repo, --env, or AIM_REPO env variable?'

    ## Setup Callbacks ##
    callbacks=[]

    validation_results_callbacks = [
        LogNormalizedLoss(),
    ]

    # TODO plotting
    plotting_callbacks = []
    callbacks.extend(plotting_callbacks)

    if args.epoch_stop:  # Early Stopping
        callbacks.append( EarlyStopping('val_loss', mode='min', patience=args.epoch_stop) )

    if args.freeze:  # custom show-grad model summary callback, overwrites default
        callbacks.append( ModelSummaryWithGradCallback(max_depth=2) )

    # Checkpointing
    print(args.checkpoints_path, args.experiment, args.run)
    chkpt_path = os.path.join(args.checkpoints_path, args.experiment, args.run)
    chkpt_kwargs = dict(
        dirpath=chkpt_path,
        filename='loss-{val_normloss:3.3f}_ep-{epoch:03.0f}',
        monitor='val_loss', mode='min',
        save_last='link', save_top_k=3,
        auto_insert_metric_name=False,
    )
    ckpt_callback = ModelCheckpoint(**chkpt_kwargs)
    callbacks.append(ckpt_callback)

    if args.onnx:
        chkpt_kwargs['save_top_k'] = 1
        chkpt_kwargs['save_last'] = None
        onnx_callbacks = make_onnx_callbacks(args.onnx, chkpt_kwargs)
        callbacks.extend(onnx_callbacks)

    ## Setup Trainer  ##
    trainer = pl.Trainer(num_sanity_val_steps=0,
                         deterministic=True,
                         accelerator='auto', devices='auto', num_nodes=1,
                         max_epochs=args.epoch_max, min_epochs=args.epoch_min,
                         precision=args.precision,
                         logger=logger,
                         log_every_n_steps=-1,
                         callbacks=callbacks,
                         fast_dev_run=args.fast_dev_run,
                         default_root_dir='/tmp/classifier',
                        )

    # auto-tune batch-size
    if args.autobatch:
        tuner = Tuner(trainer)
        found_batch_size = tuner.scale_batch_size(model, datamodule=datamodule,
            mode=args.autobatch, method='fit', max_trials=10, init_val=args.batch_size)
        args.batch_size_init, args.batch_size = args.batch_size, min([found_batch_size, args.autobatch_max or float('inf')])
        model.save_hyperparameters()

    # Do Training
    trainer.fit(model, datamodule=datamodule)

    print('DONE!')


if __name__ == '__main__':
    parser = argparse_init()
    args = parser.parse_args()
    argparse_runtime_args(args)
    main(args)



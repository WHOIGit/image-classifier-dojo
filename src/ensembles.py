import os
import random
import argparse

import coolname
import torch
from torchvision.models import Inception3, GoogLeNet

#https://ensemble-pytorch.readthedocs.io/en/latest/
import torchensemble

ENSEMBLE_MAPPING = dict(
    Voting = torchensemble.VotingClassifier,
    Bagging = torchensemble.BaggingClassifier,
    Boosting = torchensemble.GradientBoostingClassifier,
    Fusion = torchensemble.FusionClassifier,
    Snapshot = torchensemble.SnapshotEnsembleClassifier,
    FastGeometric = torchensemble.FastGeometricClassifier,
    Adversarial = torchensemble.AdversarialTrainingClassifier)


if __name__ == '__main__':
    import sys, pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))

from src.multiclass.models import check_model_name
from src.train import setup_model_and_datamodule, args_subsetter_factory


def argparse_init(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Train, Run, and perform other tasks related to ifcb and general image classification!')

    # DATASET #
    dataset = parser.add_argument_group(title='Dataset', description=None)
    dataset.add_argument('--classlist', required=True, help='A text file, each line is a class label (the label order is significant), or a json dict')
    dataset.add_argument('--trainlist', required=True, help='A text file, one sample per line, with class-index and image path per line, or a json dict')
    dataset.add_argument('--vallist', required=True, help='Like trainlist, but for validation metrics and early-stopping/overfit-prevention')

    # TRAINING TRANSFORMS #
    dataset.add_argument('--flip', choices=['x', 'y', 'xy'],
                      help='Training images have 50%% chance of being flipped along the designated axis: (x) vertically, (y) horizontally, (xy) either/both. May optionally specify "+V" to include Validation dataset')

    # ENSEMBLE
    ensemble = parser.add_argument_group(title='Model Parameters')
    ensemble.add_argument('--ensemble', metavar='METHOD', choices=ENSEMBLE_MAPPING.keys(), help='Ensemble Classifier Methods from torchensemble')
    ensemble.add_argument('--num-ensembles', type=int, default=4, help='number of ensembles')
    ensemble.add_argument('--voting-strategy', default='soft', choices=('soft','hard'), help='Only for "Voting" and "Snapshot" METHODs')
    #ensemble.add_argument('--voting-strategy', default='soft', choices=('soft', 'hard'), help='Only for "Voting" and "Snapshot" METHODs')

    # HYPER PARAMETERS #
    model = parser.add_argument_group(title='Model Parameters')
    model.add_argument('--model', help='Model Class/Module Name or torch model checkpoint file', required=True)  # TODO checkopint file, also check loading from s3
    model.add_argument('--weights', default='DEFAULT', help='''Specify a model's weights. Either "DEFAULT", some specific identifier, or "None" for no-pretrained-weights''')
    model.add_argument('--seed', type=int, help='Set a specific seed for deterministic output')
    model.add_argument('--batch', dest='batch_size', metavar='SIZE', default=256, type=int, help='Number of images per batch. Defaults is 256')
    model.add_argument('--num-classes', type=int, help=argparse.SUPPRESS)
    model.add_argument('--freeze', metavar='LAYERS', help='Freezes a models leading feature layers. '
        'Positive int freezes the first N layers/features/blocks. A negative int like "-1" freezes all but the last feature/layer/block. '
        'A positive float like "0.8" freezes the leading 80%% of features/layers/blocks. fc or final classifier layers are never frozen.')

    model.add_argument('--loss-function', metavar='MODULE.CLASS', default='CrossEntropyLoss', help=argparse.SUPPRESS)
    model.add_argument('--loss-weights', default=False, help='If "normalize", rare class instances will be boosted. Else a filepath to a perclass list of loss weights. Default is None')
    model.add_argument('--loss-weights-tensor', help=argparse.SUPPRESS)
    model.add_argument('--loss-smoothing', nargs='?', default=0.0, const=0.1, type=float, help='Label Smoothing Regularization arg. Range is 0-1. Default is 0. Const is 0.1')

    model.add_argument('--epoch-max', metavar='MAX', default=100, type=int, help='Maximum number of training epochs. Default is 100')

    model.add_argument('--early-stopping-rounds', metavar='EPOCHS', default=10, type=int, help='Early Stopping for Boosting. Default is 10')

    parser.add_argument('--workers', dest='num_workers', metavar='N', type=int, help='Total number of dataloader worker threads. If set, overrides --workers-per-gpu')
    parser.add_argument('--workers_per_gpu', metavar='N', default=4, type=int, help='Number of data-loading threads per GPU. 4 per GPU is typical. Default is 4')
    parser.add_argument('--gpus', nargs='+', type=int, help=argparse.SUPPRESS) # CUDA_VISIBLE_DEVICES
    parser.add_argument('--run', help='The name of this run. A run name is automatically generated by default')
    parser.add_argument('--experiment', default='ENSEMBLE', help='The broader category/grouping this RUN belongs to')
    parser.add_argument('--outdir', default='./experiments/{EXPERIMENT}__{METHOD}__{MODEL}/{RUN}')

    return parser


def argparse_runtime_args(args):
    # Record GPUs
    if not args.gpus:
        args.gpus = [int(gpu) for gpu in os.environ.get('CUDA_VISIBLE_DEVICES','UNSET').split(',') if gpu!='UNSET']
    if not args.num_workers:
        args.num_workers = len(args.gpus)*args.workers_per_gpu

    if not args.run:
        args.run = coolname.generate_slug(3)
    print(f'RUN: {args.run}')

    # Set Seed. If args.seed is 0 ie None, a random seed value is used and stored
    if args.seed is None:
        args.seed = random.randint(0,2**32-1)
    args.seed = torch.manual_seed(args.seed)

    # format Freeze to int or float
    if args.freeze:
        args.freeze = float(args.freeze) if '.' in args.freeze else int(args.freeze)

    args.model = check_model_name(args.model)
    args.outdir = args.outdir.format(EXPERIMENT=args.experiment,
            METHOD=args.ensemble, MODEL=args.model, RUN=args.run)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'OUTDIR: {args.outdir}')


from typing import Union, Optional
from torch import Tensor
from torch.nn import CrossEntropyLoss, functional as F
from collections import namedtuple
class CleverCrossEntropyLoss(CrossEntropyLoss):
    INCEPTION_AUXLOSS_WEIGHT = 0.4
    GOOGLENET_AUXLOSS_WEIGHT = 0.3

    def forward(self, input: Union[Tensor,namedtuple], target: Tensor) -> Tensor:
        if isinstance(input, tuple) and hasattr(input, 'aux_logits'):
            loss = self.loss(input.logits, target)
            loss_aux = self.loss(input.aux_logits, target)
            batch_loss = loss + self.INCEPTION_AUXLOSS_WEIGHT * loss_aux
        elif isinstance(input, tuple) and hasattr(input, 'aux_logits2'):
            loss = self.loss(input.logits, target)
            loss_aux2 = self.loss(input.aux_logits2, target)
            loss_aux1 = self.loss(input.aux_logits1, target)
            batch_loss = loss + self.GOOGLENET_AUXLOSS_WEIGHT * loss_aux1 + self.GOOGLENET_AUXLOSS_WEIGHT * loss_aux2
        else:
            batch_loss = self.loss(input, target)
        return batch_loss

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

from collections import namedtuple
import warnings
def patch_iv3(model):
    InceptionOutputsPatch = namedtuple("InceptionOutputs", ["logits", "aux_logits", 'data', 'softmax', 'size'])
    InceptionOutputsPatch.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputsPatch:
        if self.training and self.aux_logits:
            return InceptionOutputsPatch(x, aux, x, x.softmax, x.size)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputsPatch:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputsPatch(x, aux, x, x.softmax, x.size)
        else:
            return self.eager_outputs(x, aux)

    model.eager_outputs = eager_outputs.__get__(model, Inception3)
    model.forward = forward.__get__(model, Inception3)
    return model


def patch_goog(model):
    GoogLeNetOutputsPatch = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1", 'data', 'softmax', 'size'])
    GoogLeNetOutputsPatch.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputsPatch:
        if self.training and self.aux_logits:
            return GoogLeNetOutputsPatch(x, aux2, aux1 , x, x.softmax, x.size)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> GoogLeNetOutputsPatch:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputsPatch(x, aux2, aux1, x, x.softmax, x.size)
        else:
            return self.eager_outputs(x, aux2, aux1)

    model.eager_outputs = eager_outputs.__get__(model, GoogLeNet)
    model.forward = forward.__get__(model, GoogLeNet)
    return model


def main(args):
    torch.set_float32_matmul_precision('medium')

    try:
        logger = torchensemble.utils.logging.set_logger(log_file=args.run,
            log_console_level="info", log_file_level='info', use_tb_logger=True)
    except ImportError:
        logger = torchensemble.utils.logging.set_logger(log_file=args.run,
            log_console_level="info", log_file_level='info', use_tb_logger=False)
        tb_logdir = logger.root.handlers[0].baseFilename.replace('.log','_tb_logger')
        logger.info(f'tensorboard not installed, removing tb_logdir: {tb_logdir}')
        os.rmdir(tb_logdir)

    # Setup Model & Data Module

    lightning_module, datamodule = setup_model_and_datamodule(args)
    basemodel_args = {k: getattr(args, k) for k in 'model batch_size num_classes freeze weights'.split()}
    logger.info(f'base_model: {basemodel_args}')
    base_model = lightning_module.model
    if isinstance(base_model, Inception3):
        base_model = patch_iv3(base_model)
        #base_model.aux_logits = False
    if isinstance(base_model, GoogLeNet):
        base_model = patch_goog(base_model)
        #base_model.aux_logits = False
    datamodule.setup('fit', without_source=True)

    EnsembleMethod = ENSEMBLE_MAPPING[args.ensemble]
    ensemble_args = dict(n_estimators=args.num_ensembles)
    fit_args = dict()
    match EnsembleMethod:
        case torchensemble.VotingClassifier:
            ensemble_args['voting_strategy'] = args.voting_strategy
        case torchensemble.SnapshotEnsembleClassifier:
            ensemble_args['voting_strategy'] = args.voting_strategy
            fit_args['lr_clip'] = None
        case torchensemble.AdversarialTrainingClassifier:
            fit_args['epsilon'] = 0.5
        case torchensemble.GradientBoostingClassifier:
            base_model.aux_logits = False
            fit_args['use_reduction_sum'] = True
            fit_args['early_stopping_rounds'] = args.early_stopping_rounds
        case torchensemble.FastGeometricClassifier:
            fit_args['cycle'] = 4
            fit_args['lr_1'] = 5e-2
            fit_args['lr_2'] = 1e-4
        case torchensemble.FusionClassifier:
            base_model.aux_logits = False

    logger.info(f'ensemble_args: {ensemble_args}')
    ensemble = EnsembleMethod(estimator=base_model, **ensemble_args)

    # Loss Function
    if args.loss_function == 'CrossEntropyLoss':
        Criterion = torch.nn.CrossEntropyLoss
        if isinstance(base_model,(Inception3,GoogLeNet)):
            Criterion = CleverCrossEntropyLoss
    else:
        raise NotImplemented(f'--loss-function {args.loss_function}')
    criterion_args = dict(weight=args.loss_weights_tensor, label_smoothing=args.loss_smoothing)
    logger.info(f'criterion_args: {criterion_args}')
    criterion = Criterion(**criterion_args)
    ensemble.set_criterion(criterion)

    # Gradient Descent Optimizer
    optimizer_args = dict(optimizer_name="Adam", lr=0.001, weight_decay=0)
    logger.info(f'optimizer_args: {optimizer_args}')
    ensemble.set_optimizer(**optimizer_args)

    # Set the learning rate scheduler
    #ensemble.set_scheduler("CosineAnnealingLR", T_max=...)

    # FIT THE MODEL
    fit_args['epochs'] = args.epoch_max
    fit_args['save_dir'] = args.outdir
    #fit_args['log_interval'] = 100
    logger.info(f'fit_args: {fit_args}')
    ensemble.fit(train_loader = datamodule.train_dataloader(),
                 test_loader = datamodule.val_dataloader(),
                 **fit_args)


# if file is called directly, must set import paths to project root
if __name__ == '__main__':
    parser = argparse_init()
    args = parser.parse_args()
    argparse_runtime_args(args)
    main(args)


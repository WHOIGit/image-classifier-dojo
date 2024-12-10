import os
import argparse

import torch
from torchvision.models import Inception3, GoogLeNet
import lightning.pytorch as pl

#https://ensemble-pytorch.readthedocs.io/en/latest/
from torchensemble import VotingClassifier, BaggingClassifier, GradientBoostingClassifier, FusionClassifier, SnapshotEnsembleClassifier, FastGeometricClassifier, AdversarialTrainingClassifier

if __name__ == '__main__':
    import sys, pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))

from src.multiclass.models import check_model_name, MulticlassClassifier
import src.train
from src.multiclass.callbacks import BarPlotMetricAim, PlotPerclassDropdownAim, PlotConfusionMetricAim

from src.homogenous_ensembles.patches import patch_iv3, patch_goog, CleverCrossEntropyLoss, patch_save, \
    patch_boosting_earlystopping, disable_argument

ENSEMBLE_CHOICES = (
    'Voting',
    'Bagging',
    'Boosting',
    'Fusion',
    'Snapshot',
    'FastGeometric',
    'Adversarial')


def argparse_init(parser):
    parser.description = 'Train an image classifier ensemble of homogenous ensembles!'
    # ENSEMBLE
    ensemble = parser.add_argument_group(title='Ensemble Parameters')
    ensemble.add_argument('--ensemble', metavar='METHOD', required=True, choices=ENSEMBLE_CHOICES, help='Ensemble Classifier Methods from torchensemble')
    ensemble.add_argument('--num-ensembles', metavar='N', type=int, default=4, help='number of ensembles')
    ensemble.add_argument('--epochs', type=int, default=20, help='number or epochs to run. Default is 20')
    ensemble.add_argument('--outdir', default='./experiments/{EXPERIMENT}__{METHOD}__{MODEL}/{RUN}', help='Local ensemble-saving directory')

    # Voting
    voting = parser.add_argument_group('ENSEMBLE method: Voting (and Snapshot).')
    voting_help = 'Voting and Snapshot only. hard voting is majority voting, soft voting is score averaging. Default is "soft"'
    voting.add_argument('--voting-strategy', default='soft', choices=('soft','hard'), help=voting_help)

    # Snapshot
    snapshot = parser.add_argument_group('ENSEMBLE method: Snapshot.')
    snapshot.add_argument('--voting–strategy', dest='voting_strategy', default='FAKE', choices=('soft','hard'), help=voting_help)
    snapshot.add_argument('--lr-clip', metavar=('LOWER','UPPER'), nargs=2, type=float, help='Snapshot only. Learning Rate Lower and Upper Bounds. Default is None')

    # Bagging
    bagging = parser.add_argument_group('ENSEMBLE method: Bagging.')
    bagging.add_argument('---–', metavar='', default='FAKE2', dest='voting_strategy', help='No adl. args.')

    # Fusion
    fusion = parser.add_argument_group('ENSEMBLE method: Fusion.')
    fusion.add_argument('--–-', metavar='', default='FAKE', dest='voting_strategy', help='No adl. args.')

    # Adversarial
    adversarial = parser.add_argument_group('ENSEMBLE method: Adversarial.')
    adversarial.add_argument('--epsilon', metavar='E', default=0.5, type=float, help='Adversarial only. Default is 0.5')

    # Boosting
    boosting = parser.add_argument_group('ENSEMBLE method: Boosting.')
    boosting.add_argument('--early-stopping-rounds', metavar='EPOCHS', default=10, type=int, help='Boosting only Default is 10')
    boosting.add_argument('--use_reduction_sum', action='store_true', default=False, help='Boosting only. Default is False')

    # FastGeometric
    geometric = parser.add_argument_group('ENSEMBLE method: FastGeometric.')
    geometric.add_argument('--cycle', type=int, default=4, help='FastGeometric only. Default is 4')
    geometric.add_argument('--lr-pair', metavar=('lr1','lr2'), nargs=2, default=(5e-2,1e-4), type=float, help='FastGeometric only. Default is  5e-2 1e-4')

    return parser


def argparse_runtime_args(args):
    args.model = check_model_name(args.model)
    args.outdir = args.outdir.format(EXPERIMENT=args.experiment,
            METHOD=args.ensemble, MODEL=args.model, RUN=args.run)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'OUTDIR: {args.outdir}')


def argparse_disable_arguments(parser):
    parser.disable_argument = disable_argument.__get__(parser, argparse.ArgumentParser)
    arguments = '--epoch-max --epoch-min --epoch-stop ' \
                '--autobatch --autobatch-max --fast-dev-run ' \
                '--swa --swa-lr --swa-annealing'.split()
    for arg in arguments:
        parser.disable_argument(arg)
    return parser


def main(args):
    torch.set_float32_matmul_precision('medium')

    # setting up torchensemble "logger"
    import torchensemble  # unclear why this has to be re-imported here for utils.logging to work
    cwd = os.getcwd()
    os.chdir(args.outdir) # logs always get written to ./logs/ so we temporarily chdir
    logger = torchensemble.utils.logging.set_logger(log_file=args.run,
        log_console_level="info", log_file_level='info', use_tb_logger=False)
    os.chdir(cwd)

    # Setup Model & Data Module
    lightning_module, datamodule = src.train.setup_model_and_datamodule(args)
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

    # Setup Ensemble instantiation and fit args
    ensemble_args = dict(n_estimators=args.num_ensembles, cuda=torch.cuda.is_available())
    fit_args = dict(epochs=args.epochs, save_dir=args.outdir, log_interval=100)
    match args.ensemble:
        case 'Voting':
            EnsembleMethod = VotingClassifier
            ensemble_args['voting_strategy'] = args.voting_strategy
        case 'Snapshot':
            EnsembleMethod = SnapshotEnsembleClassifier
            ensemble_args['voting_strategy'] = args.voting_strategy
            fit_args['lr_clip'] = args.lr_clip
        case 'Adversarial':
            EnsembleMethod = AdversarialTrainingClassifier
            fit_args['epsilon'] = args.epsilon
        case 'Boosting':
            EnsembleMethod = GradientBoostingClassifier
            base_model.aux_logits = False
            fit_args['use_reduction_sum'] = args.use_reduction_sum
            fit_args['early_stopping_rounds'] = args.early_stopping_rounds
        case 'FastGeometric':
            EnsembleMethod = FastGeometricClassifier
            fit_args['cycle'] = args.cycle
            fit_args['lr_1'] = args.lr_pair[0]
            fit_args['lr_2'] = args.lr_pair[1]
        case 'Fusion':
            EnsembleMethod = FusionClassifier
            base_model.aux_logits = False
        case 'Bagging':
            EnsembleMethod = BaggingClassifier
        case _:
            raise ValueError

    # instantiating ensemble
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

    ## LIGHTNING TRAINER SAVE PATCH ##
    # aim logger
    contexts = dict(averaging={'macro': '_macro', 'micro': '_micro', 'weighted': '_weighted',
                               'none': '_perclass'},  # f1, precision, recall
                   normalized={'no': '_summed', 'yes': '_normalized'})  # confusion matrix
    aim_logger = src.train.setup_aimlogger(args, context_postfixes=contexts)

    # Plotting Callbacks
    callbacks = [
        BarPlotMetricAim('f1_perclass', order_by='f1_perclass', best_only=True),
        PlotConfusionMetricAim(order_by='classes', normalize=True, best_only=True),
        PlotConfusionMetricAim(order_by='f1_perclass', normalize=True, best_only=True),
        PlotPerclassDropdownAim(best_only=True),
        ]

    # Lightning Trainer and Multiclass module for validation metrics and plots and experiment tracking
    trainer = pl.Trainer(num_sanity_val_steps=0,
                         accelerator='auto', devices='auto', num_nodes=1,
                         logger=aim_logger,
                         log_every_n_steps=-1,
                         callbacks=callbacks,
    )
    lightning_module_with_ensemble = MulticlassClassifier(args, model=ensemble)

    # save training artifacts
    if args.artifacts_location or ( 'AIM_ARTIFACTS_URI' in os.environ and os.environ['AIM_ARTIFACTS_URI'] ):
        if os.path.isfile(args.classlist):
            trainer.logger.experiment.log_artifact(args.classlist, name=os.path.basename(args.classlist))
        if os.path.isfile(args.vallist):
            trainer.logger.experiment.log_artifact(args.trainlist, name=os.path.basename(args.vallist))
        if os.path.isfile(args.trainlist):
            trainer.logger.experiment.log_artifact(args.trainlist, name=os.path.basename(args.trainlist))

    # Patching torchensemble.utils.io.save such that a lightning validation epoch + callbacks
    # are called whenever the ensemble model decides to save a ckpt (at the end of each epoch)
    # This is a hack and very cursed.
    import torchensemble.utils.io
    torchensemble.utils.io.save = patch_save(trainer, lightning_module_with_ensemble, datamodule)
    if args.ensemble == 'Boosting':
        # boosting only saves its ckpt at the very end of the run for some reason
        # so we modify the following function to instead also save on every epoch
        ensemble._handle_early_stopping = patch_boosting_earlystopping().__get__(ensemble, GradientBoostingClassifier)

    # FIT THE MODEL
    # using torchensemble's classic-pytorch-style "fit" method
    logger.info(f'fit_args: {fit_args}')
    ensemble.fit(train_loader = datamodule.train_dataloader(),
                 **fit_args)


if __name__ == '__main__':
    parser = src.train.argparse_init()
    parser = argparse_init(parser)
    argparse_disable_arguments(parser)
    args = parser.parse_args()
    src.train.argparse_runtime_args(args)
    argparse_runtime_args(args)
    main(args)


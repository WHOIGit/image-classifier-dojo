import os
import types
from typing import Union, Optional
from collections import namedtuple
import warnings
import argparse

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, functional as F

import lightning.pytorch as pl
from aim.pytorch_lightning import AimLogger

from torchvision.models import Inception3, GoogLeNet  # models with Aux Logits
from torchensemble import GradientBoostingClassifier


def disable_argument(parser: argparse.ArgumentParser, arg: str, error_msg: str = 'Has been disabled!') -> None:
    """Disable an argument from a parser.

    Args:
        :param parser: the ArgumentParser instance from which to remove an argument
        :param arg: Argument to be removed. Eg: "--foo"
        :param error_msg: What to display on-error. Default is "Has been disabled!"
    """
    def raise_disabled_error(action):
        """Raise an argument error."""
        def raise_disabled_error_wrapper(*args) -> str:
            """Raise an exception."""
            raise argparse.ArgumentError(action, error_msg)
        return raise_disabled_error_wrapper

    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            action.type = raise_disabled_error(action)
            action.help = argparse.SUPPRESS
            break


# CleverCrossEntropyLoss knows how to handle aux_logits
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


# returns a patched save function that can do lightning_module validation and callbacks
# to do this is must follow the initialization steps of lightning.trainer.fit(...)
def patch_save(trainer: pl.Trainer, lightning_module, datamodule):
    from lightning.pytorch.trainer.trainer import TrainerFn, TrainerStatus, call, _verify_loop_configurations, _log_hyperparams
    from lightning.pytorch.loops.evaluation_loop import _set_sampler_epoch
    from torchensemble.utils.io import save as f
    torchensemble_save_deepcopy = types.FunctionType(
        f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    model = lightning_module
    model.trainer = trainer
    #model.metrics_to('cuda' if torch.cuda.is_available() else 'cpu')
    #model.trainer.fit()
    trainer.state.fn = TrainerFn.FITTING
    trainer.state.status = TrainerStatus.RUNNING
    trainer.training = True
    from copy import deepcopy
    trainer._data_connector.attach_data(model, datamodule=deepcopy(datamodule))
    trainer.fit_loop.min_epochs = 0
    trainer.fit_loop.max_epochs = -1
    trainer.strategy.connect(model)
    trainer._callback_connector._attach_model_callbacks()
    trainer._callback_connector._attach_model_logging_functions()
    _verify_loop_configurations(trainer)
    # SET UP THE TRAINER
    trainer.strategy.setup_environment()
    trainer._data_connector.prepare_data()
    call._call_setup_hook(trainer)
    call._call_configure_model(trainer)
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    trainer.strategy.setup(trainer)
    #if self.state.fn == TrainerFn.FITTING:
    #    call._call_callback_hooks(self, "on_fit_start")
    #    call._call_lightning_module_hook(self, "on_fit_start")
    _log_hyperparams(trainer)
    trainer._checkpoint_connector.restore_training_state()
    trainer._checkpoint_connector.resume_end()
    trainer._signal_connector.register_signal_handlers()
    # RUN THE TRAINER
    #results = self._run_stage()
    #self.fit_loop.run()
    trainer.fit_loop.setup_data()
    trainer.fit_loop.reset()
    trainer.fit_loop.on_run_start()
    ## self.on_advance_start() ##
    # update the epoch value for all samplers
    assert trainer.fit_loop._combined_loader is not None
    for i, dl in enumerate(trainer.fit_loop._combined_loader.flattened):
        _set_sampler_epoch(dl, trainer.fit_loop.epoch_progress.current.processed)
    trainer.fit_loop.epoch_progress.increment_ready()
    # call._call_callback_hooks(trainer, "on_train_epoch_start")
    # call._call_lightning_module_hook(trainer, "on_train_epoch_start")
    trainer.fit_loop.epoch_progress.increment_started()
    ## self.advance() ##
    combined_loader = trainer.fit_loop._combined_loader
    ### self.epoch_loop.run(self._data_fetcher)
    trainer.fit_loop.epoch_loop.reset()
    trainer.fit_loop.epoch_loop.on_run_start(trainer.fit_loop._data_fetcher)
    def run_single_lightning_validation_epoch():
        #self.epoch_loop.advance(self._data_fetcher)        # training WOULD happen here.
        trainer.fit_loop.epoch_loop.on_advance_end(trainer.fit_loop._data_fetcher)  # VALIDATION HAPPENS HERE!
        trainer.fit_loop._restarting = False
        ## self.on_advance_end() ##
        trainer._logger_connector.epoch_end_reached()
        trainer.fit_loop.epoch_progress.increment_processed()
        call._call_callback_hooks(trainer, "on_train_epoch_end", monitoring_callbacks=False)
        call._call_lightning_module_hook(trainer, "on_train_epoch_end")
        call._call_callback_hooks(trainer, "on_train_epoch_end", monitoring_callbacks=True)
        trainer._logger_connector.on_epoch_end()
        trainer.fit_loop.epoch_loop._batches_that_stepped -= 1
        trainer._logger_connector.update_train_epoch_metrics()
        trainer.fit_loop.epoch_loop._batches_that_stepped += 1
        trainer.fit_loop.epoch_progress.increment_completed()
        trainer.fit_loop._restarting = False
    def save_bestepoch_ckpt(model, save_dir, logger):
        # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
        filename = "{}_{}_{}_ckpt.pth".format(
            type(model).__name__,
            model.base_estimator_.__class__.__name__,
            model.n_estimators,
        )
        fullpath = os.path.join(save_dir,filename)
        if lightning_module.best_epoch == lightning_module.current_epoch-1:
            if isinstance(lightning_module.logger,AimLogger):
                if 'AIM_ARTIFACTS_URI' in os.environ and os.environ['AIM_ARTIFACTS_URI']:
                    logger.info(f'Saving model to {lightning_module.logger.experiment.artifacts_uri}/{filename}')
                    lightning_module.logger.experiment.log_artifact(fullpath, filename)
    def new_save(model, save_dir, logger):
        torchensemble_save_deepcopy(model, save_dir, logger)
        run_single_lightning_validation_epoch()
        save_bestepoch_ckpt(model, save_dir, logger)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        return
    return new_save


def patch_boosting_earlystopping(save_dir='/tmp/torchensemble_boosting'):
    # must be called AFTER patch_save()
    import types
    from torchensemble.utils import io
    f = GradientBoostingClassifier._handle_early_stopping
    _handle_early_stopping__deepcopy = types.FunctionType(
        f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    def new_earlystopping(self, test_loader, est_idx):
        flag, acc = _handle_early_stopping__deepcopy(self, test_loader, est_idx)
        io.save(self, save_dir, self.logger)
        return flag, acc
    return new_earlystopping
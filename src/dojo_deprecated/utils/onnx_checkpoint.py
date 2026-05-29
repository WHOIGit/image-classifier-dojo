import copy
from typing import Sequence, Optional, Literal

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from typing_extensions import override

from dojo.multiclass.models import get_model_resize
from dojo.schemas import OnnxCheckpointConfig, ModelCheckpointConfig


class OnnxCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".onnx"

    def __init__(self, dirpath, filename, monitor,
                    half: bool,
                    batch_size: Optional[int],
                    device: Literal["cpu", "cuda"] = None,
                    opset: Optional[int] = None,
                    input_names: Sequence[str] = ('input',),
                    output_names: Sequence[str] = ('output',),
                 **kwargs):

        super().__init__(dirpath, filename, monitor, **kwargs)

        self.batch_size = batch_size
        self.half = half
        self.device = device
        self.export_args = dict(input_names = input_names,
                                output_names = output_names)
        if batch_size is None:
            # if dynamo:
            #     if 'dynamic_shapes' not in export_args:
            #         export_args['dynamic_shapes'] = {'x': {0: 'batch_size'}}
            self.export_args['dynamic_axes'] = {}
            for input_name in self.export_args['input_names']:
                self.export_args['dynamic_axes'][input_name] = {0: 'batch_size'}
            for output_name in self.export_args['output_names']:
                self.export_args['dynamic_axes'][output_name] = {0: 'batch_size'}

    @property
    @override
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
            batch_size = self.batch_size,
            half = self.half,
            export_args = self.export_args,
            device = self.device
        )

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        model = copy.deepcopy(trainer.lightning_module.model)
        model.eval()

        if self.device is None:
            self.device = next(model.parameters()).device
        model.to(self.device)

        dummy_batch_size = self.batch_size or 10
        input_size = get_model_resize(trainer.lightning_module.hparams['model_name'])
        dummy_input = torch.randn(dummy_batch_size, 3, input_size, input_size, device=self.device)
        if self.half:
            model.half()
            dummy_input = dummy_input.half()

        with torch.no_grad():
            torch.onnx.export(model, dummy_input, filepath, **self.export_args)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

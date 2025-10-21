import copy

import torch
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from typing_extensions import override

from dojo.multiclass.models import get_model_resize


class OnnxCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".onnx"

    def __init__(self, *args, batch_size:int=None, half:bool=False, device:str=None, export_args:dict=None, dynamo=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.half = half
        self.export_args = export_args if export_args else {}
        if batch_size is None:
            if dynamo or ('dynamo' in export_args and export_args['dynamo']):
                if 'dynamic_shapes' not in export_args:
                    export_args['dynamic_shapes'] = {'x': {0: 'batch_size'}}
            else:
                if 'dynamic_axes' not in self.export_args:
                    self.export_args['dynamic_axes'] = {}
                if 'input_names' not in self.export_args:
                    self.export_args['input_names'] = ['input']
                if 'output_names' not in self.export_args:
                    self.export_args['output_names'] = ['output']
                for input_name in self.export_args['input_names']:
                    self.export_args['dynamic_axes'][input_name] = {0: 'batch_size'}
                for output_name in self.export_args['output_names']:
                    self.export_args['dynamic_axes'][output_name] = {0: 'batch_size'}

        self.device = device

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

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
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

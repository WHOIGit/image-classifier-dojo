from typing import List, Tuple

from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary, get_human_readable_count
from lightning.pytorch.callbacks import ModelSummary as ModelSummaryCallback
import lightning.pytorch as pl

class ModelSummaryWithGrad(ModelSummary):
    @property
    def grads_required(self) -> List[str]:
        def grad_true_false_mixed_blank(name, layer):
            if not name.startswith('model.'): return ''
            grads_required = [p.requires_grad for p in layer._module.parameters()]
            if all(grads_required): return True
            elif any(grads_required): return 'Mixed'
            else: return False  # all grads_required are false
        return [grad_true_false_mixed_blank(k, v) for k, v in self._layer_summary.items()]

    def _get_summary_data(self) -> List[Tuple[str, List[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size, requires_grad

        """
        arrays = [
            (" ", list(map(str, range(len(self._layer_summary))))),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", list(map(get_human_readable_count, self.param_nums))),
            ("Mode", ["train" if mode else "eval" for mode in self.training_modes]),
            ('requires_grad', self.grads_required),
        ]
        if self._model.example_input_array is not None:
            arrays.append(("In sizes", [str(x) for x in self.in_sizes]))
            arrays.append(("Out sizes", [str(x) for x in self.out_sizes]))

        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)

        return arrays

class ModelSummaryWithGradCallback(ModelSummaryCallback):
    def _summary(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> ModelSummaryWithGrad:
        return ModelSummaryWithGrad(pl_module, max_depth=self._max_depth)
import os
from typing import Union

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision as tv
from torchvision.models import AlexNet, DenseNet,  ResNet, SqueezeNet, VGG, \
    ConvNeXt, EfficientNet, MNASNet, MobileNetV2, MobileNetV3, RegNet, ShuffleNetV2
from torchvision.models import Inception3, InceptionOutputs, GoogLeNet, GoogLeNetOutputs
from torchvision.models import VisionTransformer, MaxVit, SwinTransformer
import lightning as L

from src.utils.focal_loss import FocalLoss
from src.multiclass.models import freeze_model_features

INCEPTION_AUXLOSS_WEIGHT = 0.4
GOOGLENET_AUXLOSS_WEIGHT = 0.3

def get_namebrand_model(model_name, num_classes:list[int], weights:Union[None,str]=None, freeze=None):
    Model = tv.models.get_model_builder(model_name.lower())
    weights_enum = tv.models.get_model_weights(Model)
    ckpt_path = None
    if weights is None:
        pass
    elif os.path.isfile(weights):
        ckpt_path = weights
        weights = None
    elif weights == 'DEFAULT':
        weights = weights_enum.DEFAULT
    else:
        assert weights in weights_enum.__members__, f'args.weights "{weights}" not in {weights_enum.__members__}'
        weights = getattr(weights_enum, weights)

    model = Model(weights=weights if weights else None)

    # modify for num_classes
    fc_models = (Inception3,ResNet,GoogLeNet,RegNet,ShuffleNetV2)
    classifierNeg1_models = (AlexNet,VGG,ConvNeXt,EfficientNet,MNASNet,MobileNetV2,MobileNetV3,MaxVit)

    class MultiHead(nn.Module):
        def __init__(
                self,
                in_features:int,
                num_classes: list[int],
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = num_classes
            self.heads = nn.ModuleList(
                [nn.Linear(self.in_features, out_features, device='cuda')
                    for out_features in num_classes]
            )

        def forward(self, x: Tensor) -> list[Tensor]:
            return [head(x) for head in self.heads]


    if isinstance(model, fc_models):
        model.fc = MultiHead(model.fc.in_features, num_classes)

    elif isinstance(model, classifierNeg1_models):
        model.classifier[-1] = MultiHead(model.classifier[-1].in_features, num_classes)

    elif isinstance(model, SqueezeNet):
        raise NotImplementedError
        # model.classifier is nn.Sequential( nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)) )
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = -1

    elif isinstance(model, DenseNet):
        model.classifier = MultiHead(model.classifier.in_features, num_classes)

    elif isinstance(model, VisionTransformer):
        model.heads.head = MultiHead(model.heads.head.in_features, num_classes)

    elif isinstance(model, SwinTransformer):
        model.head = MultiHead(model.head.in_features, num_classes)

    else:
        raise ValueError(f'Model name "{model_name}" UNKNOWN')

    # Models with Aux Logits
    if isinstance(model, Inception3):
        model.AuxLogits.fc = MultiHead(model.AuxLogits.fc.in_features, num_classes)
    elif isinstance(model, GoogLeNet):
        model.aux1.fc1 = MultiHead(model.aux1.fc1.in_features, num_classes)
        model.aux2.fc2 = MultiHead(model.aux2.fc2.in_features, num_classes)

    if ckpt_path:
        weights = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(weights)

    if freeze:
        freeze_model_features(model, freeze)

    return model


class MultilabelClassifier(L.LightningModule):
    def __init__(self,
                 model_name:str,
                 num_classes:list[int],
                 model_weights:str='DEFAULT',
                 model_freeze:Union[int,float]=None,
                 loss_function:str='CrossEntropyLoss',
                 loss_kwargs:list[dict]=None,
                 optimizer:str='Adam',
                 optimizer_kwargs:dict=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        if loss_function == 'CrossEntropyLoss':
            if not loss_kwargs: loss_kwargs = [{}]*len(num_classes)
            self.criterion = [nn.CrossEntropyLoss(**kwargs) for kwargs in loss_kwargs]
        elif loss_function == 'FocalLoss':
            self.criterion = [FocalLoss(**kwargs) for kwargs in loss_kwargs]
        else: raise NotImplemented

        if optimizer == 'Adam':
            self.Optimizer = torch.optim.Adam
        elif optimizer == 'AdamW':
            self.Optimizer = torch.optim.AdamW
        elif optimizer == 'SGD':
            self.Optimizer = torch.optim.SGD
        else: raise NotImplemented
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.model = get_namebrand_model(model_name, num_classes, model_weights, model_freeze)


        # Instance Variables
        self.best_epoch = 0
        self.best_epoch_val_loss = np.inf
        self.training_loss_by_epoch: dict[int,float] = {}
        self.validation_loss_by_epoch: dict[int,float] = {}
        self.training_loss_by_epoch_per_head = {}
        self.validation_loss_by_epoch_per_head = {}
        self.validation_preds = []
        self.validation_targets = []
        self.validation_sources = []
        self.test_preds = []
        self.test_targets = []
        self.test_sources = []

#    def to(self, *args: Any, **kwargs: Any) -> Self:
#        from lightning_fabric.utilities.device_dtype_mixin import _update_properties
#        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
#        _update_properties(self, device=device, dtype=dtype)
#        return super().to(*args, **kwargs)

    def configure_optimizers(self):
        return self.Optimizer(self.parameters(), **self.optimizer_kwargs)


    def forward(self, x):
        return self.model(x)


    def loss(self, y_hat, y, head_idx=0):
        loss_fn = self.criterion[head_idx]
        if isinstance(y_hat,InceptionOutputs) and self.model.aux_logits:
            loss = loss_fn(y_hat.logits, y)
            loss_aux = loss_fn(y_hat.aux_logits, y)
            batch_loss = loss + INCEPTION_AUXLOSS_WEIGHT*loss_aux
        elif isinstance(y_hat, GoogLeNetOutputs) and self.model.aux_logits:
            loss = loss_fn(y_hat.logits, y)
            loss_aux2 = loss_fn(y_hat.aux_logits2, y)
            loss_aux1 = loss_fn(y_hat.aux_logits1, y)
            batch_loss = loss + GOOGLENET_AUXLOSS_WEIGHT*loss_aux1 + GOOGLENET_AUXLOSS_WEIGHT*loss_aux2
        else:
            batch_loss = loss_fn(y_hat, y)
        return batch_loss


    def logits_only(self, y_hat):
        if isinstance(y_hat, (InceptionOutputs, GoogLeNetOutputs)):
            return y_hat.logits
        return y_hat


    def on_fit_start(self):
        # reset, eg after autobatch
        self.best_epoch = 0
        self.best_epoch_val_loss = np.inf
        self.training_loss_by_epoch = {}
        self.validation_loss_by_epoch = {}
        self.training_loss_by_epoch_per_head = {}
        self.validation_loss_by_epoch_per_head = {}


    def on_train_epoch_start(self) -> None:
        # initializing step loss for epoch
        self.training_loss_by_epoch[self.current_epoch] = 0
        self.training_loss_by_epoch_per_head[self.current_epoch] = [0]*len(self.num_classes)


    def on_validation_epoch_start(self) -> None:
        # Clearing previous epoch's values
        self.validation_preds = []
        self.validation_targets = []
        self.validation_sources = []
        self.validation_loss_by_epoch[self.current_epoch] = 0
        self.validation_loss_by_epoch_per_head[self.current_epoch] = [0]*len(self.num_classes)
        #self.reset_metrics()


    def training_step(self, batch, batch_idx):
        input_data, input_targets = batch[0], batch[1]
        outputs = self.forward(input_data)
        # .view(-1) flattens the tensor, if it wasn't already flat
        losses = [self.loss(output, input_target, i) for i,(output,input_target) in enumerate(zip(outputs,input_targets))]
        for i,head_loss in enumerate(losses):
            self.training_loss_by_epoch_per_head[self.current_epoch][i] += head_loss.item()
        loss = sum(losses)
        self.training_loss_by_epoch[self.current_epoch] += loss.item()
        self.log('train_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.sum)
        for i,label in enumerate(self.trainer.datamodule.labels):
            self.log(f"train_loss_{label}", losses[i], on_step=False, on_epoch=True, reduce_fx=torch.sum)
        return loss


    def eval_step(self, input_data, input_targets):
        outputs = self.forward(input_data)
        losses = [self.loss(output, input_target, i) for i,(output,input_target) in enumerate(zip(outputs,input_targets))]
        preds = [self.logits_only(output.detach()) for output in outputs]
        preds = [F.softmax(pred, dim=1) for pred in preds]
        return losses, preds


    def validation_step(self, batch, batch_idx):
        input_data, input_targets = batch[0], batch[1]
        losses, preds = self.eval_step(input_data, input_targets)
        for i, head_loss in enumerate(losses):
            self.validation_loss_by_epoch_per_head[self.current_epoch][i] += head_loss.item()
        loss = sum(losses)
        self.validation_loss_by_epoch[self.current_epoch] += loss.item()
        self.validation_preds.append([pred.cpu().numpy() for pred in preds])
        self.validation_targets.append([input_target.cpu().numpy() for input_target in input_targets])
        if len(batch)==3:  # TODO document what this 3rd batch item is
            input_srcs = batch[2]
            self.validation_sources.append(input_srcs)

        # METRICS and LOGGING
        #self.update_metrics(preds,input_targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, reduce_fx=torch.sum)
        for i,label in enumerate(self.trainer.datamodule.labels):
            self.log(f"val_loss_{label}", losses[i], on_step=False, on_epoch=True, reduce_fx=torch.sum)


    def on_validation_epoch_end(self):
        self.validation_targets = list(map(np.concatenate, zip(*self.validation_targets)))

        self.validation_preds = list(map(np.concatenate, zip(*self.validation_preds)))
        #pred_classes = np.argmax(self.validation_preds, axis=1)
        if self.validation_sources:
            self.validation_sources = [item for sublist in self.validation_sources for item in sublist]

        # Is it a best epoch?
        val_loss = self.validation_loss_by_epoch[self.current_epoch]
        if val_loss < self.best_epoch_val_loss:
            self.best_epoch_val_loss = val_loss
            self.best_epoch = self.current_epoch

        # METRICS & LOGGING
        #self.log_metrics(stage='val')
        self.log_dict(dict(val_best_epoch=self.best_epoch), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_data, input_targets = batch[0],batch[1]
        losses, preds = self.eval_step(input_data, input_targets)
        loss = sum(losses)
        self.test_preds.append([pred.cpu().numpy() for pred in preds])
        self.test_targets.append([input_target.cpu().numpy() for input_target in input_targets])
        if len(batch)>2:
            input_srcs = batch[2]
            self.test_sources.append(input_srcs)
        # METRICS
        #self.update_metrics(preds,input_targets)
        return loss

    def on_test_epoch_end(self):
        #test_loss = sum(self.test_loss)
        #target_classes = torch.cat(self.test_targets, dim=0)
        preds = [torch.cat(pred, dim=0) for pred in self.test_preds]
        #pred_classes = torch.max(preds, dim=1)
        if self.test_sources:
            sources = [item for sublist in self.test_sources for item in sublist]


    def predict_step(self):
        ...
    def on_predict_model_eval(self):
        ...



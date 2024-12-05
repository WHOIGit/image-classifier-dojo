import argparse
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.models import AlexNet, DenseNet,  ResNet, SqueezeNet, VGG, \
    ConvNeXt, EfficientNet, MNASNet, MobileNetV2, MobileNetV3, RegNet, ShuffleNetV2
from torchvision.models import Inception3, InceptionOutputs, GoogLeNet, GoogLeNetOutputs
import lightning as L
import torchmetrics as tm

INCEPTION_AUXLOSS_WEIGHT = 0.4
GOOGLENET_AUXLOSS_WEIGHT = 0.3

def check_model_name(model_name):
    correct_model_name = tv.models.list_models(include=model_name.lower())
    if correct_model_name: return correct_model_name[0]
    possible_models = tv.models.list_models(include=f'*{model_name.lower()}*')
    if len(possible_models) > 1:
        raise ValueError(f'Model name "{model_name}" ambiguous. Did you mean: {"|".join(possible_models)}?')
    elif len(possible_models) == 0:
        raise ValueError(f'Model name "{model_name}" UNKNOWN')
    else:
        correct_model_name = possible_models[0]
        print(f'Adjusting Model name from "{model_name}" to "{correct_model_name}"')
    return correct_model_name


def get_model_base_transforms(model_name):
    Model = tv.models.get_model_builder(model_name.lower())
    weights = tv.models.get_model_weights(Model).DEFAULT
    resize = weights.transforms().crop_size[0]
    return [tv.transforms.Resize((resize,resize)), tv.transforms.ToTensor()]


def get_namebrand_model(model_name, num_classes, pretrained:Union[None,str]=None, freeze=None):
    Model = tv.models.get_model_builder(model_name.lower())
    weights_enum = tv.models.get_model_weights(Model)
    weights = weights_enum.DEFAULT
    if pretrained and pretrained!='DEFAULT':
        assert pretrained in weights_enum.__members__, f'args.weights "{pretrained}" not in {weights_enum.__members__}'
        weights = getattr(weights_enum, pretrained)
    model = Model(weights=weights if pretrained else None)

    # modify for num_classes
    fc_models = (Inception3,ResNet,GoogLeNet,RegNet,ShuffleNetV2)
    classifierNeg1_models = (AlexNet,VGG,ConvNeXt,EfficientNet,MNASNet,MobileNetV2,MobileNetV3)

    if isinstance(model, fc_models):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, classifierNeg1_models):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif isinstance(model, SqueezeNet):
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif isinstance(model, DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Models with Aux Logits
    if isinstance(model, Inception3):
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    elif isinstance(model, GoogLeNet):
        model.aux1.fc1 = nn.Linear(model.aux1.fc1.in_features, num_classes)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

    if freeze:
        freeze_model_features(model, freeze)

    return model


def freeze_model_features(model, freeze):
    """

    :param model:
    :param freeze:
    :return:
    """
    features_models = (AlexNet, VGG, ConvNeXt, EfficientNet, MobileNetV2, MobileNetV3, SqueezeNet, DenseNet)
    fc_models = (Inception3, ResNet, GoogLeNet, RegNet, ShuffleNetV2)
    def freeze_float2int(freeze, feature_count):
        if isinstance(freeze, int): return freeze
        return int(freeze * feature_count) or 1

    if isinstance(model, features_models):
        freeze = freeze_float2int(freeze,len(model.features))
        for param in model.features[:freeze].parameters():
            param.requires_grad = False  # freeze!
    elif isinstance(model, MNASNet):
        freeze = freeze_float2int(freeze, len(model.layers))
        for param in model.layers[:freeze].parameters():
            param.requires_grad = False
    elif isinstance(model, RegNet):
        freeze = freeze_float2int(freeze, len(model.trunk_output))
        for param in model.stem.parameters():
            param.requires_grad = False
        for param in model.trunk_output[:freeze].parameters():
            param.requires_grad = False
    elif isinstance(model, fc_models):
        pseudo_features = []
        param_names = [n for n,_ in model.named_parameters()]
        param_topnames = [n.split('.')[0] for n in param_names]
        [pseudo_features.append(n) for n in param_topnames if n not in pseudo_features]
        pseudo_features.pop(pseudo_features.index('fc'))
        if isinstance(model, Inception3):
            pseudo_features.pop(pseudo_features.index('AuxLogits'))
        elif isinstance(model, GoogLeNet):
            pseudo_features.pop(pseudo_features.index('aux1'))
            pseudo_features.pop(pseudo_features.index('aux2'))
        freeze = freeze_float2int(freeze, len(pseudo_features))
        for feature in pseudo_features[:freeze]:
            for param in getattr(model, feature).parameters():
                param.requires_grad = False  # freeze!
    else:
        raise ValueError(f'Model name "{type(model)}" UNKNOWN')


class SupervisedModel(L.LightningModule):
    def __init__(self, args: Union[dict,argparse.Namespace]):
        super().__init__()
        if isinstance(args,dict):
            args = argparse.Namespace(**args)
        self.save_hyperparameters(args)

        if args.loss_function == 'CrossEntropyLoss':
            Criterion = nn.CrossEntropyLoss
        else: raise NotImplemented

        self.criterion = Criterion(weight=args.loss_weights_tensor, label_smoothing=args.loss_smoothing)
        self.model = get_namebrand_model(args.model, args.num_classes, args.weights, args.freeze)

        # Instance Variables
        self.best_epoch = 0
        self.best_epoch_val_loss = np.inf
        self.training_loss_by_epoch: dict[int,float] = {}
        self.validation_loss_by_epoch: dict[int,float] = {}
        self.validation_preds = []
        self.validation_targets = []
        self.validation_sources = []
        self.test_preds = []
        self.test_targets = []
        self.test_sources = []

        # Metrics
        self.metrics = nn.ModuleDict()
        self.setup_metrics()

    def on_fit_start(self):
        # reset, eg after autobatch
        self.best_epoch = 0
        self.best_epoch_val_loss = np.inf
        self.training_loss_by_epoch = {}
        self.validation_loss_by_epoch = {}
        self.validation_preds = []
        self.validation_targets = []
        self.validation_sources = []

    def setup_metrics(self):
        num_classes = self.hparams.num_classes
        for mode in ['weighted','micro','macro',None]:
            for stat,MetricClass in zip(['f1','recall','accuracy','precision'],[tm.F1Score,tm.Recall,tm.Accuracy,tm.Precision]):
                key = f'{stat}_{mode or "perclass"}'
                self.metrics[key] = MetricClass(task='multiclass', num_classes=num_classes, average=mode)
        self.metrics['confusion_matrix'] = tm.ConfusionMatrix(task='multiclass', num_classes=num_classes)

    def update_metrics(self, preds, targets):
        for mode in ['weighted','micro','macro',None]:
            for stat in ['f1','recall','precision']:
                key = f'{stat}_{mode or "perclass"}'
                self.metrics[key].update(preds,targets)
        self.metrics['confusion_matrix'].update(preds,targets)

    def reset_metrics(self):
        for mode in ['weighted','micro','macro',None]:
            for stat in ['f1','recall','precision']:
                key = f'{stat}_{mode or "perclass"}'
                self.metrics[key].reset()
        self.metrics['confusion_matrix'].reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_hat, y):
        if isinstance(y_hat,InceptionOutputs) and self.model.aux_logits:
            loss = self.criterion(y_hat.logits, y)
            loss_aux = self.criterion(y_hat.aux_logits, y)
            batch_loss = loss + INCEPTION_AUXLOSS_WEIGHT*loss_aux
        elif isinstance(y_hat, GoogLeNetOutputs) and self.model.aux_logits:
            loss = self.criterion(y_hat.logits, y)
            loss_aux2 = self.criterion(y_hat.aux_logits2, y)
            loss_aux1 = self.criterion(y_hat.aux_logits1, y)
            batch_loss = loss + GOOGLENET_AUXLOSS_WEIGHT*loss_aux1 + GOOGLENET_AUXLOSS_WEIGHT*loss_aux2
        else:
            batch_loss = self.criterion(y_hat, y)
        return batch_loss

    def logits_only(self, y_hat):
        if isinstance(y_hat, (InceptionOutputs, GoogLeNetOutputs)):
            return y_hat.logits
        return y_hat

    def on_train_epoch_start(self) -> None:
        # Clearing previous epoch's values
        self.validation_preds = []
        self.validation_targets = []
        self.validation_sources = []
        # initializing step loss for epoch
        self.training_loss_by_epoch[self.current_epoch] = 0
        self.validation_loss_by_epoch[self.current_epoch] = 0
        self.reset_metrics()

    def training_step(self, batch, batch_idx):
        input_data, input_targets = batch[0], batch[1]
        outputs = self.forward(input_data)
        loss = self.loss(outputs, input_targets.view(-1))  # .view(-1) flattens the tensor, if it wasn't already flat
        self.training_loss_by_epoch[self.current_epoch] += loss.item()
        self.log('train_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.sum)
        return loss

    def eval_step(self, input_data, input_targets):
        outputs = self.forward(input_data)
        loss = self.loss(outputs, input_targets.view(-1))
        preds = self.logits_only(outputs)
        preds = F.softmax(preds, dim=1)
        return loss, preds

    def validation_step(self, batch, batch_idx):
        input_data, input_targets = batch[0], batch[1]
        loss, preds = self.eval_step(input_data, input_targets)
        self.validation_loss_by_epoch[self.current_epoch] += loss.item()
        self.validation_preds.append(preds.cpu().numpy())
        self.validation_targets.append(input_targets.cpu().numpy())
        if len(batch)==3: # TODO document what this 3rd batch item is
            input_srcs = batch[2]
            self.validation_sources.append(input_srcs)

        # METRICS and LOGGING
        self.update_metrics(preds,input_targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, reduce_fx=torch.sum)

    def on_validation_epoch_end(self):
        val_loss = self.validation_loss_by_epoch[self.current_epoch]
        self.validation_targets = np.concatenate(self.validation_targets, axis=0)
        self.validation_preds = np.concatenate(self.validation_preds, axis=0)
        pred_classes = np.argmax(self.validation_preds, axis=1)
        if self.validation_sources:
            self.validation_sources = [item for sublist in self.validation_sources for item in sublist]

        # Is it a best epoch?
        if val_loss < self.best_epoch_val_loss:
            self.best_epoch_val_loss = val_loss
            self.best_epoch = self.current_epoch

        # METRICS & LOGGING
        for mode in ['weighted','micro','macro']:  # perclass not available to log as metric
            for stat in ['f1','recall','precision']:
                key = f'{stat}_{mode}'
                datum = self.metrics[key].compute()
                #if mode=='micro' and stat in ['recall','precision']: continue  # identical to macro
                self.log(f'val_{stat}_{mode}', datum, on_epoch=True)

        # train and val loss already handled, normalized is nice to view overfitting
        # TODO as callback, not in main loop
        if self.validation_loss_by_epoch[0] and self.training_loss_by_epoch[0]: # else errors on autobatch
            val_normloss = val_loss/self.validation_loss_by_epoch[0]
            train_normloss = self.training_loss_by_epoch[self.current_epoch]/self.training_loss_by_epoch[0]
            self.log('val_normloss', val_normloss, on_epoch=True)
            self.log('train_normloss', train_normloss, on_epoch=True)

        self.log_dict(dict(best_epoch=self.best_epoch), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_data, input_targets = batch[0],batch[1]
        loss, preds = self.eval_step(input_data, input_targets)
        self.test_preds.append(preds.cpu().numpy())
        self.test_targets.append(input_targets.cpu().numpy())
        if len(batch)>2:
            input_srcs = batch[2]
            self.test_sources.append(input_srcs)
        # METRICS
        self.update_metrics(preds,input_targets)
        return loss

    def on_test_epoch_end(self):
        #test_loss = sum(self.test_loss)
        target_classes = torch.cat(self.test_targets, dim=0)
        preds = torch.cat(self.test_preds, dim=0)
        pred_classes = torch.max(preds, dim=1)
        if self.test_sources:
            sources = [item for sublist in self.test_sources for item in sublist]


    def predict_step(self):
        ...
    def on_predict_model_eval(self):
        ...


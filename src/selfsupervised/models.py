import copy
import os.path
from typing import Any, List, Optional, Tuple, Literal, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.models import AlexNet, DenseNet, ResNet, SqueezeNet, VGG, \
    ConvNeXt, EfficientNet, MNASNet, MobileNetV2, MobileNetV3, RegNet, ShuffleNetV2
from torchvision.models import Inception3, GoogLeNet
from torchvision.models import MaxVit, VisionTransformer, SwinTransformer

import lightly.models
from lightly.utils.dist import gather as lightly_gather
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.benchmarking.knn import knn_predict

import torchmetrics as tm

# see lightly.utils.benchmarking.knn.knn_predict
def knn_scores(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> Tensor:
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    #pred_labels = pred_scores.argsort(dim=-1, descending=True)  # edited out from original knn_predict(...)
    return pred_scores


def get_namebrand_beheaded_model(model_name, weights:Union[None,str]=None) -> (torch.nn.Module,int):
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
    if isinstance(Model, (Inception3,GoogLeNet)):
        model = Model(weights=weights if weights else None, aux_logits=False)
    else:
        model = Model(weights=weights if weights else None)

    # chop head of model to make backbone
    fc_models = (Inception3,ResNet,GoogLeNet,RegNet,ShuffleNetV2)
    classifierNeg1_models = (AlexNet,VGG,ConvNeXt,EfficientNet,MNASNet,MobileNetV2,MobileNetV3,MaxVit)

    if isinstance(model, fc_models):
        out_features = model.fc.in_features
        #backbone = torch.nn.Sequential(*list(model.children())[:-1])
        model.fc = nn.Identity()
        backbone = model
    elif isinstance(model, classifierNeg1_models):
        out_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        backbone = model
    elif isinstance(model, SqueezeNet):
        out_features = model.classifier[1].in_channels
        model.classifier = model.classifier[:1]
        backbone = model
    elif isinstance(model, DenseNet):
        out_features = model.classifier.in_features
        #backbone = list(model.children())[0]
        model.classifier = nn.Identity()
        backbone = model
    elif isinstance(model, VisionTransformer):
        out_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        backbone = model
    elif isinstance(model, SwinTransformer):
        out_features = model.heads.head.in_features
        model.head = nn.Identity()
        backbone = model
    else:
        raise ValueError(f'Model name "{model_name}" UNKNOWN')

    if ckpt_path:
        weights = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(weights)

    return backbone, out_features


class SSLValidationModule(pl.LightningModule):
    def __init__(
        self,
        knn_dataloader: DataLoader[Any],
        knn_k: int = 200,
        knn_t: float = 0.1,
    ):
        super().__init__()
        self.backbone = nn.Module()
        self.knn_dataloader = knn_dataloader
        self.num_classes = len(knn_dataloader.dataset.classes) if knn_dataloader else 0
        self.knn_k = knn_k
        self.knn_t = knn_t

        self._train_features: Optional[Tensor] = None
        self._train_targets: Optional[Tensor] = None
        self._val_predicted_labels: List[Tensor] = []
        self._val_targets: List[Tensor] = []

        # Instance Variables
        self.best_epoch = 0
        self.best_epoch_val_loss = np.inf
        self.training_loss_by_epoch: dict[int, float] = {}
        self.validation_loss_by_epoch: dict[int, float] = {}

        # Metrics
        self.metrics = nn.ModuleDict()
        self.setup_metrics()

    def setup_metrics(self):
        num_classes = self.num_classes
        for mode in ['weighted','micro','macro',None]:
            for stat,MetricClass in zip(['f1','recall','accuracy','precision','accuracy'],
                                        [tm.F1Score,tm.Recall,tm.Accuracy,tm.Precision,tm.Accuracy]):
                key = f'{stat}_{mode or "perclass"}'
                self.metrics[key] = MetricClass(task='multiclass', num_classes=num_classes, average=mode)
        self.metrics['confusion_matrix'] = tm.ConfusionMatrix(task='multiclass', num_classes=num_classes)

    def update_metrics(self, preds, targets):
        for mode in ['weighted','micro','macro',None]:
            for stat in ['f1','recall','precision','accuracy']:
                key = f'{stat}_{mode or "perclass"}'
                self.metrics[key].update(preds,targets)
        self.metrics['confusion_matrix'].update(preds,targets)

    def log_metrics(self, stage:Literal['val','test','train']):
        for mode in ['weighted','micro','macro']:  # perclass not available to log as metric
            for stat in ['f1','recall','precision','accuracy']:
                key = f'{stat}_{mode}'
                datum = self.metrics[key].compute()
                #if mode=='micro' and stat in ['recall','precision']: continue  # identical to macro
                self.log(f'{stage}_{stat}_{mode}', datum, on_epoch=True)

    def reset_metrics(self):
        for mode in ['weighted','micro','macro',None]:
            for stat in ['f1','recall','precision','accuracy']:
                key = f'{stat}_{mode or "perclass"}'
                self.metrics[key].reset()
        self.metrics['confusion_matrix'].reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_fit_start(self):
        # reset, eg after autobatch
        self.best_epoch = 0
        self.best_epoch_val_loss = np.inf
        self.training_loss_by_epoch = {}
        self.validation_loss_by_epoch = {}
        self.max_accuracy = 0.0

    def on_train_epoch_start(self) -> None:
        # initializing step loss for epoch
        self.training_loss_by_epoch[self.current_epoch] = 0

    def on_validation_epoch_start(self) -> None:
        # Clearing previous epoch's values
        self.validation_loss_by_epoch[self.current_epoch] = 0
        self.reset_metrics()

        # build embeddings
        train_features = []
        train_targets = []
        with torch.no_grad():
            for data in self.knn_dataloader:
                img, target, _ = data
                img = img.to(self.device)
                target = target.to(self.device)
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                train_features.append(feature)
                train_targets.append(target)
        self._train_features = torch.cat(train_features, dim=0).t().contiguous()
        self._train_targets = torch.cat(train_targets, dim=0).t().contiguous()

    def validation_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> None:
        # we can only do kNN predictions once we have a feature bank
        if self._train_features is not None and self._train_targets is not None:
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            predicted_scores = knn_scores(
                feature,
                self._train_features,
                self._train_targets,
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )

            if dist.is_initialized() and dist.get_world_size() > 0:
                # gather predictions and targets from all processes
                predicted_scores = torch.cat(lightly_gather(predicted_scores), dim=0)
                targets = torch.cat(lightly_gather(targets), dim=0)

            preds = F.softmax(predicted_scores, dim=1)
            #for i in range(len(targets)):
            #    print(f'target={targets[i].item()} score={torch.max(preds[i])} class={torch.argmax(preds[i])}' )

            # METRICS and LOGGING
            val_batchloss = torch.nn.CrossEntropyLoss()(preds, targets)  # sidney secret sauce
            self.validation_loss_by_epoch[self.current_epoch] += val_batchloss.item()
            self.log('val_loss', val_batchloss, on_step=False, on_epoch=True, reduce_fx=torch.sum)
            self.update_metrics(preds, targets)

            predicted_labels = predicted_scores.argsort(dim=-1, descending=True)
            self._val_predicted_labels.append(predicted_labels.cpu())
            self._val_targets.append(targets.cpu())

    def on_validation_epoch_end(self) -> None:
        if self._val_predicted_labels and self._val_targets:

            # METRICS & LOGGING
            val_loss = self.validation_loss_by_epoch[self.current_epoch]
            if val_loss < self.best_epoch_val_loss:
                self.best_epoch_val_loss = val_loss
                self.best_epoch = self.current_epoch
            self.log_dict(dict(val_best_epoch=self.best_epoch), on_epoch=True, prog_bar=True)
            self.log_metrics(stage='val')

        self._val_predicted_labels.clear()
        self._val_targets.clear()

    #todo add tests

# build a SimCLR model #
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
class SimCLR(SSLValidationModule):
    def __init__(self, backbone_name:str, backbone_weights:Union[str,None],
                 output_dim:int=128, hidden_dim:Union[int,Literal['input_dim']]='input_dim',
                 knn_dataloader=[], knn_k: int = 100, knn_t: float = 0.1):
        super().__init__(knn_dataloader, knn_k, knn_t)

        self.save_hyperparameters(ignore='dataloader_kNN')

        self.backbone, backbone_features_num = get_namebrand_beheaded_model(backbone_name, backbone_weights)
        self.projection_head = SimCLRProjectionHead(
            input_dim = backbone_features_num,
            hidden_dim = hidden_dim if isinstance(hidden_dim,int) else backbone_features_num,
            output_dim = output_dim,
            num_layers = 2,
        )
        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), img_id = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.training_loss_by_epoch[self.current_epoch] += loss.item()
        self.log("train_loss", loss)
        return loss


from lightly.models.modules.heads import VICRegProjectionHead
from lightly.loss.vicreg_loss import VICRegLoss
class VICReg(SSLValidationModule):
    def __init__(self, backbone_name, backbone_weights,
                 output_dim:int=2048, hidden_dim:Union[int,Literal['input_dim']]='input_dim',
                 knn_dataloader=[], knn_k:int=100, knn_t:float=0.1):
        super().__init__(knn_dataloader, knn_k, knn_t)

        self.save_hyperparameters(ignore='dataloader_kNN')

        self.backbone, backbone_features_num = get_namebrand_beheaded_model(backbone_name, backbone_weights)

        self.projection_head = VICRegProjectionHead(
            input_dim = backbone_features_num,
            hidden_dim = hidden_dim if isinstance(hidden_dim,int) else backbone_features_num,
            output_dim = output_dim,
            num_layers = 2,
        )
        self.criterion = VICRegLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.training_loss_by_epoch[self.current_epoch] += loss.item()
        self.log("train_loss", loss)
        return loss


from lightly.models import utils
from lightly.loss import PMSNLoss
from lightly.models.modules import MAEBackbone
from lightly.models.modules.heads import MSNProjectionHead
class PMSN(SSLValidationModule):
    def __init__(self, backbone_name, backbone_weights,
                 output_dim:int=256, hidden_dim:Union[int,Literal['input_dim']]='input_dim',
                 knn_dataloader=[], knn_k:int=100, knn_t:float=0.1):
        super().__init__(knn_dataloader, knn_k, knn_t)

        # ViT small configuration (ViT-S/16)
        self.mask_ratio = 0.15

        assert backbone_name.startswith('vit')
        headless_vit,backbone_features_num = get_namebrand_beheaded_model(backbone_name, weights=backbone_weights)
        self.backbone = MAEBackbone.from_vit(headless_vit)
        self.projection_head = MSNProjectionHead(
            input_dim = backbone_features_num,
            hidden_dim = hidden_dim if isinstance(hidden_dim,int) else backbone_features_num,
            output_dim = output_dim)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(output_dim, 1024, bias=False).weight
        self.criterion = PMSNLoss()

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views = batch[0]
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(images=targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        self.training_loss_by_epoch[self.current_epoch] += loss.item()
        self.log("train_loss", loss)
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(images=anchors, idx_keep=idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(params, lr=1.5e-4)
        return optim




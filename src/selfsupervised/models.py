from typing import Any, List, Optional, Tuple, Literal, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader

from lightly.utils.dist import gather as lightly_gather
from lightly.utils.debug import std_of_l2_normalized
import torchmetrics as tm

from lightly.utils.benchmarking.knn import knn_predict
def knn_scores(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (D, N) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.

    Returns:
        A tensor containing the kNN prediction scores

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """
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


class SSLValidationModule(pl.LightningModule):
    def __init__(
        self,
        dataloader_kNN: DataLoader[Any],
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
    ):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.num_classes = num_classes
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
            for data in self.dataloader_kNN:
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
            predicted_labels = torch.cat(self._val_predicted_labels, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            top1 = (predicted_labels[:, 0] == targets).float().sum()
            acc = top1 / len(targets)
            if acc > self.max_accuracy:
                self.max_accuracy = float(acc.item())
            self.log("val_accuracy_knn", acc, on_epoch=True)

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
    def __init__(self, backbone, hidden_dim, out_dim,
                 dataloader_kNN=[], knn_k: int = 200, knn_t: float = 0.1):

        # todo backbone creation here
        # todo hparams
        if isinstance(dataloader_kNN, DataLoader):
            num_classes = len(dataloader_kNN.dataset.classes)
        else: num_classes = 0
        super().__init__(dataloader_kNN, num_classes, knn_k, knn_t)

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
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

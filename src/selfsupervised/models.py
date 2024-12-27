from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader

from lightly.utils.benchmarking.knn import knn_predict
from lightly.utils.dist import gather as lightly_gather


class BenchmarkModule(pl.LightningModule):
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

    def on_validation_epoch_start(self) -> None:
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
            predicted_labels = knn_predict(
                feature,
                self._train_features,
                self._train_targets,
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )

            if dist.is_initialized() and dist.get_world_size() > 0:
                # gather predictions and targets from all processes

                predicted_labels = torch.cat(lightly_gather(predicted_labels), dim=0)
                targets = torch.cat(lightly_gather(targets), dim=0)

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

            # TODO experiment with val_loss and metrics
            #val_loss = torch.nn.CrossEntropyLoss()(predicted_labels, targets)  # sidney secret sauce
            #self.log('val_loss', val_loss, on_epoch=True)

        self._val_predicted_labels.clear()
        self._val_targets.clear()


# build a SimCLR model #
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
class SimCLR(BenchmarkModule):
    def __init__(self, backbone, hidden_dim, out_dim,
                 dataloader_kNN=[], knn_k: int = 200, knn_t: float = 0.1):

        if isinstance(dataloader_kNN, DataLoader):
            num_classes = len(dataloader_kNN.dataset.classes)
        else: num_classes = 0
        super().__init__(dataloader_kNN, num_classes, knn_k, knn_t)

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = NTXentLoss()
        self.training_loss_by_epoch: dict[int,float] = {}

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def on_fit_start(self):
        # reset, eg after autobatch
        self.max_accuracy = 0.0
        self.training_loss_by_epoch = {}

    def on_train_epoch_start(self) -> None:
        # initializing step loss for epoch
        self.training_loss_by_epoch[self.current_epoch] = 0

    def training_step(self, batch, batch_idx):
        (x0, x1), img_id = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.training_loss_by_epoch[self.current_epoch] += loss.item()
        self.log("train_sslloss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
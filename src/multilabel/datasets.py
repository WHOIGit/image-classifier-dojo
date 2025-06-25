import json
import csv

import tqdm
import torchvision
from torchvision.transforms import v2
import lightning as L

from src.multiclass.datasets import ImageDatasetWithSource, ImageListsWithLabelIndex



# Dataset
class MultilabelDataset(ImageDatasetWithSource):

    @property
    def targets_as_str(self):
        output = []
        for target_as_indicies in self.targets:
            target_labels = [self.classes[i][idx] for i,idx in enumerate(target_as_indicies)]
            output.append(target_labels)
        return output

    @property
    def flat_classes(self):
        output = []
        for labels in self.classes:
            output.extend(labels)
        return output



# LightningDataModule
class MultilabelDataModule(ImageListsWithLabelIndex):
    def __init__(self, train_src, val_src, classlist,
                 base_transforms, training_transforms=None,
                 test_src=None, batch_size=108, num_workers=4,
                 ):
        super().__init__(train_src, val_src, classlist,
                 base_transforms, training_transforms,
                 test_src, batch_size, num_workers)
        self.labels, self.classes = self.classes


    @staticmethod
    def parse_classes_file(classlist: str):
        with open(classlist) as f:
            classes = json.load(f)
        labels = list(classes.keys())  # [Z1, A1, ... N9]
        classes = list(classes.values())  # [[Z1.0, Z1.1, ...], [A1.0, ...], [N9.0, ..., N9.9]]
        return labels, classes


    def parse_targets_file(self, listfile, source_column='filepath'):  # source_column otherwise None
        errors = []
        sources_targets = []
        with open(listfile) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if source_column is None:  # use first column
                    source_column = list(row.keys())[0]
                source = row[source_column]
                targets = []
                for i,label in enumerate(self.labels):
                    class_of_label = row[label]
                    target = self.classes[i].index(class_of_label)
                    targets.append(target)
                sources_targets.append( (source, tuple(targets)) )
        return sources_targets, errors


    def count_perclass(self, datasets='fit'):
        if datasets=='fit':
            sources = self.training_dataset.sources+self.validation_dataset.sources
            targets = self.training_dataset.targets+self.validation_dataset.targets
        else:
            raise NotImplementedError
        ipc = {c: set() for c in self.training_dataset.flat_classes}
        for src_img, target in zip(sources,targets):
            for i,label in enumerate(target):
                ipc[self.classes[i][target]].add(src_img)
        cpc = {cls:len(src_imgs) for cls,src_imgs in ipc.items()}
        return cpc
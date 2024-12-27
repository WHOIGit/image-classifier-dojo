import os
import random

from tqdm import tqdm
import lightning as L
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import v2


def parse_listfile(classlist):
    with open(classlist) as f:
        return f.read().splitlines()


def parse_listfile_with_targets(listfile, num_classes: int = None):
    bad_class = []
    bad_file = []
    bad_ext = []
    sources_targets = []
    with open(listfile) as f:
        for line in tqdm(f.read().splitlines(), desc=listfile):
            if not line: continue
            trg, src = line.split()
            trg = int(trg)
            if num_classes and not trg < num_classes:
                print(f'BAD TRG: trg > num_classes {trg} < {num_classes} for {src}')
                bad_class.append(line)
            elif not torchvision.datasets.folder.is_image_file(src):
                print(f'BAD EXT: {src}')
                bad_ext.append(line)
            elif not os.path.isfile(src):
                print(f'BAD FILE: {src}')
                bad_file.append(line)
            else:
                sources_targets.append((src, trg))
    errors = bad_file + bad_ext + bad_class
    return sources_targets, errors


class ImageDatasetWithSource(Dataset):
    """
    Custom dataset that includes source file paths in addition to data and target. Also checks for valid image extention.
    Example setup:     dataloader = torch.utils.DataLoader(ImageDatasetWithPaths(list_of_image_paths))
    Example usage:     for inputs,labels,source in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """
    def __init__(self, sources_targets, classes, transform=None, without_source=False):
        self.sources, self.targets = zip(*sources_targets)
        self.classes = classes
        self.transform = transform
        self.without_source = without_source

    @property
    def labels(self):
        return [self.classes[idx] for idx in self.targets]

    def __getitem__(self, index):
        src = self.sources[index]
        target = self.targets[index]
        sample = torchvision.datasets.folder.default_loader(src)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.without_source:
            return sample, target
        return sample, target, src

    def __len__(self):
        return len(self.targets)

    @property
    def images_perclass(self):
        ipc = {c:[] for c in self.classes}
        for src_img, target in zip(self.sources, self.targets):
            ipc[self.classes[target]].append(src_img)
        return ipc

    @property
    def count_perclass(self):
        cpc = {c:0 for c in self.classes} # initialize list at 0-counts
        for class_label in self.labels:
            cpc[class_label] += 1
        return cpc


class ImageListsWithLabelIndex(L.LightningDataModule):
    def __init__(self, train_src, val_src, classlist,
                 base_transforms, training_transforms=[],
                 test_src=None, batch_size=108, num_workers=4,
                 ):
        super().__init__()
        self.training_source = train_src
        self.validation_source = val_src
        self.test_source = test_src
        self.classes = parse_listfile(classlist)
        self.base_transforms = base_transforms
        self.training_transforms = training_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        assert len(set(self.classes))==len(self.classes), 'Class list has duplicate labels!'

    def setup(self, stage: str, without_source=False):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            training_samples, train_ds_errors = parse_listfile_with_targets(self.training_source, len(self.classes))
            validation_samples, val_ds_errors = parse_listfile_with_targets(self.validation_source, len(self.classes))

            if train_ds_errors or val_ds_errors:
                raise RuntimeError(f'BAD SAMPLES: {len(train_ds_errors)+len(val_ds_errors)}')

            validation_transform = v2.Compose(self.base_transforms)
            training_transform = v2.Compose(self.training_transforms + self.base_transforms)

            self.training_dataset = ImageDatasetWithSource(sources_targets=training_samples, classes=self.classes,
                transform=training_transform, without_source=without_source)
            self.validation_dataset = ImageDatasetWithSource(sources_targets=validation_samples, classes=self.classes,
                transform=validation_transform, without_source=without_source)


        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.testing_dataset = ...


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.testing_dataset is None:
            self.setup('test')
        return DataLoader(self.testing_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def count_perclass(self, datasets='fit'):
        if datasets=='fit':
            ipc = {c: set() for c in self.classes}
            for src_img, target in zip(self.training_dataset.sources+self.validation_dataset.sources,
                                       self.training_dataset.targets+self.validation_dataset.targets):
                ipc[self.classes[target]].add(src_img)
            cpc = {cls:len(src_imgs) for cls,src_imgs in ipc.items()}
            return cpc


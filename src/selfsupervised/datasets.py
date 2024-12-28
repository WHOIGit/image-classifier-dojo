import os
import math
import random
import itertools
from functools import cache
from typing import Union

from tqdm import tqdm
import humanize

import lightning as L
import lightning.pytorch as pl
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2
from torch.utils.data.datapipes.iter import Shuffler

import ifcb
from ifcb.data.adc import SCHEMA_VERSION_1
from ifcb.data.stitching import InfilledImages

# set import paths to project root
if __name__ == '__main__':
    import sys, pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))

from src.multiclass.datasets import parse_listfile, parse_listfile_with_targets, ImageDatasetWithSource


class IfcbBinsDataset(IterableDataset):
    def __init__(self,
                 bin_dirs:list,
                 transform,
                 with_sources=True,
                 shuffle=True,
                 use_len:Union[bool,int]=False):
        bin_dirs = [ifcb.data.files.list_data_dirs(bin_dir, blacklist=['bad','skip','beads','temp','data_temp']) for bin_dir in bin_dirs]
        self.bin_dirs = sorted(itertools.chain.from_iterable(bin_dirs), key=lambda p:os.path.basename(p))  # flatten
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.bin_dirs)
        self.transform = v2.Compose(transform) if isinstance(transform,list) else transform
        self.with_sources = with_sources
        self.use_len = use_len
        if self.use_len is True:
            self.calculate_len()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dds = [ifcb.DataDirectory(bin_dir) for bin_dir in self.bin_dirs]

        if worker_info is None:
            if self.shuffle: # shuffle order of all data directories
                random.shuffle(dds)
            iter_chunk = dds  # all of it
        else:
            # split up the work
            N = worker_info.num_workers
            n, m = divmod(len(dds), N)
            per_worker = [dds[i*n+min(i,m):(i+1)*n+min(i+1,m)] for i in range(N)]
            worker_id = worker_info.id
            iter_chunk = per_worker[worker_id]
            if self.shuffle:
                random.shuffle(iter_chunk)

        for dd in iter_chunk:
            for binfileset in dd:

                # old-style bins need to be stitched and infilled
                if binfileset.schema == SCHEMA_VERSION_1:
                    bin_images = InfilledImages(binfileset)
                else:
                    bin_images = binfileset.images

                for target_number, roi in bin_images.items():
                    target_pid = binfileset.pid.with_target(target_number)
                    img = v2.ToPILImage(mode='L')(roi)
                    img = img.convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img)
                    if self.with_sources:
                        yield img, target_pid
                    else:
                        yield img

    @cache
    def calculate_len(self):
        count_sum = 0
        bin_sum = 0
        pbar = tqdm(self.bin_dirs, desc='caching dataset length')
        for bin_dir in pbar:
            for ifcbbin in ifcb.DataDirectory(bin_dir):
                bin_sum += 1
                bin_count = len(ifcbbin.images)  # vs len(ifcbbin) ?
                count_sum += bin_count
            pbar.set_postfix(dict(BINs=humanize.intcomma(bin_sum),
                                  ROIs=humanize.intword(count_sum)))
        return count_sum

    def __len__(self):
        if self.use_len is True:
            return self.calculate_len()
        elif self.use_len:
            return self.use_len


class IfcbDatamodule(L.LightningDataModule):
    def __init__(self,
                 ifcb_bin_dirs_src:Union[str,list],
                 ssl_transform,
                 knn_src:str=None, val_src:str=None,
                 eval_classlist:str=None, eval_transform=None,
                 test_src:str=None,
                 batch_size:int=108, num_workers:int=4, shuffler_buffer_size:int=100,
                 use_len: Union[bool, int] = False,
                 ):
        super().__init__()

        self.training_sources = ifcb_bin_dirs_src
        self.training_transforms = ssl_transform
        self.use_len = use_len

        if val_src:
            assert knn_src and eval_classlist and eval_transform
        if test_src:
            assert knn_src and eval_classlist and eval_transform
        self.eval_classes = parse_listfile(eval_classlist) if eval_classlist else None
        self.eval_transform = eval_transform
        self.knn_source = knn_src
        self.validation_source = val_src
        self.test_source = test_src

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffler_buffer_size = shuffler_buffer_size

        self.training_dataset = None
        self.knn_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        assert len(set(self.eval_classes))==len(self.eval_classes), 'Class list has duplicate labels!'


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if (stage == "fit" or stage=='train') and self.training_dataset is None:
            if isinstance(self.training_sources, list):
                assert all([os.path.isdir(d) for d in self.training_sources])
                training_sources = self.training_sources
            elif os.path.isdir(self.training_sources):
                training_sources = [self.training_sources]
            else:
                training_sources = parse_listfile(self.training_sources)

            if not self.shuffler_buffer_size:
                self.training_dataset = IfcbBinsDataset(training_sources, self.training_transforms, shuffle=False, use_len=self.use_len)
            else:
                training_dataset = IfcbBinsDataset(training_sources, self.training_transforms, shuffle=True, use_len=self.use_len)
                self.training_dataset = Shuffler(training_dataset, buffer_size=self.shuffler_buffer_size)
                # Iterable datasets like IfcbBinsDataset cannot be access its elements at random;
                # Shuffler collects a buffer of dataset return values and shuffles them before returning them

        if stage == 'knn' and self.knn_dataset is None:
            knn_samples, knn_ds_errors = parse_listfile_with_targets(self.knn_source, len(self.eval_classes))
            if knn_ds_errors:
                raise RuntimeError(f'BAD KNN SAMPLES: {len(knn_ds_errors)}')
            self.knn_dataset = ImageDatasetWithSource(sources_targets=knn_samples,
                                    classes=self.eval_classes, transform=self.eval_transform)

        if (stage == 'fit' or stage == 'validate') and self.validation_dataset is None:
            if self.validation_source:
                validation_samples, val_ds_errors = parse_listfile_with_targets(self.validation_source, len(self.eval_classes))
                if val_ds_errors:
                    raise RuntimeError(f'BAD VAL SAMPLES: {len(val_ds_errors)}')
                self.validation_dataset = ImageDatasetWithSource(sources_targets=validation_samples,
                    classes=self.eval_classes, transform=self.eval_transform)
                self.setup('knn')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" and self.test_dataset is None:
            if self.test_source:
                test_samples, test_ds_errors = parse_listfile_with_targets(self.test_source, len(self.eval_classes))
                if test_ds_errors:
                    raise RuntimeError(f'BAD TEST SAMPLES: {len(test_ds_errors)}')
                self.validation_dataset = ImageDatasetWithSource(sources_targets=test_samples,
                    classes=self.eval_classes, transform=self.eval_transform)
                self.setup('knn')


    def train_dataloader(self, stage='fit'):
        if self.training_dataset is None:
            self.setup(stage)
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True if self.shuffler_buffer_size else False)

    def val_dataloader(self):
        if self.validation_dataset is None:
            self.setup('validate')
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def knn_dataloader(self):
        if self.knn_dataset is None:
            self.setup('knn')
        return DataLoader(self.knn_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup('test')
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
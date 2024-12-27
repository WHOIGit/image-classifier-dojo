import os
import math
import random
import itertools
from functools import cache

from tqdm import tqdm
import humanize

import lightning as L
import lightning.pytorch as pl
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

import ifcb
from ifcb.data.adc import SCHEMA_VERSION_1
from ifcb.data.stitching import InfilledImages


class IfcbBinsDataset(IterableDataset):
    def __init__(self, bin_dirs:list, transform, with_sources=True, shuffle=True):
        bin_dirs = [ifcb.data.files.list_data_dirs(bin_dir, blacklist=['bad','skip','beads']) for bin_dir in bin_dirs]
        self.bin_dirs = sorted(itertools.chain.from_iterable(bin_dirs), key=lambda p:os.path.basename(p))  # flatten
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.bin_dirs)
        self.transform = transforms.Compose(transform) if isinstance(transform,list) else transform
        self.with_sources = with_sources

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
                    img = transforms.ToPILImage(mode='L')(roi)
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
        pbar = tqdm(self.bin_dirs, desc='caching dataset length')
        for bin_dir in pbar:
            for ifcbbin in ifcb.DataDirectory(bin_dir):
                bin_count = len(ifcbbin.images)  # vs len(ifcbbin) ?
                count_sum += bin_count
            pbar.set_postfix(dict(ROIs=humanize.intword(count_sum)))
        return count_sum

    def __len__(self):
        return self.calculate_len()

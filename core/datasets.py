import os
import random

import lightning as L
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision import transforms


class ImageDatasetWithSource(Dataset):
    """
    Custom dataset that includes source file paths in addition to data and target. Also checks for valid image extention.
    Example setup:     dataloader = torch.utils.DataLoader(ImageDatasetWithPaths(list_of_image_paths))
    Example usage:     for inputs,labels,source in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """
    def __init__(self, sources_targets, classes, transform=None):
        self.sources, self.targets = zip(*sources_targets)
        self.classes = classes
        self.transform = transform

    @property
    def labels(self):
        return [self.classes[idx] for idx in self.targets]

    def __getitem__(self, index):
        src = self.sources[index]
        target = self.targets[index]
        sample = torchvision.datasets.folder.default_loader(src)
        if self.transform is not None:
            sample = self.transform(sample)
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
    def __init__(self, train_src, val_src, classlist, base_transforms, training_transforms=[], test_src=None, batch_size=108, num_workers=4):
        super().__init__()
        self.training_source = train_src
        self.validation_source = val_src
        self.test_source = test_src
        self.classes = self.parse_classlist(classlist)
        self.base_transforms = base_transforms
        self.training_transforms = training_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        assert len(set(self.classes))==len(self.classes), 'Class list has duplicate labels!'

    @staticmethod
    def parse_classlist(classlist):
        with open(classlist) as f:
            return f.read().splitlines()

    def parse_listfile_with_targets(self, listfile):
        num_classes = len(self.classes)
        bad_class = []
        bad_file = []
        bad_ext = []
        sources_targets = []
        with open(listfile) as f:
            for line in f.read().splitlines():
                if not line: continue
                trg, src = line.split()
                trg = int(trg)
                if not trg < num_classes:
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

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            training_samples, train_ds_errors = self.parse_listfile_with_targets(self.training_source)
            validation_samples, val_ds_errors = self.parse_listfile_with_targets(self.validation_source)

            if train_ds_errors or val_ds_errors:
                print('BAD SAMPLES:', len(train_ds_errors)+len(val_ds_errors))
                raise RuntimeError

            validation_transform = transforms.Compose(self.base_transforms)
            training_transform = transforms.Compose(self.training_transforms + self.base_transforms)

            self.training_dataset = ImageDatasetWithSource(sources_targets=training_samples, classes=self.classes, transform=training_transform)
            self.validation_dataset = ImageDatasetWithSource(sources_targets=validation_samples, classes=self.classes, transform=validation_transform)


        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.testing_dataset = ...


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
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



class BalancedPerClassImageFolder(L.LightningDataModule):
    def __init__(self, trainval_dir, test_dir=None,
                 minimum_images_per_class=1,
                 maximum_images_per_class=None,
                 batch_size=100, num_workers=4,
                 base_transforms=None, adl_train_transforms=None):
        super().__init__()
        self.trainval_dir = trainval_dir
        self.test_dir = test_dir
        self.minimum_images_per_class = max(1, minimum_images_per_class)  # always at least 1.
        self.maximum_images_per_class = maximum_images_per_class
        self.num_workers = num_workers

        self.base_transforms = base_transforms

        self.batch_size = batch_size  # may get updated by autobatch Tuner

        # declaring variables that get set elsewhere
        self.classes = None
        self.classes_ignored_from_too_few_samples = None
        self.classes_limited_from_too_many_samples = None
        self.training_dataset, self.validation_dataset = None, None
        self.setup_fit_complete = False
        self.setup_test_complete = False

    def prepare_data(self):
        # download data if it lives remote
        ...

    @staticmethod
    def fetch_images_perclass(src:str):
        """ folders in src are the classes """
        # classic behavior
        if os.path.isdir(src):
            classes = [d.name for d in os.scandir(src) if d.is_dir()]
            classes.sort()

            images_perclass = {}
            for subdir in classes:
                files = os.listdir(os.path.join(src, subdir))
                files = sorted([f for f in files if torchvision.datasets.folder.is_image_file(f)])
                images_perclass[subdir] = [os.path.join(src, subdir, i) for i in files]
            return images_perclass

    def limit_images_perclass(self, images_perclass):
        # CLASS MINIMUM CUTTOFF
        images_perclass__minthresh = {label: images for label, images in images_perclass.items() if
                                      len(images) >= self.minimum_images_per_class}
        classes_ignored = sorted(set(images_perclass.keys()) - set(images_perclass__minthresh.keys()))
        self.classes_ignored_from_too_few_samples = [(c, len(images_perclass[c])) for c in classes_ignored]
        self.classes = sorted(images_perclass__minthresh.keys())

        # CLASS MAXIMUM LIMITING
        if self.maximum_images_per_class:
            assert self.maximum_images_per_class >= self.minimum_images_per_class
            images_perclass__maxlimited = {
                label: sorted(random.sample(images, self.maximum_images_per_class)) \
                    if self.maximum_images_per_class < len(images) \
                    else images \
                for label, images in images_perclass__minthresh.items()
            }
            images_perclass__final = images_perclass__maxlimited
            self.classes_limited_from_too_many_samples = \
                [c for c in self.classes if len(images_perclass__maxlimited[c]) < len(images_perclass__minthresh[c])]
        else:
            images_perclass__final = images_perclass__minthresh
            self.classes_limited_from_too_many_samples = None

        # sort perclass images internally, just because its nice.
        images_perclass__final = {label: sorted(images) for label, images in images_perclass__final.items()}
        return images_perclass__final

    def split(self, images_perclass, ratio1, ratio2, seed=None):
        assert ratio1+ratio2 == 100, f'ratio1:ratio2 must sum to 100, instead got {ratio1}:{ratio2} (total: {ratio1+ratio2})'
        total_image_count = sum(len(vals) for vals in images_perclass.values())
        d1_perclass = {}
        d2_perclass = {}
        for class_label, images in images_perclass.items():
            #1) determine output lengths
            d1_len = int(ratio1*len(images)/100+0.5)
            if d1_len == len(images) and self.minimum_images_per_class>1:
            # make sure that at least one image gets put in d2
                d1_len -= 1

            #2) split images as per distribution
            if seed:
                random.seed(seed)
            d1_images = random.sample(images, d1_len)
            d2_images = sorted(list(set(images)-set(d1_images)))
            assert len(d1_images)+len(d2_images) == len(images)

            #3) put images into perclass_sets at the right class
            d1_perclass[class_label] = d1_images
            d2_perclass[class_label] = d2_images

        #4) create and return new datasets
        assert set(d1_perclass.keys()) == set(d2_perclass.keys()), \
            f'd1-d2_classes:{set(d1_perclass.keys())-set(d2_perclass.keys())}, d2-d1_classes:{set(d2_perclass.keys())-set(d1_perclass.keys())}' # possibly fails due to edge case thresholding?
        assert set(self.classes) == set(d1_perclass.keys()), f'd0-d1_classes:{set(self.classes)-set(d1_perclass.keys())}'
        training_samples = [(src, self.classes.index(label)) for label,sources in d1_perclass.items() for src in sources]
        validation_samples = [(src, self.classes.index(label)) for label,sources in d2_perclass.items() for src in sources]

        training_transform = transforms.Compose(self.base_transforms)
        validation_transform = training_transform
        training_dataset = ImageDatasetWithSource(training_samples, self.classes, transform=training_transform)
        validation_dataset = ImageDatasetWithSource(validation_samples, self.classes, transform=validation_transform)

        assert len(training_dataset)+len(validation_dataset) == total_image_count  # make sure we don't lose any images somewhere
        return training_dataset, validation_dataset

    def setup_fit(self):
        images_perclass = self.fetch_images_perclass(self.trainval_dir)
        images_perclass = self.limit_images_perclass(images_perclass)
        self.classes = sorted(images_perclass.keys())
        self.training_dataset, self.validation_dataset = self.split(images_perclass, 80,20, seed=None)
        self.setup_fit_complete = True

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" and not self.setup_fit_complete:
            self.setup_fit()
            self.setup_fit_complete = True

        # Assign test dataset for use in dataloader(s)
        if stage == "test" and not self.setup_test_complete:
            self.testing_dataset = ...
            self.setup_test_complete = True

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

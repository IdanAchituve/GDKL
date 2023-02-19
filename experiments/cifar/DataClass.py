import logging
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from experiments.utils import set_logger

set_logger()

class CIFARSubSet(Subset):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class CIFARData:
    """Source: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

    Create train, valid, test iterators for CIFAR-10 [1].
    [1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4

    """

    def __init__(self, dataset='cifar10', data_dir=None):
        if data_dir is None:
            data_dir = Path(os.getcwd()) / "datasets"

        self.dataset = dataset
        self.data_dir = data_dir

    def get_datasets(self, val_pct=0.1, train_pct=1.0, return_index=False, augment_train=True,
                     resize_size=32, random_state=42):

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )

        # define transforms
        valid_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            normalize,
        ])
        if augment_train:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = valid_transform

        CIFAR = CIFAR10 if self.dataset == 'cifar10' else CIFAR100

        # load the dataset
        train_dataset = CIFAR(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        clean_train_dataset = CIFAR(  # no augmentations
            root=self.data_dir,
            train=True,
            download=True,
            transform=valid_transform,
        )

        # valid
        valid_dataset = CIFAR(
            root=self.data_dir,
            train=True,
            download=True,
            transform=valid_transform,  # no aug.
        )
        # test
        test_dataset = CIFAR(
            root=self.data_dir,
            train=False,
            download=True,
            transform=valid_transform,
        )

        # take validation set from train
        indices = np.arange(len(train_dataset.targets))
        targets = np.array(train_dataset.targets)
        if val_pct > 0.0:
            train_indices, val_indices = train_test_split(indices, test_size=val_pct, stratify=targets,
                                                          random_state=random_state)
            valid_dataset = Subset(valid_dataset, indices=val_indices)
        else:
            train_indices = indices
            valid_dataset = None
            val_indices = None

        # sample from training set
        if train_pct < 1.0:
            targets = targets[train_indices]
            train_indices, _ = train_test_split(train_indices, train_size=train_pct, stratify=targets,
                                                random_state=random_state)

        SubSetClass = CIFARSubSet if return_index else Subset
        train_dataset = SubSetClass(train_dataset, indices=train_indices)
        clean_train_dataset = SubSetClass(clean_train_dataset, indices=train_indices)
        num_train = len(train_dataset)
        num_test = len(test_dataset)

        logging.info(
            f"\nTrain size = {num_train}, Sum (Err. Check) = {np.sum(train_indices)}"
            f"\nTest size = {num_test}"
        )
        if val_pct > 0.0:
            num_val = len(valid_dataset)
            logging.info(f"\nValidation size = {num_val}, Sum (Err. Check) = {np.sum(val_indices)}")

        return train_dataset, clean_train_dataset, valid_dataset, test_dataset

    def get_loaders(self, val_pct=0.1, train_pct=1.0, return_index=False, batch_size=256, test_batch_size=512,
                    shuffle_train=False, augment_train=True, resize_size=32, num_workers=4, pin_memory=True,
                    random_state=42):

        train_dataset, clean_train_dataset, valid_dataset, test_dataset = \
            self.get_datasets(val_pct, train_pct, return_index, augment_train, resize_size, random_state)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = None
        if val_pct > 0.0:
            val_loader = DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        if return_index:
            ordered_train_loader = DataLoader(
                clean_train_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, ordered_train_loader, val_loader, test_loader

        return train_loader, val_loader, test_loader
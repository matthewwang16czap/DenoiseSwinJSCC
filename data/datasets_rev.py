from torch.utils.data import Dataset
from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler

NUM_DATASET_WORKERS = 8


# ================================================================
# CIFAR10 Wrapper
# ================================================================
class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self.len]

    def __len__(self):
        return self.len * 10


# ================================================================
# Generic Image Dataset with Resize/Crop to config.image_dims
# Used for DIV2K / Kodak / custom folders
# ================================================================
class ImageFolderWithResize(Dataset):
    def __init__(self, dirs, image_dims, train):
        """
        dirs: a list of directories
        image_dims: (C,H,W)
        train: True = Random crop, False = Center crop or resize
        """
        self.paths = []
        for d in dirs:
            self.paths += glob(os.path.join(d, "*.png"))
            self.paths += glob(os.path.join(d, "*.jpg"))
        self.paths.sort()

        _, H, W = image_dims

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((H, W), scale=(0.7, 1.0)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(min(H, W)),  # preserves aspect ratio
                    transforms.CenterCrop((H, W)),  # exact final size
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ================================================================
# Worker Seed Function
# ================================================================
def worker_init_fn_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ================================================================
# Dataset Factory
# ================================================================
def get_dataset(name, data_dirs, config, train):
    """
    name: "DIV2K", "CIFAR10", "KODAK", or folder dataset name
    data_dirs: list of folders
    config: config object containing image_dims
    train: True/False
    """
    name = name.upper()

    # ------------------------------------------------------------
    # CIFAR10
    # ------------------------------------------------------------
    if name == "CIFAR10":
        if train:
            if config.norm:
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
        else:
            if config.norm:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )

        dataset = datasets.CIFAR10(
            root=data_dirs,
            train=train,
            transform=transform,
            download=False,
        )

        # repeat 10× for training (matches original logic)
        if train:
            dataset = CIFAR10(dataset)

        return dataset

    # ------------------------------------------------------------
    # Kodak, DIV2K, or any folder dataset
    # ------------------------------------------------------------
    return ImageFolderWithResize(
        dirs=data_dirs, image_dims=config.image_dims, train=train
    )


# ================================================================
# Main Loader (rewritten cleanly)
# ================================================================
def get_loader(args, config, rank=None, world_size=None):

    # ------------------------ Train Dataset ------------------------ #
    train_dataset = get_dataset(
        name=args.trainset, data_dirs=config.train_data_dir, config=config, train=True
    )

    # ------------------------ Test Dataset ------------------------- #
    test_dataset = get_dataset(
        name=args.testset, data_dirs=config.test_data_dir, config=config, train=False
    )

    # ------------------------ Sampler (DDP) ------------------------ #
    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # ------------------------ Train Loader ------------------------- #
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn_seed,
        drop_last=True,
        persistent_workers=False,
    )

    # ------------------------ Test Loader -------------------------- #
    # Large batch only for CIFAR10 test
    test_batch = 1024 if args.testset.upper() == "CIFAR10" else 1

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, train_sampler

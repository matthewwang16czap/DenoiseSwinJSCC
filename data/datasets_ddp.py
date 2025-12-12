from torch.utils.data import Dataset
from PIL import Image
import random

# import cv2
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, DistributedSampler

NUM_DATASET_WORKERS = 8  # Further reduced for DDP stability
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, "*.jpg"))
            self.imgs += glob(os.path.join(dir, "*.png"))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(
        self,
    ):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            # transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, "*.jpg"))
            self.imgs += glob(os.path.join(dir, "*.png"))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert("RGB")
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop((self.im_width, self.im_height)),
                transforms.ToTensor(),
            ]
        )
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)

    def __len__(self):
        return self.len * 10


# ---------------- Worker Seed ---------------- #
def worker_init_fn_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(args, config, rank=None, world_size=None):
    # ---------------- Dataset ---------------- #
    if args.trainset == "DIV2K":
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)

    elif args.trainset == "CIFAR10":
        dataset_ = datasets.CIFAR10
        if config.norm:
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            )
            transform_test = transforms.Compose([transforms.ToTensor()])

        train_dataset = dataset_(
            root=config.train_data_dir,
            train=True,
            transform=transform_train,
            download=False,
        )
        test_dataset = dataset_(
            root=config.test_data_dir,
            train=False,
            transform=transform_test,
            download=False,
        )
        train_dataset = CIFAR10(train_dataset)

    else:
        train_dataset = Datasets(config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)

    # ---------------- Sampler ---------------- #
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

    # ---------------- DataLoaders ---------------- #
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=True,  # Set to False to avoid memory issues with DDP
        worker_init_fn=worker_init_fn_seed,
        drop_last=True,
        persistent_workers=False,  # Set to False to avoid hangs
    )

    if args.trainset == "CIFAR10":
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, train_sampler

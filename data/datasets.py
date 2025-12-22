from torch.utils.data import Dataset
from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler
import cv2
from data.letterbox import LetterBox

NUM_DATASET_WORKERS = 8


class ImageDataset(Dataset):
    def __init__(self, dirs, image_dims):
        """
        dirs: paths to COCO train2017 or val2017
        image_dims: (C, H, W)
        """
        self.paths = []
        for d in dirs:
            self.paths += glob(os.path.join(d, "*.png"))
            self.paths += glob(os.path.join(d, "*.jpg"))
        self.paths.sort()
        assert len(self.paths) > 0, f"No images found in {dirs}"

        C, H, W = image_dims

        self.letterbox = LetterBox(
            new_shape=(H, W),
            auto=False,
            scale_fill=False,
            scaleup=True,
            center=True,
            padding_value=0,
            interpolation=cv2.INTER_CUBIC,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # ---------------- Load ---------------- #
        img = cv2.imread(self.paths[idx])  # BGR, uint8
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.paths[idx]}")

        # ---------------- Letterbox ---------------- #
        img, valid = self.letterbox(image=img)  # still BGR, uint8

        # ---------------- Convert to RGB ---------------- #
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------------- To Tensor ---------------- #
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        img = img.float() / 255.0  # [0,1]
        valid = torch.from_numpy(valid).float()

        return img, valid


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
        img = self.transform(img)
        return img, torch.ones_like(img).float()


def worker_init_fn_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(name, data_dirs, config, train):
    """
    name: "DIV2K", "KODAK", or folder dataset name
    data_dirs: list of folders
    config: config object containing image_dims
    """
    name = name.upper()
    return ImageFolderWithResize(
        dirs=data_dirs, image_dims=config.image_dims, train=train
    )


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
    test_batch = 1

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, train_sampler

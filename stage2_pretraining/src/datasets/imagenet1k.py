# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for sonar flat-folder pretraining
#
# Drop-in replacement for imagenet1k.py
# Reads images from a flat folder — no class subfolders needed
# because I-JEPA pretraining is self-supervised (no labels required)

import os
import torch
from PIL import Image
from logging import getLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')


class FlatFolderDataset(Dataset):
    """
    Reads all images from a flat directory — no class subfolders.
    Used for self-supervised pretraining where labels are not needed.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]
        self.samples.sort()
        logger.info(f'FlatFolderDataset: {len(self.samples)} images from {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert('RGB')   # convert grayscale → RGB for ViT
        if self.transform is not None:
            img = self.transform(img)
        return img, 0   # label=0 dummy — not used in I-JEPA training


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=4,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    """
    Drop-in replacement for original make_imagenet1k.
    Uses FlatFolderDataset instead of ImageNet class.
    Same function signature — main.py calls this without modification.
    """
    # Use root_path directly — ignore image_folder suffix logic
    data_path = root_path
    logger.info(f'Sonar flat dataset path: {data_path}')

    dataset = FlatFolderDataset(root=data_path, transform=transform)
    logger.info(f'Sonar dataset created: {len(dataset)} images')

    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )

    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )

    logger.info('Sonar data loader created')
    return dataset, data_loader, dist_sampler

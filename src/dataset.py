# dataset.py — Dataset & DataLoader Utils

import os
import glob
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import src.config as cfg


# Dataset class
# ---------------------------------------------------------------------------
class ImageFolderFlat(Dataset):
    """Loads all images found under DATA_ROOT. Returns clean tensors in [-1, 1]."""
    # Corruption is applied on-the-fly in the training loop (not here), so that dataset remains pure.
    EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        root_dir: str,
        img_size: int = cfg.IMG_SIZE,
        limit: Optional[int] = cfg.DATASET_LIMIT,
        augment: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment  = augment

        # Collect all image paths
        all_paths: list[str] = []
        for ext in self.EXTENSIONS:
            all_paths += glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
            all_paths += glob.glob(os.path.join(root_dir, "**", f"*{ext.upper()}"), recursive=True)
        all_paths = sorted(set(all_paths))

        if not all_paths:
            raise FileNotFoundError(f"No images found under '{root_dir}'")

        if limit is not None:
            all_paths = all_paths[:limit]

        self.paths = all_paths
        print(f"[Dataset] Found {len(self.paths):,} images in '{root_dir}'")

        # Transforms
        base_tf = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [-1, 1]
        ]
        if augment:
            aug_tf = [transforms.RandomHorizontalFlip()]
        else:
            aug_tf = []

        self.transform = transforms.Compose(aug_tf + base_tf)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return torch.Tensor(self.transform(img))



# Helper: build train, val DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(
    root_dir: str = cfg.DATA_ROOT,
    img_size: int = cfg.IMG_SIZE,
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
    train_split: float = cfg.TRAIN_SPLIT,
    limit: Optional[int] = cfg.DATASET_LIMIT,
    seed: int = cfg.SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader)."""
    full_dataset = ImageFolderFlat(root_dir, img_size=img_size, limit=limit, augment=True)

    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # Disable augmentation for validation split by overriding transform
    val_ds.dataset = _clone_dataset_no_aug(full_dataset, img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[DataLoader] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    return train_loader, val_loader


def _clone_dataset_no_aug(original: ImageFolderFlat, img_size: int) -> ImageFolderFlat:
    """Create a copy with augmentation disabled (for val/test sets)."""
    import copy
    ds = copy.copy(original)
    ds.augment = False
    ds.transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return ds

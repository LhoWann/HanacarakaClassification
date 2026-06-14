from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps

from config import Config, CLASS_TO_IDX, CLASSES
from tqdm.auto import tqdm


class AksaraJawaDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        valid_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root tidak ada: {self.root}")

        self.samples: list[tuple[Path, int]] = []
        for cls in CLASSES:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in tqdm(list(cls_dir.iterdir()), desc=f"Index {cls}", leave=False):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, CLASS_TO_IDX[cls]))

        if not self.samples:
            raise RuntimeError(f"Tidak ada gambar valid di {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        img = ImageOps.invert(img)
        img = ImageOps.pad(img, (max(img.size), max(img.size)), color=0, centering=(0.5, 0.5))

        if self.transform:
            img = self.transform(img)

        return img, label


def build_transforms(cfg: Config) -> tuple[Callable, Callable]:
    common_end = [
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ]

    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomAffine(
            degrees=cfg.aug_rotation_deg,
            translate=(cfg.aug_translate, cfg.aug_translate),
            scale=cfg.aug_scale,
            shear=cfg.aug_shear_deg,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=cfg.aug_brightness,
            contrast=cfg.aug_contrast,
        ),
        *common_end,
        transforms.RandomErasing(
            p=cfg.aug_erasing_prob,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        *common_end,
    ])

    return train_tf, eval_tf


def build_dataloaders(
    cfg: Config,
) -> tuple[dict[str, DataLoader], dict[str, AksaraJawaDataset]]:
    train_tf, eval_tf = build_transforms(cfg)

    datasets = {
        "train": AksaraJawaDataset(cfg.data_dir / "train", transform=train_tf),
        "val":   AksaraJawaDataset(cfg.data_dir / "val",   transform=eval_tf),
        "test":  AksaraJawaDataset(cfg.data_dir / "test",  transform=eval_tf),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        ),
    }

    return loaders, datasets
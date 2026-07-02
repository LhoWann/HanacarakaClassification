from dataclasses import dataclass, field
from pathlib import Path
import random
import numpy as np
import torch


CLASSES = [
    "ha", "na", "ca", "ra", "ka",
    "da", "ta", "sa", "wa", "la",
    "pa", "dha", "ja", "ya", "nya",
    "ma", "ga", "ba", "tha", "nga",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


@dataclass
class Config:
    data_dir:        Path = Path("dataset")
    artifact_dir:    Path = Path("artifacts")
    checkpoint_name: str  = "best_model.pt"

    image_size:   int  = 96
    in_channels:  int  = 1
    num_workers:  int  = 0
    pin_memory:   bool = True

    norm_mean: tuple = (0.10,)
    norm_std:  tuple = (0.25,)

    batch_size:      int   = 64
    epochs:          int   = 100
    learning_rate:   float = 1.2e-3
    weight_decay:    float = 1e-4
    warmup_epochs:   int   = 2
    label_smoothing: float = 0.05
    grad_clip_norm:  float = 1.0

    aug_rotation_deg: float = 8.0
    aug_translate:    float = 0.08
    aug_scale:        tuple = (0.90, 1.10)
    aug_shear_deg:    float = 5.0
    aug_brightness:   float = 0.10
    aug_contrast:     float = 0.10
    aug_erasing_prob: float = 0.15

    dropout:                 float = 0.4
    early_stopping_patience: int   = 20

    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config() -> Config:
    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    return cfg
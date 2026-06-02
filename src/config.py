from dataclasses import dataclass, field
from pathlib import Path


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
    # Paths
    data_dir:        Path = Path("dataset")
    artifact_dir:    Path = Path("artifacts")
    checkpoint_name: str  = "best_model.pt"

    # Data
    image_size:   int = 96        # Naik dari 64 → 96. Stroke Aksara Jawa sangat tipis;
                                  # 64x64 menghilangkan terlalu banyak detail saat resize
                                  # dari median 345x300 px.
    in_channels:  int = 1         # grayscale: warna tidak informatif untuk handwriting
    num_workers:  int = 0         # Windows: 0 lebih stabil; Linux: bisa 4-8
    pin_memory:   bool = True

    # Stats SETELAH invert + autocontrast preprocessing.
    # Dengan invert: background hitam (0.0), stroke putih (1.0)
    # → mean rendah karena background dominan (~95% piksel)
    # Perhitungan std: untuk distribusi di mana ~90% piksel = 0 dan ~10% piksel = 1,
    #   std = sqrt(p × (1-p)) ≈ sqrt(0.10 × 0.90) ≈ 0.30
    # Dengan std=0.25, range normalisasi: [(-0.10)/0.25, (0.90)/0.25] = [-0.40, 3.60]
    # (lebih terkontrol dibanding std=0.20 yang memberi range [-0.50, 4.50])
    norm_mean: tuple = (0.10,)
    norm_std:  tuple = (0.25,)

    # Training
    batch_size:    int   = 64
    epochs:        int   = 50        # Naik dari 40 — lebih banyak iterasi untuk konvergensi stabil
    # LR 1.2e-3 (turun dari 2e-3): osilasi val_acc pada epoch 12-15 (0.54→0.48→0.67→0.60)
    # mengindikasikan LR 2e-3 terlalu agresif di fase cosine decay.
    learning_rate: float = 1.2e-3
    weight_decay:  float = 1e-4      # Naik dari 5e-5 — regularisasi lebih kuat untuk generalisasi
    warmup_epochs: int   = 2         # Naik dari 1 — warmup lebih panjang stabilkan BN di awal
    # Label smoothing 0.05: cegah model terlalu confident pada kelas mudah (ha/la F1 rendah).
    # Dengan ε=0.05: label [1,0,...,0] → [0.9525, 0.0025,...,0.0025]
    label_smoothing: float = 0.05
    grad_clip_norm:  float = 1.0

    # Augmentasi (training only)
    aug_rotation_deg:   float = 8.0
    aug_translate:      float = 0.08
    aug_scale:          tuple = (0.90, 1.10)
    aug_shear_deg:      float = 5.0
    aug_brightness:     float = 0.10
    aug_contrast:       float = 0.10
    # RandomErasing: secara acak menghapus patch piksel → model tidak bergantung pada
    # satu bagian karakter saja. Sangat efektif untuk handwriting classification.
    aug_erasing_prob:   float = 0.15
    # CATATAN: TIDAK pakai HorizontalFlip / VerticalFlip
    # → Aksara Jawa tidak simetris; flip akan merusak label semantik.

    # Regularisasi & Early Stopping
    dropout:                 float = 0.4   # Naik sedikit dari 0.3, sesuai ImprovedCNN two-layer FC
    early_stopping_patience: int   = 12   # Naik dari 10 — beri ruang untuk konvergensi lebih lambat

    # Reproducibility
    seed: int = 42


def get_config() -> Config:
    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    return cfg

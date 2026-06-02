# Klasifikasi Aksara Jawa (Hanacaraka) dengan CNN PyTorch

Proyek tugas mata kuliah **Pembelajaran Mesin Semester 4** — membangun Convolutional Neural Network (CNN) untuk mengklasifikasikan 20 karakter aksara Jawa (*Hanacaraka*) dari gambar tulisan tangan.

---

## Daftar Isi

1. [Latar Belakang](#latar-belakang)
2. [Dataset](#dataset)
3. [Struktur Proyek](#struktur-proyek)
4. [Instalasi](#instalasi)
5. [Cara Penggunaan](#cara-penggunaan)
6. [Arsitektur Model](#arsitektur-model)
7. [Preprocessing & Augmentasi](#preprocessing--augmentasi)
8. [Strategi Training](#strategi-training)
9. [Hasil Eksperimen](#hasil-eksperimen)
10. [Explainability (Grad-CAM)](#explainability-grad-cam)
11. [Konfigurasi Hyperparameter](#konfigurasi-hyperparameter)

---

## Latar Belakang

Aksara Jawa (*Hanacaraka*) adalah sistem tulisan tradisional yang digunakan untuk menulis bahasa Jawa. Terdapat 20 karakter dasar yang disebut *carakan*:

```
ha  na  ca  ra  ka
da  ta  sa  wa  la
pa  dha ja  ya  nya
ma  ga  ba  tha nga
```

Tantangan utama klasifikasi aksara ini:
- Beberapa karakter secara visual sangat mirip (contoh: `ha` vs `na`, `la` vs `wa`)
- Variasi tulisan tangan antar-penulis sangat tinggi
- Stroke tipis rentan hilang saat resize gambar

---

## Dataset

### Statistik

| Split | Jumlah Gambar |
|-------|--------------|
| Train | 3,898 |
| Val   | 836 |
| Test  | 848 |
| **Total** | **5,582** |

- **20 kelas** (satu per karakter Hanacaraka)
- **Format input**: 96×96 grayscale (setelah preprocessing)
- **Distribusi kelas**: relatif seimbang, ~280 gambar per kelas

### Sumber Data

Dataset diunduh dari dua sumber:

1. **GitHub vzrenggamani** (`aksarajawa-hanacaraka`) — sumber utama, gambar tulisan tangan *in-distribution*
2. **Roboflow fawwaz** (opsional) — dataset object-detection yang di-crop per bounding box menjadi gambar klasifikasi individual

### Download Dataset Otomatis

```bash
cd src
python download_dataset.py
```

Untuk menggunakan sumber Roboflow (opsional):

```bash
python download_dataset.py --roboflow-key YOUR_API_KEY
```

Atau skip Roboflow:

```bash
python download_dataset.py --skip-roboflow
```

Script ini akan:
1. Mengunduh dari GitHub secara otomatis
2. Menghapus duplikat (berdasarkan MD5 hash)
3. Memvalidasi integritas setiap gambar
4. Membagi dataset ke train/val/test (70%/15%/15%) dengan stratified split

**Struktur dataset setelah download:**

```
dataset/
├── train/
│   ├── ha/   (ha_0001.jpg, ...)
│   ├── na/
│   └── ...   (20 folder kelas)
├── val/
│   └── ...
├── test/
│   └── ...
└── raw/      (raw download — di-gitignore)
```

---

## Struktur Proyek

```
KlasifikasiHanacaraka/
│
├── src/
│   ├── config.py           # Hyperparameter terpusat (dataclass Config)
│   ├── model.py            # SimpleCNN (baseline) + ImprovedCNN
│   ├── dataset.py          # PyTorch Dataset, preprocessing, augmentasi
│   ├── engine.py           # Training loop, evaluasi, early stopping
│   ├── explainability.py   # Grad-CAM implementation
│   └── download_dataset.py # Script download & persiapan dataset
│
├── train_and_evaluate.ipynb  # Notebook utama: training + evaluasi lengkap
├── aksara_jawa_eda.ipynb     # Exploratory Data Analysis (EDA)
│
├── artifacts/              # Checkpoint model (auto-created saat training)
│   └── best_model.pt
│
├── dataset/                # Dataset (di-gitignore kecuali struktur folder)
├── .venv/                  # Virtual environment Python
└── .gitignore
```

---

## Instalasi

### Prasyarat

- Python 3.11+
- (Opsional) GPU NVIDIA dengan CUDA untuk training lebih cepat

### Setup Virtual Environment

```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Aktivasi (Windows CMD)
.venv\Scripts\activate.bat

# Aktivasi (Linux/macOS)
source .venv/bin/activate
```

### Install Dependensi

```bash
# Dengan GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio

# Library lainnya
pip install scikit-learn pillow tqdm matplotlib requests pyyaml jupyter
```

### Dependensi Utama

| Paket | Kegunaan |
|-------|----------|
| `torch` / `torchvision` | Framework deep learning, model, transforms |
| `scikit-learn` | Metrics (accuracy, F1, confusion matrix), train-test split |
| `Pillow` | Preprocessing gambar (autocontrast, invert, padding) |
| `tqdm` | Progress bar training |
| `matplotlib` | Visualisasi kurva training, Grad-CAM overlay |
| `requests` | Download dataset dari GitHub |

---

## Cara Penggunaan

### 1. Persiapan Dataset

```bash
cd src
python download_dataset.py --skip-roboflow
```

### 2. Training via Notebook (Direkomendasikan)

Buka dan jalankan `train_and_evaluate.ipynb`. Notebook ini berisi:
- Inisialisasi konfigurasi
- Build DataLoader
- Inisialisasi model (SimpleCNN atau ImprovedCNN)
- Training loop dengan monitoring loss/accuracy
- Evaluasi komprehensif pada test set
- Visualisasi confusion matrix
- Grad-CAM untuk interpretasi

### 3. Training via Script Python

```python
import sys
sys.path.insert(0, "src")

import torch
from config import get_config, CLASSES
from dataset import build_dataloaders
from model import ImprovedCNN
from engine import fit, detailed_evaluation

cfg = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaders, _ = build_dataloaders(cfg)

model = ImprovedCNN(
    num_classes=20,
    in_channels=cfg.in_channels,
    dropout=cfg.dropout,
).to(device)

metrics, ckpt_path = fit(model, loaders, cfg, device)

# Evaluasi pada test set
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
results = detailed_evaluation(model, loaders["test"], device, CLASSES)
print(results["classification_report"])
```

### 4. EDA

Buka `aksara_jawa_eda.ipynb` untuk analisis distribusi dataset, visualisasi sampel per kelas, dan statistik ukuran gambar original.

---

## Arsitektur Model

### SimpleCNN (Baseline)

```
Input: (B, 1, 96, 96)

Block 1: Conv2d(1→32,  3×3, pad=1) → BN2d(32)  → ReLU → MaxPool(2×2)  →  (B, 32,  48, 48)
Block 2: Conv2d(32→64, 3×3, pad=1) → BN2d(64)  → ReLU → MaxPool(2×2)  →  (B, 64,  24, 24)
Block 3: Conv2d(64→128,3×3, pad=1) → BN2d(128) → ReLU → MaxPool(2×2)  →  (B, 128, 12, 12)
Block 4: Conv2d(128→256,3×3,pad=1) → BN2d(256) → ReLU                 →  (B, 256, 12, 12)
         AdaptiveAvgPool2d(1×1)                                         →  (B, 256,  1,  1)

Classifier:
  Flatten → Dropout(0.3) → Linear(256 → 20)

Output: (B, 20)  ← raw logits
```

**Perhitungan Parameter:**

| Layer | Formula | Parameter |
|-------|---------|-----------|
| Block 1 Conv | 3×3×1×32 | 288 |
| Block 1 BN | 2×32 | 64 |
| Block 2 Conv | 3×3×32×64 | 18,432 |
| Block 2 BN | 2×64 | 128 |
| Block 3 Conv | 3×3×64×128 | 73,728 |
| Block 3 BN | 2×128 | 256 |
| Block 4 Conv | 3×3×128×256 | 294,912 |
| Block 4 BN | 2×256 | 512 |
| FC Linear | 256×20 + 20 | 5,140 |
| **Total** | | **393,460** |

---

### ImprovedCNN

Feature extraction identik dengan SimpleCNN, perbedaan hanya di classifier head:

```
...
AdaptiveAvgPool2d(1×1) → (B, 256, 1, 1)

Classifier:
  Flatten
  Dropout(0.4)
  Linear(256 → 128) → BatchNorm1d(128) → ReLU
  Dropout(0.2)
  Linear(128 → 20)

Output: (B, 20)
```

**Mengapa two-layer FC head lebih baik?**

- Single layer `Linear(256→20)` memaksa mapping langsung dari fitur spasial ke keputusan kelas tanpa ruang untuk kombinasi non-linear
- Hidden layer 128 memberi CNN ruang untuk membentuk representasi intermediate yang lebih kaya sebelum classification
- `BatchNorm1d` setelah FC pertama menstabilkan distribusi hidden representation antar batch

**Perhitungan Parameter ImprovedCNN:**

| Layer | Formula | Parameter |
|-------|---------|-----------|
| Conv layers (sama) | — | 388,320 |
| FC1 Linear | 256×128 + 128 | 33,024 |
| FC1 BN1d | 2×128 | 256 |
| FC2 Linear | 128×20 + 20 | 2,580 |
| **Total** | | **424,180** |

---

### Rumus Dasar CNN

$$O_{\text{conv}} = \left\lfloor\frac{I - K + 2P}{S}\right\rfloor + 1 \qquad O_{\text{pool}} = \left\lfloor\frac{I - K}{S}\right\rfloor + 1$$

| Komponen | Rumus Parameter |
|----------|----------------|
| Conv2d (`bias=False`) | $K \times K \times C_{in} \times C_{out}$ |
| BatchNorm | $2 \times C$ (gamma + beta) |
| Linear | $C_{in} \times C_{out} + C_{out}$ |

### Keputusan Desain

| Komponen | Pilihan | Alasan |
|----------|---------|--------|
| `bias=False` di Conv | Hemat parameter | BatchNorm sudah punya shift (β) yang setara bias |
| `ReLU(inplace=False)` | Non-inplace | `inplace=True` merusak tensor yang di-hook Grad-CAM → `RuntimeError` |
| `AdaptiveAvgPool` | Output 1×1 | Robust terhadap perubahan input size |
| Weight init Conv | Kaiming `fan_out` | Sesuai karakteristik aktivasi ReLU |
| Weight init Linear | Xavier | Output logit lebih stabil dengan Xavier |

---

## Preprocessing & Augmentasi

### Pipeline Preprocessing (Semua Split)

Dieksekusi di dalam `AksaraJawaDataset.__getitem__` menggunakan PIL, *sebelum* transform:

**1. Konversi ke Grayscale** — `img.convert("L")`

Warna tidak informatif untuk handwriting; mengurangi input dari 3 channel menjadi 1 channel.

**2. AutoContrast** — `ImageOps.autocontrast(img, cutoff=2)`

Stretch dynamic range ke [0, 255]. Tanpa ini, stroke tipis dan light-gray akan hilang setelah resize.

**3. Invert** — `ImageOps.invert(img)`

Background hitam (0), stroke putih (255). "Fitur positif" (stroke) di-amplify oleh ReLU, bukan ditekan.

**4. Square Padding** — `ImageOps.pad(img, ...)`

Preserve aspect ratio sebelum resize. Tanpa ini, karakter landscape akan ter-squish dan stroke terdistorsi.

### Normalisasi

```python
norm_mean = (0.10,)   # background dominan (~90% piksel = 0) → mean rendah
norm_std  = (0.25,)   # std diestimasi: sqrt(0.10 * 0.90) ≈ 0.30, dibulatkan 0.25
                      # range: [(-0.10)/0.25, (0.90)/0.25] = [-0.40, 3.60]
```

### Augmentasi Training

| Augmentasi | Parameter | Alasan |
|------------|-----------|--------|
| `RandomAffine` | rotation=8°, translate=8%, scale=90–110%, shear=5° | Simulasi variasi cara menulis natural |
| `ColorJitter` | brightness=0.10, contrast=0.10 | Variasi ketebalan stroke / kualitas scan |
| `RandomErasing` | p=0.15, scale=(2%, 15%), value=0 | Model belajar dari seluruh karakter, bukan satu stroke khas |

> **Catatan penting:** `HorizontalFlip` dan `VerticalFlip` **tidak digunakan** karena aksara Jawa tidak simetris — flip akan menghasilkan karakter yang bukan aksara valid dan merusak semantik label.

---

## Strategi Training

### Optimizer & Loss

| Komponen | Pilihan | Nilai |
|----------|---------|-------|
| Optimizer | AdamW | lr=1.2e-3, weight_decay=1e-4 |
| Loss | CrossEntropyLoss | label_smoothing=0.05 |
| Gradient clipping | `clip_grad_norm_` | max_norm=1.0 |

**Label Smoothing (ε=0.05):**

Mengubah label one-hot `[1, 0, ..., 0]` menjadi `[0.9525, 0.0025, ..., 0.0025]`. Mencegah model terlalu *overconfident* pada kelas mudah dan meningkatkan generalisasi pada kelas sulit (`ha`, `la`).

### Learning Rate Schedule

**Linear Warmup (2 epoch) lalu Cosine Annealing (sisa epoch):**

$$\text{lr}(t) = \frac{t}{t_{\text{warmup}}} \qquad (t < t_{\text{warmup}})$$

$$\text{lr}(t) = \frac{1}{2}\left(1 + \cos\!\left(\pi \cdot \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right) \qquad (t \geq t_{\text{warmup}})$$

- **Warmup**: mencegah loss spike di iterasi awal ketika BatchNorm belum stabil
- **Cosine Annealing**: lebih baik dari step-decay untuk CNN dengan dataset kecil
- Scheduler di-step **per-batch** (bukan per-epoch) agar warmup bekerja benar di epoch pertama

### Early Stopping & Checkpoint

- Patience: **12 epoch** tanpa peningkatan `val_accuracy`
- Best model disimpan otomatis ke `artifacts/best_model.pt`
- Checkpoint menyimpan: `epoch`, `model_state`, `val_acc`, `val_loss`

### Mixed Precision (AMP)

Aktif otomatis jika GPU tersedia (`device.type == "cuda"`):

```python
scaler = torch.amp.GradScaler(device.type)

with torch.amp.autocast(device_type=device.type):
    logits = model(images)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Keuntungan: ~2x speedup, hemat VRAM ~50%

---

## Hasil Eksperimen

### SimpleCNN (Baseline)

| Metrik | Nilai |
|--------|-------|
| Test Accuracy | **89.98%** |
| F1 Macro | **89.54%** |

**Kelas paling sulit (F1 terendah):**

| Kelas | F1-score |
|-------|----------|
| `ha` | 0.575 |
| `la` | 0.629 |

### ImprovedCNN — Perubahan vs Baseline

| Komponen | SimpleCNN | ImprovedCNN | Alasan perubahan |
|----------|-----------|-------------|-----------------|
| FC Head | 256 → 20 | 256 → 128 → 20 | Representasi intermediate lebih kaya |
| Label Smoothing | 0.0 | 0.05 | Regularisasi pada kelas sulit |
| Learning Rate | 2e-3 | 1.2e-3 | Osilasi val_acc pada epoch 12-15 |
| norm_std | 0.20 | 0.25 | Range normalisasi lebih terkontrol |
| Weight Decay | 5e-5 | 1e-4 | Regularisasi lebih kuat |
| RandomErasing | — | p=0.15 | Invariansi terhadap oklusi parsial |
| Warmup Epochs | 1 | 2 | BatchNorm lebih stabil di awal |
| Total Epochs | 40 | 50 | Konvergensi lebih stabil |
| Dropout | 0.3 | 0.4 / 0.2 | Berlapis sesuai two-layer FC head |

---

## Explainability (Grad-CAM)

### Konsep

Grad-CAM (*Gradient-weighted Class Activation Mapping*) menunjukkan **bagian mana dari gambar yang paling mempengaruhi prediksi model** dengan cara melihat gradien yang mengalir ke feature map di layer konvolusi terakhir.

**Rumus matematis:**

$$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}$$

$$L^c = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right)$$

Di mana:
- `A^k` = feature map ke-k pada layer target
- `y^c` = logit untuk kelas `c`
- `α_k^c` = importance weight: seberapa besar feature map `k` mempengaruhi kelas `c`
- `ReLU` = hanya ambil kontribusi **positif** terhadap kelas target (negatif = kontra-prediksi)

### Penggunaan

```python
import sys
sys.path.insert(0, "src")

from explainability import GradCAM, overlay_cam_on_image

# Target layer: Conv2d terakhir sebelum AdaptiveAvgPool
target_layer = model.features[-3]   # nn.Conv2d(128, 256, ...)

gradcam = GradCAM(model, target_layer)

# input_tensor shape: (1, 1, 96, 96)
cam, pred_class, confidence = gradcam(input_tensor)

# Overlay heatmap pada gambar asli
overlay = overlay_cam_on_image(image_np, cam, alpha=0.4, colormap="jet")

gradcam.remove_hooks()  # Wajib: bersihkan hooks setelah selesai
```

> **Catatan implementasi:** `ReLU(inplace=False)` di arsitektur model **wajib** untuk Grad-CAM. Mode `inplace=True` memodifikasi tensor in-place sehingga backward hooks tidak bisa mengambil nilai aktivasi yang benar → `RuntimeError`.

---

## Konfigurasi Hyperparameter

Semua hyperparameter terpusat di `src/config.py` menggunakan Python dataclass:

```python
import sys
sys.path.insert(0, "src")
from config import get_config

cfg = get_config()
cfg.learning_rate = 1e-3   # Override sebelum training
```

### Daftar Parameter Lengkap

| Parameter | Default | Keterangan |
|-----------|---------|-----------|
| `image_size` | 96 | Ukuran input (px) — stroke aksara hilang pada 64px |
| `in_channels` | 1 | Grayscale; warna tidak informatif untuk handwriting |
| `batch_size` | 64 | — |
| `epochs` | 50 | — |
| `learning_rate` | 1.2e-3 | Diturunkan dari 2e-3 untuk mengurangi osilasi val_acc |
| `weight_decay` | 1e-4 | L2 regularisasi dalam AdamW |
| `warmup_epochs` | 2 | Linear warmup sebelum cosine annealing |
| `label_smoothing` | 0.05 | Mencegah overconfidence pada kelas mudah |
| `grad_clip_norm` | 1.0 | Max norm gradient clipping |
| `dropout` | 0.4 | Regularisasi classifier head |
| `early_stopping_patience` | 12 | Epoch tanpa peningkatan val_acc sebelum berhenti |
| `norm_mean` | (0.10,) | Mean setelah invert: background dominan |
| `norm_std` | (0.25,) | Std terkontrol; range [-0.40, 3.60] |
| `aug_rotation_deg` | 8.0 | Rotasi maksimum (derajat) |
| `aug_translate` | 0.08 | Translasi maksimum (fraksi ukuran gambar) |
| `aug_scale` | (0.90, 1.10) | Scale range augmentasi |
| `aug_shear_deg` | 5.0 | Shear maksimum (derajat) |
| `aug_erasing_prob` | 0.15 | Probabilitas RandomErasing |
| `seed` | 42 | Reproducibility (torch, numpy, dataset split) |
| `num_workers` | 0 | 0 untuk Windows; 4-8 untuk Linux |

---

## Catatan Teknis

- **Windows `num_workers=0`**: multiprocessing PyTorch DataLoader pada Windows menggunakan `spawn` context yang sering timeout — `num_workers=0` lebih stabil untuk development lokal
- **Reproducibility**: seed 42 di-set untuk `torch`, `numpy`, dan `sklearn.train_test_split` — hasil training dapat direproduksi selama hardware sama
- **Dataset lazy loading**: gambar dibuka satu per satu saat diakses (`__getitem__`), bukan di-load semua ke RAM saat inisialisasi — aman untuk dataset besar
- **Deduplication berbasis MD5**: script download menggunakan hash MD5 untuk menghapus gambar duplikat antar sumber, mencegah data leakage train/test

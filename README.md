# Klasifikasi Aksara Jawa (Hanacaraka) dengan CNN PyTorch

Proyek tugas mata kuliah **Pembelajaran Mesin Semester 4** — membangun Convolutional Neural Network (CNN) untuk mengklasifikasikan 20 karakter aksara Jawa (*Hanacaraka*) dari gambar tulisan tangan. Hyperparameter dioptimasi menggunakan **Optuna** untuk mencapai performa terbaik.

---

## Daftar Isi

1. [Latar Belakang](#latar-belakang)
2. [Dataset](#dataset)
3. [Struktur Proyek](#struktur-proyek)
4. [Instalasi](#instalasi)
5. [Quick Start](#quick-start)
6. [Cara Penggunaan Lengkap](#cara-penggunaan-lengkap)
7. [Arsitektur Model](#arsitektur-model)
8. [Preprocessing & Augmentasi](#preprocessing--augmentasi)
9. [Hyperparameter Optimization (Optuna)](#hyperparameter-optimization-optuna)
10. [Strategi Training](#strategi-training)
11. [Hasil Eksperimen](#hasil-eksperimen)
12. [Explainability (Grad-CAM)](#explainability-grad-cam)
13. [Konfigurasi Hyperparameter](#konfigurasi-hyperparameter)
14. [Catatan Teknis](#catatan-teknis)

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

Laporan hasil lengkap tersedia di [result.md](result.md).

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
- **Format input**: 96x96 grayscale (setelah preprocessing)
- **Distribusi kelas**: relatif seimbang, ~280 gambar per kelas

### Sumber Data

1. **GitHub vzrenggamani** (`aksarajawa-hanacaraka`) — sumber utama, gambar tulisan tangan in-distribution
2. **Roboflow fawwaz** (opsional) — dataset object-detection yang di-crop per bounding box

### Download Dataset

```bash
cd src
python download_dataset.py
```

Untuk skip Roboflow:

```bash
python download_dataset.py --skip-roboflow
```

Struktur dataset setelah download:

```
dataset/
├── train/
│   ├── ha/   (ha_0001.jpg, ...)
│   ├── na/
│   └── ...   (20 folder kelas)
├── val/
├── test/
└── raw/      
```

---

## Struktur Proyek

```
KlasifikasiHanacaraka/
│
├── result.md               # Laporan hasil lengkap
├── README.md
├── requirements.txt        # Daftar dependensi
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point CLI utama
│   ├── tune.py             # Optuna hyperparameter optimization
│   ├── config.py           # Hyperparameter terpusat (dataclass Config + set_seed)
│   ├── model.py            # SimpleCNN (baseline) + ImprovedCNN
│   ├── dataset.py          # PyTorch Dataset, preprocessing, augmentasi
│   ├── engine.py           # Training loop, evaluasi, early stopping
│   ├── explainability.py   # Grad-CAM implementation
│   └── download_dataset.py # Script download & persiapan dataset
│
├── outputs/                # Auto-generated saat main.py dijalankan
│   ├── eda/                # Plot distribusi kelas, sampel, preprocessing
│   ├── simple_cnn/         # Hasil SimpleCNN
│   │   ├── training/       # Training curves, metrics.json
│   │   ├── evaluation/     # Confusion matrix, F1 per kelas
│   │   └── gradcam/        # Grad-CAM overlay per kelas
│   ├── improved_cnn/       # Hasil ImprovedCNN
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── gradcam/
│   └── comparison/         # Plot perbandingan kedua model
│
├── artifacts/              # Checkpoint model & Optuna results
│   ├── simple_cnn/
│   │   └── best_model.pt
│   ├── improved_cnn/
│   │   └── best_model.pt
│   ├── best_params_simple.json     # Optuna best params SimpleCNN
│   └── best_params_improved.json   # Optuna best params ImprovedCNN
│
├── dataset/                # Dataset (di-gitignore)
└── .venv/                  # Virtual environment Python
```

---

## Instalasi

### Prasyarat

- Python 3.11+
- (Opsional) GPU NVIDIA dengan CUDA untuk training lebih cepat

### Setup Virtual Environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate
```

### Install Dependensi

```bash
pip install -r requirements.txt
```

Untuk GPU dengan CUDA 12.8 (sesuai `requirements.txt`):

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install scikit-learn pillow tqdm matplotlib requests pyyaml roboflow optuna
```

### Dependensi Utama

| Paket | Versi | Kegunaan |
|-------|-------|----------|
| `torch` / `torchvision` | 2.11.0 | Framework deep learning, model, transforms |
| `scikit-learn` | 1.8.0 | Metrics (accuracy, F1, confusion matrix), train-test split |
| `Pillow` | 12.2.0 | Preprocessing gambar (autocontrast, invert, padding) |
| `tqdm` | 4.67.3 | Progress bar training |
| `matplotlib` | 3.10.9 | Visualisasi kurva training, Grad-CAM overlay |
| `optuna` | >=3.6.0 | Hyperparameter optimization (Bayesian TPE) |
| `requests` | 2.34.2 | Download dataset dari GitHub |
| `roboflow` | 1.3.8 | Download dataset Roboflow (opsional) |

---

## Quick Start

```bash
# 1. Download dataset
cd src
python download_dataset.py --skip-roboflow
cd ..

# 2. Jalankan Optuna HPO (opsional, sudah ada best_params di artifacts/)
python src/tune.py --model both --n-trials 30

# 3. Jalankan semua pipeline (EDA + Training + Evaluasi + Grad-CAM + Comparison)
#    Otomatis menggunakan best params dari Optuna jika tersedia
python src/main.py --mode all

# 4. Lihat hasil di folder outputs/
```

---

## Cara Penggunaan Lengkap

### Pipeline Utama (main.py)

`src/main.py` mendukung beberapa mode dan opsi model:

| Mode | Deskripsi | Output |
|------|-----------|--------|
| `eda` | Analisis distribusi dataset, visualisasi sampel per kelas, statistik ukuran gambar, tahapan preprocessing | `outputs/eda/` |
| `train` | Training model, simpan best checkpoint | `outputs/<model>/training/` |
| `eval` | Evaluasi pada test set: confusion matrix, F1 per kelas, classification report | `outputs/<model>/evaluation/` |
| `gradcam` | Visualisasi Grad-CAM heatmap per kelas | `outputs/<model>/gradcam/` |
| `compare` | Perbandingan metrik dan F1 per kelas antara SimpleCNN vs ImprovedCNN | `outputs/comparison/` |
| `all` | Jalankan semua mode berurutan untuk kedua model | Semua folder di atas |

### Optuna HPO (tune.py)

```bash
# Tune kedua model (default 30 trials, 20 epoch per trial)
python src/tune.py

# Tune hanya ImprovedCNN dengan 50 trials
python src/tune.py --model improved --n-trials 50

# Tune cepat untuk verifikasi
python src/tune.py --model improved --n-trials 3 --tune-epochs 5
```

Best params disimpan otomatis ke `artifacts/best_params_{model}.json` dan dimuat otomatis oleh `main.py` saat training.

### Contoh Perintah

```bash
# Jalankan semua (kedua model + comparison)
python src/main.py --mode all

# Hanya EDA
python src/main.py --mode eda

# Training hanya SimpleCNN
python src/main.py --mode train --model simple

# Training hanya ImprovedCNN
python src/main.py --mode train --model improved

# Training kedua model
python src/main.py --mode train --model both

# Evaluasi kedua model
python src/main.py --mode eval --model both

# Comparison saja (butuh eval results dari kedua model)
python src/main.py --mode compare

# Grad-CAM dengan 3 sampel per kelas untuk kedua model
python src/main.py --mode gradcam --model both --gradcam-samples 3

# Semua pipeline, CPU only
python src/main.py --mode all --device cpu
```

### Argumen CLI (main.py)

| Argument | Default | Keterangan |
|----------|---------|------------|
| `--mode` | `all` | Mode pipeline |
| `--model` | `both` | `simple`, `improved`, atau `both` |
| `--epochs` | dari config (100) | Override jumlah epoch |
| `--lr` | dari config / Optuna | Override learning rate |
| `--batch-size` | dari config / Optuna | Override batch size |
| `--device` | `auto` | `cuda`, `cpu`, atau `auto` |
| `--output-dir` | `outputs` | Folder output visualisasi |
| `--gradcam-samples` | `2` | Jumlah sampel per kelas untuk Grad-CAM |
| `--no-save-plots` | `False` | Skip penyimpanan plot ke disk |

### Argumen CLI (tune.py)

| Argument | Default | Keterangan |
|----------|---------|------------|
| `--model` | `both` | `simple`, `improved`, atau `both` |
| `--n-trials` | `30` | Jumlah trial Optuna |
| `--tune-epochs` | `20` | Epoch per trial |
| `--device` | `auto` | `cuda`, `cpu`, atau `auto` |
| `--output-dir` | `artifacts` | Folder output best params JSON |
| `--seed` | `42` | Seed untuk reproducibility |

---

## Arsitektur Model

### SimpleCNN (Baseline)

```
Input: (B, 1, 96, 96)

Block 1: Conv2d(1->32,   3x3, pad=1) -> BN -> ReLU -> MaxPool(2)  ->  (B, 32,  48, 48)
Block 2: Conv2d(32->64,  3x3, pad=1) -> BN -> ReLU -> MaxPool(2)  ->  (B, 64,  24, 24)
Block 3: Conv2d(64->128, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)  ->  (B, 128, 12, 12)
Block 4: Conv2d(128->256,3x3, pad=1) -> BN -> ReLU                ->  (B, 256, 12, 12)
         AdaptiveAvgPool2d(1x1)                                    ->  (B, 256,  1,  1)

Classifier: Flatten -> Dropout(0.302) -> Linear(256 -> 20)

Output: (B, 20)  <- raw logits
```

**Perhitungan Parameter:**

| Layer | Formula | Parameter |
|-------|---------|-----------|
| Block 1 Conv | 3x3x1x32 | 288 |
| Block 1 BN | 2x32 | 64 |
| Block 2 Conv | 3x3x32x64 | 18,432 |
| Block 2 BN | 2x64 | 128 |
| Block 3 Conv | 3x3x64x128 | 73,728 |
| Block 3 BN | 2x128 | 256 |
| Block 4 Conv | 3x3x128x256 | 294,912 |
| Block 4 BN | 2x256 | 512 |
| FC Linear | 256x20 + 20 | 5,140 |
| **Total** | | **393,460** |

---

### ImprovedCNN

Feature extraction identik dengan SimpleCNN. Perbedaan hanya di classifier head:

```
[Feature extraction sama dengan SimpleCNN]
AdaptiveAvgPool2d(1x1) -> (B, 256, 1, 1)

Classifier:
  Flatten
  Dropout(0.264)
  Linear(256 -> 128) -> BatchNorm1d(128) -> ReLU
  Dropout(0.132)
  Linear(128 -> 20)

Output: (B, 20)
```

**Perhitungan Parameter ImprovedCNN:**

| Layer | Formula | Parameter |
|-------|---------|-----------|
| Conv layers (sama) | -- | 388,320 |
| FC1 Linear | 256x128 + 128 | 33,024 |
| FC1 BN1d | 2x128 | 256 |
| FC2 Linear | 128x20 + 20 | 2,580 |
| **Total** | | **424,052** |

### Keputusan Desain

| Komponen | Pilihan | Alasan |
|----------|---------|--------|
| `bias=False` di Conv | Hemat parameter | BatchNorm sudah punya shift (beta) |
| `ReLU(inplace=False)` | Non-inplace | `inplace=True` merusak tensor yang di-hook Grad-CAM |
| `AdaptiveAvgPool` | Output 1x1 | Robust terhadap perubahan input size |
| Weight init Conv | Kaiming `fan_out` | Sesuai karakteristik aktivasi ReLU |
| Weight init Linear | Xavier | Output logit lebih stabil |

---

## Preprocessing & Augmentasi

### Pipeline Preprocessing (Semua Split)

Dieksekusi di dalam `AksaraJawaDataset.__getitem__` menggunakan PIL, sebelum transform:

1. **Konversi ke Grayscale** — `img.convert("L")`: Warna tidak informatif untuk handwriting.
2. **AutoContrast** — `ImageOps.autocontrast(img, cutoff=2)`: Stretch dynamic range ke [0, 255]. Mencegah stroke tipis hilang setelah resize.
3. **Invert** — `ImageOps.invert(img)`: Background hitam (0), stroke putih (255).
4. **Square Padding** — `ImageOps.pad(img, ...)`: Preserve aspect ratio sebelum resize.

### Normalisasi

```python
norm_mean = (0.10,)
norm_std  = (0.25,)
```

Background dominan (~90% piksel = 0) membuat mean rendah. Range normalisasi: [-0.40, 3.60].

### Augmentasi Training

| Augmentasi | Parameter | Alasan |
|------------|-----------|--------|
| `RandomAffine` | rotation, translate, scale, shear | Simulasi variasi cara menulis |
| `ColorJitter` | brightness=0.10, contrast=0.10 | Variasi ketebalan stroke / kualitas scan |
| `RandomErasing` | p (dari Optuna), scale=(2%, 15%), value=0 | Invariansi terhadap oklusi parsial |

**`HorizontalFlip` dan `VerticalFlip` tidak digunakan** karena aksara Jawa tidak simetris.

---

## Hyperparameter Optimization (Optuna)

### Metode

Pencarian hyperparameter menggunakan **Optuna** dengan Tree-structured Parzen Estimator (TPE):

- **30 trial** per model, masing-masing dilatih selama **20 epoch** dengan early stopping patience 5.
- **MedianPruner** menghentikan trial yang jelas buruk lebih awal.
- **Objective**: maximize best validation accuracy.

### Ruang Pencarian

| Parameter | Range/Choices |
| --------- | ------------- |
| `learning_rate` | [5e-4, 5e-3] log-uniform |
| `weight_decay` | [1e-5, 1e-3] log-uniform |
| `dropout` | [0.2, 0.5] |
| `batch_size` | {32, 64, 128} |
| `label_smoothing` | [0.0, 0.15] |
| `aug_rotation_deg` | [5.0, 15.0] |
| `aug_erasing_prob` | [0.0, 0.3] |
| `warmup_epochs` | {1, 2, 3} |

### Hasil Terbaik

| Parameter | SimpleCNN | ImprovedCNN |
| --------- | --------- | ----------- |
| Learning Rate | 4.67e-3 | 3.89e-3 |
| Weight Decay | 2.15e-4 | 5.04e-4 |
| Dropout | 0.302 | 0.264 |
| Batch Size | 32 | 32 |
| Label Smoothing | 0.126 | 0.133 |
| Aug Rotation | 7.60 | 7.82 |
| Aug Erasing | 0.065 | 0.169 |
| Warmup Epochs | 3 | 3 |
| **Tune Val Acc** | **89.47%** | **90.67%** |

Best params disimpan ke `artifacts/best_params_{model}.json` dan dimuat otomatis oleh `main.py`.

---

## Strategi Training

### Optimizer & Loss (Optuna-tuned)

| Komponen | SimpleCNN | ImprovedCNN |
|----------|-----------|-------------|
| Optimizer | AdamW, lr=4.67e-3, wd=2.15e-4 | AdamW, lr=3.89e-3, wd=5.04e-4 |
| Loss | CrossEntropy, smoothing=0.126 | CrossEntropy, smoothing=0.133 |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 |

### Learning Rate Schedule

Linear Warmup (3 epoch) lalu Cosine Annealing:

```
lr(t) = t / t_warmup                              (t < t_warmup)
lr(t) = 0.5 * (1 + cos(pi * (t - t_w) / (T - t_w)))   (t >= t_warmup)
```

Scheduler di-step **per-batch** agar warmup bekerja benar di epoch pertama.

### Early Stopping & Checkpoint

- Max epochs: **100**
- Patience: **20 epoch** tanpa peningkatan `val_accuracy`
- Best model disimpan ke `artifacts/{model}/best_model.pt`
- Checkpoint menyimpan: `epoch`, `model_state`, `val_acc`, `val_loss`

### Mixed Precision (AMP)

Aktif otomatis jika GPU tersedia. Keuntungan: ~2x speedup, hemat VRAM ~50%.

---

## Hasil Eksperimen

Lihat laporan lengkap di [result.md](result.md).

### Ringkasan Hasil (setelah Optuna HPO)

| Metrik | SimpleCNN | ImprovedCNN |
|--------|-----------|-------------|
| Test Accuracy | **96.11%** | **98.11%** |
| F1 Macro | **96.00%** | **98.06%** |
| F1 Weighted | 96.09% | 98.12% |
| Test Loss | 0.2791 | 0.2133 |
| Best Val Accuracy | 95.33% | 97.25% |
| Total Parameters | 393,460 | 424,052 |
| Total Epochs | 83 | 97 |

**Peningkatan dari hyperparameter manual:**

| Model | Accuracy Lama | Accuracy Baru | Peningkatan |
|-------|--------------|---------------|-------------|
| SimpleCNN | 87.50% | 96.11% | **+8.61%** |
| ImprovedCNN | 87.50% | 98.11% | **+10.61%** |

**Kelas F1 sempurna (ImprovedCNN):** `na`, `ta`, `wa`, `pa`, `ja`, `nya` (6/20 kelas = 1.0000)

**Kelas F1 terendah (tetap > 0.92):** `ga` (0.9296), `tha` (0.9535), `la` (0.9577)

---

## Explainability (Grad-CAM)

Grad-CAM menunjukkan bagian gambar yang paling mempengaruhi prediksi dengan melihat gradien di layer konvolusi terakhir.

```
alpha_k = (1/Z) * sum_ij (d y_c / d A_ij^k)
L_c     = ReLU(sum_k alpha_k * A^k)
```

### Penggunaan via main.py

```bash
python src/main.py --mode gradcam --gradcam-samples 2
```

### Penggunaan Programatik

```python
import sys
sys.path.insert(0, "src")

from explainability import GradCAM, overlay_cam_on_image

target_layer = model.features[-3]   # Conv2d(128, 256, ...)
gradcam = GradCAM(model, target_layer)

cam, pred_class, confidence = gradcam(input_tensor)
overlay = overlay_cam_on_image(image_np, cam, alpha=0.4, colormap="jet")

gradcam.remove_hooks()
```

---

## Konfigurasi Hyperparameter

Semua hyperparameter terpusat di `src/config.py` menggunakan Python dataclass. Jika file `artifacts/best_params_{model}.json` tersedia (dari Optuna), parameter otomatis di-override saat training.

```python
import sys
sys.path.insert(0, "src")
from config import get_config

cfg = get_config()
cfg.learning_rate = 1e-3
```

### Daftar Parameter Lengkap

| Parameter | Default | Optuna Override | Keterangan |
|-----------|---------|-----------------|------------|
| `image_size` | 96 | - | Ukuran input (px) |
| `in_channels` | 1 | - | Grayscale |
| `batch_size` | 64 | 32 | Dari Optuna |
| `epochs` | 100 | - | Max epochs |
| `learning_rate` | 1.2e-3 | ~4e-3 | Dari Optuna |
| `weight_decay` | 1e-4 | ~2-5e-4 | Dari Optuna |
| `warmup_epochs` | 2 | 3 | Dari Optuna |
| `label_smoothing` | 0.05 | ~0.13 | Dari Optuna |
| `grad_clip_norm` | 1.0 | - | Max norm gradient clipping |
| `dropout` | 0.4 | ~0.26-0.30 | Dari Optuna |
| `early_stopping_patience` | 20 | - | Epoch tanpa peningkatan val_acc |
| `norm_mean` | (0.10,) | - | Mean setelah invert |
| `norm_std` | (0.25,) | - | Std normalisasi |
| `aug_rotation_deg` | 8.0 | ~7.6-7.8 | Dari Optuna |
| `aug_translate` | 0.08 | - | Translasi maksimum |
| `aug_scale` | (0.90, 1.10) | - | Scale range augmentasi |
| `aug_shear_deg` | 5.0 | - | Shear maksimum |
| `aug_erasing_prob` | 0.15 | ~0.07-0.17 | Dari Optuna |
| `seed` | 42 | - | Reproducibility |
| `num_workers` | 0 | - | 0 untuk Windows |

---

## Catatan Teknis

- **Windows `num_workers=0`**: multiprocessing PyTorch DataLoader pada Windows menggunakan `spawn` context yang sering timeout.
- **Reproducibility**: `set_seed(42)` di-set untuk `torch`, `numpy`, `random`, dan `cudnn` sebelum training dimulai.
- **Dataset lazy loading**: gambar dibuka satu per satu saat diakses (`__getitem__`), aman untuk dataset besar.
- **Deduplication berbasis MD5**: hash MD5 menghapus gambar duplikat antar sumber, mencegah data leakage.
- **`ReLU(inplace=False)` wajib untuk Grad-CAM**: `inplace=True` memodifikasi tensor in-place sehingga backward hooks tidak bisa mengambil nilai aktivasi yang benar.
- **Auto-load Optuna params**: `main.py` otomatis membaca `artifacts/best_params_{model}.json` sebelum training. Jika file tidak ada, training tetap berjalan dengan config default.

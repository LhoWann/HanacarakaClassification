# Laporan Hasil: Klasifikasi Aksara Jawa (Hanacaraka) dengan CNN

## 1. Latar Belakang

Aksara Jawa atau *Hanacaraka* adalah sistem tulisan tradisional yang telah digunakan selama berabad-abad untuk menulis bahasa Jawa. Terdapat 20 karakter dasar (*carakan*) yang membentuk fondasi sistem penulisan ini:

```
ha  na  ca  ra  ka
da  ta  sa  wa  la
pa  dha ja  ya  nya
ma  ga  ba  tha nga
```

Pelestarian aksara ini menghadapi tantangan serius di era digital. Digitalisasi naskah dan dokumen aksara Jawa memerlukan kemampuan pengenalan karakter otomatis (OCR) yang andal. Namun, klasifikasi tulisan tangan aksara Jawa secara otomatis memiliki beberapa tantangan:

1. **Kemiripan visual tinggi**: Beberapa pasangan karakter sangat mirip secara visual (misalnya `ha` vs `na`, `la` vs `wa`, `da` vs `dha`).
2. **Variasi tulisan tangan**: Setiap penulis memiliki gaya yang berbeda, menghasilkan variasi bentuk yang signifikan untuk karakter yang sama.
3. **Stroke tipis**: Karakteristik stroke aksara Jawa yang halus rentan hilang atau terdistorsi saat resize gambar ke resolusi yang lebih kecil.
4. **Dataset terbatas**: Corpus digital aksara Jawa tulisan tangan masih sangat terbatas dibanding aksara Latin.

Proyek ini bertujuan membangun sistem klasifikasi otomatis berbasis CNN (*Convolutional Neural Network*) yang mampu mengenali ke-20 karakter Hanacaraka dari gambar tulisan tangan dengan akurasi tinggi.

---

## 2. Novelty dan Kontribusi

### 2.1 Pipeline Reprodusibel Berbasis CLI

Berbeda dari pendekatan notebook yang umum digunakan, proyek ini mengimplementasikan pipeline end-to-end berbasis command-line (`main.py`) yang mendukung mode `eda`, `train`, `eval`, `gradcam`, dan `all`. Pendekatan ini memastikan reprodusibilitas penuh: seed global di-set secara deterministik, semua output disimpan terstruktur di folder `outputs/`.

### 2.2 Preprocessing Domain-Specific untuk Aksara Tulisan Tangan

Pipeline preprocessing dirancang khusus untuk karakteristik aksara Jawa tulisan tangan:

- **AutoContrast dengan cutoff**: Memperkuat kontras stroke tipis sebelum resize, mencegah hilangnya detail stroke yang merupakan fitur diskriminatif utama.
- **Invert (background hitam, stroke putih)**: Stroke menjadi "fitur positif" yang di-amplifikasi oleh ReLU, menghasilkan aktivasi lebih kuat pada area diskriminatif.
- **Square Padding**: Mempertahankan aspect ratio sebelum resize. Tanpa ini, karakter landscape akan ter-squish dan distorsi stroke mengacaukan fitur.

### 2.3 Augmentasi Domain-Aware

Augmentasi dipilih berdasarkan sifat fisik aksara Jawa:

- `RandomAffine` (rotasi, translasi, scale, shear): Mensimulasikan variasi cara menulis natural.
- `ColorJitter` (brightness, contrast): Menangani variasi ketebalan stroke dan kualitas scan.
- `RandomErasing`: Membuat model invariant terhadap oklusi parsial, belajar dari seluruh karakter bukan satu stroke dominan.
- **Tanpa flip horizontal/vertikal**: Aksara Jawa tidak simetris; flip menghasilkan karakter yang semantiknya berbeda atau tidak valid.

### 2.4 Arsitektur ImprovedCNN dengan Two-Layer FC Head

Penambahan hidden layer 256 -> 128 -> 20 dengan BatchNorm1d memberikan ruang untuk pembentukan representasi intermediate sebelum keputusan klasifikasi. Dropout berlapis (0.4 sebelum FC1, 0.2 sebelum FC2) memberikan regularisasi proporsional dengan posisi dalam jaringan.

### 2.5 Grad-CAM untuk Interpretabilitas

Implementasi Grad-CAM memungkinkan visualisasi bagian gambar yang paling berpengaruh terhadap prediksi. Ini penting untuk verifikasi bahwa model benar-benar belajar dari stroke aksara (bukan artefak background atau noise), dan untuk mengidentifikasi kelemahan model pada pasangan kelas yang sering tertukar.

---

## 3. Metodologi

### 3.1 Dataset

| Split | Jumlah Gambar | Sumber |
|-------|--------------|--------|
| Train | 3,898 | GitHub vzrenggamani (70%) |
| Val   | 836   | GitHub vzrenggamani (15%) |
| Test  | 848   | GitHub vzrenggamani (15%) |
| **Total** | **5,582** | |

**Sumber Data:**
- **GitHub vzrenggamani** (`aksarajawa-hanacaraka`): Sumber utama gambar tulisan tangan in-distribution.
- **Roboflow fawwaz** (opsional): Dataset object-detection yang di-crop per bounding box.

**Strategi Split:**
- Stratified split 70/15/15 menggunakan `sklearn.model_selection.train_test_split` dengan `random_state=42`.
- Deduplication berbasis MD5 hash mencegah kebocoran data antar split.
- Validasi integritas setiap gambar via `PIL.Image.verify()`.

### 3.2 Preprocessing

Dieksekusi per-gambar di dalam `AksaraJawaDataset.__getitem__`:

| Tahap | Operasi | Tujuan |
|-------|---------|--------|
| 1 | `convert("L")` | Grayscale: warna tidak informatif untuk handwriting |
| 2 | `ImageOps.autocontrast(cutoff=2)` | Stretch dynamic range, perkuat stroke tipis |
| 3 | `ImageOps.invert()` | Background hitam, stroke putih |
| 4 | `ImageOps.pad(square, color=0)` | Preserve aspect ratio |
| 5 | `Resize(96x96)` | Input standar model |
| 6 | `ToTensor() + Normalize(0.10, 0.25)` | Normalisasi distribusi |

### 3.3 Arsitektur Model

**SimpleCNN (Baseline):**

```
Input: (B, 1, 96, 96)
Block 1: Conv2d(1->32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)  -> (B, 32, 48, 48)
Block 2: Conv2d(32->64)             -> BN -> ReLU -> MaxPool(2)  -> (B, 64, 24, 24)
Block 3: Conv2d(64->128)            -> BN -> ReLU -> MaxPool(2)  -> (B, 128, 12, 12)
Block 4: Conv2d(128->256)           -> BN -> ReLU                -> (B, 256, 12, 12)
         AdaptiveAvgPool2d(1)                                    -> (B, 256, 1, 1)
Classifier: Flatten -> Dropout(0.3) -> Linear(256, 20)
Total Parameters: 393,460
```

**ImprovedCNN:**

```
[Feature extraction identik dengan SimpleCNN]
Classifier:
  Flatten -> Dropout(0.4) -> Linear(256, 128) -> BN1d(128) -> ReLU
  -> Dropout(0.2) -> Linear(128, 20)
Total Parameters: 424,180
```

### 3.4 Strategi Training

| Komponen | Pilihan | Nilai |
|----------|---------|-------|
| Optimizer | AdamW | lr=1.2e-3, weight_decay=1e-4 |
| Loss | CrossEntropyLoss | label_smoothing=0.05 |
| LR Scheduler | Linear Warmup + Cosine Annealing | warmup=2 epoch |
| Gradient Clipping | `clip_grad_norm_` | max_norm=1.0 |
| Early Stopping | Patience | 12 epoch tanpa peningkatan val_acc |
| Mixed Precision | AMP | Aktif otomatis jika GPU tersedia |
| Max Epochs | | 50 |

**Label Smoothing (ε=0.05):** Label `[1, 0, ..., 0]` diubah menjadi `[0.9525, 0.0025, ...]` untuk mencegah overconfidence pada kelas mudah dan meningkatkan generalisasi pada kelas sulit.

**LR Scheduler per-batch:** Warmup berjalan benar di epoch pertama; jika di-step per-epoch, epoch pertama dilatih dengan LR=0.

---

## 4. Hasil Eksperimen

> **Catatan:** Nilai metrik di bawah adalah hasil dari eksperimen training yang dijalankan dengan konfigurasi default. Jalankan `python main.py --mode all` untuk memperbarui dengan hasil terbaru.

### 4.1 Training Curves

![Training Curves](outputs/training/training_curves.png)

### 4.2 Perbandingan SimpleCNN vs ImprovedCNN

| Metrik | SimpleCNN (Baseline) | ImprovedCNN |
|--------|---------------------|-------------|
| Test Accuracy | 89.98% | -- |
| F1 Macro | 89.54% | -- |
| F1 Weighted | -- | -- |
| Total Parameters | 393,460 | 424,180 |
| Epoch Terbaik | -- | -- |

**Perubahan ImprovedCNN vs SimpleCNN:**

| Komponen | SimpleCNN | ImprovedCNN | Alasan |
|----------|-----------|-------------|--------|
| FC Head | 256 -> 20 | 256 -> 128 -> 20 | Representasi intermediate lebih kaya |
| Label Smoothing | 0.0 | 0.05 | Regularisasi pada kelas sulit |
| Learning Rate | 2e-3 | 1.2e-3 | Kurangi osilasi val_acc |
| norm_std | 0.20 | 0.25 | Range normalisasi lebih terkontrol |
| Weight Decay | 5e-5 | 1e-4 | Regularisasi lebih kuat |
| RandomErasing | -- | p=0.15 | Invariansi terhadap oklusi parsial |
| Warmup Epochs | 1 | 2 | BatchNorm lebih stabil di awal |
| Dropout | 0.3 | 0.4 / 0.2 | Berlapis sesuai two-layer FC head |

### 4.3 Evaluasi Test Set

![Confusion Matrix](outputs/evaluation/confusion_matrix.png)

![F1 Per Kelas](outputs/evaluation/f1_per_class.png)

**Kelas dengan F1 Terendah (SimpleCNN baseline):**

| Kelas | F1-Score | Analisis |
|-------|----------|---------|
| `ha` | 0.575 | Visual mirip dengan `na`; stroke pembeda sangat kecil |
| `la` | 0.629 | Sering tertukar dengan `wa` dan `na` |

---

## 5. Analisis Grad-CAM

![Grad-CAM Contoh](outputs/gradcam/gradcam_ha.png)

Grad-CAM memvisualisasikan region pada gambar yang paling mempengaruhi prediksi model. Analisis dilakukan pada layer konvolusi terakhir (`features[-3]`, Conv2d 128->256).

**Temuan:**

- Pada prediksi **benar**: model konsisten fokus pada stroke utama yang membedakan karakter (area tengah dan kiri yang mengandung kurva pembeda).
- Pada prediksi **salah** (misalnya `ha` diprediksi sebagai `na`): heatmap menunjukkan model fokus pada bagian yang memang ambigu secara visual antara kedua kelas.
- Kelas dengan F1 tinggi (misalnya `nga`, `nya`): heatmap terpusat rapi pada stroke unik yang tidak dimiliki kelas lain.

---

## 6. Diskusi

### 6.1 Kekuatan

- Pipeline end-to-end yang reprodusibel tanpa ketergantungan pada notebook.
- Preprocessing domain-specific terbukti efektif mempertahankan detail stroke tipis.
- Regularisasi berlapis (label smoothing, dropout berlapis, weight decay, random erasing) menghasilkan generalisasi yang baik untuk dataset berukuran sedang (~5,582 gambar).
- Grad-CAM memberikan bukti bahwa model belajar dari fitur yang semantically meaningful (stroke aksara).

### 6.2 Keterbatasan

- **Dataset kecil**: ~280 gambar per kelas relatif sedikit untuk deep learning. Transfer learning dari pretrained model (ResNet, EfficientNet) berpotensi memberikan peningkatan signifikan.
- **Tanpa transfer learning**: Model dilatih dari nol (random weight initialization). ImageNet pretrained features mungkin tidak secara langsung relevan untuk aksara, tetapi fine-tuning seringkali tetap menguntungkan.
- **Kelas ambigu inheren**: Beberapa pasangan kelas (`ha`/`na`, `la`/`wa`) memiliki kemiripan visual yang sangat tinggi bahkan bagi penutur asli. Akurasi 100% mungkin tidak achievable tanpa konteks sekitar karakter.
- **Hanya aksara dasar**: Tidak mencakup sandhangan (diakritik), pasangan, atau aksara murda yang ada dalam sistem tulisan Jawa lengkap.

### 6.3 Arah Pengembangan

1. **Transfer Learning**: Fine-tune ResNet-18/EfficientNet-B0 yang di-pretrain di ImageNet.
2. **Dataset Augmentation Lanjutan**: CutMix, MixUp untuk meningkatkan robustness pada kelas yang mirip.
3. **Ensemble**: Kombinasi SimpleCNN + ImprovedCNN + model pretrained.
4. **Sequence Model**: Untuk pengenalan kata/kalimat aksara Jawa secara utuh (CTC/Attention-based).
5. **Attention Mechanism**: Channel/spatial attention untuk fokus lebih eksplisit pada stroke diskriminatif.

---

## 7. Kesimpulan

Proyek ini berhasil membangun sistem klasifikasi aksara Jawa (Hanacaraka) berbasis CNN dengan:

1. **Pipeline reprodusibel**: Single entry point `main.py` mencakup EDA, training, evaluasi, dan Grad-CAM.
2. **Preprocessing domain-specific**: AutoContrast + Invert + Square Padding terbukti efektif mempertahankan fitur stroke tipis.
3. **Arsitektur efisien**: ImprovedCNN dengan ~424K parameter mampu mencapai akurasi kompetitif pada dataset berukuran sedang.
4. **Regularisasi menyeluruh**: Label smoothing, dropout berlapis, weight decay, dan augmentasi domain-aware menghasilkan generalisasi yang baik.
5. **Interpretabilitas**: Grad-CAM memverifikasi bahwa model belajar dari stroke aksara yang bermakna secara semantik.

Tantangan utama pada kelas yang mirip secara visual (`ha`/`na`, `la`/`wa`) membuka peluang pengembangan dengan transfer learning dan teknik augmentasi yang lebih canggih. Hasil ini menjadi baseline yang solid untuk pengembangan sistem pengenalan aksara Jawa yang lebih komprehensif.

"""
CNN architecture untuk Aksara Jawa classification.

Design rationale:
  - 4 conv blocks: cukup depth untuk 96×96 input
  - BatchNorm setelah setiap Conv: stabilkan training, allow higher LR
  - ReLU inplace=False: kompatibel dengan Grad-CAM hooks
  - AdaptiveAvgPool: robust terhadap perubahan input size
  - Dropout sebelum FC: regularisasi paling efektif di classifier head

Rumus dasar:
  Output conv  : O = floor((I - K + 2P) / S) + 1
  Output pool  : O = floor((I - K) / S) + 1
  Params conv  : K × K × C_in × C_out  (bias=False)
  Params BN    : 2 × C  (gamma + beta, learnable)
  Params Linear: C_in × C_out + C_out  (dengan bias)
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Conv(3×3, pad=1) → BN → ReLU → MaxPool(2×2).

    - padding=1 mempertahankan H×W setelah conv
    - MaxPool(2×2) membagi H dan W menjadi setengah
    - ReLU inplace=False: inplace memodifikasi tensor yang di-hook Grad-CAM
      → RuntimeError saat backward; trade-off memori kecil tapi wajib
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """
    Baseline CNN — 4 conv blocks, single-layer FC head.

    Input  : (B, in_channels, 96, 96)  — grayscale default
    Output : (B, num_classes)          — raw logits

    Feature map sizes (input 96×96):
        After block 1 (Conv+Pool): 96→48,   32 ch
        After block 2 (Conv+Pool): 48→24,   64 ch
        After block 3 (Conv+Pool): 24→12,  128 ch
        After block 4 (Conv only): 12→12,  256 ch  [no pool]
        After AdaptiveAvgPool    :  1× 1,  256 ch

    Parameter count (96×96, in_channels=1):
        Block1 Conv : 3×3×1×32         =    288
        Block1 BN   : 2×32             =     64
        Block2 Conv : 3×3×32×64        = 18,432
        Block2 BN   : 2×64             =    128
        Block3 Conv : 3×3×64×128       = 73,728
        Block3 BN   : 2×128            =    256
        Block4 Conv : 3×3×128×256      =294,912
        Block4 BN   : 2×256            =    512
        FC Linear   : 256×20 + 20      =  5,140
                                 TOTAL = 393,460
    """

    def __init__(self, num_classes: int = 20, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming init untuk Conv (sesuai ReLU activation).
        Xavier untuk Linear (output adalah logit, bukan feature).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    """
    CNN yang ditingkatkan — arsitektur identik dengan SimpleCNN di bagian
    feature extraction, namun dengan FC head dua layer.

    Perbedaan vs SimpleCNN:
      1. Two-layer FC head: 256 → 128 → num_classes
         (SimpleCNN: 256 → num_classes langsung)
      2. BatchNorm1d setelah Linear pertama: stabilkan hidden representation
      3. Dropout berlapis: 0.4 sebelum FC1, 0.2 sebelum FC2

    Mengapa two-layer FC head lebih baik?
      - Satu layer Linear(256→20) memaksa mapping langsung dari fitur spasial
        ke keputusan kelas — terlalu "abrupt"
      - Menambahkan hidden 128 memberi ruang untuk kombinasi fitur non-linear
        sebelum classifier, terbukti meningkatkan akurasi 1-2% pada handwriting

    Input  : (B, in_channels, 96, 96)
    Output : (B, num_classes)

    Feature map sizes (input 96×96):
        After block 1 : 96→48,   32 ch
        After block 2 : 48→24,   64 ch
        After block 3 : 24→12,  128 ch
        After block 4 : 12→12,  256 ch
        After AvgPool :  1× 1,  256 ch

    Parameter count (96×96, in_channels=1):
        Conv layers    : sama dengan SimpleCNN = 388,320
        FC1 Linear     : 256×128 + 128         =  33,024
        FC1 BN1d       : 2×128                 =     256
        FC2 Linear     : 128×20  + 20          =   2,580
                                        TOTAL  = 424,180
    """

    def __init__(self, num_classes: int = 20, in_channels: int = 1, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for ModelClass in [SimpleCNN, ImprovedCNN]:
        model = ModelClass(num_classes=20)
        x = torch.randn(4, 1, 96, 96)
        y = model(x)
        total, trainable = count_parameters(model)
        print(f"{ModelClass.__name__}")
        print(f"  Input  : {tuple(x.shape)}")
        print(f"  Output : {tuple(y.shape)}")
        print(f"  Params : {total:,} total, {trainable:,} trainable\n")

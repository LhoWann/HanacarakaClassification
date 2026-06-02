"""
Grad-CAM implementasi from scratch

Konsep:
  Grad-CAM = Class Activation Map berbasis gradien.
  Ide: bobot setiap feature map berdasarkan seberapa besar
  pengaruhnya terhadap skor kelas target.

Rumus matematis:
  α_k^c = (1/Z) Σᵢⱼ ∂y^c / ∂A_ij^k         (global avg pool of gradients)
  L^c   = ReLU( Σₖ α_k^c · A^k )            (weighted combination)

Dimana:
  A^k       = feature map ke-k pada layer target
  y^c       = logit untuk kelas c
  α_k^c     = importance weight feature map k untuk kelas c
  L^c       = Grad-CAM heatmap untuk kelas c
  ReLU      = hanya ambil aktivasi positif (kontribusi positif ke kelas)
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Hooks ke layer target (biasanya conv terakhir) untuk extract
    feature maps (forward) + gradients (backward).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        # register_full_backward_hook lebih akurat dari register_backward_hook (deprecated)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output adalah tuple; ambil elemen pertama
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    @torch.enable_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Args:
            input_tensor: (1, C, H, W) — single image batch
            target_class: kelas yang mau dijelaskan; None = pakai prediksi top-1

        Returns:
            cam:          numpy (H, W), normalized [0, 1]
            pred_class:   index kelas yang dijelaskan
            confidence:   softmax probability untuk kelas tersebut
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Expected (1, C, H, W), got {tuple(input_tensor.shape)}")

        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)              # (1, num_classes)
        probs  = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(1).item())

        confidence = float(probs[0, target_class].item())

        # Backward dari skor kelas target
        # Penting: zero gradient dulu untuk tidak terkontaminasi dari forward sebelumnya
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Sanity check hooks fired
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks tidak ter-trigger. Periksa target_layer.")

        # Hitung Grad-CAM
        # gradients: (1, K, h, w)
        # weights α_k = global avg pool gradients per channel → (1, K, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Linear combination of activations dengan weights
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU: drop negative contributions
        cam = F.relu(cam)

        # Upsample ke input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalize ke [0, 1] untuk visualisasi
        # Edge case: kalau cam semua nol (extremely rare), avoid div-by-zero
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Args:
        image: (H, W) atau (H, W, 3), values [0, 1] atau [0, 255]
        cam:   (H, W), values [0, 1]
        alpha: blending weight untuk heatmap

    Returns:
        overlay: (H, W, 3) uint8 numpy
    """
    import matplotlib.cm as cm

    # Normalize image ke [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Convert grayscale → RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Convert cam → colored heatmap via colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[..., :3]  # drop alpha channel → (H, W, 3)

    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay

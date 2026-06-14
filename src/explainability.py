from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
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
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Expected (1, C, H, W), got {tuple(input_tensor.shape)}")

        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(1).item())

        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks tidak ter-trigger. Periksa target_layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

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
    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    cmap = matplotlib.colormaps[colormap]
    heatmap = cmap(cam)[..., :3]

    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay
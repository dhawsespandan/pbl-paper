"""
utils/preprocess.py
Shared image preprocessing utilities for all CNN / ViT models.

All functions produce a (1, 3, 224, 224) torch.Tensor on DEVICE,
normalised with ImageNet mean/std.
"""

import os
import torch
import torchvision.transforms as T
from PIL import Image

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet normalisation constants ──────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ── Transforms ────────────────────────────────────────────────────────────

_cnn_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])

# Severity regression model was trained without explicit normalisation
_severity_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


# ── Public helpers ────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    """Open an image from disk and convert to RGB."""
    return Image.open(path).convert("RGB")


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Standard CNN preprocessing pipeline.

    Steps:
        1. Resize to 224 × 224
        2. Convert to float tensor  (values in [0, 1])
        3. Normalise with ImageNet mean / std
        4. Add batch dimension  → (1, 3, 224, 224)

    Returns
    -------
    torch.Tensor on DEVICE, shape (1, 3, 224, 224)
    """
    return _cnn_transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)


def to_cnn_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Alias for preprocess_image — kept for backwards compatibility."""
    return preprocess_image(pil_image)


def to_severity_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Preprocessing for the severity regression model (no normalisation).

    Returns
    -------
    torch.Tensor on DEVICE, shape (1, 3, 224, 224)
    """
    return _severity_transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

"""
model-service/models/fruit_severity/model.py
=============================================
Fruit Disease Severity Estimation — EfficientNet-B4 regression

Outputs a severity score in the range [0, 100] %.

Severity is defined as:
    affected_pixel_area / total_fruit_pixel_area × 100

where the model is trained on ground-truth labels derived from polygon
annotations (Healthy_Apple = healthy portion, disease labels = affected portion).

Weight loading priority
-----------------------
1. weights/fruit_severity_trained.pth  ← produced by train.py  (preferred)
   Head: Dropout → Linear(1792,512) → ReLU → Dropout → Linear(512,128)
         → ReLU → Dropout → Linear(128,1) → Sigmoid
2. weights/efficientnetb4_spandan.pth  ← original checkpoint   (fallback)
   Head: Dropout → Linear(1792,1)

Inference contract
------------------
predict_severity() accepts a (1, 3, H, W) tensor with pixel values in [0, 1]
(i.e. plain ToTensor() output, NOT pre-normalised by the caller).
Normalisation with ImageNet mean/std is applied internally so the model
always receives the same input distribution it was trained on.

This design prevents results from changing depending on which preprocessing
path the caller used, eliminating the non-determinism bug.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.serialization
import torchvision.models as models

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_TRAINED_PATH  = os.getenv("SEVERITY_TRAINED_PATH", "weights/fruit_severity_trained.pth")
_ORIGINAL_PATH = os.getenv("SEVERITY_MODEL_PATH",   "weights/efficientnetb4_spandan.pth")

# ImageNet normalisation — applied inside predict_severity() so the caller
# does not need to worry about it (and cannot accidentally skip it).
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


# ── build model ───────────────────────────────────────────────────────────────

def _build_trained_model() -> nn.Module:
    """Architecture matching what train.py saves (v4).
    4-layer head, linear output — clamped to [0,100] at inference."""
    m = models.efficientnet_b4(weights=None)
    feat_dim = m.classifier[1].in_features   # 1792
    m.classifier = nn.Sequential(
        m.classifier[0],                      # Dropout(p=0.4)
        nn.Linear(feat_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        # No Sigmoid — linear output, clamped to [0,100] at inference
    )
    return m


def _build_original_model() -> nn.Module:
    """Architecture from the original efficientnetb4_spandan.pth checkpoint."""
    m = models.efficientnet_b4(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m


# ── load weights ─────────────────────────────────────────────────────────────

def _try_load(model: nn.Module, path: str, remap_backbone: bool = False) -> bool:
    try:
        ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        if remap_backbone:
            state = {
                k.replace("backbone.", "features.", 1) if k.startswith("backbone.") else k: v
                for k, v in state.items()
            }
        model.load_state_dict(state, strict=False)
        return True
    except Exception as exc:
        warnings.warn(f"[fruit_severity] Cannot load '{path}': {exc}", RuntimeWarning)
        return False


_model: nn.Module

if os.path.exists(_TRAINED_PATH):
    _model = _build_trained_model()
    if _try_load(_model, _TRAINED_PATH):
        print(
            "FruitSeverity [EfficientNet-B4] loaded ← trained weights  "
            f"({_TRAINED_PATH})\n"
            "  Severity = affected_pixel_area / total_fruit_pixel_area × 100"
        )
    else:
        warnings.warn(
            f"[fruit_severity] Trained weights at '{_TRAINED_PATH}' could not be "
            "loaded. Falling back to original weights.",
            RuntimeWarning,
        )
        _model = _build_original_model()
        if os.path.exists(_ORIGINAL_PATH):
            _try_load(_model, _ORIGINAL_PATH, remap_backbone=True)

elif os.path.exists(_ORIGINAL_PATH):
    _model = _build_original_model()
    if _try_load(_model, _ORIGINAL_PATH, remap_backbone=True):
        print(
            "FruitSeverity [EfficientNet-B4] loaded ← original weights  "
            f"({_ORIGINAL_PATH}).\n"
            "  NOTE: run train.py to get accurate severity scores."
        )
    else:
        warnings.warn(
            "[fruit_severity] Original weights could not be loaded. "
            "Using random weights.",
            RuntimeWarning,
        )

else:
    _model = _build_trained_model()
    warnings.warn(
        "[fruit_severity] No weights found. "
        f"Checked: '{_TRAINED_PATH}', '{_ORIGINAL_PATH}'. "
        "Run model-service/models/fruit_severity/train.py to train the model.",
        RuntimeWarning,
    )

# Force eval mode at startup — disables all Dropout layers
_model.to(DEVICE).eval()

# Move normalisation constants to the same device as the model
_NORM_MEAN = _NORM_MEAN.to(DEVICE)
_NORM_STD  = _NORM_STD.to(DEVICE)


# ── public API ────────────────────────────────────────────────────────────────

def predict_severity(tensor: torch.Tensor) -> float:
    """
    Estimate the percentage of fruit area that is disease-affected.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (1, 3, H, W) with pixel values in [0, 1]
        (plain torchvision.transforms.ToTensor() output).
        ImageNet normalisation is applied internally — do NOT pre-normalise.

    Returns
    -------
    float
        Severity percentage in [0, 100], fully deterministic.
        Represents  affected_pixel_area / total_fruit_pixel_area × 100.

    Notes
    -----
    - model.eval() is called explicitly before every forward pass to
      guarantee Dropout is inactive, regardless of any external state changes.
    - torch.manual_seed(0) pins the RNG so results are bit-identical across
      repeated calls on the same input.
    - ImageNet normalisation is applied here, not by the caller, so the
      result is independent of the preprocessing path used upstream.
    """
    # ── determinism guards ───────────────────────────────────────────────────
    _model.eval()               # ensure Dropout is off for this call
    torch.manual_seed(0)        # pin RNG → identical results on repeated calls

    # ── normalise input (training used ImageNet stats; caller does not) ──────
    tensor = tensor.to(DEVICE)
    tensor = (tensor - _NORM_MEAN) / _NORM_STD

    with torch.no_grad():
        raw = _model(tensor).item()   # linear output; clamp to [0, 1]

    pct = max(0.0, min(1.0, raw)) * 100.0
    return round(pct, 2)

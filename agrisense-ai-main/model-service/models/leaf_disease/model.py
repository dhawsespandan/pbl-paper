"""
models/leaf_disease/model.py
============================
Leaf Disease Detection — Two-Stage Pipeline

Stage 1 — Apple Leaf / Not-Apple-Leaf (leaf_detector.py)
    Uses CLIP zero-shot classification to decide whether the image actually
    contains an apple leaf.  Non-apple-leaf images are rejected immediately
    with a clear label so main.py can return a 422 with a user-friendly message.

Stage 2 — Disease Classification
    EfficientNet-B0 CNN classifies a confirmed apple-leaf image into one of
    5 classes:
        alternaria leaf spot | brown spot | gray spot | healthy leaf | rust

    Priority weight lookup:
        1. weights/leaf_disease_v2.pt  — fine-tuned with rust-weighted loss
        2. weights/efficientnetb0_astha.pt  — original checkpoint (fallback)
        3. Random EfficientNet-B0 weights  — dev fallback with warning

Environment variables
---------------------
LEAF_MODEL_PATH         : override weight file path
LEAF_DETECTOR_THRESHOLD : CLIP threshold for the leaf gate (default 0.50)
DEVICE                  : "cpu" or "cuda"
"""

import os
import warnings
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image

from .leaf_detector import is_apple_leaf

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

LEAF_CLASSES = [
    "alternaria leaf spot",
    "brown spot",
    "gray spot",
    "healthy leaf",
    "rust",
]

NOT_LEAF_LABEL = "Not an apple leaf image."

# Weight file priority
_WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "weights"))
_V2_CKPT     = os.path.join(_WEIGHTS_DIR, "leaf_disease_v2.pt")
_ORIG_CKPT   = os.path.join(_WEIGHTS_DIR, "efficientnetb0_astha.pt")
_CKPT_PATH   = os.getenv("LEAF_MODEL_PATH",
                          _V2_CKPT if os.path.exists(_V2_CKPT) else _ORIG_CKPT)


def _build_model() -> nn.Module:
    model = tv_models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(LEAF_CLASSES)
    )
    return model


def _load_state(path: str) -> dict | None:
    """Load a raw state dict from a checkpoint file, regardless of format."""
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        if "model_state" in raw:
            return raw["model_state"], raw.get("val_acc", "?"), raw.get("arch", "unknown")
        if "state_dict" in raw:
            return raw["state_dict"], raw.get("val_acc", "?"), "unknown"
        if "model" in raw:
            return raw["model"], raw.get("val_acc", "?"), "unknown"
        # assume the dict itself is the state dict
        return raw, "?", "unknown"
    return None, "?", "unknown"


# ── load model ────────────────────────────────────────────────────────────

_model = _build_model()

if os.path.exists(_CKPT_PATH):
    _state, _val_acc, _arch = _load_state(_CKPT_PATH)
    if _state is not None:
        missing, unexpected = _model.load_state_dict(_state, strict=False)
        _label = "v2 (rust-tuned)" if "v2" in os.path.basename(_CKPT_PATH) else "original"
        print(
            f"LeafDisease [EfficientNet-V2-S | {_label}] loaded — "
            f"classes: {LEAF_CLASSES}  val_acc: {_val_acc}"
        )
        if missing:
            print(f"  [warn] {len(missing)} missing keys (expected for head-only mismatch)")
else:
    warnings.warn(
        f"[leaf_disease] No weights found. Using random EfficientNet-B0 — "
        "predictions are meaningless until weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )

_model.to(DEVICE).eval()

# ── pre-processing ─────────────────────────────────────────────────────────

from torchvision import transforms as _T

_transform = _T.Compose([
    _T.Resize(256),
    _T.CenterCrop(224),
    _T.ToTensor(),
    _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── public API ─────────────────────────────────────────────────────────────

def predict_leaf(image: Image.Image) -> tuple[str, float]:
    """
    Two-stage leaf disease prediction.

    Parameters
    ----------
    image : PIL.Image.Image
        Already-opened RGB image.

    Returns
    -------
    (label, confidence)
        NOT_LEAF_LABEL                          — image is not an apple leaf
        "alternaria leaf spot" | "brown spot"
        | "gray spot" | "healthy leaf" | "rust" — confirmed leaf + disease class
    """
    # ── Stage 1: Is it an apple leaf? ─────────────────────────────────
    leaf_ok, leaf_conf = is_apple_leaf(image)
    if not leaf_ok:
        return NOT_LEAF_LABEL, leaf_conf

    # ── Stage 2: Which disease? ────────────────────────────────────────
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return LEAF_CLASSES[idx], round(probs[idx].item(), 4)

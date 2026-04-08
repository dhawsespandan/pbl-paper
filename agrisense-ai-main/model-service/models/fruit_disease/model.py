"""
models/fruit_disease/model.py
==============================
Fruit Disease Detection — Two-Stage Pipeline

Stage 1 — Apple / Not-Apple (apple_detector.py)
    Uses CLIP zero-shot classification to decide whether the image
    actually contains an apple fruit.  Non-apple images are rejected
    immediately and never reach Stage 2.

Stage 2 — Disease Classification (this file)
    EfficientNet-B2 CNN classifies a confirmed apple image into one
    of 5 classes:  Anthracnose | Blotch | Healthy | Rot | Scab

If the weights file is missing (development mode) the model is initialised
with random weights and a warning is printed — the API will still respond
with mock-shaped output so the pipeline can be tested end-to-end.

Environment variables
---------------------
FRUIT_MODEL_PATH          : path to the EfficientNet checkpoint
FRUIT_CONFIDENCE_THRESHOLD: minimum disease-classifier confidence (default 0.50)
                            Lower than before because Stage 1 already confirms
                            the image is an apple — we only need the classifier
                            to be decisive about which disease it is.
APPLE_DETECTOR_THRESHOLD  : see apple_detector.py (default 0.60)
DEVICE                    : "cpu" or "cuda"
"""

import os
import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .apple_detector import is_apple_image

DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_CKPT_PATH = os.getenv("FRUIT_MODEL_PATH", "weights/fruit_efficientnet.pt")

DEFAULT_CLASSES = ["Anthracnose", "Blotch", "Healthy", "Rot", "Scab"]

NOT_APPLE_LABEL = "Not an apple image."

# ── disease confidence threshold (Stage 2 only) ───────────────────────────
# Because Stage 1 already verified the image is an apple we only need the
# disease classifier to be reasonably decisive.
DISEASE_CONFIDENCE_THRESHOLD = float(
    os.getenv("FRUIT_CONFIDENCE_THRESHOLD", "0.50")
)

# ── load EfficientNet-B2 checkpoint ───────────────────────────────────────

if os.path.exists(_CKPT_PATH):
    _ckpt    = torch.load(_CKPT_PATH, map_location=DEVICE)
    _CLASSES = _ckpt["classes"]
    _model   = models.efficientnet_b2(weights=None)
    _model.classifier[1] = nn.Linear(
        _model.classifier[1].in_features, len(_CLASSES)
    )
    _model.load_state_dict(_ckpt["model_state"])
    print(f"FruitDisease [EfficientNet-B2] loaded — "
          f"classes: {_CLASSES}  val_acc: {_ckpt.get('val_acc', '?'):.4f}")
else:
    warnings.warn(
        f"[fruit_disease] Weights not found at '{_CKPT_PATH}'. "
        "Using random EfficientNet-B2 weights — predictions are meaningless "
        "until real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )
    _CLASSES = DEFAULT_CLASSES
    _model   = models.efficientnet_b2(weights=None)
    _model.classifier[1] = nn.Linear(
        _model.classifier[1].in_features, len(_CLASSES)
    )

_model.to(DEVICE).eval()

# ── pre-processing transform ──────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── public API ────────────────────────────────────────────────────────────

def predict_fruit(image_path: str) -> tuple[str, float]:
    """
    Two-stage fruit disease prediction.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.

    Returns
    -------
    (label, confidence)

    label is one of:
        NOT_APPLE_LABEL                       — image is not an apple
        "Anthracnose" | "Blotch" | "Healthy"
        | "Rot" | "Scab"                      — confirmed apple + disease class

    confidence is the probability score for the returned label ∈ [0, 1].
    For NOT_APPLE_LABEL it is the CLIP apple-probability (how "apple-like"
    the image was, which will be low).

    Pipeline
    --------
    1. CLIP zero-shot check: is this an apple fruit?
       → No  → return (NOT_APPLE_LABEL, clip_apple_score)
       → Yes → continue to Stage 2

    2. EfficientNet-B2 disease classification.
       → max softmax prob < DISEASE_CONFIDENCE_THRESHOLD
             → return (NOT_APPLE_LABEL, disease_confidence)
       → otherwise → return (disease_class, disease_confidence)
    """

    # ── Stage 1: Is it an apple? ──────────────────────────────────────────
    apple, apple_conf = is_apple_image(image_path)

    if not apple:
        return NOT_APPLE_LABEL, apple_conf

    # ── Stage 2: Which disease? ───────────────────────────────────────────
    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    idx            = probs.argmax().item()
    max_confidence = probs[idx].item()

    # Sanity check: even after CLIP approval, reject if the disease
    # classifier itself is very uncertain
    if max_confidence < DISEASE_CONFIDENCE_THRESHOLD:
        return NOT_APPLE_LABEL, round(max_confidence, 4)

    return _CLASSES[idx], round(max_confidence, 4)

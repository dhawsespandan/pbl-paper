"""
models/leaf_disease/leaf_detector.py
======================================
Gate: Apple Leaf / Not-Apple-Leaf binary check using CLIP zero-shot classification.

Runs before the leaf disease CNN so that non-apple-leaf images (other plants,
objects, animals, etc.) are rejected with a clear message rather than being
misclassified as one of the 5 apple leaf disease classes.

Environment variables
---------------------
LEAF_DETECTOR_THRESHOLD : float  (default 0.50)
    Minimum CLIP probability for the "apple leaf" label to accept the image.
"""

import os
import warnings
from PIL import Image

DEVICE = os.getenv("DEVICE", "cpu")
_HF_DEVICE = 0 if DEVICE == "cuda" else -1

LEAF_THRESHOLD = float(os.getenv("LEAF_DETECTOR_THRESHOLD", "0.50"))

_LABELS     = ["apple leaf", "not an apple leaf"]
_LEAF_LABEL = "apple leaf"

_pipeline = None


def _load_pipeline() -> None:
    global _pipeline
    if _pipeline is not None:
        return
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=_HF_DEVICE,
        )
        print("[leaf_detector] CLIP model loaded (openai/clip-vit-base-patch32)")
    except Exception as exc:
        warnings.warn(
            f"[leaf_detector] Could not load CLIP model: {exc}\n"
            "Leaf detection will be skipped — all images passed to CNN (fail-open).",
            RuntimeWarning,
            stacklevel=2,
        )
        _pipeline = None


def is_apple_leaf(image: Image.Image) -> tuple[bool, float]:
    """
    Determine whether *image* contains an apple leaf.

    Parameters
    ----------
    image : PIL.Image.Image
        Already-opened RGB image.

    Returns
    -------
    (is_leaf, confidence)
        is_leaf    – True if CLIP believes this is an apple leaf image.
        confidence – CLIP probability for the "apple leaf" label ∈ [0, 1].
                     Returns -1.0 and True if CLIP failed to load (fail-open).
    """
    _load_pipeline()

    if _pipeline is None:
        return True, -1.0

    img_rgb = image.convert("RGB")
    results = _pipeline(img_rgb, candidate_labels=_LABELS)

    leaf_score = next(
        (r["score"] for r in results if r["label"] == _LEAF_LABEL), 0.0
    )

    return leaf_score >= LEAF_THRESHOLD, round(leaf_score, 4)

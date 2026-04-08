"""
models/router_transformer/crop_prefilter.py
=============================================
Gate-0: Apple Crop / Not-Apple-Crop binary check using CLIP zero-shot classification.

Runs BEFORE the router so that images that are clearly not apple leaves,
fruits, or flower clusters are rejected immediately — without even reaching
the EfficientNet/LoRA router, which was trained only on apple crop imagery
and will blindly assign one of the three crop classes to anything.

Pipeline
--------
   image  →  CLIP (general check)
                  P("apple leaf, fruit or flower") >= CROP_PREFILTER_THRESHOLD  → pass
                  else → CLIP (fruit-specific check)
                              P("apple fruit") >= FRUIT_PREFILTER_THRESHOLD  → pass
                              else → reject with 'unknown'

Two-stage design ensures blurry or dark fruit images that score low on the
general label still get a second chance via a fruit-specific prompt, while
clearly non-crop images (e.g. random photos) are still rejected.

Environment variables
---------------------
CROP_PREFILTER_THRESHOLD : float  (default 0.50)
    Minimum CLIP probability for the general apple-crop label.
FRUIT_PREFILTER_THRESHOLD : float  (default 0.30)
    Minimum CLIP probability for the fruit-specific fallback check.
    Lower than the general threshold because blurry/dark fruit images
    naturally score lower on composite labels.
"""

import os
import warnings
import torch
from PIL import Image

DEVICE = os.getenv("DEVICE", "cpu")
_HF_DEVICE = 0 if DEVICE == "cuda" else -1

CROP_THRESHOLD  = float(os.getenv("CROP_PREFILTER_THRESHOLD",  "0.50"))
FRUIT_THRESHOLD = float(os.getenv("FRUIT_PREFILTER_THRESHOLD", "0.30"))

# General candidate labels
_LABELS = [
    "apple leaf, fruit or flower cluster",
    "not an apple crop image",
]
_CROP_LABEL = "apple leaf, fruit or flower cluster"

# Fruit-specific fallback labels — shorter, more concrete prompts that CLIP
# handles better for close-up or blurry apple fruit images.
_FRUIT_LABELS = [
    "apple fruit",
    "not an apple",
]
_FRUIT_LABEL = "apple fruit"

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
        print("[crop_prefilter] CLIP model loaded (openai/clip-vit-base-patch32)")
    except Exception as exc:
        warnings.warn(
            f"[crop_prefilter] Could not load CLIP model: {exc}\n"
            "Crop pre-filtering will be skipped — all images will be forwarded "
            "to the router (fail-open behaviour).",
            RuntimeWarning,
            stacklevel=2,
        )
        _pipeline = None


def is_apple_crop(image: Image.Image) -> tuple[bool, float]:
    """
    Check whether *image* contains apple leaf, fruit, or flower content.

    Parameters
    ----------
    image : PIL.Image.Image
        Already-opened RGB image.

    Returns
    -------
    (is_crop, confidence)
        is_crop    – True if CLIP believes this is an apple crop image.
        confidence – CLIP score for the positive label ∈ [0, 1].
                     Returns -1.0 and True if CLIP failed to load
                     (fail-open so the rest of the pipeline still works).
    """
    _load_pipeline()

    if _pipeline is None:
        return True, -1.0

    img_rgb = image.convert("RGB")

    # ── Stage 1: general crop check ──────────────────────────────────────
    results = _pipeline(img_rgb, candidate_labels=_LABELS)
    crop_score = next(
        (r["score"] for r in results if r["label"] == _CROP_LABEL), 0.0
    )

    if crop_score >= CROP_THRESHOLD:
        return True, round(crop_score, 4)

    # ── Stage 2: fruit-specific fallback ─────────────────────────────────
    # Blurry, dark, or close-up fruit images can score below the general
    # threshold even though they are clearly apple fruit.  A targeted
    # two-class prompt gives CLIP a much better signal.
    fruit_results = _pipeline(img_rgb, candidate_labels=_FRUIT_LABELS)
    fruit_score = next(
        (r["score"] for r in fruit_results if r["label"] == _FRUIT_LABEL), 0.0
    )

    if fruit_score >= FRUIT_THRESHOLD:
        print(f"[crop_prefilter] General check failed ({crop_score:.3f}), "
              f"but fruit-specific check passed ({fruit_score:.3f}) — forwarding to router")
        return True, round(fruit_score, 4)

    return False, round(crop_score, 4)

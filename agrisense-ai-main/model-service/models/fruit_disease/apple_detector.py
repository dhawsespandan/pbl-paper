"""
models/fruit_disease/apple_detector.py
=======================================
Stage-1: Apple / Not-Apple binary detector using CLIP zero-shot classification.

CLIP (Contrastive Language-Image Pre-Training) was trained on hundreds of
millions of image-text pairs from the internet.  It genuinely understands
the concept of "apple fruit", so it can correctly reject vegetables, people,
animals, landscapes, etc. — things the closed-set disease classifier cannot
distinguish because it has only ever seen apple images.

Pipeline
--------
   image  →  CLIP  →  P("apple fruit")  ≥ threshold  →  True / False
                       P("not an apple")

The CLIP model is downloaded from HuggingFace on first use (~350 MB) and
cached automatically in ~/.cache/huggingface.  Subsequent startups are fast.

Environment variables
---------------------
APPLE_DETECTOR_THRESHOLD : float  (default 0.60)
    Minimum CLIP probability for the "apple fruit" label to accept the image.
    Raise it to be stricter; lower it if real apples are getting rejected.
"""

import os
import warnings
import torch
from PIL import Image

DEVICE = os.getenv("DEVICE", "cpu")
_HF_DEVICE = 0 if DEVICE == "cuda" else -1

APPLE_THRESHOLD = float(os.getenv("APPLE_DETECTOR_THRESHOLD", "0.60"))

# Candidate labels fed to the CLIP zero-shot pipeline.
# CLIP converts these to text embeddings and finds which one is most similar
# to the image embedding.
_LABELS = ["apple fruit", "not an apple"]
_APPLE_LABEL = "apple fruit"

# Lazy-loaded; initialised on first call to is_apple_image()
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
        print("[apple_detector] CLIP model loaded (openai/clip-vit-base-patch32)")
    except Exception as exc:
        warnings.warn(
            f"[apple_detector] Could not load CLIP model: {exc}\n"
            "Apple detection will be skipped — all images will be forwarded "
            "to the disease classifier.",
            RuntimeWarning,
            stacklevel=2,
        )
        _pipeline = None          # stays None → detection disabled


def is_apple_image(image_path: str) -> tuple[bool, float]:
    """
    Determine whether *image_path* contains an apple fruit.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.

    Returns
    -------
    (is_apple, confidence)
        is_apple   – True if the image is classified as apple fruit.
        confidence – CLIP probability for the "apple fruit" label ∈ [0, 1].
                     If the CLIP model could not be loaded this is -1.0 and
                     is_apple is True (fail-open so the pipeline still works).
    """
    _load_pipeline()

    # If CLIP couldn't load, fail open so the rest of the pipeline still works
    if _pipeline is None:
        return True, -1.0

    img = Image.open(image_path).convert("RGB")
    results = _pipeline(img, candidate_labels=_LABELS)

    # results is a list of {"label": ..., "score": ...} sorted by score desc
    apple_score = next(
        (r["score"] for r in results if r["label"] == _APPLE_LABEL), 0.0
    )

    return apple_score >= APPLE_THRESHOLD, round(apple_score, 4)

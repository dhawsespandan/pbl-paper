"""
models/flower_cluster/model.py
================================
Flower Cluster Detection CNN — YOLOv8-m

Detects individual apple flowers in an image and returns a health label:
    Healthy Clusters  — ≥ 15 detections (abundant)
    Adequate Clusters — 6–14 detections (moderate)
    Sparse Clusters   — 1–5 detections (sparse)
    No Clusters Detected — 0 detections

Detection tuning notes:
  - conf=0.10   : lower than default (0.25) to catch partially open buds
  - iou=0.50    : moderate NMS to separate adjacent overlapping flowers
  - imgsz=1280  : higher resolution improves small/distant flower detection

If the weights file is missing (development mode) a mock predictor is used
that always returns "Sparse Clusters" with a confidence of 0.5 and a
warning is printed.
"""

import os
import warnings

_MODEL_PATH = os.getenv("FLOWER_MODEL_PATH", "weights/yolo26m_abhirami.pt")

# ── YOLO inference hyper-parameters ──────────────────────────────────────
_CONF    = 0.10   # lower threshold catches partially-open buds
_IOU     = 0.50   # moderate NMS — separates adjacent flowers without over-splitting
_IMGSZ   = 1280   # high resolution helps with wide/distant shots

# ── load YOLO or create mock fallback ────────────────────────────────────

_yolo = None

if os.path.exists(_MODEL_PATH):
    try:
        from ultralytics import YOLO
        _yolo = YOLO(_MODEL_PATH)
        print(f"FlowerCluster [YOLOv8-m] loaded — weight: {_MODEL_PATH}")
    except Exception as exc:
        warnings.warn(
            f"[flower_cluster] YOLO load failed ({exc}). "
            "Using mock predictor — predictions are meaningless until "
            "real weights are supplied.",
            RuntimeWarning, stacklevel=2,
        )
else:
    warnings.warn(
        f"[flower_cluster] Weights not found at '{_MODEL_PATH}'. "
        "Using mock predictor — predictions are meaningless until "
        "real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )


def _count_to_health(count: int) -> tuple[str, float]:
    """
    Map raw detection count → (health_label, pseudo_confidence).

    Thresholds are calibrated against real apple-blossom images using
    conf=0.10, iou=0.50, imgsz=1280 YOLO settings.
    """
    if count >= 15:
        return "Healthy Clusters", 0.92
    elif count >= 6:
        return "Adequate Clusters", 0.85
    elif count >= 1:
        return "Sparse Clusters", 0.80
    else:
        return "No Clusters Detected", 0.88


# ── public API ────────────────────────────────────────────────────────────

def predict_flowers(image_path: str) -> tuple[str, float, int]:
    """
    Parameters
    ----------
    image_path : str — path to the image on disk

    Returns
    -------
    (health_label, confidence, flower_count)
        health_label  : 'Healthy Clusters' | 'Adequate Clusters' |
                        'Sparse Clusters' | 'No Clusters Detected'
        confidence    : float in [0, 1]
        flower_count  : int — individual flowers detected by YOLO
    """
    if _yolo is None:
        # mock fallback for development
        return "Sparse Clusters", 0.50, 0

    results = _yolo(image_path, conf=_CONF, iou=_IOU, imgsz=_IMGSZ, verbose=False)
    count   = int(len(results[0].boxes))
    label, conf = _count_to_health(count)
    return label, conf, count

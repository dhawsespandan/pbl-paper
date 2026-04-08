"""
models/router_transformer/model.py
===================================
Router — HuggingFace ViT fine-tuned with LoRA/QLoRA (PEFT)

Classifies an input image into one of four categories:
    fruit | leaf | flower_cluster | unknown

Primary  : DINOv2-base + LoRA adapter  (ROUTER_LORA_DIR in .env)
Fallback : EfficientNet-B0             (ROUTER_EFFICIENTNET_PATH in .env)

If router confidence is below ROUTER_UNKNOWN_THRESHOLD the prediction is
overridden and  ("unknown", confidence)  is returned, triggering the
422 error response in main.py.

If neither weights file is present the router initialises with random
EfficientNet-B0 weights and logs a warning (development fallback).
"""

import json
import os
import warnings
import torch
import torch.nn as nn
from PIL import Image

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_LORA_DIR        = os.getenv("ROUTER_LORA_DIR",         "weights/router_model_final")
_EFFNET_PT       = os.getenv("ROUTER_EFFICIENTNET_PATH", "weights/router_efficientnet.pt")
_UNKNOWN_THRESH  = float(os.getenv("ROUTER_UNKNOWN_THRESHOLD", "0.60"))

# Internal class names that map to external API names
_INTERNAL_TO_API = {
    "flower":         "flower_cluster",   # trained label → API label
    "flower_cluster": "flower_cluster",
    "fruit":          "fruit",
    "leaf":           "leaf",
}

_predict_fn = None   # callable: PIL Image → (api_label: str, conf: float)
_CLASSES    = None


# ── 1. Try DINOv2 + LoRA ─────────────────────────────────────────────────

def _load_lora() -> bool:
    global _predict_fn, _CLASSES

    meta_path = os.path.join(_LORA_DIR, "router_meta.json")
    cfg_path  = os.path.join(_LORA_DIR, "adapter_config.json")

    if not os.path.isdir(_LORA_DIR) or not (
        os.path.exists(meta_path) or os.path.exists(cfg_path)
    ):
        return False

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from peft import PeftModel

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            classes   = meta["classes"]
            base_name = meta.get("base_model", "facebook/dinov2-base")
            val_acc   = meta.get("best_val_acc", "?")
        else:
            with open(cfg_path) as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path", "facebook/dinov2-small")
            classes   = ["flower", "fruit", "leaf"]
            val_acc   = "?"

        n         = len(classes)
        processor = AutoImageProcessor.from_pretrained(_LORA_DIR)
        base      = AutoModelForImageClassification.from_pretrained(
            base_name,
            num_labels=n,
            id2label={i: c for i, c in enumerate(classes)},
            label2id={c: i for i, c in enumerate(classes)},
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, _LORA_DIR).to(DEVICE)
        model.eval()

        _CLASSES = classes
        print(f"Router [DINOv2+LoRA | {base_name}] loaded — "
              f"classes: {_CLASSES}  val_acc: {val_acc}")

        def _fn(pil_image: Image.Image):
            inputs = processor(images=pil_image.convert("RGB"),
                               return_tensors="pt")
            with torch.no_grad():
                logits = model(
                    pixel_values=inputs["pixel_values"].to(DEVICE)
                ).logits
            probs = torch.softmax(logits, dim=1)[0]
            idx   = probs.argmax().item()
            conf  = probs[idx].item()
            label = _CLASSES[idx]
            api_label = _INTERNAL_TO_API.get(label, label)
            # Always route fruit images even when confidence is low —
            # blurry/close-up fruit shots score below the general threshold
            # but should still reach the fruit disease module.
            if api_label == "fruit":
                return "fruit", conf
            if conf < _UNKNOWN_THRESH:
                return "unknown", conf
            return api_label, conf

        _predict_fn = _fn
        return True

    except Exception as exc:
        print(f"[router] LoRA load failed ({exc}) — trying EfficientNet fallback")
        return False


# ── 2. EfficientNet-B0 (trained or random-weight fallback) ───────────────

def _load_efficientnet():
    global _predict_fn, _CLASSES

    from torchvision import transforms, models

    DEFAULT_CLASSES = ["flower", "fruit", "leaf"]

    net = models.efficientnet_b0(weights=None)

    if os.path.exists(_EFFNET_PT):
        ckpt    = torch.load(_EFFNET_PT, map_location=DEVICE)
        classes = ckpt["classes"]
        net.classifier[1] = nn.Linear(
            net.classifier[1].in_features, len(classes)
        )
        net.load_state_dict(ckpt["model_state"])
        print(f"Router [EfficientNet-B0] loaded — "
              f"classes: {classes}  val_acc: {ckpt.get('val_acc', '?'):.4f}")
    else:
        warnings.warn(
            f"[router] Weights not found at '{_EFFNET_PT}'. "
            "Falling back to random EfficientNet-B0 weights — "
            "predictions will be meaningless until real weights are supplied.",
            RuntimeWarning, stacklevel=2,
        )
        classes = DEFAULT_CLASSES
        net.classifier[1] = nn.Linear(
            net.classifier[1].in_features, len(classes)
        )

    net.to(DEVICE).eval()
    _CLASSES = classes

    _tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def _fn(pil_image: Image.Image):
        t = _tf(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = net(t)
        probs = torch.softmax(logits, dim=1)[0]
        idx   = probs.argmax().item()
        conf  = probs[idx].item()
        label = _CLASSES[idx]
        api_label = _INTERNAL_TO_API.get(label, label)
        # Always route fruit images even when confidence is low —
        # blurry/close-up fruit shots score below the general threshold
        # but should still reach the fruit disease module.
        if api_label == "fruit":
            return "fruit", conf
        if conf < _UNKNOWN_THRESH:
            return "unknown", conf
        return api_label, conf

    _predict_fn = _fn


# ── initialise on import ──────────────────────────────────────────────────

if not _load_lora():
    _load_efficientnet()


# ── public API ────────────────────────────────────────────────────────────

def predict_route(pil_image: Image.Image) -> tuple[str, float]:
    """
    Classify a PIL image.

    Returns
    -------
    (label, confidence)
        label      : 'fruit' | 'leaf' | 'flower_cluster' | 'unknown'
        confidence : float in [0, 1]
    """
    return _predict_fn(pil_image)

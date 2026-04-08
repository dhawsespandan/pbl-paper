"""
eval_router.py
==============
Evaluates the DINOv2-small + LoRA router against the EfficientNet-B0 baseline.

Produces:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix (saved as confusion_matrix.png)
  - Trainable parameter counts for both models
  - Inference latency on CPU (mean ± std over N_TIMING_RUNS)
  - results_summary.json — machine-readable results for the paper

Dataset layout expected:
    router_dataset/
        train/
            flower/   (or flower_cluster/)
            fruit/
            leaf/
        val/
            flower/
            fruit/
            leaf/

Usage:
    python eval_router.py --data_dir /path/to/router_dataset
"""

import os
import sys
import json
import time
import argparse
import warnings

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",       type=str, required=True,
                    help="Path to router dataset root (contains train/ and val/)")
parser.add_argument("--lora_dir",       type=str,
                    default="model-service/weights/router_model_final",
                    help="Path to DINOv2+LoRA weights directory")
parser.add_argument("--effnet_path",    type=str,
                    default="model-service/weights/router_efficientnet.pt",
                    help="Path to EfficientNet-B0 weights .pt file")
parser.add_argument("--batch_size",     type=int, default=16)
parser.add_argument("--n_timing_runs",  type=int, default=50,
                    help="Number of single-image runs for latency measurement")
parser.add_argument("--output_dir",     type=str, default="eval_results",
                    help="Directory to write results and plots")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
DEVICE = "cpu"

# ── helpers ───────────────────────────────────────────────────────────────────

CLASSES = ["flower", "fruit", "leaf"]

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def evaluate(predict_fn, val_dir, transform=None):
    """Run predict_fn over val_dir and return metrics."""
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score
    )

    all_preds, all_labels = [], []
    class_names = sorted(os.listdir(val_dir))
    label_map   = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            img_path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                pred_label, _ = predict_fn(img)
                # Normalise flower_cluster → flower for comparison
                pred_label = pred_label.replace("flower_cluster", "flower")
                all_preds.append(pred_label)
                all_labels.append(cls.replace("flower_cluster", "flower"))
            except Exception as e:
                warnings.warn(f"Skipping {img_path}: {e}")

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds, labels=CLASSES)
    return acc, report, cm, class_names


def measure_latency(predict_fn, sample_img, n=50):
    """Warm-up then measure mean ± std latency in ms."""
    for _ in range(5):
        predict_fn(sample_img)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict_fn(sample_img)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def save_confusion_matrix(cm, class_names, title, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 1. Load DINOv2 + LoRA ────────────────────────────────────────────────────

print("\n" + "="*60)
print("Loading DINOv2-small + LoRA router ...")
print("="*60)

from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel

lora_processor = AutoImageProcessor.from_pretrained(args.lora_dir)
lora_base = AutoModelForImageClassification.from_pretrained(
    "facebook/dinov2-small",
    num_labels=3,
    id2label={0: "flower", 1: "fruit", 2: "leaf"},
    label2id={"flower": 0, "fruit": 1, "leaf": 2},
    ignore_mismatched_sizes=True,
)
lora_model = PeftModel.from_pretrained(lora_base, args.lora_dir).to(DEVICE)
lora_model.eval()

lora_trainable = count_trainable_params(lora_model)
lora_total     = count_total_params(lora_model)
print(f"LoRA trainable params : {lora_trainable:,}  ({100*lora_trainable/lora_total:.2f}% of total)")
print(f"LoRA total params     : {lora_total:,}")

def lora_predict(pil_image):
    inputs = lora_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        logits = lora_model(pixel_values=inputs["pixel_values"].to(DEVICE)).logits
    probs    = torch.softmax(logits, dim=1)[0]
    idx      = probs.argmax().item()
    label_map = {0: "flower", 1: "fruit", 2: "leaf"}
    return label_map[idx], probs[idx].item()


# ── 2. Load EfficientNet-B0 ───────────────────────────────────────────────────

print("\n" + "="*60)
print("Loading EfficientNet-B0 baseline ...")
print("="*60)

effnet = models.efficientnet_b0(weights=None)
ckpt   = torch.load(args.effnet_path, map_location=DEVICE)
classes_effnet = ckpt["classes"]
effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, len(classes_effnet))
effnet.load_state_dict(ckpt["model_state"])
effnet.to(DEVICE).eval()

effnet_total = count_total_params(effnet)
print(f"EfficientNet-B0 total params : {effnet_total:,}  (all trainable, full fine-tune)")
print(f"Val acc from checkpoint      : {ckpt.get('val_acc', 'N/A')}")

_eff_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def effnet_predict(pil_image):
    t = _eff_tf(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = effnet(t)
    probs = torch.softmax(logits, dim=1)[0]
    idx   = probs.argmax().item()
    return classes_effnet[idx], probs[idx].item()


# ── 3. Evaluate both on val set ───────────────────────────────────────────────

val_dir = os.path.join(args.data_dir, "val")
print(f"\nEvaluating on: {val_dir}")

print("\n--- DINOv2 + LoRA ---")
lora_acc, lora_report, lora_cm, cn = evaluate(lora_predict, val_dir)
print(f"Accuracy: {lora_acc:.4f}")
print(json.dumps({k: v for k, v in lora_report.items() if k in CLASSES + ["macro avg", "weighted avg"]}, indent=2))
save_confusion_matrix(lora_cm, CLASSES, "DINOv2-small + LoRA",
                      os.path.join(args.output_dir, "cm_lora.png"))

print("\n--- EfficientNet-B0 ---")
eff_acc, eff_report, eff_cm, _ = evaluate(effnet_predict, val_dir)
print(f"Accuracy: {eff_acc:.4f}")
print(json.dumps({k: v for k, v in eff_report.items() if k in CLASSES + ["macro avg", "weighted avg"]}, indent=2))
save_confusion_matrix(eff_cm, CLASSES, "EfficientNet-B0 (baseline)",
                      os.path.join(args.output_dir, "cm_effnet.png"))


# ── 4. Latency measurement ────────────────────────────────────────────────────

print("\n" + "="*60)
print("Measuring inference latency (CPU) ...")
print("="*60)

# Use a random sample image from val
sample_path = next(
    p for cls in CLASSES
    for p in Path(val_dir).glob(f"{cls}/*.jpg")
)
sample_img = Image.open(sample_path).convert("RGB")

lora_mean, lora_std = measure_latency(lora_predict,   sample_img, args.n_timing_runs)
eff_mean,  eff_std  = measure_latency(effnet_predict, sample_img, args.n_timing_runs)

print(f"DINOv2+LoRA   : {lora_mean:.1f} ± {lora_std:.1f} ms")
print(f"EfficientNet  : {eff_mean:.1f} ± {eff_std:.1f} ms")


# ── 5. Save results ───────────────────────────────────────────────────────────

results = {
    "lora": {
        "accuracy": lora_acc,
        "macro_f1": lora_report["macro avg"]["f1-score"],
        "per_class": {c: lora_report[c] for c in CLASSES if c in lora_report},
        "trainable_params": lora_trainable,
        "total_params": lora_total,
        "trainable_pct": round(100 * lora_trainable / lora_total, 4),
        "latency_mean_ms": lora_mean,
        "latency_std_ms": lora_std,
        "confusion_matrix": lora_cm.tolist(),
    },
    "efficientnet_b0": {
        "accuracy": eff_acc,
        "macro_f1": eff_report["macro avg"]["f1-score"],
        "per_class": {c: eff_report[c] for c in CLASSES if c in eff_report},
        "total_params": effnet_total,
        "latency_mean_ms": eff_mean,
        "latency_std_ms": eff_std,
        "confusion_matrix": eff_cm.tolist(),
    }
}

out_path = os.path.join(args.output_dir, "results_summary.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {out_path}")
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Metric':<30} {'DINOv2+LoRA':>15} {'EfficientNet-B0':>17}")
print("-"*62)
print(f"{'Accuracy':<30} {lora_acc:>15.4f} {eff_acc:>17.4f}")
print(f"{'Macro F1':<30} {lora_report['macro avg']['f1-score']:>15.4f} {eff_report['macro avg']['f1-score']:>17.4f}")
print(f"{'Trainable params':<30} {lora_trainable:>15,} {effnet_total:>17,}")
print(f"{'Trainable %':<30} {lora_trainable/lora_total*100:>14.2f}% {'100.00%':>17}")
print(f"{'Latency (ms, CPU)':<30} {lora_mean:>12.1f} ms {eff_mean:>14.1f} ms")

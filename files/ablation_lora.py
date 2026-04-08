"""
ablation_lora.py
================
Ablation study for the LoRA router paper.

Experiments:
    A. Rank sweep          — r in {4, 8, 16, 32}  (alpha fixed at 2r)
    B. Target module sweep — Q+V vs Q+K+V vs all linear layers
    C. Confidence threshold sweep — threshold in {0.50, 0.55, 0.60, 0.65, 0.70, 0.80}
       (measured as unknown-rejection rate + routing accuracy on known samples)

Each experiment trains from scratch using the same base model and dataset split.

Requirements:
    pip install transformers peft safetensors torch torchvision scikit-learn

Dataset layout:
    router_dataset/
        train/
            flower/
            fruit/
            leaf/
        val/
            flower/
            fruit/
            leaf/

Usage:
    python ablation_lora.py --data_dir /path/to/router_dataset --experiment all
    python ablation_lora.py --data_dir /path/to/router_dataset --experiment rank
    python ablation_lora.py --data_dir /path/to/router_dataset --experiment modules
    python ablation_lora.py --data_dir /path/to/router_dataset --experiment threshold
"""

import os
import json
import time
import argparse
import warnings

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str, required=True)
parser.add_argument("--output_dir", type=str, default="ablation_results")
parser.add_argument("--experiment", type=str, default="all",
                    choices=["rank", "modules", "threshold", "all"])
parser.add_argument("--epochs",     type=int, default=10,
                    help="Training epochs per ablation run (10 is enough for convergence here)")
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
DEVICE  = "cpu"
CLASSES = ["flower", "fruit", "leaf"]
BASE_MODEL = "facebook/dinov2-small"

# ── shared transforms ─────────────────────────────────────────────────────────

TRAIN_TF = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── dataset loader ────────────────────────────────────────────────────────────

def get_loaders(data_dir, batch_size):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=VAL_TF)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

# ── model builder ─────────────────────────────────────────────────────────────

def build_lora_model(rank, target_modules, alpha=None):
    """Build DINOv2-small with a LoRA adapter with given rank and target modules."""
    from transformers import AutoModelForImageClassification
    from peft import get_peft_model, LoraConfig, TaskType

    if alpha is None:
        alpha = rank * 2  # Standard ratio

    base = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CLASSES),
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)},
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(base, lora_cfg).to(DEVICE)
    return model


# ── training loop ─────────────────────────────────────────────────────────────

def train_and_eval(model, train_loader, val_loader, epochs, lr=2e-4, save_path=None):
    """Train classifier head + LoRA weights, return best val accuracy and F1."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_f1 = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(pixel_values=imgs).logits
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(DEVICE)
                logits = model(pixel_values=imgs).logits
                preds  = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="macro")
        if acc > best_acc:
            best_acc = acc
            best_f1  = f1
            if save_path:
                model.save_pretrained(save_path)
                from transformers import AutoImageProcessor
                processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
                processor.save_pretrained(save_path)
        print(f"  Epoch {epoch+1:02d}/{epochs} — val_acc={acc:.4f}  macro_f1={f1:.4f}")

    return best_acc, best_f1


def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = sum(p.numel() for p in model.parameters())
    return t, n


# ── Experiment A: Rank sweep ──────────────────────────────────────────────────

def experiment_rank(train_loader, val_loader):
    print("\n" + "="*60)
    print("EXPERIMENT A: LoRA Rank Sweep (r ∈ {4, 8, 16, 32})")
    print("Target modules fixed: query, value")
    print("="*60)

    ranks   = [4, 8, 16, 32]
    results = []

    for r in ranks:
        print(f"\n--- Rank r={r} ---")
        model = build_lora_model(rank=r, target_modules=["query", "value"])
        trainable, total = count_trainable(model)
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.3f}%)")

        t0 = time.time()
        save_path = os.path.join(args.output_dir, "best_r8_checkpoint") if r == 8 else None
        acc, f1 = train_and_eval(model, train_loader, val_loader, args.epochs, save_path=save_path)
        elapsed = time.time() - t0

        entry = {
            "rank": r,
            "alpha": r * 2,
            "target_modules": ["query", "value"],
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "best_val_acc": round(acc, 4),
            "best_macro_f1": round(f1, 4),
            "training_time_s": round(elapsed, 1),
        }
        results.append(entry)
        print(f"Best val_acc={acc:.4f}  macro_f1={f1:.4f}  time={elapsed:.0f}s")

    out = os.path.join(args.output_dir, "ablation_rank.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n{:<8} {:<12} {:<20} {:<12} {:<12}".format(
        "Rank", "Alpha", "Trainable Params", "Val Acc", "Macro F1"))
    print("-"*64)
    for r in results:
        print("{:<8} {:<12} {:<20,} {:<12.4f} {:<12.4f}".format(
            r["rank"], r["alpha"], r["trainable_params"],
            r["best_val_acc"], r["best_macro_f1"]))

    return results


# ── Experiment B: Target module sweep ─────────────────────────────────────────

def experiment_modules(train_loader, val_loader):
    print("\n" + "="*60)
    print("EXPERIMENT B: Target Module Sweep (rank fixed at r=8)")
    print("="*60)

    configs = [
        {"label": "Q+V only",        "modules": ["query", "value"]},
        {"label": "Q+K+V",           "modules": ["query", "key", "value"]},
        {"label": "Q+K+V+Proj",      "modules": ["query", "key", "value", "projection"]},
        {"label": "All linear",       "modules": None},  # None = all linear layers
    ]

    results = []

    for cfg in configs:
        label   = cfg["label"]
        modules = cfg["modules"]
        print(f"\n--- {label} ---")

        model = build_lora_model(rank=8, target_modules=modules)
        trainable, total = count_trainable(model)
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.3f}%)")

        acc, f1 = train_and_eval(model, train_loader, val_loader, args.epochs)

        entry = {
            "label": label,
            "target_modules": modules,
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 4),
            "best_val_acc": round(acc, 4),
            "best_macro_f1": round(f1, 4),
        }
        results.append(entry)

    out = os.path.join(args.output_dir, "ablation_modules.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n{:<20} {:<20} {:<12} {:<12}".format(
        "Config", "Trainable Params", "Val Acc", "Macro F1"))
    print("-"*64)
    for r in results:
        print("{:<20} {:<20,} {:<12.4f} {:<12.4f}".format(
            r["label"], r["trainable_params"],
            r["best_val_acc"], r["best_macro_f1"]))

    return results


# ── Experiment C: Confidence threshold sweep ──────────────────────────────────

def experiment_threshold(lora_weights_dir, val_dir):
    """
    Loads the trained LoRA model and sweeps the unknown-rejection threshold.
    Reports routing accuracy on accepted samples and rejection rate at each threshold.

    Requires the trained LoRA weights to already exist (run eval_router.py first,
    or use the weights from model-service/weights/router_model_final/).
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: Confidence Threshold Sweep")
    print(f"Using weights: {lora_weights_dir}")
    print("="*60)

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from peft import PeftModel
    from sklearn.metrics import accuracy_score

    processor = AutoImageProcessor.from_pretrained(lora_weights_dir)
    base = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label={0: "flower", 1: "fruit", 2: "leaf"},
        label2id={"flower": 0, "fruit": 1, "leaf": 2},
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, lora_weights_dir).to(DEVICE)
    model.eval()

    # Collect all predictions and confidences on val set
    raw_preds, raw_confs, raw_labels = [], [], []

    for cls in CLASSES:
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                logits = model(pixel_values=inputs["pixel_values"].to(DEVICE)).logits
            probs = torch.softmax(logits, dim=1)[0]
            idx   = probs.argmax().item()
            conf  = probs[idx].item()
            raw_preds.append(CLASSES[idx])
            raw_confs.append(conf)
            raw_labels.append(cls)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []

    for thresh in thresholds:
        accepted_preds, accepted_labels, n_rejected = [], [], 0
        for pred, conf, label in zip(raw_preds, raw_confs, raw_labels):
            if conf < thresh:
                n_rejected += 1
            else:
                accepted_preds.append(pred)
                accepted_labels.append(label)

        n_total    = len(raw_preds)
        reject_pct = 100 * n_rejected / n_total
        acc        = accuracy_score(accepted_labels, accepted_preds) if accepted_preds else 0.0
        f1         = f1_score(accepted_labels, accepted_preds, average="macro",
                              zero_division=0) if accepted_preds else 0.0

        entry = {
            "threshold": thresh,
            "n_accepted": len(accepted_preds),
            "n_rejected": n_rejected,
            "rejection_rate_pct": round(reject_pct, 2),
            "routing_accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        }
        results.append(entry)
        print(f"  threshold={thresh:.2f}  accepted={len(accepted_preds):4d}  "
              f"rejected={n_rejected:3d} ({reject_pct:.1f}%)  "
              f"acc={acc:.4f}  f1={f1:.4f}")

    out = os.path.join(args.output_dir, "ablation_threshold.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}

    if args.experiment in ("rank", "modules", "all"):
        train_loader, val_loader = get_loaders(args.data_dir, args.batch_size)

    if args.experiment in ("rank", "all"):
        all_results["rank"] = experiment_rank(train_loader, val_loader)

    if args.experiment in ("modules", "all"):
        all_results["modules"] = experiment_modules(train_loader, val_loader)

    if args.experiment in ("threshold", "all"):
        # Threshold sweep uses existing trained weights — no retraining needed
        lora_dir = os.path.join(args.output_dir, "best_r8_checkpoint")
        val_dir = os.path.join(args.data_dir, "val")
        all_results["threshold"] = experiment_threshold(lora_dir, val_dir)

    out = os.path.join(args.output_dir, "ablation_all.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll ablation results saved to: {out}")

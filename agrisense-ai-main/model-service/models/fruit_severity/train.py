"""
model-service/models/fruit_severity/train.py
============================================
Train EfficientNet-B4 regression head for fruit severity estimation.

Key design decisions (v4)
--------------------------
1. Subsample the Healthy (0 %) class to MAX_HEALTHY images to prevent the
   large healthy cohort from anchoring all predictions toward zero.
2. WeightedRandomSampler gives every high-severity sample a proportionally
   higher chance of appearing in each mini-batch — fixing the critical
   under-representation of the 60–100 % severity bucket.
3. Severity-weighted MSE loss: samples with higher true severity receive a
   larger gradient signal, forcing the head to learn the upper tail rather
   than regressing to the mean.
4. Linear (no Sigmoid) final layer so the model can freely output across
   the full [0, 1] range; predictions are clamped at inference.

Dataset layout expected
-----------------------
fruit_disease_dataset/
    <ClassName>/
        images/      ← image files (.jpg / .JPG / .jpeg / .png)
        annotations/ ← labelme JSON files (same stem as image)

Usage (from repo root):
    python model-service/models/fruit_severity/train.py
"""

import os, sys, glob, json, warnings
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

# ── paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_MS_ROOT    = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))

DATA_ROOT  = os.path.join(_REPO_ROOT, "fruit_disease_dataset")
SAVE_PATH  = os.path.join(_MS_ROOT, "weights", "fruit_severity_trained.pth")
CACHE_PATH = os.path.join(_MS_ROOT, "weights", ".severity_features_cache_v4.npz")

# ── hyper-parameters ──────────────────────────────────────────────────────────

IMG_SIZE     = 224
BATCH_SIZE   = 64
N_AUG        = 2         # colour-augmented feature extractions per image
HEAD_EPOCHS  = 150
LR           = 5e-4
VAL_SPLIT    = 0.15
DEVICE       = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Cap on healthy (0 %) images — prevents zero-anchor bias.
# Kept at ~20 % of remaining disease samples.
MAX_HEALTHY  = 40

# ── label sets ────────────────────────────────────────────────────────────────

HEALTHY_LABELS = {"Healthy_Apple", "healthy_apple", "Healthy", "healthy"}
IMAGE_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

# ── severity from polygon annotations ─────────────────────────────────────────

def compute_severity(json_path: str) -> float:
    with open(json_path) as f:
        data = json.load(f)
    w, h = data.get("imageWidth", 512), data.get("imageHeight", 512)
    affected_mask = Image.new("L", (w, h), 0)
    fruit_mask    = Image.new("L", (w, h), 0)
    draw_aff      = ImageDraw.Draw(affected_mask)
    draw_fruit    = ImageDraw.Draw(fruit_mask)
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        pts   = [tuple(p) for p in shape.get("points", [])]
        if len(pts) < 3:
            continue
        if label in HEALTHY_LABELS:
            draw_fruit.polygon(pts, fill=255)
        else:
            draw_aff.polygon(pts,   fill=255)
            draw_fruit.polygon(pts, fill=255)
    fruit_px    = float(np.array(fruit_mask).sum())    / 255.0
    affected_px = float(np.array(affected_mask).sum()) / 255.0
    if fruit_px < 1.0:
        return 0.0
    return min(100.0, round(affected_px / fruit_px * 100.0, 2))


# ── collect (image_path, severity) pairs ─────────────────────────────────────

def collect_samples(data_root: str):
    """
    Walk fruit_disease_dataset/<ClassName>/images/ and pair with annotations.
    Healthy images without annotations → 0.0 severity (capped at MAX_HEALTHY).
    Non-healthy images without annotations are skipped.
    """
    disease_samples = []
    healthy_samples = []

    if not os.path.isdir(data_root):
        sys.exit(f"[train] Data root not found: {data_root}")

    for class_dir in sorted(os.listdir(data_root)):
        full_class_dir = os.path.join(data_root, class_dir)
        if not os.path.isdir(full_class_dir):
            continue
        images_dir = os.path.join(full_class_dir, "images")
        ann_dir    = os.path.join(full_class_dir, "annotations")
        if not os.path.isdir(images_dir):
            continue
        is_healthy = class_dir.lower() == "healthy"

        for img_file in sorted(os.listdir(images_dir)):
            if not img_file.lower().endswith(IMAGE_EXTS):
                continue
            img_path  = os.path.join(images_dir, img_file)
            stem      = os.path.splitext(img_file)[0]
            json_path = os.path.join(ann_dir, stem + ".json")

            if os.path.exists(json_path):
                severity = compute_severity(json_path)
                if is_healthy:
                    healthy_samples.append((img_path, severity))
                else:
                    disease_samples.append((img_path, severity))
            elif is_healthy:
                healthy_samples.append((img_path, 0.0))

    # Subsample healthy to prevent zero-anchor bias
    rng = random.Random(42)
    rng.shuffle(healthy_samples)
    healthy_samples = healthy_samples[:MAX_HEALTHY]

    samples = disease_samples + healthy_samples
    samples.sort(key=lambda x: x[0])   # deterministic order for cache key
    return samples


# ── colour-augmented transforms ───────────────────────────────────────────────

_BASE_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _augmented_tf(seed: int):
    rng = random.Random(seed)

    def transform(img: Image.Image) -> torch.Tensor:
        if rng.random() > 0.5: img = TF.hflip(img)
        if rng.random() > 0.5: img = TF.vflip(img)
        img = TF.rotate(img, rng.uniform(-20, 20))
        img = TF.adjust_brightness(img, rng.uniform(0.6, 1.4))
        img = TF.adjust_contrast(img,   rng.uniform(0.7, 1.3))
        img = TF.adjust_saturation(img, rng.uniform(0.5, 1.5))
        img = TF.adjust_hue(img,        rng.uniform(-0.25, 0.25))
        if rng.random() < 0.15:
            img = TF.to_grayscale(img, num_output_channels=3)
        if rng.random() < 0.3:
            import PIL.ImageFilter as IF
            img = img.filter(IF.GaussianBlur(radius=rng.choice([1, 2])))
        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        t   = TF.to_tensor(img)
        t   = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t

    return transform


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(samples, backbone, device):
    backbone.eval()
    all_feats, all_labels = [], []
    n = len(samples)
    with torch.no_grad():
        for aug_idx in range(N_AUG):
            label_str = "original" if aug_idx == 0 else f"augment-{aug_idx}"
            print(f"  Pass {aug_idx+1}/{N_AUG}  ({label_str}) …")
            batch_imgs, batch_lbls = [], []
            for i, (img_path, severity) in enumerate(samples):
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
                t = _BASE_TF(img) if aug_idx == 0 else _augmented_tf(aug_idx * 100000 + i)(img)
                batch_imgs.append(t)
                batch_lbls.append(severity / 100.0)
                if len(batch_imgs) == BATCH_SIZE or i == n - 1:
                    tensor = torch.stack(batch_imgs).to(device)
                    feats  = backbone(tensor).cpu().numpy()
                    all_feats.append(feats)
                    all_labels.extend(batch_lbls)
                    batch_imgs, batch_lbls = [], []
            print(f"    {n} images done for pass {aug_idx+1}")
    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    labels   = np.array(all_labels, dtype=np.float32)
    return features, labels


# ── weighted MSE loss ─────────────────────────────────────────────────────────

def severity_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE with per-sample weight proportional to true severity.
    Samples at 0 % receive weight 0.4 (still trained on, but down-weighted).
    Samples at 100 % receive weight 3.0 — 7.5× more gradient signal.
    This forces the head to correctly fit the upper tail of the distribution
    instead of collapsing to the mean.
    """
    # weight = 0.4 + 2.6 * label   →  0%→0.4,  50%→1.7,  100%→3.0
    weight = 0.4 + 2.6 * target
    return (weight * (pred - target).pow(2)).mean()


# ── dataset ───────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── main ──────────────────────────────────────────────────────────────────────

def train():
    print(f"[train] Device     : {DEVICE}")
    print(f"[train] N_AUG      : {N_AUG}")
    print(f"[train] MAX_HEALTHY: {MAX_HEALTHY}  (zero-anchor cap)")
    print(f"[train] Output     : {SAVE_PATH}")

    # 1. Collect samples
    print(f"\n[1/3] Collecting samples from {DATA_ROOT} …")
    samples = collect_samples(DATA_ROOT)
    if not samples:
        sys.exit("[train] No samples found — check DATA_ROOT path.")

    raw_labels = np.array([s[1] for s in samples], dtype=np.float32)
    print(f"      {len(samples)} samples  |  mean={raw_labels.mean():.1f}%  "
          f"max={raw_labels.max():.1f}%  non-zero={(raw_labels > 0).sum()}")
    for lo, hi in [(0,0),(0.1,20),(20,40),(40,60),(60,80),(80,100)]:
        n = int(((raw_labels >= lo) & (raw_labels <= hi)).sum())
        print(f"      [{lo:4.0f}–{hi:3.0f}%]: {n:3d} samples")

    # 2. Build frozen backbone
    print("\n[2/3] Building EfficientNet-B4 backbone (pretrained, frozen) …")
    full_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    feat_dim   = full_model.classifier[1].in_features   # 1792
    backbone   = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten()).to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # 3. Feature extraction / cache
    cache_valid = False
    if os.path.exists(CACHE_PATH):
        print(f"      Found cache at {CACHE_PATH} — validating …")
        try:
            cache = np.load(CACHE_PATH)
            expected_rows = len(samples) * N_AUG
            if (cache["features"].shape == (expected_rows, feat_dim)
                    and int(cache["n_aug"]) == N_AUG
                    and list(cache["paths"]) == [s[0] for s in samples]):
                features, labels = cache["features"], cache["labels"]
                cache_valid = True
                print(f"      Cache valid: {features.shape}")
        except Exception as e:
            print(f"      Cache corrupt ({e}) — re-extracting.")

    if not cache_valid:
        print(f"      Extracting {N_AUG} × {len(samples)} = {N_AUG*len(samples)} vectors …")
        features, labels = extract_features(samples, backbone, DEVICE)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, features=features, labels=labels,
                 paths=np.array([s[0] for s in samples]), n_aug=N_AUG)
        print(f"      Saved feature cache → {CACHE_PATH}")

    print(f"      Feature bank: {features.shape}  "
          f"labels min={labels.min():.3f} max={labels.max():.3f} mean={labels.mean():.3f}")

    # 4. Train/val split (stratified by severity quartile)
    print(f"\n[3/3] Training regression head ({HEAD_EPOCHS} epochs) …")
    n_orig  = len(samples)
    n_val   = max(1, int(n_orig * VAL_SPLIT))
    n_train = n_orig - n_val

    rng   = torch.Generator().manual_seed(42)
    idx   = torch.randperm(n_orig, generator=rng).numpy()
    t_idx, v_idx = idx[:n_train], idx[n_train:]

    aug_t_idx = np.concatenate([t_idx + aug * n_orig for aug in range(N_AUG)])
    aug_v_idx = v_idx   # validation: original images only

    train_labels_used = labels[aug_t_idx]

    # WeightedRandomSampler — oversample high-severity cases
    # weight = sqrt(severity + 0.05) so 0 % → 0.22, 50 % → 0.73, 100 % → 1.02
    sample_weights = torch.tensor(
        np.sqrt(train_labels_used + 0.05).astype(np.float32)
    )
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = FeatureDataset(features[aug_t_idx], labels[aug_t_idx])
    val_ds   = FeatureDataset(features[aug_v_idx], labels[aug_v_idx])

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    print(f"      Train vectors: {len(train_ds)}  |  Val images: {len(val_ds)}")

    # Regression head — linear output, no Sigmoid
    head = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HEAD_EPOCHS, eta_min=1e-5
    )

    best_val_mae = float("inf")
    best_epoch   = 0

    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        train_loss = 0.0
        for feats, targets in train_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = head(feats)
            loss  = severity_weighted_mse(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_ds)

        head.eval()
        val_mae = 0.0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
                preds = head(feats).clamp(0.0, 1.0)
                val_mae += (preds - targets).abs().sum().item() * 100.0
        val_mae /= len(val_ds)
        scheduler.step()

        if epoch % 15 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{HEAD_EPOCHS}  "
                  f"train_loss={train_loss:.5f}  val_MAE={val_mae:.2f}%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            # Assemble full model for saving
            full_model.classifier = nn.Sequential(
                full_model.classifier[0],   # original Dropout(p=0.4)
                head[0],  head[1],  head[2],   # Linear 1792→512, ReLU, Dropout
                head[3],  head[4],  head[5],   # Linear 512→256, ReLU, Dropout
                head[6],  head[7],             # Linear 256→64, ReLU
                head[8],                       # Linear 64→1
            )
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({"model_state": full_model.state_dict()}, SAVE_PATH)

    print(f"\n[train] Done.  Best val MAE = {best_val_mae:.2f}%  "
          f"(epoch {best_epoch})  →  {SAVE_PATH}")


if __name__ == "__main__":
    train()

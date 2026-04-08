"""
models/leaf_disease/train.py
============================
Fine-tunes the leaf disease classifier to improve rust detection.

Strategy (CPU-friendly):
  1. Feature extraction  — run all images through the frozen EfficientNet-V2-S
     backbone ONCE and cache the feature vectors on disk.  This is fast and
     memory-efficient because only the lightweight classifier head needs
     gradients during training.

  2. Head fine-tuning   — train a new classifier head on the cached features
     with rust-weighted CrossEntropy loss.

  3. Weight injection   — graft the new head weights back into the original
     EfficientNet-V2-S model and save a full checkpoint.

Usage (run from the model-service directory):
    python models/leaf_disease/train.py

Output: weights/leaf_disease_v2.pt
"""

import os
import time
import copy
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, TensorDataset

# ── paths ──────────────────────────────────────────────────────────────────

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.path.abspath(os.path.join(SERVICE_ROOT, "..", "leaf_disease_dataset"))
WEIGHTS_DIR  = os.path.join(SERVICE_ROOT, "weights")
OLD_CKPT     = os.path.join(WEIGHTS_DIR, "efficientnetb0_astha.pt")
NEW_CKPT     = os.path.join(WEIGHTS_DIR, "leaf_disease_v2.pt")

# ── config ─────────────────────────────────────────────────────────────────

DEVICE       = "cpu"   # CPU-only, but feature caching makes this fast
BATCH_SIZE   = 16
EPOCHS       = 60      # head-only; very fast because backbone is frozen
LR           = 1e-3
WEIGHT_DECAY = 1e-4

LEAF_CLASSES = [
    "alternaria leaf spot",
    "brown spot",
    "gray spot",
    "healthy leaf",
    "rust",
]
RUST_IDX = LEAF_CLASSES.index("rust")

# Rust gets 3× weight in the loss — the main lever for fixing recall
RUST_WEIGHT  = 3.0
_w = [1.0] * len(LEAF_CLASSES)
_w[RUST_IDX] = RUST_WEIGHT
CLASS_WEIGHTS = torch.tensor(_w, dtype=torch.float32)

# ── transforms ─────────────────────────────────────────────────────────────

# For feature extraction — deterministic
EXTRACT_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# For training augmentation — run multiple augmented passes to expand the
# effective dataset.  We'll extract features from N_AUG augmented copies.
N_AUG = 0   # feature extraction only — augmentation handled by weighted sampler
AUG_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── model builder ──────────────────────────────────────────────────────────

def load_backbone() -> tuple[nn.Module, nn.Module]:
    """
    Load the existing EfficientNet-V2-S checkpoint and split into
    (backbone, old_head).  Returns backbone in eval mode, frozen.
    """
    net = models.efficientnet_v2_s(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(LEAF_CLASSES))

    # Load checkpoint
    raw = torch.load(OLD_CKPT, map_location="cpu")
    state = raw if not isinstance(raw, dict) or "model_state" not in raw else raw["model_state"]
    net.load_state_dict(state, strict=False)

    # Split backbone (features + avgpool) from head (classifier)
    backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten())
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    return backbone, net


# ── feature extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone: nn.Module, root: str, augmented: bool) -> tuple:
    """
    Extract feature vectors from all images under *root*.
    Returns (features_tensor, labels_tensor).
    """
    tf_list = [EXTRACT_TF] + ([AUG_TF] * N_AUG if augmented else [])
    all_feats, all_labels = [], []

    for tf in tf_list:
        ds     = datasets.ImageFolder(root=root, transform=tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        for imgs, labs in loader:
            feats = backbone(imgs)
            all_feats.append(feats)
            all_labels.append(labs)

    return torch.cat(all_feats), torch.cat(all_labels)


# ── head training ──────────────────────────────────────────────────────────

def make_head(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


def train_head(head, train_feats, train_labels, val_feats, val_labels):
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Weighted sampler: rust samples drawn 3× more often than others
    sample_weights = torch.ones(len(train_labels))
    sample_weights[train_labels == RUST_IDX] = RUST_WEIGHT
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_labels), replacement=True
    )

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds   = TensorDataset(val_feats,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    best_acc   = 0.0
    best_state = copy.deepcopy(head.state_dict())

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        head.train()
        correct, total = 0, 0
        for feats, labs in train_loader:
            optimizer.zero_grad()
            out  = head(feats)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labs).sum().item()
            total   += labs.size(0)
        tr_acc = correct / total

        head.eval()
        val_correct, val_total = 0, 0
        cls_correct = [0] * len(LEAF_CLASSES)
        cls_total   = [0] * len(LEAF_CLASSES)
        with torch.no_grad():
            for feats, labs in val_loader:
                preds = head(feats).argmax(1)
                val_correct += (preds == labs).sum().item()
                val_total   += labs.size(0)
                for i in range(len(LEAF_CLASSES)):
                    m = (labs == i)
                    cls_correct[i] += (preds[m] == labs[m]).sum().item()
                    cls_total[i]   += m.sum().item()
        vl_acc   = val_correct / val_total
        rust_acc = cls_correct[RUST_IDX] / max(cls_total[RUST_IDX], 1)
        scheduler.step()

        improved = ""
        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = copy.deepcopy(head.state_dict())
            improved   = " ← best"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}  tr={tr_acc:.3f}  val={vl_acc:.3f}  "
                  f"rust_recall={rust_acc:.3f}  ({time.time()-t0:.1f}s){improved}")

    return best_state, best_acc


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Dataset : {DATASET_ROOT}")
    print(f"Device  : {DEVICE}")

    backbone, net = load_backbone()
    in_features   = net.features[-1][-1].out_channels if hasattr(net.features[-1][-1], 'out_channels') else 1280

    # Try to get in_features from the backbone output
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        in_features = backbone(dummy).shape[1]
    print(f"Feature dim: {in_features}")

    # ── extract features ───────────────────────────────────────────────
    print("\nExtracting train features (with augmentation)…")
    t0 = time.time()
    train_feats, train_labels = extract_features(
        backbone, os.path.join(DATASET_ROOT, "train"), augmented=True
    )
    print(f"  {len(train_feats)} samples in {time.time()-t0:.1f}s")

    print("Extracting val features…")
    t0 = time.time()
    val_feats, val_labels = extract_features(
        backbone, os.path.join(DATASET_ROOT, "val"), augmented=False
    )
    print(f"  {len(val_feats)} samples in {time.time()-t0:.1f}s")

    # ── class distribution ─────────────────────────────────────────────
    print("\nClass distribution in train features:")
    for i, cls in enumerate(LEAF_CLASSES):
        n = (train_labels == i).sum().item()
        print(f"  {cls:25s}: {n:4d}")

    # ── train head ─────────────────────────────────────────────────────
    print(f"\nTraining head ({EPOCHS} epochs, rust_weight={RUST_WEIGHT}×)…")
    head      = make_head(in_features, len(LEAF_CLASSES))
    best_state, best_acc = train_head(head, train_feats, train_labels,
                                       val_feats, val_labels)
    print(f"\nBest val accuracy: {best_acc:.4f}")

    # ── inject head back into full model ───────────────────────────────
    head.load_state_dict(best_state)
    # net.classifier = [Dropout, Linear] — replace with new head
    net.classifier = head
    full_state = net.state_dict()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(
        {
            "model_state": full_state,
            "classes":     LEAF_CLASSES,
            "val_acc":     best_acc,
            "arch":        "efficientnet_v2_s",
        },
        NEW_CKPT,
    )
    print(f"✅ Saved → {NEW_CKPT}")

    # ── test set eval ──────────────────────────────────────────────────
    print("\nTest set evaluation…")
    net.eval()
    full_backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten())
    test_feats, test_labels = extract_features(
        full_backbone, os.path.join(DATASET_ROOT, "test"), augmented=False
    )
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_feats), 64):
            out = net.classifier(test_feats[i:i+64])
            preds.append(out.argmax(1))
    preds = torch.cat(preds)
    test_acc = (preds == test_labels).float().mean().item()
    print(f"Test accuracy: {test_acc:.4f}")
    for i, cls in enumerate(LEAF_CLASSES):
        m = (test_labels == i)
        cls_acc = (preds[m] == test_labels[m]).float().mean().item() if m.sum() > 0 else 0.0
        marker = " ← rust" if i == RUST_IDX else ""
        print(f"  {cls:25s}: {cls_acc:.3f}{marker}")


if __name__ == "__main__":
    main()

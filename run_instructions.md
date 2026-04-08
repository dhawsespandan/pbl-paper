# Run Instructions — LoRA Router Evaluation & Ablation

## Prerequisites

```bash
pip install torch torchvision transformers peft safetensors scikit-learn matplotlib
```

---

## Step 1 — Fix the missing preprocessor_config.json

Copy the provided file into the weights directory:

```bash
cp preprocessor_config.json model-service/weights/router_model_final/preprocessor_config.json
```

This is the only file that was blocking the LoRA router from loading in production.
After copying it, the DINOv2+LoRA model will load correctly in the live app as well.

---

## Step 2 — Run the main evaluation (both models, val set)

```bash
python eval_router.py \
  --data_dir /path/to/router_dataset \
  --lora_dir model-service/weights/router_model_final \
  --effnet_path model-service/weights/router_efficientnet.pt \
  --output_dir eval_results
```

This produces:
- `eval_results/results_summary.json` — all numbers for the paper's results table
- `eval_results/cm_lora.png` — confusion matrix for DINOv2+LoRA (Figure for paper)
- `eval_results/cm_effnet.png` — confusion matrix for EfficientNet-B0

---

## Step 3 — Run ablation studies (one or all)

**All experiments:**
```bash
python ablation_lora.py --data_dir /path/to/router_dataset --experiment all --epochs 10
```

**Individual experiments:**
```bash
# Rank sweep only
python ablation_lora.py --data_dir /path/to/router_dataset --experiment rank

# Target module sweep only  
python ablation_lora.py --data_dir /path/to/router_dataset --experiment modules

# Confidence threshold sweep (uses existing weights — no training required)
python ablation_lora.py --data_dir /path/to/router_dataset --experiment threshold
```

Outputs go to `ablation_results/`:
- `ablation_rank.json`
- `ablation_modules.json`
- `ablation_threshold.json`
- `ablation_all.json`

---

## Dataset

If you do not have the router dataset split already, create it from your images:

```
router_dataset/
    train/
        flower/      ← apple flower cluster images
        fruit/       ← apple fruit images
        leaf/        ← apple leaf images
    val/
        flower/
        fruit/
        leaf/
```

Recommended split: 80% train / 20% val. A minimum of 50 images per class per split
is needed for reliable accuracy metrics.

---

## What each result maps to in the paper

| Output | Paper section |
|---|---|
| `results_summary.json` → accuracy, F1 | Section 6 results table |
| `results_summary.json` → trainable_params, trainable_pct | Section 5.3 and Table 2 |
| `results_summary.json` → latency_mean_ms | Section 6.4 inference cost comparison |
| `cm_lora.png`, `cm_effnet.png` | Figure in Section 7 |
| `ablation_rank.json` | Section 6.2 ablation table |
| `ablation_modules.json` | Section 6.2 ablation table |
| `ablation_threshold.json` | Section 6.3 threshold analysis |

---

## Expected runtime (CPU)

| Task | Estimated time |
|---|---|
| eval_router.py (100 val images) | 5–15 minutes |
| ablation rank sweep (4 runs × 10 epochs) | 2–4 hours on CPU |
| ablation module sweep (4 runs × 10 epochs) | 2–4 hours on CPU |
| ablation threshold sweep | < 5 minutes (no training) |

GPU (CUDA) reduces training times by ~5–8×.

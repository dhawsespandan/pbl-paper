# AgriSense AI — Apple Orchard Health Monitoring

An AI-powered full-stack web application for monitoring the health of apple orchards. Users upload a photo of an apple leaf, fruit, or flower cluster, and a multi-stage deep learning pipeline identifies diseases, estimates severity, counts flowers, and returns actionable agronomic recommendations — all in seconds.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [ML Pipeline — Deep Dive](#4-ml-pipeline--deep-dive)
5. [Project Structure](#5-project-structure)
6. [API Reference](#6-api-reference)
7. [Environment Variables](#7-environment-variables)
8. [Running Locally](#8-running-locally)
9. [Deployment Guide](#9-deployment-guide)
10. [Model Details & Weights](#10-model-details--weights)
11. [Known Limitations](#11-known-limitations)

---

## 1. Project Overview

AgriSense AI solves a real agricultural problem: apple growers need fast, reliable information about crop health without needing specialist expertise on-site. The application accepts a single image and returns:

- **What it is** — leaf, fruit, or flower cluster (automatic routing, no user selection needed)
- **Disease diagnosis** — specific disease or health label with confidence score
- **Severity** — how bad the condition is (percentage area affected for fruit, density level for flowers)
- **Flower count** — individual flowers detected in the frame (flower images only)
- **Recommendations** — concrete, actionable treatment advice
- **Details** — scientific background on the identified condition

### Supported Conditions

| Category | Detectable Conditions |
|---|---|
| **Fruit** | Anthracnose, Blotch, Rot, Scab, Healthy |
| **Leaf** | Alternaria Leaf Spot, Brown Spot, Gray Spot, Rust, Healthy Leaf |
| **Flower Cluster** | Healthy Clusters, Adequate Clusters, Sparse Clusters, No Clusters Detected |

---

## 2. Architecture

The system uses a **three-tier architecture** where each service runs independently:

```
┌─────────────────────────────────────────────────────────┐
│                    User's Browser                        │
│              React + Vite  (Port 5000)                  │
└────────────────────────┬────────────────────────────────┘
                         │  POST /api/analyze  (multipart image)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Express API Gateway  (Port 3001)            │
│  • Validates & buffers the uploaded image               │
│  • Enforces 10 MB file size limit                       │
│  • Accepts JPG / PNG / WebP only                        │
│  • Proxies the request to the Python service            │
└────────────────────────┬────────────────────────────────┘
                         │  POST /predict  (multipart image)
                         ▼
┌─────────────────────────────────────────────────────────┐
│           FastAPI Model Service  (Port 8000)             │
│  • Loads all ML models into memory at startup           │
│  • Runs the full inference pipeline per request         │
│  • Returns structured JSON diagnosis                    │
└─────────────────────────────────────────────────────────┘
```

**Why this split?**

The Express gateway decouples file handling and validation from ML logic. It also allows each tier to be hosted on a different platform — the frontend on Vercel, the gateway on Render, and the heavy model service on HuggingFace Spaces. All models are kept hot in memory after startup so there is no per-request loading overhead.

---

## 3. Tech Stack

### Frontend (`client/`)

| | |
|---|---|
| Framework | React 18 |
| Build Tool | Vite 6 |
| Language | TypeScript |
| Styling | Tailwind CSS v4 (Vite plugin) |
| Dev Server | `0.0.0.0:5000`, proxy `/api` → `localhost:3001` |

### API Gateway (`server/`)

| | |
|---|---|
| Runtime | Node.js 20 |
| Framework | Express 4 |
| File Handling | Multer (memory storage, 10 MB limit) |
| HTTP Client | Axios |
| Port | `3001` (localhost only) |

### Model Service (`model-service/`)

| | |
|---|---|
| Language | Python 3.12 |
| Framework | FastAPI + Uvicorn |
| Deep Learning | PyTorch + Torchvision |
| Object Detection | Ultralytics YOLOv8 |
| Transformer | HuggingFace Transformers + PEFT (LoRA) |
| Image Processing | Pillow, NumPy |
| Port | `8000` (localhost only) |

---

## 4. ML Pipeline — Deep Dive

Every uploaded image passes through the same pipeline regardless of content:

```
Input Image
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 0 — Crop Pre-filter           │
│  CLIP zero-shot check:               │
│  "apple leaf, fruit or flower"       │
│  Rejects unrelated images → HTTP 422 │
└──────────────────┬───────────────────┘
                   │ passes
                   ▼
┌──────────────────────────────────────┐
│  Stage 1 — Router                    │
│  Primary:  DINOv2-base + LoRA        │
│  Fallback: EfficientNet-B0           │
│  Output: fruit | leaf | flower       │
└──────────────────┬───────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   [fruit]      [leaf]   [flower]
        │          │          │
        ▼          ▼          ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────────┐
│ FruitDisease │ │LeafDisease│ │  FlowerCluster        │
│ EfficientNet │ │EfficientNet│ │  YOLOv8-m             │
│ B2 (5 class) │ │V2-S(5 cls)│ │  conf=0.10            │
└──────┬───────┘ └─────┬────┘ │  iou=0.50             │
       │               │      │  imgsz=1280            │
       ▼               │      └──────────┬─────────────┘
┌──────────────┐       │                 │
│ FruitSeverity│       │          count  →  label
│ EfficientNet │       │          ≥ 15  →  Healthy
│ B4 (% area)  │       │          6-14  →  Adequate
└──────┬───────┘       │          1-5   →  Sparse
       │               │          0     →  No Clusters
       └───────────────┘
                   │
                   ▼
          PredictionResponse JSON
```

### Stage 0 — Crop Pre-filter

Uses a CLIP-based zero-shot classifier to confirm the image contains apple crop content before any heavy model runs. Images that fail are rejected with HTTP 422, saving compute on irrelevant uploads (e.g. selfies, random objects).

### Stage 1 — Router

**Primary path:** DINOv2-base fine-tuned with LoRA (PEFT) — a vision transformer that distinguishes between the three image categories with high accuracy.

**Fallback path:** If the LoRA adapter is missing or fails to load, an EfficientNet-B0 classifier takes over automatically and prints a warning. The fallback achieves `val_acc = 1.0` on the held-out test set and is what runs in the current Replit deployment (the LoRA preprocessor config is absent from the weights directory).

Output: one of `fruit`, `leaf`, `flower_cluster`, or `unknown` (rejected with 422).

### Stage 2a — Fruit Branch

1. **FruitDisease (EfficientNet-B2):** Classifies into Anthracnose, Blotch, Healthy, Rot, or Scab. Validation accuracy ~88%.
2. **FruitSeverity (EfficientNet-B4):** A regression model estimating the percentage of the fruit's surface area affected by disease. Output is a human-readable string: `Mild — 12% area affected`, `Moderate — 35% area affected`, `Critical — 74% area affected`, etc.

### Stage 2b — Leaf Branch

**LeafDisease (EfficientNet-V2-S):** Classifies into Alternaria Leaf Spot, Brown Spot, Gray Spot, Healthy Leaf, or Rust. Preprocessing uses standard ImageNet normalisation at 224×224.

### Stage 2c — Flower Cluster Branch

**FlowerCluster (YOLOv8-m):** Object detection model that draws bounding boxes around individual apple flowers. The raw detection count determines the health label:

| Detected Flowers | Health Label | Pseudo-Confidence |
|---|---|---|
| ≥ 15 | Healthy Clusters | 92% |
| 6 – 14 | Adequate Clusters | 85% |
| 1 – 5 | Sparse Clusters | 80% |
| 0 | No Clusters Detected | 88% |

**Detection settings tuned for accuracy:**

| Parameter | Value | Reason |
|---|---|---|
| `conf` | `0.10` | Default (0.25) misses partially open buds — lower threshold catches them |
| `iou` | `0.50` | Moderate NMS separates adjacent overlapping flowers without over-splitting |
| `imgsz` | `1280` | Higher resolution than default (640) significantly improves detection in wide/distance shots |

At default settings (`conf=0.25`) the healthy test image returned 9 flowers. With tuned settings (`conf=0.10`, `imgsz=1280`) it returns 17 — an 89% improvement without retraining.

> **Note:** Tightly overlapping flowers in very dense clusters and partially open buds can still be undercounted. The UI displays a caveat: *"Partially open buds and tightly overlapping flowers in dense clusters may not be counted."*

---

## 5. Project Structure

```
agrisense-ai/
│
├── client/                              # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx               # Top navigation bar
│   │   │   ├── ImageInput.tsx           # Drag-and-drop / click-to-upload UI
│   │   │   ├── LoadingIndicator.tsx     # Spinner shown during inference
│   │   │   └── OutputBox.tsx            # Full results display card
│   │   ├── pages/
│   │   │   └── Home.tsx                 # Main page — upload state machine
│   │   ├── types/
│   │   │   └── index.ts                 # TypeScript interfaces
│   │   ├── constants/
│   │   │   └── index.ts                 # Shared constants
│   │   ├── App.tsx                      # Root component
│   │   ├── main.tsx                     # Vite entry point
│   │   └── index.css                    # Tailwind base styles
│   ├── index.html
│   ├── vite.config.ts                   # Dev server: host, port, /api proxy
│   ├── tsconfig.json
│   └── package.json
│
├── server/                              # Node.js / Express API gateway
│   ├── routes/
│   │   └── analyze.js                   # POST /api/analyze — validates + proxies
│   ├── middleware/
│   │   └── upload.js                    # Multer: 10 MB limit, JPG/PNG/WebP only
│   ├── app.js                           # Express app entry point
│   ├── .env                             # PORT, PYTHON_SERVICE_URL
│   └── package.json
│
├── model-service/                       # Python FastAPI ML inference service
│   ├── main.py                          # App entry, pipeline orchestration, disease KB
│   ├── models/
│   │   ├── router_transformer/
│   │   │   ├── model.py                 # DINOv2+LoRA router + EfficientNet-B0 fallback
│   │   │   └── crop_prefilter.py        # CLIP-based pre-filter (Stage 0)
│   │   ├── fruit_disease/
│   │   │   └── model.py                 # EfficientNet-B2 fruit disease classifier
│   │   ├── fruit_severity/
│   │   │   └── model.py                 # EfficientNet-B4 severity regression
│   │   ├── leaf_disease/
│   │   │   └── model.py                 # EfficientNet-V2-S leaf classifier
│   │   └── flower_cluster/
│   │       └── model.py                 # YOLOv8-m flower detector + counter
│   ├── schemas/
│   │   └── response.py                  # Pydantic models for request/response
│   ├── utils/
│   │   └── preprocess.py                # Shared image preprocessing utilities
│   ├── weights/                         # Pre-trained model weight files (~476 MB total)
│   │   ├── router_efficientnet.pt       # EfficientNet-B0 router (fallback, active)
│   │   ├── router_model_final/          # DINOv2+LoRA adapter weights (primary)
│   │   ├── fruit_efficientnet.pt        # EfficientNet-B2 fruit disease
│   │   ├── fruit_severity_trained.pth   # EfficientNet-B4 severity regression
│   │   ├── efficientnetb0_astha.pt      # Leaf disease weights (v1)
│   │   ├── leaf_disease_v2.pt           # Leaf disease weights (v2, active)
│   │   ├── efficientnetb4_spandan.pth   # Severity alternate weights
│   │   └── yolo26m_abhirami.pt          # YOLOv8-m flower detector
│   ├── requirements.txt
│   └── .env                             # DEVICE + weight file paths
│
├── start.sh                             # Launches all three services
├── replit.md                            # Replit-specific environment notes
└── README.md                            # This file
```

---

## 6. API Reference

### Express Gateway

#### `POST /api/analyze`

The only endpoint the frontend calls.

**Request**
```
Content-Type: multipart/form-data
Field name:   image
File types:   JPG, PNG, WebP
Max size:     10 MB
```

**Success — `200 OK`**
```json
{
  "image_type": "fruit",
  "disease": "Apple Scab",
  "severity": "Moderate — 35% area affected",
  "confidence": "91.4%",
  "recommendation": "Apply myclobutanil or captan fungicide at 7–10 day intervals during wet periods. Remove fallen leaves to break the disease cycle.",
  "details": "Apple scab (Venturia inaequalis) is a fungal disease causing dark, scabby lesions on fruit surfaces.",
  "flower_count": null
}
```

For flower cluster images, `flower_count` is populated:
```json
{
  "image_type": "flower_cluster",
  "disease": "Healthy Clusters",
  "severity": "None — clusters are abundant and healthy",
  "confidence": "92.0%",
  "recommendation": "No intervention needed. Monitor for pest pressure and ensure adequate pollinator access during bloom.",
  "details": "Flower clusters are abundant and dense. Conditions are highly favourable for pollination and fruit set.",
  "flower_count": 17
}
```

**Error — `422 Unprocessable Entity`**
```json
{
  "error": "Could not classify image. Please upload a clear image of an apple leaf, fruit, or flower cluster."
}
```

**Other status codes**

| Code | Cause |
|---|---|
| `400` | No image file in the request |
| `503` | Model service is not running (connection refused) |
| `502` | Model service returned an unexpected error |

#### `GET /health`

```json
{ "status": "ok", "service": "agrisense-ai-server" }
```

---

### FastAPI Model Service

Called internally by the Express gateway. Can be hit directly for testing.

**Base URL (local):** `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service name and version |
| `GET` | `/health` | Health check + active device (`cpu`/`cuda`) |
| `POST` | `/predict` | Run full ML inference pipeline |
| `GET` | `/docs` | Swagger UI (interactive, auto-generated) |
| `GET` | `/redoc` | ReDoc documentation |

**`POST /predict` — Request**
```
Content-Type: multipart/form-data
Field name:   file
File types:   JPG, PNG, WebP
```

Response schema is identical to the Express gateway's `200 OK` response above.

---

## 7. Environment Variables

### `server/.env`

```env
PORT=3001
PYTHON_SERVICE_URL=http://localhost:8000
```

| Variable | Description | Default |
|---|---|---|
| `PORT` | Port the Express server listens on | `3001` |
| `PYTHON_SERVICE_URL` | Base URL of the FastAPI model service | `http://localhost:8000` |

### `model-service/.env`

```env
DEVICE=cpu

ROUTER_LORA_DIR=weights/router_model_final
ROUTER_EFFICIENTNET_PATH=weights/router_efficientnet.pt

FRUIT_MODEL_PATH=weights/fruit_efficientnet.pt
SEVERITY_MODEL_PATH=weights/efficientnetb4_spandan.pth
LEAF_MODEL_PATH=weights/efficientnetb0_astha.pt
FLOWER_MODEL_PATH=weights/yolo26m_abhirami.pt
```

| Variable | Description |
|---|---|
| `DEVICE` | `cpu` or `cuda` — PyTorch inference device for all models |
| `ROUTER_LORA_DIR` | Directory of the DINOv2+LoRA adapter (primary router) |
| `ROUTER_EFFICIENTNET_PATH` | EfficientNet-B0 router weights (fallback, always loaded) |
| `FRUIT_MODEL_PATH` | EfficientNet-B2 fruit disease classifier weights |
| `SEVERITY_MODEL_PATH` | EfficientNet-B4 severity regression weights |
| `LEAF_MODEL_PATH` | EfficientNet leaf disease classifier weights |
| `FLOWER_MODEL_PATH` | YOLOv8-m flower detection weights |

---

## 8. Running Locally

### Prerequisites

- Node.js 20+
- Python 3.12+
- pip

### Step 1 — Clone

```bash
git clone <your-repo-url>
cd agrisense-ai
```

### Step 2 — Install dependencies

```bash
# Frontend
cd client && npm install && cd ..

# API gateway
cd server && npm install && cd ..

# Model service — core packages
pip install fastapi uvicorn[standard] python-multipart python-dotenv pydantic pillow numpy

# PyTorch (CPU build — faster to download, ~200 MB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Transformers and YOLO
pip install transformers peft accelerate safetensors ultralytics
```

Or install everything at once from the requirements file:
```bash
pip install -r model-service/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — Configure environment

```bash
# API gateway
cp server/.env.example server/.env      # edit PORT and PYTHON_SERVICE_URL if needed

# Model service
cp model-service/.env.example model-service/.env   # edit DEVICE and weight paths if needed
```

The defaults work out of the box if you haven't moved any files.

### Step 4 — Start all services

**Option A — single command:**
```bash
bash start.sh
```

**Option B — three separate terminals:**

```bash
# Terminal 1 — Model service (takes 20–60 s to load all models)
cd model-service
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — API gateway
cd server
node app.js

# Terminal 3 — Frontend
cd client
npm run dev
```

### Step 5 — Open the app

- **Frontend:** `http://localhost:5000`
- **FastAPI Swagger UI:** `http://localhost:8000/docs`

> The model service takes 20–60 seconds to start — all models are loaded into memory at startup. Wait for the `All models loaded — service ready` message in the terminal before uploading images.

---

## 9. Deployment Guide

The three services are designed to run on separate platforms. The recommended split:

| Tier | Platform | Notes |
|---|---|---|
| Frontend | **Vercel** | Free tier, automatic Vite builds |
| API Gateway | **Render** | Free web service, Node.js |
| Model Service | **HuggingFace Spaces** | Best for large weight files via Git LFS |

---

### Frontend → Vercel

1. Push the repo to GitHub.
2. Import the project in [Vercel](https://vercel.com). Set **Root Directory** to `client/`.
3. Vercel auto-detects Vite. Build command: `npm run build`. Output: `dist/`.
4. Update `client/vite.config.ts` to point the API proxy at your deployed Express URL:

```ts
proxy: {
  "/api": {
    target: "https://your-gateway.onrender.com",
    changeOrigin: true,
  },
},
```

Or set `VITE_API_BASE_URL` as an environment variable in Vercel and use it in the fetch calls.

---

### API Gateway → Render

1. Create a new **Web Service** on [Render](https://render.com). Point it to the `server/` directory.
2. **Build command:** `npm install`
3. **Start command:** `node app.js`
4. Add environment variables in the Render dashboard:

```
PORT=3001
PYTHON_SERVICE_URL=https://your-space.hf.space
```

---

### Model Service → HuggingFace Spaces

HuggingFace Spaces handles large binary files via Git LFS, making it the best fit for the ~476 MB weights directory.

1. Create a new Space — choose **Docker** as the SDK.
2. In the space repository, create a `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart \
    python-dotenv pydantic pillow numpy transformers peft accelerate \
    safetensors ultralytics
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN mkdir -p temp

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

3. Copy the entire `model-service/` folder contents into the space repository.
4. Track the weight files with Git LFS before pushing:

```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.safetensors" "*.npz"
git add .gitattributes
git add .
git commit -m "add model service with weights"
git push
```

5. Set environment variables in the Space settings:

```
DEVICE=cpu
FLOWER_MODEL_PATH=weights/yolo26m_abhirami.pt
FRUIT_MODEL_PATH=weights/fruit_efficientnet.pt
SEVERITY_MODEL_PATH=weights/efficientnetb4_spandan.pth
LEAF_MODEL_PATH=weights/efficientnetb0_astha.pt
ROUTER_EFFICIENTNET_PATH=weights/router_efficientnet.pt
```

> **HuggingFace note:** The Space will expose your service at `https://your-username-your-space-name.hf.space`. Use this URL as `PYTHON_SERVICE_URL` in your Render gateway.

---

### Alternative: Single VM (Render or Railway)

For simplicity during early testing, all three services can run on one VM:

1. **Build the frontend first:**
   ```bash
   cd client && npm run build
   ```

2. **Serve the built frontend from Express** by adding a static file handler to `server/app.js`:
   ```js
   const path = require("path");
   app.use(express.static(path.join(__dirname, "../client/dist")));
   ```

3. **Start command:** `bash start.sh` (with the Vite `dev` command replaced by the static serve).
4. Set all environment variables from both `.env` files in the platform dashboard.

---

## 10. Model Details & Weights

| Weight File | Architecture | Task | Approx. Size |
|---|---|---|---|
| `router_efficientnet.pt` | EfficientNet-B0 | Route image → fruit / leaf / flower (fallback, **active**) | ~21 MB |
| `router_model_final/` | DINOv2-base + LoRA | Route image → fruit / leaf / flower (primary) | ~330 MB |
| `fruit_efficientnet.pt` | EfficientNet-B2 | Fruit disease classification — 5 classes, val_acc ~88% | ~35 MB |
| `fruit_severity_trained.pth` | EfficientNet-B4 | Fruit severity regression — % pixel area affected | ~76 MB |
| `efficientnetb4_spandan.pth` | EfficientNet-B4 | Severity (alternate checkpoint) | ~76 MB |
| `leaf_disease_v2.pt` | EfficientNet-V2-S | Leaf disease classification — 5 classes (**active**) | varies |
| `efficientnetb0_astha.pt` | EfficientNet-V2-S | Leaf disease (v1 checkpoint) | varies |
| `yolo26m_abhirami.pt` | YOLOv8-m | Flower detection & counting | ~52 MB |

### Image Preprocessing

All CNN classifiers share the same pipeline (implemented in `utils/preprocess.py`):

```
PIL Image → Resize 224×224 → ToTensor → Normalize(ImageNet mean/std) → (1, 3, 224, 224)
```

ImageNet constants: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

The severity regression model uses the same resize but **no normalisation** (it was trained without it).

The YOLO flower model uses its own internal preprocessing pipeline at `imgsz=1280`.

### Device Selection

All models run on **CPU** by default (`DEVICE=cpu` in `.env`). Set `DEVICE=cuda` if a GPU is available for 5–10× faster inference. No code changes required — the device is read from the environment variable at startup.

---

## 11. Known Limitations

| Limitation | Detail |
|---|---|
| **Flower count underestimation** | Tightly overlapping flowers in dense clusters merge into single detections. Very partially open buds may be below the `conf=0.10` threshold. The true count is always ≥ the reported count. |
| **Confidence is pseudo-calibrated (flowers)** | For flower cluster images, the confidence value is assigned by count bucket — not a neural network softmax output. It does not reflect model uncertainty. |
| **No leaf severity** | Severity estimation (% area) is only available for fruit images. Leaf results show a qualitative severity string derived from the disease label, not a measured value. |
| **Zoom / distance sensitivity** | The flower count is frame-relative. A wide-angle shot of a full tree will detect far fewer flowers than a close-up of the same branch, even though the tree is identically healthy. |
| **LoRA router fallback** | The DINOv2+LoRA primary router needs a `preprocessor_config.json` in `weights/router_model_final/`. In the current deployment this file is absent, so the EfficientNet-B0 fallback runs. Since the fallback has `val_acc=1.0` on the test set, this does not affect output quality. |
| **CPU inference speed** | Loading all models takes 20–60 s at startup. Per-request inference takes 3–15 s on CPU depending on which branch runs (YOLO at `imgsz=1280` is the slowest). A GPU reduces both figures by ~5×. |
| **No streaming / progress** | The UI shows a spinner but does not stream intermediate results. The user must wait for the full pipeline to complete before seeing any output. |
| **Single image only** | There is no batch upload mode. Each request processes exactly one image. |

# AgriSense AI

An AI-powered full-stack web application for monitoring apple orchard health using deep learning.

## Architecture

Three-service architecture:

1. **Frontend** (`agrisense-ai-main/client/`) — React 18 + Vite 6 + TypeScript + Tailwind CSS v4
   - Runs on port **5000** (host `0.0.0.0`)
   - Proxies `/api` requests to the Node.js backend at port 3001

2. **API Gateway** (`agrisense-ai-main/server/`) — Node.js + Express 4
   - Runs on port **3001** (host `localhost`)
   - Validates uploads and proxies to the Python model service
   - Configured via `agrisense-ai-main/server/.env`

3. **Model Service** (`agrisense-ai-main/model-service/`) — Python 3.12 + FastAPI + Uvicorn
   - Runs on port **8000** (host `localhost`)
   - Loads PyTorch / HuggingFace / YOLOv8 models at startup (CPU mode)
   - Weights stored in `agrisense-ai-main/model-service/weights/`

## ML Pipeline

- **Stage 0**: CLIP zero-shot crop pre-filter (leaf / fruit / flower)
- **Stage 1 Router**: DINOv2-base + LoRA (PEFT) with EfficientNet-B0 fallback
- **Stage 2 Branches**:
  - Fruit: EfficientNet-B2 (disease) + EfficientNet-B4 (severity)
  - Leaf: EfficientNet-V2-S (disease)
  - Flower: YOLOv8-m (detection/counting)

## Starting the App

```bash
bash start.sh
```

This launches all three services in order:
1. Python model service (uvicorn)
2. Node.js API server
3. React dev server (Vite)

## Environment Variables

- `agrisense-ai-main/server/.env`: Sets `PORT=3001` and `PYTHON_SERVICE_URL=http://localhost:8000`

## Package Managers

- JavaScript: `npm` (both client and server have separate `package.json`)
- Python: `pip` with `agrisense-ai-main/model-service/requirements.txt`

## Key Files

- `start.sh` — main startup script
- `agrisense-ai-main/client/vite.config.ts` — Vite config (host, port, proxy)
- `agrisense-ai-main/server/app.js` — Express entry point
- `agrisense-ai-main/model-service/main.py` — FastAPI entry point
- `run_instructions.md` — Notes on evaluation scripts

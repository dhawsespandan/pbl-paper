# AgriSense AI — Deployment Instructions (Replit)

## Overview

This project has three services that must all run simultaneously:

| Service | Folder | Port |
|---|---|---|
| React Frontend | `client/` | 3000 (nginx via Docker) |
| Node/Express Backend | `server/` | 5000 |
| Python FastAPI Model | `model-service/` | 8000 |

All three are orchestrated via `docker-compose.yml` at the project root.

---

## Prerequisites

Replit must have Docker available. Use a **Replit Deployments** plan or a Repl with Docker support enabled. Verify with:

```bash
docker --version
docker-compose --version
```

---

## Environment Variables

**Never hardcode values.** All configuration lives in `.env` files and Replit Secrets.

### `server/.env`
```
PORT=5000
PYTHON_SERVICE_URL=http://model-service:8000
```

> When running locally without Docker, change `PYTHON_SERVICE_URL` to `http://localhost:8000`.
> Docker Compose overrides this automatically via the `environment` block in `docker-compose.yml`.

### `model-service/.env`
```
DEVICE=cpu
CONFIDENCE_THRESHOLD=0.60
FRUIT_DISEASE_WEIGHTS=models/fruit_disease/weights/model.pth
FRUIT_SEVERITY_WEIGHTS=models/fruit_severity/weights/model.pth
LEAF_DISEASE_WEIGHTS=models/leaf_disease/weights/model.pth
FLOWER_CLUSTER_WEIGHTS=models/flower_cluster/weights/model.pth
ROUTER_WEIGHTS=models/router_transformer/weights/model.pth
```

> If weight files do not exist at the specified paths, the service falls back to
> random weights with a console warning. This is expected during development.

### Replit Secrets (for production)

Go to **Tools → Secrets** in your Replit and add:

| Key | Value |
|---|---|
| `PORT` | `5000` |
| `PYTHON_SERVICE_URL` | `http://model-service:8000` |
| `DEVICE` | `cpu` |
| `CONFIDENCE_THRESHOLD` | `0.60` |

Replit Secrets are injected as environment variables at runtime and override `.env` files.

---

## Initial Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/dhawsespandan/agrisense-ai.git
cd agrisense-ai
```

### Step 2 — Create `.env` files

Create `server/.env` and `model-service/.env` using the values above.
These files are in `.gitignore` and will not be present after cloning — you must create them manually each time.

### Step 3 — Build and start all services

```bash
docker-compose up --build
```

First build will take several minutes due to PyTorch installation.
Subsequent starts (without `--build`) are much faster:

```bash
docker-compose up
```

### Step 4 — Verify all services

Open three separate browser tabs:

- Frontend: `https://<your-repl-name>.repl.co` or `http://localhost:3000`
- Express health check: `http://localhost:5000/health`
- FastAPI docs: `http://localhost:8000/docs`

---

## Updating the Project

### After changing frontend code (`client/`)

```bash
docker-compose up --build client
```

### After changing server code (`server/`)

```bash
docker-compose up --build server
```

### After changing model code (`model-service/`)

```bash
docker-compose up --build model-service
```

### After changing `requirements.txt`

```bash
docker-compose build model-service
docker-compose up
```

### After changing `package.json` in server

```bash
docker-compose build server
docker-compose up
```

### Full rebuild (when in doubt)

```bash
docker-compose down
docker-compose up --build
```

---

## Adding Trained Model Weights

When `.pth` weight files are ready, place them at the paths defined in `model-service/.env`:

```
model-service/
└── models/
    ├── fruit_disease/weights/model.pth
    ├── fruit_severity/weights/model.pth
    ├── leaf_disease/weights/model.pth
    ├── flower_cluster/weights/model.pth
    └── router_transformer/weights/model.pth
```

Then restart the model service:

```bash
docker-compose restart model-service
```

No code changes are needed — paths are read from `.env` at startup.

---

## Switching from CPU to GPU

If Replit provides a GPU instance, update `model-service/.env`:

```
DEVICE=cuda
```

Then rebuild:

```bash
docker-compose up --build model-service
```

---

## Stopping the Project

```bash
docker-compose down
```

To also remove cached build layers:

```bash
docker-compose down --rmi all --volumes
```

---

## Deploying to Production (Post-Project)

| Service | Platform | How |
|---|---|---|
| `client/` | Vercel | Connect GitHub repo, auto-deploys on push |
| `server/` | Render | Connect GitHub repo, set env vars in Render dashboard, uses `server/Dockerfile` |
| `model-service/` | HuggingFace Spaces | Upload folder, uses `model-service/Dockerfile`, set env vars in Space settings |

### Environment variable checklist for production

- Render → set `PYTHON_SERVICE_URL` to the HuggingFace Spaces public URL
- Vercel → set `VITE_API_URL` to the Render public URL (and update the fetch call in `Home.tsx` accordingly)
- Never commit `.env` files to GitHub

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Port already in use | `docker-compose down` then retry |
| Model service crashes on startup | Check `model-service/.env` paths are correct; fallback to random weights should prevent crashes |
| Frontend shows blank page | Check `client/nginx.conf` proxy points to `server:5000` |
| Express cannot reach Python service | Ensure `PYTHON_SERVICE_URL=http://model-service:8000` (not localhost) inside Docker |
| Changes not reflecting | Run `docker-compose up --build <service>` |
| Git push rejected | Run `git pull origin main` first, resolve conflicts, then push |

---

## Git Workflow (keeping repo updated)

```bash
# After any file changes
git add .
git commit -m "describe what changed"
git push origin main
```

Replit auto-pulls on workspace open if connected to GitHub. To manually pull latest:

```bash
git pull origin main
docker-compose up --build
```

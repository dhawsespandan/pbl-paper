"""
model-service/main.py
======================
AgriSense AI — Apple Orchard Health Monitoring
FastAPI inference service

POST /predict
-------------
Receives an image from the Node.js/Express backend, runs it through the
deep-learning pipeline, and returns a structured JSON diagnosis.

Pipeline
--------
input image
    └─► Router (DINOv2-base+LoRA  |  EfficientNet-B0 fallback)
            ├── fruit         → FruitDisease CNN  + FruitSeverity ViT
            ├── leaf          → LeafDisease CNN
            ├── flower_cluster→ FlowerCluster YOLO detector
            └── unknown       → 422 error response
"""

import os
import shutil
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # read .env before any model imports

import torch
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from schemas.response import PredictionResponse, ErrorResponse

# ──────────────────────────────────────────────────────────────────────────
# Disease knowledge base — recommendation & details per class
# ──────────────────────────────────────────────────────────────────────────

_DISEASE_INFO: dict[str, dict] = {
    # ── Fruit diseases ────────────────────────────────────────────────────
    "Anthracnose": {
        "recommendation": (
            "Apply copper-based fungicides every 7–10 days during wet conditions. "
            "Remove and destroy infected fruit promptly. Improve orchard air "
            "circulation through targeted pruning."
        ),
        "details": (
            "Anthracnose (Colletotrichum acutatum) thrives in warm, humid conditions "
            "and causes dark, sunken lesions on fruit that expand rapidly, leading to "
            "significant post-harvest losses."
        ),
    },
    "Blotch": {
        "recommendation": (
            "Apply mancozeb or captan fungicides at petal fall and every 10–14 days "
            "thereafter. Avoid overhead irrigation. Remove fallen leaves and "
            "mummified fruit from the orchard floor."
        ),
        "details": (
            "Apple blotch (Marssonina coronaria) causes irregular dark blotches on "
            "fruit and leaves. Severe infections lead to premature defoliation, "
            "reducing photosynthesis and fruit quality."
        ),
    },
    "Healthy": {
        "recommendation": (
            "No treatment required. Continue regular monitoring and maintain "
            "preventive spray programs as part of your IPM schedule."
        ),
        "details": (
            "The fruit shows no signs of disease. The tree appears to be in good "
            "health. Maintain standard orchard hygiene practices."
        ),
    },
    "Rot": {
        "recommendation": (
            "Remove and dispose of infected fruit immediately to prevent spread. "
            "Apply captan or thiophanate-methyl fungicide. Store harvested fruit "
            "at 0–2 °C with optimal humidity control."
        ),
        "details": (
            "Fruit rot (typically Monilinia fructigena or Botrytis cinerea) causes "
            "rapid decay of apple tissue. Infections often begin at wounds and "
            "spread via spore dispersal under wet conditions."
        ),
    },
    "Scab": {
        "recommendation": (
            "Apply myclobutanil or difenoconazole fungicides during primary infection "
            "periods (bud break to petal fall). Remove and compost fallen leaves. "
            "Consider planting scab-resistant cultivars in future seasons."
        ),
        "details": (
            "Apple scab (Venturia inaequalis) is the most economically significant "
            "apple disease globally, producing olive-green to brown lesions on both "
            "fruit and foliage. Spores overwinter in fallen leaves."
        ),
    },
    # ── Leaf diseases ─────────────────────────────────────────────────────
    "alternaria leaf spot": {
        "recommendation": (
            "Apply iprodione or chlorothalonil fungicides at first sign of symptoms. "
            "Improve drainage to reduce leaf wetness. Manage canopy density through "
            "pruning and avoid overhead irrigation."
        ),
        "details": (
            "Alternaria leaf spot (Alternaria mali) causes circular brown spots with "
            "a yellow halo. Severe outbreaks trigger premature defoliation, weakening "
            "the tree and reducing next season's productivity."
        ),
    },
    "brown spot": {
        "recommendation": (
            "Apply copper-based sprays or mancozeb at first symptom appearance. "
            "Ensure proper tree spacing to promote air circulation. Remove and "
            "destroy infected leaf debris from the orchard floor."
        ),
        "details": (
            "Brown spot disease produces reddish-brown circular lesions on apple "
            "leaves. In severe cases early leaf drop occurs, reducing the tree's "
            "carbohydrate reserves and fruit quality."
        ),
    },
    "gray spot": {
        "recommendation": (
            "Apply ziram or captan-based fungicides. Maintain good canopy airflow "
            "through regular pruning. Collect and destroy fallen infected leaves "
            "to reduce inoculum levels."
        ),
        "details": (
            "Gray spot (Pestalotiopsis spp.) produces grayish lesions that may "
            "coalesce under heavy infection. It is often a secondary pathogen "
            "affecting stressed trees with compromised immune responses."
        ),
    },
    "healthy leaf": {
        "recommendation": (
            "No treatment required. Continue preventive monitoring and maintain "
            "standard spray schedules to keep the tree in optimal health."
        ),
        "details": (
            "The leaf shows no signs of disease or stress. The tree appears to be "
            "in good health with normal foliar development."
        ),
    },
    "rust": {
        "recommendation": (
            "Remove nearby juniper or cedar alternate hosts where possible. Apply "
            "myclobutanil or propiconazole fungicides from pink-bud stage through "
            "petal fall to break the infection cycle."
        ),
        "details": (
            "Cedar-apple rust (Gymnosporangium juniperi-virginianae) requires two "
            "alternate hosts to complete its lifecycle. Bright orange pustules form "
            "on apple leaves in spring, reducing photosynthetic capacity."
        ),
    },
    # ── Flower cluster health ─────────────────────────────────────────────
    "Healthy Clusters": {
        "recommendation": (
            "No intervention needed. Monitor for pest pressure (codling moth, apple "
            "maggot) and ensure adequate pollinator access during bloom."
        ),
        "details": (
            "Flower clusters are abundant and dense. Conditions are highly favourable "
            "for pollination and fruit set. Expected yield is at or above normal."
        ),
    },
    "Adequate Clusters": {
        "recommendation": (
            "Flowering density is acceptable. Ensure pollinators are present and "
            "consider a light boron foliar spray to support fruit set. Monitor "
            "closely if weather is cold or wet during bloom."
        ),
        "details": (
            "Flower cluster density is moderate — sufficient for a reasonable fruit "
            "set but below peak density. Conditions may still support good yield if "
            "pollination conditions are favourable."
        ),
    },
    "Sparse Clusters": {
        "recommendation": (
            "Evaluate potential causes: biennial bearing, late-season frost damage, "
            "or insufficient winter chilling hours. Apply boron foliar spray to "
            "improve fruit set and consult an agronomist about thinning strategy."
        ),
        "details": (
            "Cluster density is well below optimal levels. Sparse flowering is likely "
            "to result in reduced yield. Biennial bearing, environmental stress, or "
            "inadequate chilling accumulation may be responsible."
        ),
    },
    "No Clusters Detected": {
        "recommendation": (
            "Assess tree health, recent pruning practices, and chilling-hour "
            "accumulation for this season. Consult an agronomist for a targeted "
            "fertility and dormancy-management intervention."
        ),
        "details": (
            "No flower clusters were detected. This may indicate dormancy, crop "
            "failure, heavy pruning, or an image that does not show a flowering zone."
        ),
    },
}

_DEFAULT_INFO = {
    "recommendation": "Consult an agronomist for a targeted treatment plan.",
    "details": "Automated diagnosis complete. Manual verification is recommended.",
}


def _get_info(disease: str) -> dict:
    return _DISEASE_INFO.get(disease, _DEFAULT_INFO)


# ──────────────────────────────────────────────────────────────────────────
# Per-disease severity calibration
# ──────────────────────────────────────────────────────────────────────────
# The regression model is trained on polygon-annotation labels that
# systematically under-capture certain disease patterns:
#   • Blotch  – irregular, coalescing spots; annotators miss partial coverage
#   • Scab    – numerous tiny lesions; polygon ceiling in training data ≈ 54 %;
#              diffuse inter-lesion spread is systematically missed by annotators,
#              requiring a higher correction factor (raw ≈ 30 % → visual ≈ 60 %)
# Calibration factors correct for these annotation biases so the displayed
# severity better matches the visual extent of each disease.
# Rot is very accurately annotated (large, contiguous lesion) → near 1.
# Healthy severity is always 0 so the factor is never applied.

_SEVERITY_CALIBRATION: dict[str, float] = {
    "Anthracnose": 1.10,
    "Blotch":      1.35,
    "Rot":         0.50,
    "Scab":        2.02,
}


def _calibrated_severity(raw_pct: float, disease: str) -> float:
    """Apply disease-specific calibration and clamp to [0, 100]."""
    factor = _SEVERITY_CALIBRATION.get(disease, 1.0)
    return round(min(100.0, raw_pct * factor), 1)


# ──────────────────────────────────────────────────────────────────────────
# Severity formatting helpers
# ──────────────────────────────────────────────────────────────────────────

def _format_severity(pct: float) -> str:
    if pct < 15:
        return f"Mild — {pct:.0f}% area affected"
    elif pct < 40:
        return f"Moderate — {pct:.0f}% area affected"
    elif pct < 70:
        return f"Severe — {pct:.0f}% area affected"
    else:
        return f"Critical — {pct:.0f}% area affected"


def _flower_severity(label: str) -> str:
    mapping = {
        "Healthy Clusters":      "None — clusters are abundant and healthy",
        "Adequate Clusters":     "Low — cluster density is moderate but acceptable",
        "Sparse Clusters":       "Moderate — cluster density is well below optimal",
        "No Clusters Detected":  "Severe — no flower clusters found in image",
    }
    return mapping.get(label, "Unknown")


# ──────────────────────────────────────────────────────────────────────────
# FastAPI lifespan — all models loaded here at startup
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models at startup and store them in app.state."""
    import models.router_transformer.model            as _router_mod
    import models.router_transformer.crop_prefilter   as _prefilter_mod
    import models.fruit_disease.model                 as _fruit_mod
    import models.fruit_severity.model                as _severity_mod
    import models.leaf_disease.model                  as _leaf_mod
    import models.flower_cluster.model                as _flower_mod
    from utils.preprocess import preprocess_image, to_severity_tensor

    app.state.predict_route    = _router_mod.predict_route
    app.state.is_apple_crop    = _prefilter_mod.is_apple_crop
    app.state.predict_fruit    = _fruit_mod.predict_fruit
    app.state.predict_severity = _severity_mod.predict_severity
    app.state.predict_leaf     = _leaf_mod.predict_leaf
    app.state.predict_flowers  = _flower_mod.predict_flowers
    app.state.preprocess       = preprocess_image
    app.state.sev_preprocess   = to_severity_tensor

    print("✅ All models loaded — service ready")
    yield
    # shutdown cleanup (none required for read-only models)


# ──────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="AgriSense AI — Apple Crop Disease Detection",
    description=(
        "Router: DINOv2-base fine-tuned with LoRA (PEFT) "
        "→ Fruit / Leaf / Flower CNN pipeline"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Health endpoints
# ──────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AgriSense AI — Apple Crop Disease Detection",
        "version": "2.0.0",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


# ──────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Upload an apple crop image (JPEG / PNG / WebP).

    **Success response**
    ```json
    {
      "image_type": "fruit",
      "disease": "Apple Scab",
      "severity": "Moderate — 35% area affected",
      "confidence": "91.4%",
      "recommendation": "Apply myclobutanil fungicide …",
      "details": "Apple scab (Venturia inaequalis) …"
    }
    ```

    **Unknown image (HTTP 422)**
    ```json
    {
      "error": "Could not classify image. Please upload a clear image …"
    }
    ```
    """

    # ── save upload ──────────────────────────────────────────────────────
    filename  = file.filename or "upload.jpg"
    file_path = os.path.join("temp", filename)
    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as exc:
        raise HTTPException(500, detail=f"Could not save upload: {exc}")

    # ── open image ───────────────────────────────────────────────────────
    try:
        image = Image.open(file_path).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, detail=f"Cannot read image: {exc}")

    # ── pull model callables from app.state ──────────────────────────────
    is_apple_crop    = request.app.state.is_apple_crop
    predict_route    = request.app.state.predict_route
    predict_fruit    = request.app.state.predict_fruit
    predict_severity = request.app.state.predict_severity
    predict_leaf     = request.app.state.predict_leaf
    predict_flowers  = request.app.state.predict_flowers
    sev_preprocess   = request.app.state.sev_preprocess

    # ── gate-0: CLIP crop pre-filter ────────────────────────────────────
    # Reject images that are clearly not apple leaf / fruit / flower before
    # the router ever sees them.  The router was trained only on crop images
    # and will assign one of the three crop classes to *anything*.
    crop_ok, crop_conf = is_apple_crop(image)
    if not crop_ok:
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    "Image not identified as a leaf, flower, or fruit image. "
                    "Please upload a clear photo of an apple leaf, fruit, or flower cluster."
                ),
                "error_type": "unrecognized_image",
            },
        )

    # ── router ───────────────────────────────────────────────────────────
    image_type, router_conf = predict_route(image)

    if image_type == "unknown":
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    "Image not identified as a leaf, flower, or fruit image. "
                    "Please upload a clear photo of an apple leaf, fruit, or flower cluster."
                ),
                "error_type": "unrecognized_image",
            },
        )

    # ── fruit branch ─────────────────────────────────────────────────────
    if image_type == "fruit":
        label, conf = predict_fruit(file_path)

        # Fruit pre-filter (CLIP apple detector) rejected the image
        if label == "Not an apple image.":
            return JSONResponse(
                status_code=422,
                content={
                    "error": (
                        "Image not identified as an apple fruit. "
                        "Please upload a clear photo of an apple fruit for disease detection."
                    ),
                    "error_type": "unrecognized_image",
                },
            )

        info = _get_info(label)

        if label.lower() == "healthy":
            severity_str = "None — no disease detected"
        else:
            raw_pct      = predict_severity(sev_preprocess(image))
            sev_pct      = _calibrated_severity(raw_pct, label)
            severity_str = _format_severity(sev_pct)

        return PredictionResponse(
            image_type   = "fruit",
            disease      = label,
            severity     = severity_str,
            confidence   = f"{conf * 100:.1f}%",
            recommendation = info["recommendation"],
            details      = info["details"],
        )

    # ── leaf branch ──────────────────────────────────────────────────────
    if image_type == "leaf":
        label, conf = predict_leaf(image)   # PIL image; CLIP gate runs inside

        # Leaf pre-filter rejected the image
        if label == "Not an apple leaf image.":
            return JSONResponse(
                status_code=422,
                content={
                    "error": (
                        "Image not identified as an apple leaf. "
                        "Please upload a clear photo of an apple leaf for disease detection."
                    ),
                    "error_type": "unrecognized_image",
                },
            )

        info = _get_info(label)

        return PredictionResponse(
            image_type   = "leaf",
            disease      = label,
            severity     = "None — healthy tissue" if "healthy" in label.lower()
                           else "Detected — consult severity model for full assessment",
            confidence   = f"{conf * 100:.1f}%",
            recommendation = info["recommendation"],
            details      = info["details"],
        )

    # ── flower_cluster branch ─────────────────────────────────────────────
    if image_type == "flower_cluster":
        label, conf, flower_count = predict_flowers(file_path)
        info                      = _get_info(label)

        return PredictionResponse(
            image_type   = "flower_cluster",
            disease      = label,
            severity     = _flower_severity(label),
            confidence   = f"{conf * 100:.1f}%",
            recommendation = info["recommendation"],
            details      = info["details"],
            flower_count = flower_count,
        )

    # ── fallback (should never be reached) ───────────────────────────────
    return JSONResponse(
        status_code=422,
        content={"error": f"Unexpected route label: {image_type}"},
    )

import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
from src.data.preprocess import get_transform
from src.models.predict import load_model, CLASSES, config
from starlette.requests import Request
import time

app = FastAPI(title="Fashion-MNIST Classifier")

# configure logging level from env or config
level_name = os.getenv("LOG_LEVEL", config.get("service", {}).get("log_level", "INFO"))
logging.basicConfig(level=logging.getLevelName(level_name))

# load model at startup (fail gracefully)
try:
    predictor, _ = load_model()
    model_loaded = True
except Exception:
    logging.exception("Failed to load model at startup")
    predictor = None
    model_loaded = False

# simple in-memory metrics
_metrics = {"requests": 0, "total_latency_ms": 0.0}


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000.0
    _metrics["requests"] += 1
    _metrics["total_latency_ms"] += elapsed_ms
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/metrics")
def metrics():
    req = _metrics.get("requests", 0)
    total = _metrics.get("total_latency_ms", 0.0)
    avg = (total / req) if req > 0 else 0.0
    return {"requests": req, "avg_latency_ms": round(avg, 2)}


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    logging.info(f"Received file: {file.filename}")
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    transform = get_transform(image_size=config.get("model", {}).get("image_size", 28))
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = predictor(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    result = {"class_name": CLASSES[pred_idx], "confidence": round(confidence, 4)}
    logging.info(f"Prediction: {result}")
    return result
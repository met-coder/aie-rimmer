import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
from starlette.requests import Request

from src.config import load_config
from src.models.predict import CLASSES, load_model, predict_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor, _ = load_model()
        app.state.predictor = predictor
        app.state.model_loaded = True
        logging.info("Model loaded successfully during startup.")
    except Exception:
        logging.exception("Failed to load model during startup")
        app.state.predictor = None
        app.state.model_loaded = False
    yield


config = load_config()
log_level = os.getenv("LOG_LEVEL", config.get("service", {}).get("log_level", "INFO"))
logging.basicConfig(level=logging.getLevelName(log_level), format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="Fashion-MNIST Classifier",
    description="API-сервис для классификации одежды на Fashion-MNIST с поддержкой batch-инференса.",
    lifespan=lifespan,
)

app.state.config = config
app.state.predictor = None
app.state.model_loaded = False
app.state.metrics = {"requests": 0, "total_latency_ms": 0.0}



@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000.0
    app.state.metrics["requests"] += 1
    app.state.metrics["total_latency_ms"] += elapsed_ms
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": app.state.model_loaded,
        "available_classes": app.state.config.get("classes", CLASSES),
        "total_requests": app.state.metrics.get("requests", 0),
    }


@app.get("/metrics")
def metrics():
    req = app.state.metrics.get("requests", 0)
    total = app.state.metrics.get("total_latency_ms", 0.0)
    avg = (total / req) if req > 0 else 0.0
    return {
        "requests": req,
        "total_requests": req,
        "avg_latency_ms": round(avg, 2),
        "model_loaded": app.state.model_loaded,
    }


def validate_upload(file: UploadFile) -> None:
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG/JPEG/BMP.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")


def _predict_file(file: UploadFile) -> dict:
    validate_upload(file)
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    class_name, confidence = predict_image(img, app.state.predictor, app.state.config)
    result = {"class_name": class_name, "confidence": confidence}
    logging.info(f"Prediction result: {result}")
    return result


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    return _predict_file(file)


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch prediction.")
    return [_predict_file(file) for file in files]

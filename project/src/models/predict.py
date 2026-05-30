import logging
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from src.config import load_config, resolve_path
from src.data.preprocess import get_transform
from src.models.model import SimpleCNN


def load_model() -> Tuple[SimpleCNN, Dict[str, Any]]:
    cfg = load_config()
    model = SimpleCNN(num_classes=cfg["model"]["num_classes"])
    model_path = resolve_path(cfg["model"]["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    except Exception as exc:
        logging.exception(f"Failed to load model state from {model_path}: {exc}")
        raise

    model.eval()
    return model, cfg


def predict_image(img: Image.Image, model: SimpleCNN, cfg: Dict[str, Any]) -> Tuple[str, float]:
    transform = get_transform(cfg["model"].get("image_size", 28))
    tensor = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx].item())
    class_name = cfg["classes"][idx]
    return class_name, round(confidence, 4)


config = load_config()
CLASSES = config["classes"]

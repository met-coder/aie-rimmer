from pathlib import Path
import logging
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

def load_config():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model():
    cfg = load_config()
    model = SimpleCNN(num_classes=cfg["model"]["num_classes"])
    model_path = Path(cfg["model"]["path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if model_path.exists():
        try:
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
        except Exception as e:
            logging.exception(f"Failed to load model state from {model_path}: {e}")
            logging.warning("Using random model weights due to load failure.")
    else:
        logging.warning(f"Model file not found at {model_path}. Using random model weights.")

    model.eval()
    return model, cfg

config = load_config()
CLASSES = config["classes"]
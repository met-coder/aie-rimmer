import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(path_str: str) -> Path:
    """Resolve path relative to project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config() -> Dict[str, Any]:
    """Load configuration from configs/config.yaml and apply environment overrides."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if os.getenv("MODEL_PATH"):
        config.setdefault("model", {})["path"] = os.getenv("MODEL_PATH")
    if os.getenv("PORT"):
        config.setdefault("service", {})["port"] = int(os.getenv("PORT"))
    if os.getenv("LOG_LEVEL"):
        config.setdefault("service", {})["log_level"] = os.getenv("LOG_LEVEL").upper()

    return config

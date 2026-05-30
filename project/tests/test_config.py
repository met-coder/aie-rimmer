from src.config import load_config


def test_load_config():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "service" in cfg
    assert "classes" in cfg
    assert isinstance(cfg["classes"], list) and len(cfg["classes"]) >= 2

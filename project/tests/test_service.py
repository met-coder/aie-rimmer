from fastapi.testclient import TestClient
from src.service import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_dummy():
    # Простая проверка, что эндпоинт не падает на пустом файле
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (28, 28), color="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
    assert r.status_code == 200
    assert "class_name" in r.json()


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    j = r.json()
    assert "requests" in j and "avg_latency_ms" in j
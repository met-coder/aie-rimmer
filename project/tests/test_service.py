from fastapi.testclient import TestClient
from src.service import app

from io import BytesIO
from PIL import Image

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "model_loaded" in r.json()


def test_predict_dummy():
    img = Image.new("RGB", (28, 28), color="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
    assert r.status_code == 200
    assert "class_name" in r.json()
    assert "confidence" in r.json()


def test_predict_invalid_file():
    buf = BytesIO(b"not-an-image")
    r = client.post("/predict", files={"file": ("test.txt", buf, "text/plain")})
    assert r.status_code == 400


def test_batch_predict():
    img1 = Image.new("RGB", (28, 28), color="black")
    img2 = Image.new("RGB", (28, 28), color="white")
    buf1 = BytesIO()
    buf2 = BytesIO()
    img1.save(buf1, format="PNG")
    img2.save(buf2, format="PNG")
    buf1.seek(0)
    buf2.seek(0)
    r = client.post(
        "/batch_predict",
        files=[
            ("files", ("img1.png", buf1, "image/png")),
            ("files", ("img2.png", buf2, "image/png")),
        ],
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert all("class_name" in item for item in data)


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    j = r.json()
    assert "requests" in j and "avg_latency_ms" in j and "model_loaded" in j

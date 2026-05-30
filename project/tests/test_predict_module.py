from PIL import Image
import torch
import torch.nn as nn

from src.models.predict import predict_image


class DummyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # return logits with high score for class 0
        batch = x.shape[0]
        out = torch.zeros((batch, self.num_classes), dtype=torch.float32)
        out[:, 0] = 10.0
        return out


def test_predict_image_dummy():
    img = Image.new("RGB", (28, 28), color="black")
    model = DummyModel(num_classes=10)
    cfg = {"model": {"image_size": 28}, "classes": [str(i) for i in range(10)]}
    class_name, confidence = predict_image(img, model, cfg)
    assert isinstance(class_name, str)
    assert isinstance(confidence, float)
    assert class_name == "0"
    assert 0.0 <= confidence <= 1.0

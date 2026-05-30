from PIL import Image
import torch

from src.data.preprocess import get_transform


def test_get_transform_output_shape():
    transform = get_transform(28)
    img = Image.new("RGB", (32, 32), color="white")
    t = transform(img)
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 3
    assert t.shape[0] == 1  # grayscale channel
    assert t.shape[1] == 28 and t.shape[2] == 28

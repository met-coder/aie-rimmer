import yaml
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_transform(image_size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def train():
    cfg = load_config()
    image_size = cfg["model"].get("image_size", 28)
    num_classes = cfg["model"].get("num_classes", 10)
    model_path = PROJECT_ROOT / cfg["model"]["path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    transform = get_transform(image_size)
    dataset = datasets.FashionMNIST(
        root=PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cpu")
    model.to(device)

    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f} acc={accuracy:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    train()

import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import load_config, resolve_path
from src.data.preprocess import get_transform
from src.models.model import SimpleCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train() -> None:
    cfg = load_config()
    image_size = cfg["model"].get("image_size", 28)
    num_classes = cfg["model"].get("num_classes", 10)
    batch_size = cfg["model"].get("batch_size", 128)
    epochs = cfg["model"].get("epochs", 4)
    lr = cfg["model"].get("lr", 1e-3)
    num_workers = cfg["model"].get("num_workers", 0)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device selection
    requested_device = cfg.get("device") or os.getenv("DEVICE")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = resolve_path(cfg["model"]["path"])
    metrics_path = resolve_path("artifacts/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    transform = get_transform(image_size)
    full_train = datasets.FashionMNIST(
        root=PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=PROJECT_ROOT / "data",
        train=False,
        download=True,
        transform=transform,
    )

    # Baseline evaluation: Logistic Regression on flattened pixels
    X_train = full_train.data.numpy().reshape(-1, image_size * image_size).astype(np.float32) / 255.0
    y_train = full_train.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, image_size * image_size).astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    def create_logistic_regression(**kwargs: Any) -> LogisticRegression:
        supported_args = {
            name: value
            for name, value in kwargs.items()
            if name in inspect.signature(LogisticRegression).parameters
        }
        return LogisticRegression(**supported_args)

    print("Training baseline LogisticRegression...")
    baseline = create_logistic_regression(
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
        verbose=0,
    )
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_accuracy = float(accuracy_score(y_test, baseline_preds))
    print(f"Baseline LogisticRegression accuracy: {baseline_accuracy:.4f}")

    # split train -> train / val
    train_size = int(len(full_train) * 0.9)
    val_size = len(full_train) - train_size
    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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
        train_acc = correct / total
        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # final evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_dataset:
            images = images.unsqueeze(0).to(device)
            outputs = model(images)
            pred = int(outputs.argmax(dim=1).item())
            all_preds.append(pred)
            all_labels.append(int(labels))

    final_accuracy = float(accuracy_score(all_labels, all_preds))
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")
    print(f"Final SimpleCNN accuracy: {final_accuracy:.4f}")

    metrics: Dict[str, Any] = {
        "dataset": "Fashion-MNIST",
        "seed": seed,
        "device": str(device),
        "baseline": {
            "model": "LogisticRegression",
            "accuracy": baseline_accuracy,
            "description": "Logistic regression on flattened pixels",
        },
        "final_model": {
            "name": "SimpleCNN",
            "model_path": str(model_path.relative_to(model_path.parent)),
            "accuracy": final_accuracy,
            "description": "CNN trained in src/train.py and saved to artifacts/model.pt",
        },
        "validation_accuracy": val_acc,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    train()

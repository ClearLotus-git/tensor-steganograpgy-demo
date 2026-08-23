"""Train the baseline MNIST model used by the TensorStego demonstration."""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CLEAN_MODEL, DATA_DIR, MODEL_DIR, RANDOM_SEED


EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTClassifier(nn.Module):
    """Simple MLP classifier for handwritten digits."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def set_seed() -> None:
    """Make training behavior more reproducible."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> float:
    """Return classification accuracy as a percentage."""

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main() -> None:
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    model = MNISTClassifier().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss()

    print("=" * 60)
    print(" TensorStego Research | Baseline Model Training")
    print("=" * 60)
    print(f" Device       : {DEVICE}")
    print(f" Architecture : 784 -> 128 -> 10")
    print(f" Epochs       : {EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            output = model(images)
            loss = loss_function(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f" [{epoch:02d}/{EPOCHS:02d}] "
            f"loss={average_loss:.4f}"
        )

    accuracy = evaluate(model, test_loader)

    torch.save(model.state_dict(), CLEAN_MODEL)

    print()
    print("-" * 60)
    print(f" Accuracy     : {accuracy:.2f}%")
    print(f" Artifact     : {CLEAN_MODEL.name}")
    print(" Status       : BASELINE CREATED")
    print("-" * 60)


if __name__ == "__main__":
    main()

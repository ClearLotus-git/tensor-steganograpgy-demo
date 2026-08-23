"""Verify integrity and performance of clean vs stego models."""

from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    CLEAN_MODEL,
    STEGO_MODEL,
    DATA_DIR,
)

from train import MNISTClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_model(path: Path) -> MNISTClassifier:
    model = MNISTClassifier().to(DEVICE)

    state = torch.load(
        path,
        weights_only=True,
        map_location=DEVICE,
    )

    model.load_state_dict(state)
    model.eval()

    return model


def evaluate_pair(
    clean_model: MNISTClassifier,
    stego_model: MNISTClassifier,
    loader: DataLoader,
):
    clean_correct = 0
    stego_correct = 0
    total = 0
    prediction_changes = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            clean_predictions = clean_model(images).argmax(dim=1)
            stego_predictions = stego_model(images).argmax(dim=1)

            clean_correct += (
                clean_predictions == labels
            ).sum().item()

            stego_correct += (
                stego_predictions == labels
            ).sum().item()

            prediction_changes += (
                clean_predictions != stego_predictions
            ).sum().item()

            total += labels.size(0)

    clean_accuracy = 100.0 * clean_correct / total
    stego_accuracy = 100.0 * stego_correct / total

    return (
        clean_accuracy,
        stego_accuracy,
        prediction_changes,
        total,
    )


def compare_parameters():
    clean_state = torch.load(
        CLEAN_MODEL,
        weights_only=True,
        map_location="cpu",
    )

    stego_state = torch.load(
        STEGO_MODEL,
        weights_only=True,
        map_location="cpu",
    )

    total_changed = 0
    max_delta = 0.0
    sum_delta = 0.0
    total_values = 0

    for name in clean_state:
        clean_tensor = clean_state[name]
        stego_tensor = stego_state[name]

        delta = torch.abs(clean_tensor - stego_tensor)

        total_changed += (
            clean_tensor != stego_tensor
        ).sum().item()

        if delta.numel() > 0:
            max_delta = max(
                max_delta,
                delta.max().item(),
            )

            sum_delta += delta.sum().item()
            total_values += delta.numel()

    mean_delta = (
        sum_delta / total_values
        if total_values
        else 0.0
    )

    return total_changed, max_delta, mean_delta


def main():
    print("=" * 60)
    print(" TensorStego Research | Integrity Verification")
    print("=" * 60)

    if not CLEAN_MODEL.exists():
        raise FileNotFoundError("clean_model.pt not found")

    if not STEGO_MODEL.exists():
        raise FileNotFoundError("stego_model.pt not found")

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    clean_model = load_model(CLEAN_MODEL)
    stego_model = load_model(STEGO_MODEL)

    (
        clean_accuracy,
        stego_accuracy,
        prediction_changes,
        total_predictions,
    ) = evaluate_pair(
        clean_model,
        stego_model,
        test_loader,
    )

    changed_parameters, max_delta, mean_delta = compare_parameters()

    clean_hash = sha256_file(CLEAN_MODEL)
    stego_hash = sha256_file(STEGO_MODEL)

    print(f" Device              : {DEVICE}")
    print()

    print(" Model performance")
    print("-" * 60)
    print(f" Clean accuracy      : {clean_accuracy:.2f}%")
    print(f" Stego accuracy      : {stego_accuracy:.2f}%")
    print(
        f" Prediction changes  : "
        f"{prediction_changes:,} / {total_predictions:,}"
    )

    print()
    print(" Parameter integrity")
    print("-" * 60)
    print(f" Changed parameters  : {changed_parameters:,}")
    print(f" Maximum delta       : {max_delta:.12e}")
    print(f" Mean delta          : {mean_delta:.12e}")

    print()
    print(" Artifact hashes")
    print("-" * 60)
    print(f" Clean SHA-256       : {clean_hash}")
    print(f" Stego SHA-256       : {stego_hash}")

    print()
    print("-" * 60)

    if prediction_changes == 0:
        print(" Result              : MODEL MODIFIED")
        print(" Behavior            : PREDICTIONS PRESERVED")
    else:
        print(" Result              : MODEL MODIFIED")
        print(
            f" Behavior            : "
            f"{prediction_changes} predictions changed"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()

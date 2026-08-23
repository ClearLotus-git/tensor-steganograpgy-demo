"""Compare baseline, steganographic, and deliberately perturbed models."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CLEAN_MODEL, STEGO_MODEL, MODEL_DIR, DATA_DIR
from train import MNISTClassifier


STRESS_MODEL = MODEL_DIR / "stress_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    model = MNISTClassifier().to(DEVICE)

    state = torch.load(
        path,
        weights_only=True,
        map_location=DEVICE,
    )

    model.load_state_dict(state)
    model.eval()

    return model


def main():
    print("=" * 70)
    print(" TensorStego Research | Three-Model Behavior Comparison")
    print("=" * 70)

    dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )

    clean = load_model(CLEAN_MODEL)
    stego = load_model(STEGO_MODEL)
    stress = load_model(STRESS_MODEL)

    correct = {
        "clean": 0,
        "stego": 0,
        "stress": 0,
    }

    stego_changes = 0
    stress_changes = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            clean_pred = clean(images).argmax(dim=1)
            stego_pred = stego(images).argmax(dim=1)
            stress_pred = stress(images).argmax(dim=1)

            correct["clean"] += (clean_pred == labels).sum().item()
            correct["stego"] += (stego_pred == labels).sum().item()
            correct["stress"] += (stress_pred == labels).sum().item()

            stego_changes += (
                clean_pred != stego_pred
            ).sum().item()

            stress_changes += (
                clean_pred != stress_pred
            ).sum().item()

            total += labels.size(0)

    clean_acc = 100 * correct["clean"] / total
    stego_acc = 100 * correct["stego"] / total
    stress_acc = 100 * correct["stress"] / total

    print()
    print(" Model performance")
    print("-" * 70)

    print(f" {'Model':<18}{'Accuracy':>12}{'Changes vs Clean':>22}")
    print("-" * 70)

    print(
        f" {'Clean':<18}"
        f"{clean_acc:>11.2f}%"
        f"{'—':>22}"
    )

    print(
        f" {'LSB Stego':<18}"
        f"{stego_acc:>11.2f}%"
        f"{f'{stego_changes:,} / {total:,}':>22}"
    )

    print(
        f" {'Stress':<18}"
        f"{stress_acc:>11.2f}%"
        f"{f'{stress_changes:,} / {total:,}':>22}"
    )

    print("-" * 70)

    print()
    print(" Modification scale")
    print("-" * 70)
    print(" LSB maximum delta   : ~1.49e-08")
    print(" Stress delta        : ~1.00e-02")
    print()

    if stress_changes > 0:
        print(" Result : LARGER MODIFICATIONS ALTERED MODEL BEHAVIOR")
    else:
        print(" Result : NO PREDICTION CHANGES AT THIS PERTURBATION LEVEL")

    print("=" * 70)


if __name__ == "__main__":
    main()

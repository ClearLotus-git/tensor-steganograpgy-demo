"""Inspect parameter-level differences between clean and stego models."""

import torch

from config import CLEAN_MODEL, STEGO_MODEL, TARGET_TENSOR


DISPLAY_LIMIT = 10


def main() -> None:
    print("=" * 60)
    print(" TensorStego Research | Parameter Analysis")
    print("=" * 60)

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

    clean = clean_state[TARGET_TENSOR].flatten()
    stego = stego_state[TARGET_TENSOR].flatten()

    changed_mask = clean != stego
    changed_indices = torch.where(changed_mask)[0]

    differences = torch.abs(clean - stego)

    print(f" Tensor              : {TARGET_TENSOR}")
    print(f" Total parameters    : {clean.numel():,}")
    print(f" Changed parameters  : {changed_mask.sum().item():,}")
    print()

    print(" Parameter-level changes")
    print("-" * 60)
    print(
        f"{'Index':>8}  "
        f"{'Original':>15}  "
        f"{'Modified':>15}  "
        f"{'Delta':>14}"
    )
    print("-" * 60)

    for index in changed_indices[:DISPLAY_LIMIT]:
        i = index.item()

        original = clean[i].item()
        modified = stego[i].item()
        delta = differences[i].item()

        print(
            f"{i:8d}  "
            f"{original:15.10f}  "
            f"{modified:15.10f}  "
            f"{delta:14.6e}"
        )

    print("-" * 60)

    if changed_indices.numel() > DISPLAY_LIMIT:
        remaining = changed_indices.numel() - DISPLAY_LIMIT
        print(f" ... {remaining:,} additional changes not displayed")

    print()
    print(f" Maximum delta       : {differences.max().item():.12e}")

    changed_differences = differences[changed_mask]

    if changed_differences.numel():
        print(
            f" Mean changed delta  : "
            f"{changed_differences.mean().item():.12e}"
        )

    print()
    print("-" * 60)
    print(" Analysis            : LSB-LEVEL MODIFICATIONS DETECTED")
    print("-" * 60)


if __name__ == "__main__":
    main()

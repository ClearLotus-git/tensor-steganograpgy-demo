"""Recover embedded research metadata from a TensorStego model."""

import json

import numpy as np
import torch

from config import STEGO_MODEL, TARGET_TENSOR


def main() -> None:
    print("=" * 60)
    print(" TensorStego Research | Artifact Inspection")
    print("=" * 60)

    if not STEGO_MODEL.exists():
        raise FileNotFoundError(
            "Stego model not found. Run embed.py first."
        )

    state = torch.load(
        STEGO_MODEL,
        weights_only=True,
        map_location="cpu",
    )

    tensor = state[TARGET_TENSOR]

    values = tensor.detach().flatten().numpy().copy()
    integer_view = values.view(np.uint32)

    bits = (integer_view & 1).astype(np.uint8)

    recovered_bytes = np.packbits(bits).tobytes()

    payload_bytes = recovered_bytes.split(
        b"\x00",
        1,
    )[0]

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Embedded payload could not be decoded."
        ) from exc

    print(f" Artifact     : {STEGO_MODEL.name}")
    print(f" Tensor       : {TARGET_TENSOR}")
    print()
    print(" Embedded metadata recovered")
    print("-" * 60)

    for key, value in payload.items():
        print(f" {key:<15}: {value}")

    print("-" * 60)
    print(" Status       : RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()

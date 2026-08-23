"""Embed research metadata into float32 model parameters using LSBs."""

import json

import numpy as np
import torch

from config import CLEAN_MODEL, STEGO_MODEL, TARGET_TENSOR


PAYLOAD = {
    "classification": "INTERNAL",
    "artifact_id": "TSL-2026-0823-001",
    "project": "TensorStego Research",
    "model": "MNIST-MLP-v1",
    "environment": "LAB",
    "handling": "AUTHORIZED RESEARCH ONLY",
}


def serialize_payload() -> bytes:
    """Serialize metadata and append an extraction terminator."""

    encoded = json.dumps(
        PAYLOAD,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return encoded + b"\x00"


def main() -> None:
    print("=" * 60)
    print(" TensorStego Research | Payload Embedding")
    print("=" * 60)

    if not CLEAN_MODEL.exists():
        raise FileNotFoundError(
            "Baseline model not found. Run train.py first."
        )

    payload = serialize_payload()

    payload_bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8)
    )

    state = torch.load(
        CLEAN_MODEL,
        weights_only=True,
        map_location="cpu",
    )

    if TARGET_TENSOR not in state:
        raise KeyError(
            f"Tensor '{TARGET_TENSOR}' was not found."
        )

    tensor = state[TARGET_TENSOR]

    if tensor.dtype != torch.float32:
        raise TypeError(
            "Target tensor must contain float32 parameters."
        )

    flat = tensor.detach().flatten().numpy().copy()

    if len(payload_bits) > flat.size:
        raise ValueError(
            "Payload exceeds available tensor capacity."
        )

    integer_view = flat.view(np.uint32)

    original = integer_view[:len(payload_bits)].copy()

    # Replace only the least-significant bit.
    integer_view[:len(payload_bits)] = (
        integer_view[:len(payload_bits)]
        & np.uint32(0xFFFFFFFE)
    ) | payload_bits.astype(np.uint32)

    changed = np.count_nonzero(
        original != integer_view[:len(payload_bits)]
    )

    state[TARGET_TENSOR] = torch.from_numpy(
        flat.copy()
    ).reshape(tensor.shape)

    torch.save(state, STEGO_MODEL)

    capacity_bytes = tensor.numel() // 8

    print(f" Source       : {CLEAN_MODEL.name}")
    print(f" Target       : {TARGET_TENSOR}")
    print(f" Parameters   : {tensor.numel():,}")
    print(f" Capacity     : {capacity_bytes:,} bytes")
    print()
    print(f" Payload      : {len(payload):,} bytes")
    print(f" Embedded     : {len(payload_bits):,} bits")
    print(f" LSB changes  : {changed:,}")
    print()
    print(" Payload metadata")
    print(" -" * 30)

    for key, value in PAYLOAD.items():
        print(f" {key:<15}: {value}")

    print()
    print("-" * 60)
    print(f" Artifact     : {STEGO_MODEL.name}")
    print(" Status       : PAYLOAD EMBEDDED")
    print("-" * 60)


if __name__ == "__main__":
    main()

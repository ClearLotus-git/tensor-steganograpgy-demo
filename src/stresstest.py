"""Create a deliberately perturbed model for comparison with LSB embedding."""

import torch

from config import CLEAN_MODEL, MODEL_DIR, TARGET_TENSOR


OUTPUT_MODEL = MODEL_DIR / "stress_model.pt"

# Start modestly. We can increase this afterward.
PERTURBATION = 0.01

# Modify the same number of parameters changed in our LSB experiment.
PARAMETERS_TO_MODIFY = 701


def main() -> None:
    print("=" * 60)
    print(" TensorStego Research | Perturbation Experiment")
    print("=" * 60)

    state = torch.load(
        CLEAN_MODEL,
        weights_only=True,
        map_location="cpu",
    )

    tensor = state[TARGET_TENSOR]
    modified = tensor.flatten().clone()

    print(f" Source              : {CLEAN_MODEL.name}")
    print(f" Target tensor       : {TARGET_TENSOR}")
    print(f" Parameters modified : {PARAMETERS_TO_MODIFY:,}")
    print(f" Perturbation        : {PERTURBATION}")
    print()

    original = modified[:PARAMETERS_TO_MODIFY].clone()

    # Deliberately introduce a much larger numerical change than our
    # previous LSB-level modifications.
    modified[:PARAMETERS_TO_MODIFY] += PERTURBATION

    delta = torch.abs(
        modified[:PARAMETERS_TO_MODIFY] - original
    )

    state[TARGET_TENSOR] = modified.reshape(tensor.shape)

    torch.save(state, OUTPUT_MODEL)

    print(f" Maximum delta       : {delta.max().item():.12e}")
    print(f" Mean delta          : {delta.mean().item():.12e}")
    print()
    print(f" Artifact            : {OUTPUT_MODEL.name}")
    print(" Status              : PERTURBED MODEL CREATED")
    print("=" * 60)


if __name__ == "__main__":
    main()

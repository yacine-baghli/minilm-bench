"""Few-shot evaluation on HellaSwag and ARC-Easy benchmarks."""

# TODO: Implement HellaSwag and ARC-Easy few-shot evaluation
# This will load benchmark datasets, run inference, and compute accuracy.

import torch
from model import Transformer


@torch.no_grad()
def evaluate_hellaswag(model: Transformer, device: str = "cuda", num_examples: int = 100) -> float:
    """Evaluate on HellaSwag (0-shot sentence completion).

    Returns:
        Accuracy as a fraction.
    """
    # Placeholder — will be implemented in Phase 4
    raise NotImplementedError("HellaSwag evaluation not yet implemented")


@torch.no_grad()
def evaluate_arc_easy(model: Transformer, device: str = "cuda", num_examples: int = 100) -> float:
    """Evaluate on ARC-Easy (0-shot science questions).

    Returns:
        Accuracy as a fraction.
    """
    raise NotImplementedError("ARC-Easy evaluation not yet implemented")

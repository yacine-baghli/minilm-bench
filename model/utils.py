"""
Utility functions for model initialization and inspection.
"""

import math

import torch.nn as nn

from model.config import ModelConfig


def init_weights(module: nn.Module, config: ModelConfig) -> None:
    """Initialize weights following GPT-2 / Llama conventions.

    - Normal init (std=0.02) for most linear layers and embeddings.
    - Scaled init for residual projections: std = 0.02 / sqrt(2 * n_layers).
      This prevents residual stream magnitude from growing with depth.

    Args:
        module: Module to initialize.
        config: Model configuration (used for n_layers).
    """
    if isinstance(module, nn.Linear):
        std = 0.02
        # Check if this is a residual output projection
        if hasattr(module, "_is_residual_proj") and module._is_residual_proj:
            std = 0.02 / math.sqrt(2.0 * config.n_layers)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters by component.

    Returns:
        Dictionary mapping component names to parameter counts.
    """
    counts: dict[str, int] = {}
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Group by top-level component
        component = name.split(".")[0]
        counts[component] = counts.get(component, 0) + param.numel()
        total += param.numel()

    counts["total"] = total
    return counts


def print_model_summary(model: nn.Module, config: ModelConfig) -> None:
    """Print a summary of model architecture and parameter counts."""
    counts = count_parameters(model)
    total = counts.pop("total")

    print(f"\n{'=' * 60}")
    print(f"Model Summary: {config.attention_type.upper()} variant")
    print(f"{'=' * 60}")
    print(f"  d_model:      {config.d_model}")
    print(f"  n_layers:     {config.n_layers}")
    print(f"  n_heads:      {config.n_heads}")
    print(f"  n_kv_heads:   {config.n_kv_heads}")
    print(f"  head_dim:     {config.head_dim}")
    print(f"  d_ff:         {config.d_ff}")
    print(f"  max_seq_len:  {config.max_seq_len}")
    print(f"  vocab_size:   {config.vocab_size}")
    print(f"{'─' * 60}")

    for component, count in sorted(counts.items()):
        pct = 100.0 * count / total
        print(f"  {component:20s}: {count:>12,} params ({pct:5.1f}%)")

    print(f"{'─' * 60}")
    print(f"  {'TOTAL':20s}: {total:>12,} params ({total / 1e6:.1f}M)")
    print(f"{'=' * 60}\n")


def estimate_flops_per_step(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> int:
    """Estimate FLOPs for one forward+backward training step.

    Uses the approximation: FLOPs ≈ 6 * N * B * T
    where N = params, B = batch_size, T = seq_len.
    Factor 6 = 2 (forward multiply-add) * 3 (forward + backward).

    Args:
        config: Model configuration.
        batch_size: Batch size.
        seq_len: Sequence length.

    Returns:
        Estimated FLOPs per step.
    """
    num_params = config.num_params()
    return 6 * num_params * batch_size * seq_len

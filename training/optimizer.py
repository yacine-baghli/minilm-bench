"""
Optimizer configuration and learning rate scheduling.

Implements AdamW with separate weight decay groups and cosine annealing
with linear warmup — the standard recipe for LLM pre-training.
"""

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def configure_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Create AdamW optimizer with proper weight decay groups.

    Weight decay is applied ONLY to 2D+ parameters (linear weights).
    Biases, norms, and embeddings get zero weight decay.
    This is critical for training stability.

    Args:
        model: The model to optimize.
        lr: Peak learning rate.
        weight_decay: Weight decay coefficient.
        betas: AdamW beta coefficients.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(
        f"Optimizer: {n_decay:,} params with decay, "
        f"{n_no_decay:,} params without decay"
    )

    use_fused = torch.cuda.is_available()
    return AdamW(param_groups, lr=lr, betas=betas, fused=use_fused)


def build_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create cosine annealing scheduler with linear warmup.

    Schedule:
        - Steps [0, warmup_steps): linear warmup from 0 to peak LR
        - Steps [warmup_steps, max_steps): cosine decay to min_lr

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak (default 0.1 = 10%).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

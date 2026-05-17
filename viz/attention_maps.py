"""
Attention pattern extraction and rendering.

Hooks into attention layers to extract weights, produces heatmaps
with token labels, and supports side-by-side variant comparison.
"""

import torch

from model.transformer import Transformer


class AttentionExtractor:
    """Hooks into attention layers to capture attention weights.

    Usage:
        extractor = AttentionExtractor(model)
        logits, _ = model(input_ids)
        patterns = extractor.get_patterns()
        extractor.remove_hooks()
    """

    def __init__(self, model: Transformer) -> None:
        self.model = model
        self.patterns: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on all attention layers."""
        for idx, block in enumerate(self.model.blocks):
            attn = block.attention
            hook = attn.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a hook that captures attention weights."""
        def hook_fn(module, input, output):
            # Re-compute attention weights for visualization
            # (we don't store them during forward to save memory)
            x = input[0]
            B, T, D = x.shape

            with torch.no_grad():
                q = module.wq(x).view(B, T, module.n_heads, module.head_dim).transpose(1, 2)
                k = module.wk(x).view(B, T, module.n_kv_heads, module.head_dim).transpose(1, 2)

                if hasattr(module, 'n_groups') and module.n_kv_heads < module.n_heads:
                    n_groups = module.n_heads // module.n_kv_heads
                    k = k.repeat_interleave(n_groups, dim=1)

                scale = module.head_dim ** -0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale

                # Apply causal mask
                mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn = torch.softmax(attn, dim=-1)

                self.patterns[layer_idx] = attn.cpu()

        return hook_fn

    def get_patterns(self) -> dict[int, torch.Tensor]:
        """Return captured attention patterns. Shape: {layer: (B, H, T, T)}."""
        return self.patterns

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.patterns.clear()


def aggregate_attention(patterns: dict[int, torch.Tensor], mode: str = "mean") -> dict[int, torch.Tensor]:
    """Aggregate attention patterns across heads.

    Args:
        patterns: {layer_idx: (B, H, T, T)} attention tensors.
        mode: "mean", "max", or "entropy".

    Returns:
        {layer_idx: (B, T, T)} aggregated attention.
    """
    result = {}
    for layer_idx, attn in patterns.items():
        if mode == "mean":
            result[layer_idx] = attn.mean(dim=1)
        elif mode == "max":
            result[layer_idx] = attn.max(dim=1).values
        elif mode == "entropy":
            eps = 1e-8
            entropy = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, H, T)
            result[layer_idx] = entropy.mean(dim=1)  # (B, T)
    return result

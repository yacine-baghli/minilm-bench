"""
Core building blocks: RMSNorm, SwiGLU FFN, and TransformerBlock.

All implemented from scratch — no nn.TransformerDecoder or third-party layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler and faster than LayerNorm: no mean subtraction, no bias.
    Used in Llama, Gemma, Mistral, and most modern LLMs.

    Args:
        d_model: Hidden dimension.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).

    Uses gated activation: out = W_down(SiLU(W_gate(x)) * W_up(x))
    ~1% perplexity improvement over ReLU/GELU at same param count.
    Requires 3 linear projections instead of 2.

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension (typically 4 * d_model).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: gated activation with SiLU."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with pre-norm architecture.

    Structure: x → RMSNorm → Attention → residual → RMSNorm → FFN → residual

    Pre-norm (norm before attention/FFN) is more stable for training
    than post-norm. Used in GPT-2+, Llama, and all modern LLMs.

    Args:
        config: Model configuration.
        layer_idx: Layer index (used for scaled residual init).
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Attention (created by Transformer parent via factory)
        self.attention: nn.Module = None  # Set by Transformer.__init__

        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Feed-forward
        self.ffn = SwiGLU(config.d_model, config.d_ff)

        # Dropout (0 for pre-training, >0 for fine-tuning)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            freqs_cis: Precomputed RoPE frequencies.
            mask: Optional causal attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Attention with residual
        h = x + self.dropout(self.attention(self.attn_norm(x), freqs_cis, mask))
        # FFN with residual
        out = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return out

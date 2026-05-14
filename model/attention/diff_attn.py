"""
Differential Attention (DiffAttn).

Computes two independent attention maps and subtracts one from the other,
canceling noise (irrelevant context) like noise-canceling headphones.
Attention scores can be NEGATIVE, breaking the standard softmax constraint.

Achieves equivalent quality with ~65% of the parameters of standard MHA.

Reference: Ye et al., "Differential Transformer" (Microsoft Research, 2024).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class DifferentialAttention(BaseAttention):
    """Differential Attention mechanism.

    Splits Q and K into two halves, computes two softmax attention maps,
    and subtracts one from the other. A learnable λ per head scales the
    differential output.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of KV heads (supports GQA-style grouping).
        head_dim: Dimension per head.
        layer_idx: Layer index (used for λ initialization).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        layer_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        assert head_dim % 2 == 0, f"head_dim must be even for DiffAttn, got {head_dim}"
        self.half_head_dim = head_dim // 2
        self.layer_idx = layer_idx

        # Q, K, V projections (full head_dim for Q/K, then split internally)
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.wo._is_residual_proj = True

        # Learnable λ per head: controls differential scaling
        # Initialization: deeper layers get smaller λ (less differential)
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_param = nn.Parameter(torch.full((n_heads,), lambda_init))

        # Sublayer norm applied after differential attention (stabilizes training)
        self.sub_norm = nn.RMSNorm(head_dim, eps=1e-6) if hasattr(nn, 'RMSNorm') else _RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, T, head_dim), k: (B, n_kv_heads, T, head_dim)

        # Apply RoPE before splitting (applied to full Q and K)
        q, k = apply_rope(q, k, freqs_cis)

        # Expand KV heads if using GQA-style grouping
        if self.n_kv_heads < self.n_heads:
            n_groups = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_groups, dim=1)
            v = v.repeat_interleave(n_groups, dim=1)

        # Split Q and K into two halves along head_dim
        # q: (B, H, T, D) → q1, q2: (B, H, T, D//2)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # Compute two independent attention distributions
        scale = self.half_head_dim ** -0.5
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if mask is not None:
            attn1 = attn1.masked_fill(mask == 0, float("-inf"))
            attn2 = attn2.masked_fill(mask == 0, float("-inf"))

        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Differential attention: subtract to cancel noise
        # This is the core innovation — attention weights can be NEGATIVE
        diff_attn = attn1 - attn2  # (B, H, T, T)

        # Apply λ scaling per head
        # λ: (n_heads,) → (1, n_heads, 1, 1)
        lam = self.lambda_param.view(1, -1, 1, 1)
        diff_attn = lam * diff_attn

        # Weighted sum of values
        out = torch.matmul(diff_attn, v)  # (B, H, T, head_dim)

        # Sublayer norm for training stability
        out = self.sub_norm(out)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class _RMSNorm(nn.Module):
    """Fallback RMSNorm for PyTorch versions without nn.RMSNorm."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + float(self.eps)) * self.weight

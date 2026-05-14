"""
Grouped-Query Attention (GQA).

Groups of query heads share a single K/V head. Interpolates between
MHA (every head independent) and MQA (all heads share one KV).

This is the modern standard: Llama 3, Gemma, Mistral all use GQA.
Typically achieves 98-99.8% of MHA quality with 1.3-1.5x speedup.

Reference: Ainslie et al., "GQA: Training Generalized Multi-Query
           Transformer Models from Multi-Head Checkpoints" (2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class GroupedQueryAttention(BaseAttention):
    """Grouped-Query Attention.

    n_kv_heads < n_heads: groups of (n_heads // n_kv_heads) query heads
    share a single K/V head.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads (must divide n_heads).
        head_dim: Dimension per head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        assert n_heads % n_kv_heads == 0
        self.n_groups = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.wo._is_residual_proj = True

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Project
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # Expand KV heads to match query heads via repeat_interleave
        # (B, n_kv_heads, T, head_dim) → (B, n_heads, T, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

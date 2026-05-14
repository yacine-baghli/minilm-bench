"""
Multi-Query Attention (MQA).

All query heads share a single K/V head. Maximum KV cache compression
but potentially lower model quality and training stability.

Reference: Shazeer, "Fast Transformer Decoding: One Write-Head is
           All You Need" (2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class MultiQueryAttention(BaseAttention):
    """Multi-Query Attention.

    n_kv_heads == 1: all query heads share a single K and V head.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads (must be 1 for MQA).
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
        assert n_kv_heads == 1, f"MQA requires n_kv_heads=1, got {n_kv_heads}"

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, head_dim, bias=False)  # Single K head
        self.wv = nn.Linear(d_model, head_dim, bias=False)  # Single V head
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.wo._is_residual_proj = True

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Q: (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, T, 1, head_dim) → (B, 1, T, head_dim)
        k = self.wk(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # Expand K, V to all heads via broadcast
        # (B, 1, T, head_dim) → (B, n_heads, T, head_dim)
        k = k.expand(-1, self.n_heads, -1, -1)
        v = v.expand(-1, self.n_heads, -1, -1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

"""
Multi-Head Attention (MHA).

Standard attention where each query head has its own dedicated K and V head.
This is the baseline: maximum representational capacity, highest memory usage.

Reference: Vaswani et al., "Attention Is All You Need" (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class MultiHeadAttention(BaseAttention):
    """Standard Multi-Head Attention.

    n_kv_heads == n_heads: every query head has its own K/V pair.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads (must equal n_heads for MHA).
        head_dim: Dimension per attention head.
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

        # Q, K, V projections (no bias — modern convention)
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.wo._is_residual_proj = True  # Flag for scaled init

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, T, D).
            freqs_cis: RoPE frequencies of shape (T, head_dim // 2).
            mask: Causal mask of shape (1, 1, T, T).

        Returns:
            Output of shape (B, T, D).
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, T, head_dim), k/v: (B, n_kv_heads, T, head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rope(q, k, freqs_cis)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_weights: (B, n_heads, T, T)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)
        # out: (B, n_heads, T, head_dim)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

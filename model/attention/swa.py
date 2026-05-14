"""
Sliding Window Attention (SWA).

Restricts each query to attend only to keys within a fixed window.
Reduces attention memory from O(n^2) to O(n * w) where w = window_size.
Used in Mistral and as a component in hybrid architectures.

Reference: Beltagy et al., "Longformer: The Long-Document Transformer" (2020).
           Jiang et al., "Mistral 7B" (2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class SlidingWindowAttention(BaseAttention):
    """Sliding Window Attention with causal masking.

    Combines a causal mask with a sliding window: each position can
    only attend to the previous `window_size` positions.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        window_size: Attention window size.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        window_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        self.window_size = window_size

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.wo._is_residual_proj = True

    def _build_swa_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build combined causal + sliding window mask.

        Returns:
            Boolean mask of shape (1, 1, seq_len, seq_len).
            True = attend, False = mask out.
        """
        # Causal mask: lower triangular
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

        # Window mask: band diagonal
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        window = distance.abs() <= self.window_size

        # Combine: causal AND within window
        mask = causal & window
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # Expand KV if using GQA-style grouping within SWA
        if self.n_kv_heads < self.n_heads:
            n_groups = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_groups, dim=1)
            v = v.repeat_interleave(n_groups, dim=1)

        # Build sliding window mask (overrides the global causal mask)
        swa_mask = self._build_swa_mask(T, x.device)

        # Attention with SWA mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(~swa_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

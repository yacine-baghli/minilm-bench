"""
Abstract base class for all attention mechanisms.

Defines the shared interface that all variants must implement.
"""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseAttention(ABC, nn.Module):
    """Abstract base class for attention mechanisms.

    All attention variants must implement `forward` with this signature.
    This enables clean swapping between MHA, GQA, MQA, and SWA via config.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
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
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            freqs_cis: Precomputed RoPE frequencies of shape (seq_len, head_dim // 2).
            mask: Optional attention mask of shape (1, 1, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        ...

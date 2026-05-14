"""
Native Sparse Attention (NSA).

Hierarchical sparsity with three parallel branches:
  1. Compressed: block-pool KV for coarse global context
  2. Selected: use importance scores to pick fine-grained blocks
  3. Sliding Window: local attention for recent tokens

Branches are combined via learned per-head sigmoid gates.
Complexity: O(n * (n/b + k*b + w)) instead of O(n²).

Reference: DeepSeek (2025). "Native Sparse Attention:
           Hardware-Aligned and Natively Trainable."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class NativeSparseAttention(BaseAttention):
    """Native Sparse Attention with 3-branch hierarchical sparsity.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        block_size: Block size for compression (tokens per block).
        top_k_blocks: Number of fine-grained blocks to select.
        window_size: Sliding window size for local branch.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        top_k_blocks: int = 8,
        window_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.window_size = window_size

        # Shared Q, K, V projections
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.wo._is_residual_proj = True

        # Compression projection: map block-pooled KV to head_dim
        self.compress_k = nn.Linear(head_dim, head_dim, bias=False)
        self.compress_v = nn.Linear(head_dim, head_dim, bias=False)

        # Learned gates for combining branches (per head)
        self.gate_compress = nn.Parameter(torch.zeros(n_heads))
        self.gate_select = nn.Parameter(torch.zeros(n_heads))
        self.gate_window = nn.Parameter(torch.zeros(n_heads))

    def _compress_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Block-pool K and V into coarse compressed tokens.

        Groups consecutive tokens into blocks and averages them.

        Args:
            k: (B, H, T, D)
            v: (B, H, T, D)

        Returns:
            Compressed k, v of shape (B, H, T // block_size, D).
        """
        B, H, T, D = k.shape
        bs = self.block_size

        # Pad sequence to be divisible by block_size
        pad_len = (bs - T % bs) % bs
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        T_padded = k.size(2)
        n_blocks = T_padded // bs

        # Reshape into blocks and average
        k_blocks = k.view(B, H, n_blocks, bs, D).mean(dim=3)  # (B, H, n_blocks, D)
        v_blocks = v.view(B, H, n_blocks, bs, D).mean(dim=3)

        # Project compressed representations
        k_compressed = self.compress_k(k_blocks)
        v_compressed = self.compress_v(v_blocks)

        return k_compressed, v_compressed

    def _select_blocks(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-K important blocks based on compressed attention scores.

        Uses compressed attention as an importance proxy, then gathers
        fine-grained KV from the selected blocks.

        Args:
            q: (B, H, T, D)
            k_compressed: (B, H, n_blocks, D)
            k_full: (B, H, T, D)
            v_full: (B, H, T, D)

        Returns:
            Selected k, v of shape (B, H, top_k * block_size, D).
        """
        B, H, T, D = q.shape
        bs = self.block_size

        # Importance scores from compressed attention
        # Use mean query over time for block-level importance
        q_mean = q.mean(dim=2, keepdim=True)  # (B, H, 1, D)
        importance = torch.matmul(q_mean, k_compressed.transpose(-2, -1))  # (B, H, 1, n_blocks)
        importance = importance.squeeze(2)  # (B, H, n_blocks)

        # Select top-K blocks
        n_blocks = k_compressed.size(2)
        actual_k = min(self.top_k_blocks, n_blocks)
        _, top_idx = torch.topk(importance, k=actual_k, dim=-1)  # (B, H, top_k)

        # Gather fine-grained KV from selected blocks
        # Expand block indices to token indices
        block_starts = top_idx * bs  # (B, H, top_k)
        offsets = torch.arange(bs, device=q.device)  # (bs,)
        token_idx = (block_starts.unsqueeze(-1) + offsets.view(1, 1, 1, -1))  # (B, H, top_k, bs)
        token_idx = token_idx.reshape(B, H, -1)  # (B, H, top_k * bs)

        # Clamp indices to valid range
        T_full = k_full.size(2)
        token_idx = token_idx.clamp(0, T_full - 1)

        # Gather selected tokens
        token_idx_expanded = token_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        k_selected = torch.gather(k_full, 2, token_idx_expanded)
        v_selected = torch.gather(v_full, 2, token_idx_expanded)

        return k_selected, v_selected

    def _window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build causal sliding window mask."""
        pos = torch.arange(T, device=device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        mask = (diff >= 0) & (diff <= self.window_size)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Project Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # Expand KV heads if needed
        if self.n_kv_heads < self.n_heads:
            n_groups = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_groups, dim=1)
            v = v.repeat_interleave(n_groups, dim=1)

        # === Branch 1: Compressed global attention ===
        k_comp, v_comp = self._compress_kv(k, v)
        attn_compress = torch.matmul(q, k_comp.transpose(-2, -1)) * self.scale
        attn_compress = F.softmax(attn_compress, dim=-1)
        out_compress = torch.matmul(attn_compress, v_comp)  # (B, H, T, D)

        # === Branch 2: Selected fine-grained attention ===
        k_sel, v_sel = self._select_blocks(q, k_comp, k, v)
        attn_select = torch.matmul(q, k_sel.transpose(-2, -1)) * self.scale
        attn_select = F.softmax(attn_select, dim=-1)
        out_select = torch.matmul(attn_select, v_sel)  # (B, H, T, D)

        # === Branch 3: Sliding window local attention ===
        attn_window = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        win_mask = self._window_mask(T, x.device)
        if mask is not None:
            win_mask = win_mask & (mask.bool())
        attn_window = attn_window.masked_fill(~win_mask, float("-inf"))
        attn_window = F.softmax(attn_window, dim=-1)
        out_window = torch.matmul(attn_window, v)  # (B, H, T, D)

        # === Combine branches with learned gates ===
        g_c = torch.sigmoid(self.gate_compress).view(1, -1, 1, 1)
        g_s = torch.sigmoid(self.gate_select).view(1, -1, 1, 1)
        g_w = torch.sigmoid(self.gate_window).view(1, -1, 1, 1)

        out = g_c * out_compress + g_s * out_select + g_w * out_window

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

"""
Multi-head Latent Attention (MLA).

Compresses Key/Value into a low-rank latent space before caching,
then reconstructs full K/V on the fly via up-projections.
Achieves near-lossless quality with massive KV cache reduction (up to 8×).

Uses decoupled RoPE: position info is injected through separate dedicated
dimensions because standard RoPE is incompatible with low-rank compression
(rotation breaks the learned latent structure).

Reference: DeepSeek-V2 (2024). Used in DeepSeek-V3, DeepSeek-R1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention
from model.embeddings import apply_rope


class MultiHeadLatentAttention(BaseAttention):
    """Multi-head Latent Attention with low-rank KV compression.

    Architecture:
        x → W_DKV → c_KV (low-rank latent, d_latent dims)
        c_KV → W_UK → K (reconstructed)
        c_KV → W_UV → V (reconstructed)
        x → W_Q → Q
        RoPE applied via separate decoupled dimensions on Q and K

    Only c_KV is cached during inference → massive memory savings.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        d_latent: Latent dimension for KV compression.
        rope_head_dim: Dimension for decoupled RoPE injection.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        d_latent: int = 192,
        rope_head_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        self.d_latent = d_latent
        self.rope_head_dim = rope_head_dim

        # Query projection (standard)
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # KV compression: down-project to latent space
        self.w_dkv = nn.Linear(d_model, d_latent, bias=False)

        # KV reconstruction: up-project from latent to full K, V
        self.w_uk = nn.Linear(d_latent, n_kv_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(d_latent, n_kv_heads * head_dim, bias=False)

        # Decoupled RoPE projections: separate position-aware dimensions
        # These dimensions carry positional info and bypass the latent bottleneck
        self.wq_rope = nn.Linear(d_model, n_heads * rope_head_dim, bias=False)
        self.wk_rope = nn.Linear(d_latent, n_kv_heads * rope_head_dim, bias=False)

        # Output projection
        total_out_dim = n_heads * (head_dim + rope_head_dim)
        self.wo = nn.Linear(total_out_dim, d_model, bias=False)
        self.wo._is_residual_proj = True

        # Scale for attention
        self.scale = (head_dim + rope_head_dim) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # --- Query path ---
        q_content = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, T, head_dim)

        # --- KV compression path ---
        c_kv = self.w_dkv(x)  # (B, T, d_latent) — this is what gets cached

        # Reconstruct full K, V from latent
        k_content = self.w_uk(c_kv).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k_content, v: (B, n_kv_heads, T, head_dim)

        # --- Decoupled RoPE path ---
        # Separate position-aware projections that bypass the latent bottleneck
        q_rope = self.wq_rope(x).view(B, T, self.n_heads, self.rope_head_dim).transpose(1, 2)
        k_rope = self.wk_rope(c_kv).view(B, T, self.n_kv_heads, self.rope_head_dim).transpose(1, 2)

        # Apply RoPE only to the dedicated position dimensions
        # Need frequencies for rope_head_dim, not head_dim
        rope_freqs = freqs_cis[:T, :self.rope_head_dim // 2]
        q_rope, k_rope = apply_rope(q_rope, k_rope, rope_freqs)

        # --- Concatenate content + position dimensions ---
        # Q: (B, n_heads, T, head_dim + rope_head_dim)
        q = torch.cat([q_content, q_rope], dim=-1)

        # K: need to expand KV heads to match query heads, then concat
        if self.n_kv_heads < self.n_heads:
            n_groups = self.n_heads // self.n_kv_heads
            k_content = k_content.repeat_interleave(n_groups, dim=1)
            k_rope = k_rope.repeat_interleave(n_groups, dim=1)
            v = v.repeat_interleave(n_groups, dim=1)

        k = torch.cat([k_content, k_rope], dim=-1)
        # k: (B, n_heads, T, head_dim + rope_head_dim)

        # --- Standard attention with combined dimensions ---
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Value uses only content dimensions, but output has full dim
        # Pad v to match attention output expectations
        v_padded = F.pad(v, (0, self.rope_head_dim))  # (B, H, T, head_dim + rope_head_dim)
        out = torch.matmul(attn_weights, v_padded)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

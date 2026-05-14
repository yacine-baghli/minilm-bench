"""
Model configuration dataclass.

Defines all architectural hyperparameters for the Transformer.
Configs are typically loaded from YAML and converted to this dataclass.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a GPT-style decoder Transformer.

    Attributes:
        vocab_size: Vocabulary size for token embeddings.
        d_model: Hidden dimension (embedding size).
        n_layers: Number of Transformer blocks.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads.
            - n_kv_heads == n_heads → MHA (Multi-Head Attention)
            - 1 < n_kv_heads < n_heads → GQA (Grouped-Query Attention)
            - n_kv_heads == 1 → MQA (Multi-Query Attention)
        d_ff_mult: FFN hidden dimension multiplier (d_ff = d_model * d_ff_mult).
        max_seq_len: Maximum sequence length supported.
        dropout: Dropout rate (0.0 for pre-training).
        rope_base: Base frequency for Rotary Position Embeddings.
        norm_eps: Epsilon for RMSNorm.
        attention_type: Attention variant identifier.
        swa_window_size: Window size for Sliding Window Attention.
        d_latent: Latent dimension for MLA KV compression.
        rope_head_dim: Separate RoPE dimension for MLA decoupled RoPE.
        moh_top_k: Number of active heads in MoH routing.
        moh_n_shared: Number of always-active shared heads in MoH.
        moh_aux_loss_weight: Load balancing loss coefficient for MoH.
        nsa_block_size: Block size for NSA compression branch.
        nsa_top_k_blocks: Number of selected blocks in NSA.
        nsa_window_size: Window size for NSA local branch.
    """

    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_ff_mult: int = 4
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    attention_type: str = "mha"
    swa_window_size: int = 512
    # MLA (Multi-head Latent Attention)
    d_latent: int = 192
    rope_head_dim: int = 64
    # MoH (Mixture-of-Head)
    moh_top_k: int = 6
    moh_n_shared: int = 1
    moh_aux_loss_weight: float = 0.01
    # NSA (Native Sparse Attention)
    nsa_block_size: int = 16
    nsa_top_k_blocks: int = 8
    nsa_window_size: int = 512

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        """Feed-forward hidden dimension."""
        return self.d_model * self.d_ff_mult

    @property
    def n_query_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    def num_params(self) -> int:
        """Estimate total parameter count (non-embedding)."""
        # Attention: Q + K + V + Output projections
        attn_params = (
            self.d_model * self.head_dim * self.n_heads  # Q
            + self.d_model * self.head_dim * self.n_kv_heads  # K
            + self.d_model * self.head_dim * self.n_kv_heads  # V
            + self.head_dim * self.n_heads * self.d_model  # Output
        )
        # FFN: SwiGLU has 3 linear layers (gate, up, down)
        ffn_params = 3 * self.d_model * self.d_ff
        # Norms: 2 per layer (pre-attn, pre-ffn)
        norm_params = 2 * self.d_model
        # Per layer total
        per_layer = attn_params + ffn_params + norm_params
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        # Final norm + LM head (weight-tied with embedding)
        final = self.d_model  # final norm

        return self.n_layers * per_layer + embed_params + final

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        valid_types = {"mha", "gqa", "mqa", "swa", "diff", "mla", "moh", "nsa"}
        assert self.attention_type in valid_types, (
            f"attention_type must be one of {valid_types}, got {self.attention_type!r}"
        )

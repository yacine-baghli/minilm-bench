"""
GPT-style Decoder Transformer.

Assembles all components: token embeddings, RoPE, Transformer blocks
(with configurable attention), and language modeling head.

This is the top-level model class used for pre-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.embeddings import TokenEmbedding, precompute_rope_frequencies
from model.layers import RMSNorm, TransformerBlock
from model.attention import build_attention
from model.utils import init_weights


class Transformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.

    Architecture:
        Token Embedding → [TransformerBlock × n_layers] → RMSNorm → LM Head

    Each TransformerBlock contains:
        RMSNorm → Attention → Residual → RMSNorm → SwiGLU FFN → Residual

    The LM head shares weights with the token embedding (weight tying).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding (shared with LM head via weight tying)
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            block = TransformerBlock(config, layer_idx)
            # Create and attach the attention module
            # Pass all variant-specific params — unused ones are absorbed by **kwargs
            block.attention = build_attention(
                attention_type=config.attention_type,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                # SWA params
                window_size=config.swa_window_size,
                # MLA params
                d_latent=config.d_latent,
                rope_head_dim=config.rope_head_dim,
                # MoH params
                top_k=config.moh_top_k,
                n_shared=config.moh_n_shared,
                aux_loss_weight=config.moh_aux_loss_weight,
                # NSA params
                block_size=config.nsa_block_size,
                top_k_blocks=config.nsa_top_k_blocks,
            )
            self.blocks.append(block)

        # Final norm
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.embedding.weight  # Weight tying

        # Precompute RoPE frequencies (use max dim to support MLA's decoupled RoPE)
        rope_dim = max(config.head_dim, config.rope_head_dim)
        freqs_cis = precompute_rope_frequencies(
            rope_dim, config.max_seq_len, config.rope_base
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Precompute causal mask
        causal_mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Initialize weights
        self.apply(lambda m: init_weights(m, config))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            targets: Target token IDs of shape (batch, seq_len).
                If provided, computes cross-entropy loss.

        Returns:
            Tuple of (logits, loss):
                - logits: shape (batch, seq_len, vocab_size)
                - loss: scalar tensor if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Token embeddings
        x = self.token_emb(input_ids)  # (B, T, D)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]

        # Get causal mask for this sequence length
        mask = self.causal_mask[:, :, :T, :T]

        # Forward through Transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis, mask)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            input_ids: Prompt token IDs of shape (batch, prompt_len).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = neutral).
            top_k: If set, only sample from top-k logits.

        Returns:
            Generated token IDs of shape (batch, prompt_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Last position only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

"""
Embeddings: Token embeddings and Rotary Position Embeddings (RoPE).

RoPE (Su et al., 2021) encodes position via rotation of Q/K vectors.
Key advantage: relative position encoding that extrapolates to unseen lengths.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding layer.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimension.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token IDs to vectors.

        Args:
            x: Token IDs of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        return self.embedding(x)


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute RoPE complex frequency tensor.

    Computes e^(i * m * theta) for all positions m and frequency bands.
    Using complex exponentials makes the rotation application efficient.

    Args:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base frequency (default 10000, higher = better length extrapolation).
        device: Target device.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Frequency bands: theta_i = base^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (float(base) ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices: m = [0, 1, ..., max_seq_len - 1]
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (max_seq_len, head_dim // 2)
    freqs_matrix = torch.outer(positions, freqs)

    # Convert to complex: e^(i * m * theta)
    freqs_cis = torch.polar(torch.ones_like(freqs_matrix), freqs_matrix)

    return freqs_cis


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors.

    Reshapes Q/K to complex pairs, multiplies by rotation frequencies,
    then converts back to real. This encodes relative position information
    directly into the attention dot product.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2).

    Returns:
        Tuple of rotated (q, k) with same shapes as input.
    """
    # Reshape to complex: (..., head_dim) → (..., head_dim // 2, 2) → complex
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Slice freqs to match actual head dimension (may be smaller than precomputed)
    q_freq_dim = q_complex.shape[-1]
    k_freq_dim = k_complex.shape[-1]
    freqs_q = freqs_cis[:, :q_freq_dim].unsqueeze(0).unsqueeze(0)
    freqs_k = freqs_cis[:, :k_freq_dim].unsqueeze(0).unsqueeze(0)

    # Apply rotation via complex multiplication
    q_rotated = torch.view_as_real(q_complex * freqs_q).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs_k).flatten(-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)

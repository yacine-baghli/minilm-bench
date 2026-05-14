"""
Attention mechanism implementations.

Eight variants (4 standard + 4 advanced) share a common interface
and can be swapped via config.
"""

from model.attention.base import BaseAttention
from model.attention.mha import MultiHeadAttention
from model.attention.gqa import GroupedQueryAttention
from model.attention.mqa import MultiQueryAttention
from model.attention.swa import SlidingWindowAttention
from model.attention.diff_attn import DifferentialAttention
from model.attention.mla import MultiHeadLatentAttention
from model.attention.moh import MixtureOfHeadAttention
from model.attention.nsa import NativeSparseAttention

ATTENTION_REGISTRY: dict[str, type[BaseAttention]] = {
    # Standard variants
    "mha": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
    "mqa": MultiQueryAttention,
    "swa": SlidingWindowAttention,
    # Advanced sparse variants
    "diff": DifferentialAttention,
    "mla": MultiHeadLatentAttention,
    "moh": MixtureOfHeadAttention,
    "nsa": NativeSparseAttention,
}


def build_attention(attention_type: str, **kwargs) -> BaseAttention:
    """Factory function to create an attention module by type name.

    Args:
        attention_type: One of the registered attention types.
        **kwargs: Arguments forwarded to the attention constructor.

    Returns:
        An attention module instance.
    """
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type {attention_type!r}. "
            f"Valid types: {list(ATTENTION_REGISTRY.keys())}"
        )
    return ATTENTION_REGISTRY[attention_type](**kwargs)


__all__ = [
    "BaseAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "SlidingWindowAttention",
    "DifferentialAttention",
    "MultiHeadLatentAttention",
    "MixtureOfHeadAttention",
    "NativeSparseAttention",
    "build_attention",
    "ATTENTION_REGISTRY",
]

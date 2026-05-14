"""
MiniLM-Bench Model Module.

From-scratch GPT-style decoder Transformer with swappable attention mechanisms.
No HuggingFace transformers dependency — every component implemented manually.
"""

from model.config import ModelConfig
from model.transformer import Transformer

__all__ = ["ModelConfig", "Transformer"]

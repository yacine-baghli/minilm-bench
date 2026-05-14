"""
MiniLM-Bench Data Module.

Provides streaming download, tokenization, and memory-mapped data loading
for efficient pre-training on FineWeb-Edu.
"""

from data.tokenizer import Tokenizer
from data.dataloader import create_dataloader

__all__ = ["Tokenizer", "create_dataloader"]

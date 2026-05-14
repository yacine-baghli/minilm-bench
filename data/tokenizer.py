"""
Tokenizer wrapper for MiniLM-Bench.

Wraps tiktoken to provide a consistent interface for encoding/decoding.
Supports GPT-2 BPE encoding (50257 vocab).
"""

import tiktoken


class Tokenizer:
    """Thin wrapper around tiktoken for consistent encode/decode interface.

    Args:
        encoding_name: tiktoken encoding identifier (default: "gpt2").
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.encoding.n_vocab

    @property
    def eot_token(self) -> int:
        """Return end-of-text token ID."""
        return self.encoding.eot_token

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (ordinary tokens only, no special)."""
        return self.encoding.encode_ordinary(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.encoding.decode(token_ids)

    def __repr__(self) -> str:
        return f"Tokenizer(encoding={self.encoding_name!r}, vocab_size={self.vocab_size})"

"""
Dataset download and tokenization script.

Downloads FineWeb-Edu via streaming, tokenizes with tiktoken,
and saves as memory-mapped .bin shards for zero-copy training.

Usage:
    python -m data.download --dataset HuggingFaceFW/fineweb-edu-sample-10BT \
                            --output_dir ./data/tokenized \
                            --shard_size 100_000_000
"""

import os
import argparse
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import tiktoken


def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    """Load a tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def tokenize_document(text: str, enc: tiktoken.Encoding) -> list[int]:
    """Tokenize a single document, prepending with EOT token."""
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(text))
    return tokens


def stream_dataset(dataset_name: str, config_name: str | None = None, split: str = "train") -> Iterator[dict]:
    """Stream documents from a HuggingFace dataset without downloading fully."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, name=config_name, split=split, streaming=True)
    yield from ds


def write_shard(
    tokens: np.ndarray,
    output_dir: Path,
    shard_idx: int,
) -> Path:
    """Write a shard of tokenized data as a memory-mapped .bin file.

    File format: raw uint16 token IDs (no header).
    uint16 supports vocab sizes up to 65535 (GPT-2 vocab = 50257).
    """
    shard_path = output_dir / f"shard_{shard_idx:05d}.bin"
    tokens_u16 = tokens.astype(np.uint16)
    tokens_u16.tofile(shard_path)
    return shard_path


def download_and_tokenize(
    dataset_name: str,
    output_dir: str,
    shard_size: int = 100_000_000,
    encoding_name: str = "gpt2",
    max_shards: int | None = None,
    config_name: str | None = None,
) -> None:
    """Download dataset, tokenize, and save as memory-mapped shards.

    Args:
        dataset_name: HuggingFace dataset identifier.
        output_dir: Directory to save tokenized shards.
        shard_size: Number of tokens per shard.
        encoding_name: tiktoken encoding name.
        max_shards: Maximum number of shards to create (None = unlimited).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    enc = get_tokenizer(encoding_name)
    print(f"Tokenizer loaded: {encoding_name} (vocab_size={enc.n_vocab})")

    # Buffer for accumulating tokens before writing a shard
    token_buffer = np.empty(shard_size, dtype=np.uint16)
    buffer_idx = 0
    shard_idx = 0
    total_tokens = 0

    print(f"Streaming dataset: {dataset_name}")
    for doc in stream_dataset(dataset_name, config_name=config_name):
        text = doc.get("text", "")
        if not text:
            continue

        tokens = tokenize_document(text, enc)

        # Write tokens to buffer, flushing shards as needed
        for token in tokens:
            token_buffer[buffer_idx] = token
            buffer_idx += 1

            if buffer_idx >= shard_size:
                shard_path = write_shard(token_buffer, output_path, shard_idx)
                total_tokens += buffer_idx
                print(
                    f"  Shard {shard_idx:05d} written: {buffer_idx:,} tokens "
                    f"({shard_path.name}) | Total: {total_tokens:,}"
                )
                buffer_idx = 0
                shard_idx += 1

                if max_shards is not None and shard_idx >= max_shards:
                    print(f"Reached max_shards={max_shards}, stopping.")
                    return

    # Write remaining tokens
    if buffer_idx > 0:
        shard_path = write_shard(
            token_buffer[:buffer_idx], output_path, shard_idx
        )
        total_tokens += buffer_idx
        print(
            f"  Shard {shard_idx:05d} written: {buffer_idx:,} tokens "
            f"({shard_path.name}) | Total: {total_tokens:,}"
        )

    print(f"\nDone. {shard_idx + 1} shards, {total_tokens:,} total tokens.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and tokenize dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
    )
    parser.add_argument("--config_name", type=str, default="sample-10BT")
    parser.add_argument("--output_dir", type=str, default="./data/tokenized")
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--encoding", type=str, default="gpt2")
    parser.add_argument("--max_shards", type=int, default=None)
    args = parser.parse_args()

    download_and_tokenize(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        encoding_name=args.encoding,
        max_shards=args.max_shards,
        config_name=args.config_name,
    )

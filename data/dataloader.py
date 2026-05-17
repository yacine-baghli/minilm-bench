"""
Memory-mapped DataLoader for pre-training.

Reads tokenized .bin shards via np.memmap for zero-copy access.
Produces (input, target) pairs where target = input shifted by 1 position.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ShardedTokenDataset(Dataset):
    """Memory-mapped dataset over tokenized .bin shards.

    Each shard is a flat array of uint16 token IDs. The dataset
    concatenates all shards virtually and samples random contiguous
    subsequences of length `seq_len + 1` (input + target).

    Args:
        data_dir: Path to directory containing .bin shard files.
        seq_len: Sequence length for training.
    """

    def __init__(self, data_dir: str, seq_len: int) -> None:
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)

        # Discover and memory-map all shards
        shard_paths = sorted(self.data_dir.glob("shard_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_*.bin files found in {self.data_dir}"
            )

        self.shards: list[np.memmap] = []
        self.shard_lengths: list[int] = []
        self.cumulative_lengths: list[int] = []
        total = 0

        for path in shard_paths:
            shard = np.memmap(path, dtype=np.uint16, mode="r")
            self.shards.append(shard)
            self.shard_lengths.append(len(shard))
            total += len(shard)
            self.cumulative_lengths.append(total)

        self.total_tokens = total
        # Number of valid starting positions
        self._len = total - seq_len - 1

        print(
            f"ShardedTokenDataset: {len(shard_paths)} shards, "
            f"{self.total_tokens:,} tokens, "
            f"{self._len:,} valid positions"
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) pair of shape (seq_len,) each.

        Target is input shifted by 1 position (next-token prediction).
        """
        # Find which shard this index falls in
        chunk = self._get_contiguous_chunk(idx, self.seq_len + 1)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

    def _get_contiguous_chunk(self, start: int, length: int) -> np.ndarray:
        """Extract a contiguous chunk of tokens across shard boundaries."""
        result = np.empty(length, dtype=np.uint16)
        offset = 0
        remaining = length

        # Find starting shard
        shard_idx = 0
        local_start = start
        for i, cum_len in enumerate(self.cumulative_lengths):
            if start < cum_len:
                shard_idx = i
                local_start = start - (self.cumulative_lengths[i - 1] if i > 0 else 0)
                break

        # Read across shards if necessary
        while remaining > 0:
            shard = self.shards[shard_idx]
            available = len(shard) - local_start
            to_read = min(remaining, available)
            result[offset : offset + to_read] = shard[local_start : local_start + to_read]
            offset += to_read
            remaining -= to_read
            shard_idx += 1
            local_start = 0

        return result


def create_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for pre-training.

    Args:
        data_dir: Path to tokenized shards.
        seq_len: Training sequence length.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        shuffle: Whether to shuffle (should be True for training).
        seed: Random seed for reproducibility.

    Returns:
        DataLoader yielding (input, target) batches of shape (B, seq_len).
    """
    dataset = ShardedTokenDataset(data_dir, seq_len)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator if shuffle else None,
    )

"""DDP/FSDP distributed training setup utilities."""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int]:
    """Initialize distributed process group. Returns (rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed() -> None:
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: torch.nn.Module, device_id: int) -> DDP:
    """Wrap model in DistributedDataParallel."""
    return DDP(model, device_ids=[device_id])

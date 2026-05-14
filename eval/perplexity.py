"""Validation perplexity computation."""

import torch
from model import Transformer
from data.dataloader import create_dataloader


@torch.no_grad()
def evaluate_perplexity(model: Transformer, data_dir: str, seq_len: int, batch_size: int, eval_steps: int = 50, device: str = "cuda") -> float:
    """Compute validation perplexity over eval_steps batches.

    Returns:
        Perplexity (exp of average cross-entropy loss).
    """
    model.eval()
    loader = create_dataloader(data_dir, seq_len, batch_size, num_workers=2, shuffle=False)
    total_loss = 0.0
    count = 0

    for i, (x, y) in enumerate(loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return perplexity

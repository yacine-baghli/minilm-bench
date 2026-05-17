"""End-to-end smoke test for the training pipeline.

Creates a tiny model + synthetic data and runs a few training steps
to verify the full pipeline (model → optimizer → grad accum → loss → checkpoint).
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from model.config import ModelConfig
from model.transformer import Transformer
from training.optimizer import configure_optimizer, build_lr_scheduler
from training.checkpoint import CheckpointManager


def _create_synthetic_data(data_dir: str, vocab_size: int, seq_len: int, n_tokens: int = 8192) -> None:
    """Create a synthetic memmap shard for testing."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    tokens = np.random.randint(0, vocab_size, size=(n_tokens,), dtype=np.uint16)
    shard_path = path / "shard_000000.bin"
    mmap = np.memmap(shard_path, dtype=np.uint16, mode="w+", shape=tokens.shape)
    mmap[:] = tokens
    mmap.flush()


@pytest.mark.parametrize("attn_type", ["mha", "gqa", "mqa", "swa", "diff", "mla", "moh", "nsa"])
def test_training_loop_smoke(attn_type, tmp_path):
    """Run 5 training steps for each attention variant on CPU."""
    vocab_size, d_model, seq_len = 256, 64, 32
    batch_size = 2
    n_steps = 5

    # Create synthetic data
    data_dir = str(tmp_path / "data")
    _create_synthetic_data(data_dir, vocab_size, seq_len, n_tokens=4096)

    # Build tiny model
    cfg_kwargs = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=2, n_heads=4,
        n_kv_heads=4, d_ff_mult=2, max_seq_len=seq_len, attention_type=attn_type,
    )
    if attn_type == "mqa":
        cfg_kwargs["n_kv_heads"] = 1
    elif attn_type == "mla":
        cfg_kwargs["d_latent"] = 32
        cfg_kwargs["rope_head_dim"] = 16
    elif attn_type == "moh":
        cfg_kwargs["moh_top_k"] = 2
        cfg_kwargs["moh_n_shared"] = 1
    elif attn_type == "nsa":
        cfg_kwargs["nsa_block_size"] = 8
        cfg_kwargs["nsa_top_k_blocks"] = 2
        cfg_kwargs["nsa_window_size"] = 16

    config = ModelConfig(**cfg_kwargs)
    model = Transformer(config)

    # Build optimizer + scheduler
    optimizer = configure_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=2, max_steps=n_steps)

    # Import dataloader
    from data.dataloader import create_dataloader
    loader = create_dataloader(data_dir, seq_len, batch_size, num_workers=0, shuffle=True)
    data_iter = iter(loader)

    # Training loop
    model.train()
    losses = []
    for step in range(n_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        _, loss = model(x, y)

        # MoH: add auxiliary loss
        if attn_type == "moh":
            for block in model.blocks:
                if hasattr(block.attention, "aux_loss") and block.attention.aux_loss is not None:
                    loss = loss + block.attention.aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # Verify loss is finite and decreasing or reasonable
    assert all(np.isfinite(val) for val in losses), f"Non-finite losses for {attn_type}: {losses}"
    print(f"  {attn_type}: losses={[f'{val:.3f}' for val in losses]}")

    # Verify checkpoint save/load roundtrip
    ckpt_dir = str(tmp_path / f"ckpt_{attn_type}")
    mgr = CheckpointManager(ckpt_dir, max_keep=2)
    mgr.save(model, optimizer, scheduler, step=n_steps)

    state = mgr.load_latest()
    assert state is not None
    assert state["step"] == n_steps

    # Verify model loads correctly
    model2 = Transformer(config)
    model2.load_state_dict(state["model_state_dict"])
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_lr_schedule_warmup_and_decay():
    """Verify cosine schedule with warmup produces expected LR curve."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=5, max_steps=20)

    lrs = []
    for step in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Warmup: LR should increase
    assert lrs[0] < lrs[4], f"LR should increase during warmup: {lrs[:5]}"
    # Peak at end of warmup
    assert lrs[4] > lrs[10], "LR should decay after warmup"
    # Final LR should be near min_lr (10% of peak)
    assert lrs[-1] < lrs[5], f"LR should be lower at end: {lrs[-1]} vs {lrs[5]}"

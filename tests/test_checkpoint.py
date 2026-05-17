"""Tests for checkpoint save/resume consistency."""

import torch
import tempfile
from model.config import ModelConfig
from model.transformer import Transformer
from training.checkpoint import CheckpointManager


def test_save_and_load_roundtrip():
    cfg = ModelConfig(vocab_size=256, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, d_ff_mult=4, max_seq_len=64)
    model = Transformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir, max_keep=3)
        mgr.save(model, optimizer, None, step=100, val_loss=2.5)

        state = mgr.load_latest()
        assert state is not None
        assert state["step"] == 100
        assert state["val_loss"] == 2.5

        model2 = Transformer(cfg)
        model2.load_state_dict(state["model_state_dict"])

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

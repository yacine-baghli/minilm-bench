"""Unit tests for the full Transformer model."""

import pytest
import torch
from model.config import ModelConfig
from model.transformer import Transformer


def _make_small_config(**overrides) -> ModelConfig:
    defaults = dict(vocab_size=256, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, d_ff_mult=4, max_seq_len=64, attention_type="mha")
    defaults.update(overrides)
    return ModelConfig(**defaults)


@pytest.mark.parametrize("attn_type,n_kv", [("mha", 4), ("gqa", 2), ("mqa", 1), ("swa", 4)])
def test_forward_pass_shape(attn_type, n_kv):
    cfg = _make_small_config(attention_type=attn_type, n_kv_heads=n_kv)
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets():
    cfg = _make_small_config()
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar


def test_weight_tying():
    cfg = _make_small_config()
    model = Transformer(cfg)
    assert model.lm_head.weight is model.token_emb.embedding.weight


def test_parameter_count():
    cfg = _make_small_config()
    model = Transformer(cfg)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual > 0
    print(f"Small model params: {actual:,}")


def test_generate():
    cfg = _make_small_config()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    output = model.generate(prompt, max_new_tokens=10)
    assert output.shape == (1, 14)

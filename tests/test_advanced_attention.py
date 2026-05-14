"""Unit tests for advanced sparse attention variants (DiffAttn, MLA, MoH, NSA)."""

import pytest
import torch
from model.config import ModelConfig
from model.attention import build_attention
from model.embeddings import precompute_rope_frequencies
from model.transformer import Transformer

B, T, D = 2, 32, 512
N_HEADS, HEAD_DIM = 8, 64


@pytest.fixture
def freqs_cis():
    return precompute_rope_frequencies(HEAD_DIM, T)


@pytest.fixture
def x():
    return torch.randn(B, T, D)


# ── DiffAttn ──────────────────────────────────────────────

def test_diff_attn_shape(x, freqs_cis):
    attn = build_attention("diff", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, layer_idx=0)
    out = attn(x, freqs_cis)
    assert out.shape == (B, T, D)


def test_diff_attn_lambda_initialization():
    """λ should be depth-dependent."""
    import math
    attn0 = build_attention("diff", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, layer_idx=0)
    attn5 = build_attention("diff", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, layer_idx=5)
    # Layer 0 should have smaller λ than layer 5 (formula increases with depth)
    assert attn0.lambda_param.mean().item() < attn5.lambda_param.mean().item()


def test_diff_attn_gradient_flow(x, freqs_cis):
    attn = build_attention("diff", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, layer_idx=0)
    x_in = x.clone().requires_grad_(True)
    out = attn(x_in, freqs_cis)
    out.sum().backward()
    assert x_in.grad is not None and x_in.grad.abs().sum() > 0


# ── MLA ───────────────────────────────────────────────────

def test_mla_shape(x):
    freqs = precompute_rope_frequencies(HEAD_DIM, T)  # Must cover rope_head_dim
    attn = build_attention("mla", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, d_latent=128, rope_head_dim=HEAD_DIM)
    out = attn(x, freqs)
    assert out.shape == (B, T, D)


def test_mla_gradient_flow(x):
    freqs = precompute_rope_frequencies(HEAD_DIM, T)
    attn = build_attention("mla", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, d_latent=128, rope_head_dim=HEAD_DIM)
    x_in = x.clone().requires_grad_(True)
    out = attn(x_in, freqs)
    out.sum().backward()
    assert x_in.grad is not None and x_in.grad.abs().sum() > 0


def test_mla_latent_is_smaller():
    """MLA should have fewer KV projection params than MHA (the compression)."""
    mha = build_attention("mha", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM)
    mla = build_attention("mla", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, d_latent=64, rope_head_dim=HEAD_DIM)
    # MLA down-projection is d_model→d_latent (512→64 = 32K) vs MHA K+V (512→512 * 2 = 524K)
    mla_kv_params = sum(p.numel() for n, p in mla.named_parameters() if 'w_dkv' in n or 'w_uk' in n or 'w_uv' in n)
    mha_kv_params = sum(p.numel() for n, p in mha.named_parameters() if 'wk' in n or 'wv' in n)
    # With small d_latent, MLA KV path should be smaller
    assert mla_kv_params < mha_kv_params


# ── MoH ───────────────────────────────────────────────────

def test_moh_shape(x, freqs_cis):
    attn = build_attention("moh", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, top_k=4, n_shared=1)
    out = attn(x, freqs_cis)
    assert out.shape == (B, T, D)


def test_moh_aux_loss_during_training(x, freqs_cis):
    """MoH should produce an auxiliary loss during training."""
    attn = build_attention("moh", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, top_k=4, n_shared=1)
    attn.train()
    _ = attn(x, freqs_cis)
    assert attn.aux_loss is not None
    assert attn.aux_loss.dim() == 0  # Scalar


def test_moh_no_aux_loss_during_eval(x, freqs_cis):
    attn = build_attention("moh", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, top_k=4, n_shared=1)
    attn.eval()
    _ = attn(x, freqs_cis)
    assert attn.aux_loss is None


def test_moh_gradient_flow(x, freqs_cis):
    attn = build_attention("moh", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, top_k=4, n_shared=1)
    x_in = x.clone().requires_grad_(True)
    out = attn(x_in, freqs_cis)
    out.sum().backward()
    assert x_in.grad is not None and x_in.grad.abs().sum() > 0


# ── NSA ───────────────────────────────────────────────────

def test_nsa_shape(x, freqs_cis):
    attn = build_attention("nsa", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, block_size=8, top_k_blocks=2, window_size=16)
    out = attn(x, freqs_cis)
    assert out.shape == (B, T, D)


def test_nsa_gradient_flow(x, freqs_cis):
    attn = build_attention("nsa", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, block_size=8, top_k_blocks=2, window_size=16)
    x_in = x.clone().requires_grad_(True)
    out = attn(x_in, freqs_cis)
    out.sum().backward()
    assert x_in.grad is not None and x_in.grad.abs().sum() > 0


def test_nsa_gate_values():
    """Gates should be initialized near 0.5 (sigmoid of 0)."""
    attn = build_attention("nsa", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, block_size=8, top_k_blocks=2, window_size=16)
    for name in ['gate_compress', 'gate_select', 'gate_window']:
        gate = getattr(attn, name)
        assert torch.allclose(torch.sigmoid(gate), torch.tensor(0.5), atol=1e-5)


# ── Full Transformer with advanced variants ───────────────

@pytest.mark.parametrize("attn_type", ["diff", "mla", "moh", "nsa"])
def test_transformer_forward_advanced(attn_type):
    """Full model forward pass with each advanced variant."""
    kwargs = dict(vocab_size=256, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, d_ff_mult=4, max_seq_len=64, attention_type=attn_type)
    if attn_type == "mla":
        kwargs["d_latent"] = 64
        kwargs["rope_head_dim"] = 32
    elif attn_type == "moh":
        kwargs["moh_top_k"] = 2
        kwargs["moh_n_shared"] = 1
    elif attn_type == "nsa":
        kwargs["nsa_block_size"] = 8
        kwargs["nsa_top_k_blocks"] = 2
        kwargs["nsa_window_size"] = 16
    cfg = ModelConfig(**kwargs)
    model = Transformer(cfg)
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, 256)
    assert loss is not None and loss.dim() == 0

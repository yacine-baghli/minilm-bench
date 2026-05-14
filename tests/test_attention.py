"""Unit tests for attention variants."""

import pytest
import torch
from model.config import ModelConfig
from model.attention import build_attention
from model.embeddings import precompute_rope_frequencies

B, T, D = 2, 32, 512
N_HEADS, HEAD_DIM = 8, 64


@pytest.fixture
def freqs_cis():
    return precompute_rope_frequencies(HEAD_DIM, T)


@pytest.fixture
def x():
    return torch.randn(B, T, D)


@pytest.mark.parametrize("attn_type,n_kv", [("mha", 8), ("gqa", 4), ("gqa", 2), ("mqa", 1)])
def test_attention_output_shape(attn_type, n_kv, x, freqs_cis):
    attn = build_attention(attn_type, d_model=D, n_heads=N_HEADS, n_kv_heads=n_kv, head_dim=HEAD_DIM)
    out = attn(x, freqs_cis)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


def test_swa_output_shape(x, freqs_cis):
    attn = build_attention("swa", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM, window_size=16)
    out = attn(x, freqs_cis)
    assert out.shape == (B, T, D)


def test_mha_gqa_equivalence_when_all_heads(x, freqs_cis):
    """GQA with n_kv_heads == n_heads should behave identically to MHA (same weights)."""
    mha = build_attention("mha", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM)
    gqa = build_attention("gqa", d_model=D, n_heads=N_HEADS, n_kv_heads=N_HEADS, head_dim=HEAD_DIM)
    # Copy weights
    gqa.load_state_dict(mha.state_dict())
    out_mha = mha(x, freqs_cis)
    out_gqa = gqa(x, freqs_cis)
    torch.testing.assert_close(out_mha, out_gqa, atol=1e-5, rtol=1e-4)


def test_gradient_flow(x, freqs_cis):
    """Ensure gradients flow through all attention variants."""
    for attn_type, n_kv in [("mha", 8), ("gqa", 4), ("mqa", 1), ("swa", 8)]:
        kwargs = dict(d_model=D, n_heads=N_HEADS, n_kv_heads=n_kv, head_dim=HEAD_DIM)
        if attn_type == "swa":
            kwargs["window_size"] = 16
        attn = build_attention(attn_type, **kwargs)
        x_input = x.clone().requires_grad_(True)
        out = attn(x_input, freqs_cis)
        out.sum().backward()
        assert x_input.grad is not None, f"No gradient for {attn_type}"
        assert x_input.grad.abs().sum() > 0, f"Zero gradient for {attn_type}"

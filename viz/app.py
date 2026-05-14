"""
Streamlit dashboard for model inspection and comparison.

Three tabs:
  1. Training Comparison — loss curves, throughput, memory overlaid
  2. Attention Inspector — select variant → layer → head → heatmap
  3. Architecture Explorer — parameter breakdown, KV cache estimates
"""

try:
    import streamlit as st
except ImportError:
    raise ImportError("Install streamlit: pip install streamlit")

import json
import torch
import numpy as np
from pathlib import Path

from model.config import ModelConfig
from model.transformer import Transformer
from model.utils import count_parameters
from eval.compare import estimate_kv_cache_per_token, RunMetrics
from viz.attention_maps import AttentionExtractor


# ── Page Config ──────────────────────────────────────────
st.set_page_config(page_title="MiniLM-Bench", page_icon="🔬", layout="wide")
st.title("🔬 MiniLM-Bench: Attention Architecture Comparison")

tab1, tab2, tab3 = st.tabs(["📊 Training Comparison", "🔍 Attention Inspector", "🏗️ Architecture Explorer"])


# ── Tab 1: Training Comparison ───────────────────────────
with tab1:
    st.header("Training Comparison")

    results_path = Path("results/comparison_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "val_ppl" in df.columns:
                st.bar_chart(df.set_index("variant")["val_ppl"])
                st.caption("Validation Perplexity (lower is better)")
        with col2:
            if "tokens_per_sec" in df.columns:
                st.bar_chart(df.set_index("variant")["tokens_per_sec"])
                st.caption("Throughput — Tokens/sec (higher is better)")
    else:
        st.info("No results found. Run training experiments first, then `eval/compare.py`.")
        st.code("make train CONFIG=mha.yaml\nmake train CONFIG=diff.yaml\n# ... then run comparison")


# ── Tab 2: Attention Inspector ───────────────────────────
with tab2:
    st.header("Attention Pattern Inspector")

    col1, col2 = st.columns([1, 3])
    with col1:
        variant = st.selectbox("Variant", ["mha", "gqa", "mqa", "swa", "diff", "mla", "moh", "nsa"])
        n_layers = st.number_input("Layers", 2, 24, 4)
        layer_idx = st.slider("Layer", 0, n_layers - 1, 0)
        n_heads = st.number_input("Heads", 2, 16, 4)
        head_idx = st.slider("Head", 0, n_heads - 1, 0)
        text_input = st.text_area("Input text", "The quick brown fox jumps over the lazy dog")

    with col2:
        if st.button("Generate Attention Map"):
            with st.spinner("Computing attention patterns..."):
                # Build a small model for visualization
                cfg_kwargs = dict(
                    vocab_size=256, d_model=128, n_layers=n_layers, n_heads=n_heads,
                    n_kv_heads=n_heads, d_ff_mult=2, max_seq_len=128, attention_type=variant,
                )
                if variant == "mqa":
                    cfg_kwargs["n_kv_heads"] = 1
                elif variant == "mla":
                    cfg_kwargs["d_latent"] = 32
                    cfg_kwargs["rope_head_dim"] = 32

                config = ModelConfig(**cfg_kwargs)
                model = Transformer(config).eval()

                # Tokenize (simple char-level for demo)
                tokens = [ord(c) % 256 for c in text_input[:64]]
                input_ids = torch.tensor([tokens])

                # Extract attention
                extractor = AttentionExtractor(model)
                with torch.no_grad():
                    model(input_ids)
                patterns = extractor.get_patterns()
                extractor.remove_hooks()

                if layer_idx in patterns:
                    attn = patterns[layer_idx][0, head_idx].numpy()  # (T, T)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(attn, cmap="viridis", aspect="auto")
                    ax.set_xlabel("Key position")
                    ax.set_ylabel("Query position")
                    ax.set_title(f"{variant.upper()} — Layer {layer_idx}, Head {head_idx}")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("No attention pattern captured for this layer.")


# ── Tab 3: Architecture Explorer ─────────────────────────
with tab3:
    st.header("Architecture Explorer")

    variant = st.selectbox("Select variant", ["mha", "gqa", "mqa", "swa", "diff", "mla", "moh", "nsa"], key="arch_variant")
    d_model = st.select_slider("d_model", [256, 512, 768, 1024, 2048], value=768)
    n_layers = st.select_slider("n_layers", [4, 6, 8, 12, 16, 24], value=12)

    cfg_kwargs = dict(
        vocab_size=50257, d_model=d_model, n_layers=n_layers, n_heads=d_model // 64,
        n_kv_heads=d_model // 64, d_ff_mult=4, max_seq_len=1024, attention_type=variant,
    )
    if variant == "mqa":
        cfg_kwargs["n_kv_heads"] = 1
    elif variant == "gqa":
        cfg_kwargs["n_kv_heads"] = max(1, d_model // 64 // 4)

    config = ModelConfig(**cfg_kwargs)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{config.num_params() / 1e6:.1f}M")
    with col2:
        kv_bytes = estimate_kv_cache_per_token(config)
        st.metric("KV Cache / Token / Layer", f"{kv_bytes} bytes")
    with col3:
        total_kv = kv_bytes * n_layers * 1024  # For 1024 tokens
        st.metric("KV Cache (1024 tokens)", f"{total_kv / 1e6:.1f} MB")

    st.subheader("Config Details")
    st.json({
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "d_ff": config.d_ff,
        "attention_type": config.attention_type,
    })

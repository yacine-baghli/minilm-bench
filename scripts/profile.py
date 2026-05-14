"""
Throughput profiling entry point.

Benchmarks forward/backward pass for each attention variant,
measuring tokens/sec, step time, and peak memory.
"""

import argparse
import time
import torch

from model.config import ModelConfig
from model.transformer import Transformer
from model.utils import print_model_summary, estimate_flops_per_step


def _get_variant_overrides(variant_name: str, n_heads: int) -> dict:
    """Get variant-specific config overrides with proper head counts."""
    base = {"attention_type": variant_name, "n_kv_heads": n_heads}
    if variant_name == "gqa":
        base["n_kv_heads"] = max(1, n_heads // 4)
    elif variant_name == "mqa":
        base["n_kv_heads"] = 1
    elif variant_name == "mla":
        base["d_latent"] = 192
        base["rope_head_dim"] = 64
    elif variant_name == "moh":
        base["moh_top_k"] = max(2, n_heads // 2)
    elif variant_name == "nsa":
        base["nsa_block_size"] = 16
        base["nsa_top_k_blocks"] = 8
    return base


def profile_variant(
    variant_name: str,
    d_model: int = 768,
    n_layers: int = 12,
    batch_size: int = 4,
    seq_len: int = 1024,
    n_warmup: int = 3,
    n_steps: int = 10,
    device: str = "cpu",
) -> dict:
    """Benchmark a single attention variant."""
    n_heads = d_model // 64
    overrides = _get_variant_overrides(variant_name, n_heads)
    config = ModelConfig(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len, **overrides,
    )

    model = Transformer(config).to(device)
    model.train()

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        _, loss = model(x, y)
        loss.backward()
        model.zero_grad()

    # Benchmark
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _, loss = model(x, y)
        loss.backward()
        model.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / avg_time

    peak_mem = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "variant": variant_name,
        "avg_step_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_mem,
        "total_params_M": total_params / 1e6,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile attention variants")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    ALL_VARIANTS = ["mha", "gqa", "mqa", "swa", "diff", "mla", "moh", "nsa"]
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    args = parser.parse_args()

    print(f"Profiling on {args.device} | d={args.d_model} L={args.n_layers} B={args.batch_size} T={args.seq_len}")
    print("=" * 80)

    results = []
    for name in args.variants:
        print(f"\n--- {name.upper()} ---")
        r = profile_variant(
            name, d_model=args.d_model, n_layers=args.n_layers,
            batch_size=args.batch_size, seq_len=args.seq_len, device=args.device,
        )
        results.append(r)
        print(f"  Step: {r['avg_step_ms']:.1f}ms | {r['tokens_per_sec']:,.0f} tok/s | "
              f"Mem: {r['peak_memory_mb']:.0f}MB | Params: {r['total_params_M']:.1f}M")

    print("\n" + "=" * 80)
    print(f"{'Variant':>8} | {'ms/step':>8} | {'tok/s':>10} | {'Memory':>8} | {'Params':>8}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["tokens_per_sec"], reverse=True):
        print(f"{r['variant']:>8} | {r['avg_step_ms']:>7.1f}ms | {r['tokens_per_sec']:>10,.0f} | "
              f"{r['peak_memory_mb']:>6.0f}MB | {r['total_params_M']:>6.1f}M")

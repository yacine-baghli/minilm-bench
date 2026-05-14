# Design Decisions

This document records the rationale behind every significant architectural, algorithmic, and engineering decision in MiniLM-Bench. Each section states the decision, the alternatives considered, and the reasoning — structured for interview discussion.

---

## 1. Model Architecture

### 1.1 Normalization: RMSNorm

**Decision**: Use Root Mean Square normalization instead of LayerNorm.

**Reasoning**: LayerNorm computes both mean and variance. RMSNorm drops the mean-centering step, computing only the RMS:

```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```

This is ~10% cheaper per call and eliminates a reduction operation that can be a bottleneck in distributed settings. Empirically, the mean-centering provides no measurable quality benefit for Transformer pre-training (Zhang & Sennrich, 2019). Every modern LLM (Llama, Gemma, Mistral, DeepSeek, Qwen) uses RMSNorm.

### 1.2 Feed-Forward: SwiGLU

**Decision**: Use Swish-Gated Linear Units (3 linear layers) instead of standard 2-layer FFN with ReLU/GELU.

**Reasoning**: SwiGLU introduces a learnable gating mechanism:

```
SwiGLU(x) = (σ(W_gate · x) ⊙ (W_up · x)) · W_down
```

Shazeer (2020) demonstrated ~1% perplexity improvement at equivalent parameter count. The gating creates implicit activation sparsity — the network learns to zero out irrelevant feature dimensions. The tradeoff is 50% more parameters per FFN block (3 projections instead of 2), which we compensate by using a 2/3 scaling factor on `d_ff`.

### 1.3 Positional Encoding: RoPE via Complex Rotation

**Decision**: Rotary Position Embeddings using complex number multiplication.

**Reasoning**: RoPE encodes relative positions by rotating Q and K vectors in 2D subspaces. The rotation angle depends on position, so the dot product `q · k` naturally captures relative distance. Our implementation uses `torch.view_as_complex` for clean, vectorized rotation:

```python
q_complex = torch.view_as_complex(q.reshape(..., -1, 2))
q_rotated = q_complex * freqs_cis  # Complex multiplication = 2D rotation
```

**Key subtlety**: For MLA, standard RoPE is incompatible with low-rank KV compression. Rotation applied to K before down-projection gets destroyed during compression. We implement **decoupled RoPE** — separate positional dimensions that bypass the latent bottleneck. The `apply_rope` function slices frequency tensors to handle variable head dimensions across variants.

### 1.4 Block Structure: Pre-Norm

**Decision**: Apply normalization before attention/FFN, not after.

**Reasoning**: Pre-norm creates a cleaner gradient highway through the residual stream. In post-norm, gradients must flow through the normalization layer at every block, which can cause vanishing gradients in deep networks. Pre-norm eliminates this issue and reduces sensitivity to learning rate warmup duration.

---

## 2. Attention Mechanisms

### 2.1 Standard Variants

The four standard variants span the established KV-sharing design space:

| Variant | Innovation | Why Included |
|---------|-----------|-------------|
| **MHA** | Full independent heads | Baseline with maximum expressiveness |
| **GQA** | Grouped KV sharing | Llama-2's practical sweet spot — near-MHA quality at reduced KV cost |
| **MQA** | Single KV for all heads | Extreme compression bound — tests how far sharing can go |
| **SWA** | Bounded attention span | Orthogonal axis: restricts *where* attention looks, not *how many* KV heads |

### 2.2 Differential Attention (DiffAttn)

**Paper**: Ye et al., *Differential Transformer*, Microsoft Research (2024).

**Why this is interesting**: DiffAttn is the only mechanism that produces **negative attention weights**, breaking the fundamental softmax constraint. By computing two attention distributions and subtracting:

```
Attn = λ · (softmax(Q₁K₁ᵀ) − softmax(Q₂K₂ᵀ)) · V
```

the model can actively *suppress* irrelevant tokens rather than just assigning them small positive weights. This is conceptually closer to how biological attention works — active inhibition, not just passive de-weighting.

**Implementation detail**: The per-head λ uses depth-dependent initialization: `λ = 0.8 - 0.6·exp(-0.3·layer_idx)`. Early layers apply stronger differential (more noise cancellation), while later layers converge toward standard attention behavior. A sublayer RMSNorm after the differential computation stabilizes training.

### 2.3 Multi-head Latent Attention (MLA)

**Paper**: DeepSeek-AI, *DeepSeek-V2* (2024). Deployed in DeepSeek-V3 and R1.

**Why this is interesting**: MLA directly addresses the **KV cache memory bottleneck** — the dominant cost during inference at scale. The key insight is that K and V vectors are highly redundant across positions and can be compressed into a much lower-dimensional latent:

```
x → W_DKV → c_KV (d_latent dims)    ← only this is cached
c_KV → W_UK → K (reconstructed)
c_KV → W_UV → V (reconstructed)
```

With `d_latent = d_model/4`, this achieves ~4× KV cache reduction. At d_model/8, the reduction is 8×.

**Critical design choice — Decoupled RoPE**: RoPE applies rotation to K before the dot product with Q. But if we rotate K, then compress, then decompress — the rotation is lost. The solution is *decoupled* positional encoding: separate projection dimensions dedicated to position that bypass the latent bottleneck entirely. Our implementation concatenates content dimensions (from the latent path) with position dimensions (from a separate RoPE path) for both Q and K.

### 2.4 Mixture-of-Head Attention (MoH)

**Paper**: SkyworkAI, *Mixture-of-Head* (2025).

**Why this is interesting**: MoH applies the MoE insight to attention heads themselves. A small router network predicts which heads are most relevant for each token, and only the top-K are activated:

```
Router(x) → top-K head indices + gating weights
Output = Σᵢ gᵢ · Headᵢ(x)    (only for selected i)
```

**Critical design choice — Load balancing loss**: Without regularization, the router collapses: all tokens are routed to the same 2-3 "popular" heads, and the remaining heads receive no gradients and die. We add the standard MoE auxiliary loss:

```python
aux_loss = N_heads × Σᵢ (frac_routed_to_i × mean_router_prob_i)
```

This penalizes imbalanced routing, encouraging all heads to receive roughly equal traffic.

**Design choice — Shared heads**: We keep 1-2 heads permanently active (not routed) to capture common patterns that every token needs (e.g., local syntactic context). This stabilizes early training before the router has learned meaningful specialization.

### 2.5 Native Sparse Attention (NSA)

**Paper**: DeepSeek-AI, *NSA: Native Sparse Attention* (2025). Deployed in DeepSeek-V3.

**Why this is interesting**: NSA is the most sophisticated attention mechanism deployed at scale. It decomposes attention into three complementary branches:

```
① Compressed: block-pool K,V (16 tokens → 1) → full attention over coarse sequence
② Selected:   use ① scores as importance → gather fine-grained top-K blocks
③ Window:     standard sliding window over recent tokens

Output = σ(g₁)·Branch① + σ(g₂)·Branch② + σ(g₃)·Branch③
```

The hierarchy is the key: Branch ① provides cheap global awareness (O(n/b) cost). Branch ② provides high-fidelity attention to the *most important* distant tokens (O(k·b) cost). Branch ③ captures local context (O(w) cost). Total: O(n·(n/b + k·b + w)) vs. O(n²) for full attention.

**Design choice — Learned gates**: The per-head sigmoid gates (initialized at 0 → σ(0) = 0.5) allow each head to specialize: some heads may learn to rely heavily on global context (high g₁), others on local patterns (high g₃). This is natively trainable — no post-hoc pruning or pattern engineering.

---

## 3. Training Infrastructure

### 3.1 Atomic Checkpointing

Checkpoints are written via `tempfile.mkstemp()` → `os.replace()`. This guarantees that the checkpoint file is never in a partially-written state. If the process crashes mid-save, the temp file is simply orphaned and the previous valid checkpoint remains intact. On cloud instances with preemption, the `SIGTERM` handler triggers an immediate save before exit.

The checkpoint stores: model state dict, optimizer state dict, scheduler state dict, training step, validation loss, and `torch.rng_state` (+ CUDA RNG if available) for exact reproducibility on resume.

### 3.2 Mixed-Precision Training

We use `torch.autocast(dtype=bfloat16)` rather than FP16. BF16 has 8 exponent bits (same as FP32) vs. FP16's 5, giving it the same dynamic range while using half the memory. This eliminates the need for loss scaling — a common source of training instability in FP16.

### 3.3 Weight Decay Groups

Weight decay (L2 regularization) is applied only to 2D+ parameters (linear weights). Biases, normalization weights, and embedding vectors receive zero weight decay. Regularizing these 1D parameters provides no benefit and can destabilize training by pulling learned scales toward zero.

### 3.4 Gradient Accumulation

To achieve large effective batch sizes on limited hardware, we accumulate gradients over multiple micro-batches before taking an optimizer step. The loss is divided by `grad_accum_steps` at each micro-step to maintain correct gradient magnitude. This is mathematically equivalent to a larger batch (up to synchronization noise in batch normalization, which we don't use).

---

## 4. Data Pipeline

### 4.1 Memory-Mapped Shards

Tokenized data is stored as `uint16` numpy arrays accessed via `np.memmap`. This provides:
- **O(1) random access**: any token position is immediately available without sequential reading
- **OS-level caching**: the kernel pages in frequently-accessed regions automatically
- **Memory efficiency**: `uint16` is sufficient for GPT-2's 50,257-token vocabulary (max value 65,535) and uses 2× less memory than `int32`

### 4.2 FineWeb-Edu

We train on a 10B-token sample of FineWeb-Edu rather than unfiltered web data. FineWeb-Edu applies quality filtering for educational content, which empirically produces better downstream task performance per training token. This is especially important at our scale (50M-200M parameters), where data quality matters more than at frontier scale.

---

## 5. Software Engineering

### 5.1 Factory Pattern for Attention

The `ATTENTION_REGISTRY` maps string names to classes:

```python
ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "diff": DifferentialAttention,
    # ...
}
```

This enables: (1) attention swapping via a single config field, (2) programmatic iteration over all variants for benchmarking, (3) zero modification to training/eval code when adding new variants. Adding a ninth variant requires exactly two changes: implement the class, add one registry line.

### 5.2 Test Design

Each of our 42 tests verifies a specific, falsifiable property:

- **Shape invariants**: attention output is always `(B, T, d_model)` regardless of variant
- **Gradient flow**: every learnable parameter receives non-zero gradients (catches dead parameters)
- **Mathematical properties**: GQA with `n_kv_heads == n_heads` is numerically identical to MHA
- **Variant-specific contracts**: MoH produces aux loss during training but not eval; DiffAttn λ increases with depth; NSA gates initialize at sigmoid(0) = 0.5
- **Pipeline integration**: 5-step training loop succeeds for each variant (catches shape mismatches, device errors, dtype issues)
- **Persistence**: checkpoint save → load produces bit-identical parameters

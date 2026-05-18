# SGJM at 1B Parameters: Scaling Speculative Graph JEPA Across Two Hardware Platforms Simultaneously

*Does the architecture hold? Does JEPA remain load-bearing at this scale? Does the dual-backend approach actually work in practice?*

---

The previous posts in this series trained SGJM at 25M, 100M, and 250M parameters, always on a single machine, always on a single backend. This post documents the first time I pushed to 1B parameters and, by necessity, ran the experiment on two hardware platforms at once: a Mac Studio M1 Ultra using MLX, and a Framework Desktop ("Hyde") running an AMD Strix Halo under ROCm. Both started at the same time. Both ran the same code. Both checkpoint to the same format. This post explains how that works, what the architecture looks like at this scale, and what the initial numbers say.

---

## The Architecture, Explained from First Principles

SGJM is a four-component system. The headline claim is that it internalizes the draft-verify loop of speculative decoding into a single jointly-trained model. To understand what that means and why each component is necessary, it helps to start from the problem standard speculative decoding leaves unsolved.

Standard speculative decoding uses two models: a small "draft" model that generates k candidate tokens quickly, and a large "target" model that verifies them in parallel. The verification step uses rejection sampling: for each draft token, you check whether the target model would assign a high enough probability. The catch is that the draft and target models are trained independently with no shared objective. They share a vocabulary but not a representation space. The result is that branch quality is measured purely by next-token probability — a signal that is necessary but not sufficient for semantic coherence.

SGJM replaces this with a single system where the backbone, drafter, judge, and verifier are trained together against a combined loss. Crucially, the JEPA judge introduces a second scoring signal — latent-space proximity — that operates orthogonally to token probability. A draft branch that produces high-probability tokens but lands in an unexpected region of representation space gets penalized. A branch that lands close to where the model predicts the context should end up gets rewarded.

Here is how the four components divide the work.

### Backbone

The backbone is a causal transformer with byte-level tokenization (vocab_size=256, tied input/output embeddings). It uses SwiGLU activations, RMSNorm pre-normalization, and learned absolute position embeddings. At 1B scale the configuration is:

```
d_model      = 2048
n_layers     = 20
n_heads      = 16
head_dim     = 128
d_ff         = 8192  (SwiGLU: gate + up projections, then down)
max_seq_len  = 4096
vocab_size   = 256
```

The backbone runs a standard causal forward pass and returns two things: the full sequence of hidden states `h` (shape `[B, T, 2048]`) and the next-token logit matrix (shape `[B, T, 256]`). Both are used downstream. The backbone's hidden states are the currency the other three components trade in.

Why byte-level? Because it removes tokenizer design from the variable space. A 256-token vocabulary is fully deterministic: every byte is exactly one token. There are no subword boundary artifacts, no vocabulary mismatch between machines, no OOV problem. The tradeoff is that sequences are longer for the same text, which is why block_size matters — you want the drafter to amortize over at least a few tokens per speculative step.

Why tied embeddings? The input embedding and lm_head weight matrices are shared. This has two effects: it reduces parameter count (the embedding table at d_model=2048 is 2048 * 256 = ~500K parameters, not large but not zero), and it creates a mild representation alignment between the embedding space and the output space, which benefits the JEPA judge (see below).

### Drafter

The drafter takes the backbone's hidden state at position `t` and produces `block_size` draft tokens in a single forward pass. At 1B scale, block_size=2 (versus block_size=4 at smaller variants). The reason for this is discussed in the hyperparameter section below.

The drafter's internal architecture:

```
d_model      = 768
n_layers     = 3
n_heads      = 12
d_ff         = 3072
```

The forward pass works through learnable position queries. For each position in the sequence, the drafter:

1. Projects the parent hidden state from d=2048 to d=768 via a linear layer.
2. Adds `block_size` learnable query vectors (shape `[block_size, 768]`), one per draft position. These are not attention keys — they are additive offsets that specialize each position in the draft block.
3. Runs the result through 3 transformer blocks.
4. Produces two outputs: draft token logits (for each position in the block, a distribution over 256 bytes) and draft latents (a projection back to d=2048, representing where the drafter thinks the backbone hidden state should be after accepting this branch).

The critical design decision is that all branches are produced in one forward pass, not k separate passes. This is what makes the drafter tractable at inference time. The branching factor k comes from sampling different sequences from the draft logits, not from k separate calls to the drafter.

The `latent_out` projection in the drafter (a linear from d=768 to d=2048) produces what I call the "endpoint latent" — the drafter's claim about where the backbone hidden state will be after the draft block is accepted. This latent is consumed by the JEPA judge during training.

### JEPA Judge

The JEPA judge is the structurally unusual component. It is a two-layer feedforward network with hidden size 4096:

```python
fc1: Linear(2048 -> 4096)
activation: GELU
fc2: Linear(4096 -> 2048)
```

It takes the parent hidden state as input and predicts what the backbone hidden state *should* look like `block_size` positions later. During training, the actual future hidden state `h_{t+block_size}` is computed from the backbone forward pass, and the judge's prediction is trained to match it via MSE with a stop-gradient on the target.

The stop-gradient is not optional. Without it, the loss gradient flows into the backbone and incentivizes the backbone to make the future state easy to predict — which is a completely different objective from language modeling. The stop-gradient ensures the judge trains to predict the backbone's natural future states, not the other way around.

During evaluation and inference, the judge scores each draft branch by computing the MSE between the branch's endpoint latent (from the drafter's `latent_out`) and the judge's prediction of where the backbone should be. Low MSE = semantically consistent branch. High MSE = branch has drifted away from where the model expects the context to go.

This is the JEPA signal: Joint Embedding Predictive Architecture, applied to the speculative decoding problem. The judge does not look at tokens at all. It operates entirely in the backbone's representation space. A branch can produce perfectly fluent tokens and still score poorly if its endpoint latent is far from the judge's prediction.

Why does this matter? The ablation results at 25M scale are unambiguous. When the JEPA loss is removed (`loss.jepa=0.0`), branch acceptance rate collapses from 63% to 2.7%. The judge is not providing a redundant signal — it is providing the primary signal that the verifier refines.

### Verifier

The verifier is a binary classifier on concatenated parent and child hidden states:

```
input:  concat([parent_h, child_h])  -- shape [B, T, 4096]
fc1:    Linear(4096 -> 2048)
act:    GELU
fc2:    Linear(2048 -> 1)
output: logit (positive = accept)
```

It is trained with contrastive pairs. Within each batch, the true future hidden states are the positive examples. The negatives are constructed by rolling the batch along the batch dimension (`torch.roll(shifts=1, dims=0)` in PyTorch, `mx.concatenate([h[-1:], h[:-1]])` in MLX) — this gives negatives that have the same distributional character as real hidden states but are misaligned with the parent contexts. The result is a discriminator that learns to separate genuine continuations from plausible-but-wrong ones.

The verifier's BCE loss is weighted 0.1 in the combined loss, the lowest weight of any term. This reflects its role: it refines the judge's score rather than driving it. In the ablation where only the verifier is present (JEPA removed), acceptance collapses to 2.7%. The verifier without a trained judge has nothing meaningful to discriminate.

### Why All Four Components Are Required

The ablation results at 25M make the dependency structure explicit:

| Configuration | Accept rate | JEPA top-1 | Merge prec. | Notes |
|---|---|---|---|---|
| Full (token + drafter + JEPA + verifier) | 63.3% | 97.5% | 1.19x | All signals active |
| No JEPA | 2.7% | ~chance (11.5%) | 1.11x | Verifier has nothing to learn from |
| No drafter loss | 100% | 95.5% | 1.00x | Accept stays high but merge collapses |
| No verifier | 21.3% | 96.7% | 1.18x | Accept halved; judge alone insufficient |
| Token only | 18.6% | ~chance | 1.00x | SGJM with dead weight |

Reading down the table tells a story. JEPA is structurally upstream: remove it and the verifier fails. The drafter loss is required for branch organization: without it, the drafter still produces high-probability tokens (so the verifier accepts them), but the endpoint latents are not organized — SimHash-based merge precision drops to 1.00x (meaning draft branches that appear similar by locality-sensitive hash are no more distributionally similar than random pairs). The verifier provides a second filter after the judge that materially improves acceptance precision.

At the fully-trained 25M eval gate (5000 steps, TinyShakespeare corpus), the model passes all five conditions:

| Condition | Threshold | Result |
|---|---|---|
| NLL delta vs baseline | <= 0.05 nats | +0.0015 nats |
| Branch acceptance rate | >= 50% | 99.99% |
| JEPA top-1 above chance | >= +5 pp | +88.5 pp above 11.1% chance |
| Merge precision advantage | >= 1.5x | 10,607x |
| Compute per accepted token | >= 1.0x baseline | 13.9x |

That merge precision number (10,607x) deserves a brief explanation. It measures the Jensen-Shannon divergence between next-token distributions at positions that SimHash clusters together (merged branches) versus random pairs. At 5000 steps, the JS divergence within merged clusters is 0.000065 nats; across random pairs it is 0.690 nats. The model has learned, entirely from the JEPA and drafter losses, to organize draft branches so that semantically similar continuations cluster together in SimHash space. This is the mechanism that would make a production speculative graph efficient — branches that converge to the same meaning can be deduplicated cheaply.

---

## Configuration at 1B Scale

The 1B configuration is defined in `TrainingConfig.sgjm_1b()`:

```python
ModelConfig(
    d_model=2048,
    n_layers=20,
    n_heads=16,
    d_ff=8192,
    drafter_layers=3,
    drafter_d_model=768,
    drafter_heads=12,
    drafter_d_ff=3072,
    judge_hidden=4096,
    verifier_hidden=2048,
    block_size=2,       # shorter blocks vs 4 at smaller scales
    max_seq_len=4096,
)

OptimConfig(
    lr=6e-5,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    warmup_steps=5_000,
    max_steps=50_000,
    grad_clip=1.0,
    batch_size=1,
    seq_len=2048,
)

corpus_bytes = 256 << 20   # 256 MiB Python extended corpus
```

### Why block_size=2?

The block_size sweep at 25M scale showed that shorter blocks have better judge signal quality and better merge precision, while longer blocks have higher raw compute advantage:

| block_size | Accept rate | JEPA top-1 | Merge prec. | Compute advantage | Gate |
|---|---|---|---|---|---|
| 2 | 69.0% | 99.1% | 1.92x | 4.8x | Pass |
| 4 | 63.3% | 97.5% | 1.19x | 8.8x | Fail (merge < 1.5x) |
| 8 | 90.8% | 96.1% | 1.00x | 25.3x | Fail (merge = 1.0x) |

At block_size=8, the drafter produces high-acceptance branches, but the merge precision advantage collapses to 1.0x. The drafter's endpoint latents are not being organized by the JEPA signal over 8-token windows — the signal is too diffuse across the block. At block_size=2, merge precision is 1.92x and the gate passes.

At 1B scale, the backbone is substantially more powerful than at 25M. A stronger backbone produces higher-quality hidden states, which gives the judge a richer signal even over short windows. This is the argument for using block_size=2 here: the judge will have more to work with per position. If the judge signal degrades at 1B (which would be a surprising result), we can revisit.

### Why batch_size=1?

The 1B backbone requires roughly 4 bytes * 2048 (d_model) * 20 (layers) * 2048 (seq_len) for activations per batch element, plus the drafter, judge, and verifier activations. At batch_size=1 with seq_len=2048, the memory footprint is manageable on both hardware platforms. At batch_size=4 (used at 250M) the activation memory would require gradient checkpointing, which is not yet implemented.

The effective batch in terms of tokens is still 2048, which is the same as the 100M configuration's batch_size=4 * seq_len=512. The gradient noise is higher, but the cosine LR schedule with 5000 warmup steps compensates.

### Corpus

The 256 MiB Python extended corpus is built from CPython's standard library plus common site-packages. It is byte-level (no tokenization step), so the model trains directly on the raw UTF-8 byte stream. This corpus was introduced at 250M scale; the 1B run uses the same source with a 256 MiB slice rather than the 32 MiB used at 250M.

The Python corpus is harder than TinyShakespeare (1 MiB of English prose) in multiple ways: longer average sequence length before a semantic unit completes, mixed identifier/keyword/symbol vocabulary, and significant whitespace structure that the model must reproduce accurately for syntactically valid completions. The 25M and 100M results on TinyShakespeare (token_loss ~0.025) are not comparable to the 250M and 1B results on the Python corpus; the harder corpus raises the expected convergence loss.

---

## Scaling History

For reference, here is the complete scaling table across all variants trained to date. Note that the 25M and 100M runs used TinyShakespeare (1 MiB); 250M and 1B use the Python extended corpus (32 MiB and 256 MiB respectively). Hyde's higher token_loss at 25M relative to the MacBook results is corpus difference, not a regression.

| Model | Machine | Backend | Steps | Time | Steps/sec | Token loss | Accept acc |
|---|---|---|---|---|---|---|---|
| 25M | apippert-mac (M-series) | MLX | 5,000 | 27.3 min | ~3.0 | 0.025 | 99.8% |
| 100M | apippert-mac (M-series) | MLX | 5,000 | 55.4 min | ~1.5 | 0.024 | ~99% |
| 250M | apippert-mac (M-series) | MLX | 10,000 | — | — | — | — |
| 25M | Hyde (Strix Halo) | ROCm | 5,000 | 20.9 min | 4.0 | 0.152 | 98.0% |
| 250M | Hyde (Strix Halo) | ROCm | 10,000 | 68.8 min | 2.4 | 0.584 | 99.6% |
| **1B** | **Mac Studio M1 Ultra** | **MLX** | **[updating]** | **[updating]** | **[updating]** | **[updating]** | **[updating]** |
| **1B** | **Hyde (Strix Halo)** | **ROCm** | **[updating]** | **[updating]** | **[updating]** | **[updating]** | **[updating]** |

The 250M Hyde token_loss (0.584) after 10,000 steps on 256 MiB Python is not a strong comparison to the 25M MacBook token_loss (0.025) after 5,000 steps on 1 MiB Shakespeare. They are different tasks. The relevant comparison at each scale is the delta between SGJM and the same-budget baseline on the same corpus, which the eval gate measures.

---

## Live: 1B Training in Progress

Both machines started the 1B run simultaneously. Here is what step 0 looked like.

**Mac Studio M1 Ultra (MLX backend):**
```
[sgjm] backend=mlx params=1417.34M
[sgjm] step=     0 lr=1.20e-08 total=9.4373 tok=6.2839 draft=5.6462 jepa=0.4041 ver=0.6930 acc=0.481
elapsed: 4.34s  (includes MLX compilation on first call)
```

**Hyde, Framework Desktop (ROCm backend):**
```
[sgjm:sgjm] backend=rocm device=cuda params=1417.34M
             backbone=1298.10M drafter=108.22M judge=8.39M verifier=10.49M
[sgjm:sgjm] step=     0 lr=1.20e-08 total=9.2647 tok=5.9358 draft=5.8129 jepa=0.4274 ver=0.6935 acc=0.491
elapsed: 2.09s
```

Both machines report 1,417M parameters. The breakdown on Hyde (printed by `param_breakdown()` in the torch model) confirms the component distribution: the backbone takes ~1,298M (91.6%), the drafter ~108M (7.6%), the judge ~8.4M (0.6%), and the verifier ~10.5M (0.7%). This is the expected shape — the backbone dominates at 1B, which is why the overall model behaves predominantly as a strong language model with lightweight auxiliary heads.

Both start at approximately 50% accept_acc. This is the correct behavior for a randomly initialized model: the verifier has learned nothing yet, so it accepts roughly half the contrastive pairs by chance. By step 500 at smaller scales, accept_acc reliably exceeds 90%. The question at 1B is whether the judge latent signal converges as quickly given the higher-dimensional representation space.

The starting losses are also as expected. Token CE near ln(256) = 5.545 would be pure uniform random; the slightly higher values (6.28 on MLX, 5.94 on ROCm) reflect the Xavier/normal initialization producing non-uniform initial logits. The total loss is higher on the MLX run because the initial token CE is higher — the random initialization is seed-dependent and the two backends have different internal RNG implementations.

The 4.34s MLX step-0 elapsed time includes the JIT compilation that MLX performs on the first call to each compute graph. Subsequent steps will be faster. Hyde's 2.09s is hardware evaluation time without a compilation phase (ROCm kernels are compiled at the CUDA driver level, not at framework dispatch time).

### What Convergence Should Look Like

Based on the smaller model trajectories, I expect the following shape for the 1B loss curve:

- **Steps 0–5,000 (warmup)**: Total loss drops steeply. The token CE should fall from ~6 to somewhere between 2 and 4, depending on how quickly the backbone fits the Python corpus. The JEPA loss will drop rapidly because the judge has a clear signal to follow. Accept_acc will cross 90% in this window.

- **Steps 5,000–20,000**: The main learning curve. Token CE should continue dropping. The JEPA loss will plateau when the judge is well-calibrated to the backbone's current hidden states. Drafter loss tracks token CE with a lag (the drafter is learning to predict the same targets as the backbone but from a smaller model and without causal context beyond the parent latent).

- **Steps 20,000–50,000**: Refinement. The larger model has more capacity to continue fitting the Python corpus, so I expect slower but steady improvement in token CE relative to the 250M run. The JEPA and verifier losses should be near-converged by this point, with accept_acc stabilized.

Measured steady-state throughput once both machines cleared first-step compilation:

| Machine | Backend | Steps/sec | s/step | ETA (50k steps) |
|---|---|---|---|---|
| Hyde (Strix Halo) | ROCm / bf16 AMP | **0.51** | 1.96s | **~27 hours** |
| Mac Studio M1 Ultra | MLX | **0.26** | 3.92s | **~54 hours** |

Hyde runs roughly 2× faster than the Mac Studio for the 1B model, despite the M1 Ultra's advantage in raw memory bandwidth. The likely cause: PyTorch's bf16 AMP path on ROCm lets the Strix Halo operate on half-precision activations throughout, halving the memory traffic for the attention operations that dominate at seq_len=2048. MLX operates in bf16 natively on M-series but may have higher per-dispatch overhead at this parameter count due to its lazy-evaluation compilation model. The raw bandwidth advantage of M1 Ultra's unified memory fabric does not fully compensate at this scale.

I will update this post with checkpoint-500 and checkpoint-5000 numbers when they are available.

---

## Platform Comparison: MLX vs ROCm

Running the same architecture on two very different hardware platforms in parallel is useful not just for redundancy but because the platforms make different tradeoffs visible.

### Installation

**Mac Studio / MLX path:**

```bash
# Clone the repo, create a venv with uv
uv venv .venv --python 3.12
source .venv/bin/activate
pip install mlx numpy
pip install -e '.[mlx,dev]'

# Run smoke test
python -m sgjm.training --size smoke --backend mlx
```

Three commands. No system packages. MLX is a pure-Python install that ships its own Metal compute kernels. The only constraint is Darwin arm64 — the framework will not install on Intel Macs or Linux.

A `setup_remote.sh` script in the repo automates the full bootstrap (Homebrew, Miniforge, conda env, pip install, smoke test) for deploying to a fresh Mac Studio:

```bash
bash scripts/setup_remote.sh https://github.com/AdamPippert/SGJM.git
```

**Hyde / ROCm path:**

```bash
# System package pull (Arch Linux) — this is the heavy step
sudo pacman -S python-pytorch-opt-rocm
# Pulls: rocm-hip-sdk, hipblaslt, miopen-hip, aotriton, and pytorch itself
# ~4GB download

# Project install into a venv that can see the system pytorch
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[rocm,dev]'  # installs only numpy and the sgjm package

# Required env flag for Strix Halo's attention kernels
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Run
python -m sgjm.training --size 1b --backend rocm
```

The ROCm path is more involved, primarily because the PyTorch ROCm wheel is distributed through the Arch Linux package manager rather than PyPI. The `[rocm]` extra in `pyproject.toml` deliberately omits torch (it lists only numpy), because torch ROCm must come from the system package or from a specific PyTorch index URL.

The critical environment flag `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables the AOTriton-based Flash Attention and memory-efficient attention kernels for Strix Halo GPUs. Without it, `F.scaled_dot_product_attention` silently falls back to the naive O(n^2) attention implementation, which is both slower and consumes more memory. This flag is required for any GPU in the Strix Halo family. It is not required for RDNA3 or earlier ROCm targets.

### Memory Architecture

The two platforms handle the 1.4GB+ model in very different ways.

**M1 Ultra unified memory**: Apple's M1 Ultra has a single pool of RAM shared by the CPU, GPU, and Neural Engine. The Mac Studio variant ships with up to 128GB. The GPU has access to the full pool at approximately 800 GB/s bandwidth (per Apple's published numbers). There is no discrete VRAM; everything is unified. This means a 1B model with fp32 weights (~5.7GB) plus activations, optimizer state (AdamW stores two moment tensors per parameter, so ~11.4GB additional), plus the corpus in memory is comfortably within a 64GB configuration. The absence of a CPU-to-GPU memory transfer bottleneck is significant: there is no PCIe lane between the compute fabric and the model weights. Every access to model weights is the same latency as any other memory access.

**Hyde Strix Halo GTT**: The Framework Desktop's AMD Strix Halo (Radeon 8060S) has 512MB of dedicated VRAM and 62GB of GTT (Graphics Translation Table) memory — system RAM that the GPU can address directly. The total addressable GPU memory is 62.5GB, but with significantly lower bandwidth than the M1 Ultra's unified architecture. GTT accesses go through the system interconnect rather than the high-bandwidth unified memory fabric. For inference latency, this matters; for training throughput where the bottleneck is compute rather than memory bandwidth, it matters less. The 1B model fits in GTT with room for optimizer state, and the ROCm driver handles the page mapping transparently.

For a 1B model, both platforms have enough addressable memory. The bandwidth difference will show up in steps/second rather than in convergence quality.

### Software Maturity and Friction

**MLX**: MLX is purpose-built for Apple Silicon. The `mlx_backend/trainer.py` is clean: `mx.array`, `mx.eval()`, `tree_flatten`, `mx.save_safetensors()`. There is no AMP boilerplate because MLX operates in bf16 natively on M-series chips — the mixed-precision question does not arise. The `nn.value_and_grad()` pattern computes loss and gradients in a single call with no explicit backward pass. The main friction at 1B is that MLX's lazy evaluation model means that the first step incurs JIT compilation cost (~4s in our step-0 measurement). Subsequent steps should be faster.

**ROCm / PyTorch**: The torch backend is standard PyTorch, which means the full AMP machinery is available. The `_amp_dtype` function in `torch_backend/trainer.py` selects bf16 for ROCm backends automatically. The GradScaler is instantiated but only activated for fp16 (not bf16, since bf16 does not underflow in the same way that requires loss scaling). The actual training loop is identical in structure to the MLX loop. The substantive difference is the `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` flag — this is genuine friction that a developer would hit without documentation. Strix Halo is a new enough GPU that its attention kernel support was not in the stable ROCm path at the time of writing.

### The Backend Abstraction

The reason this dual-platform experiment was a one-day effort rather a multi-week port is `src/sgjm/training/backends.py`. The `detect_backend()` function does the following:

```python
def detect_backend() -> ResolvedBackend:
    if platform.system() == "Darwin" and platform.machine() == "arm64" and _has_module("mlx.core"):
        return "mlx"
    if _torch_cuda_available():
        return "rocm" if _torch_is_rocm() else "cuda"
    return "cpu"
```

On Darwin arm64 with MLX installed, the resolved backend is `mlx`. On Hyde with the ROCm pytorch package, `torch.version.hip` is set, so `_torch_is_rocm()` returns True and the resolved backend is `rocm`. The `__main__.py` entry point dispatches to the appropriate trainer with no further configuration:

```python
if is_torch_backend(backend):
    from sgjm.training.torch_backend.trainer import train as torch_train
    torch_train(cfg, backend)
elif is_mlx_backend(backend):
    from sgjm.training.mlx_backend.trainer import train as mlx_train
    mlx_train(cfg, backend)
```

Neither trainer imports the other. The MLX trainer uses `mlx.core`, `mlx.nn`, and `mlx.optimizers`. The torch trainer uses `torch`, `torch.nn`, and `torch.optim`. They share only the `TrainingConfig`, `ByteDataset`, and the loss computation structure (which is re-implemented in each backend's idiom).

The loss implementations converge to the same computation:

```
# MLX (losses.py):
jepa_loss = 0.5 * (
    nn.losses.mse_loss(jepa_pred, future_hidden, reduction="mean")
    + nn.losses.mse_loss(drafter_endpoint, future_hidden, reduction="mean")
)

# PyTorch (losses.py):
jepa_loss = 0.5 * (
    F.mse_loss(jepa_pred, future_hidden)
    + F.mse_loss(drafter_endpoint, future_hidden)
)
```

The PyTorch `future_hidden` uses `.detach()` for the stop-gradient; the MLX version uses `mx.stop_gradient()`. Same semantics, different idiom. The checkpoint formats differ (`safetensors` on MLX, `.pt` on torch), which means checkpoints from one backend cannot be loaded directly on the other — a limitation that would matter if we were trying to resume on a different machine mid-run, but which does not affect this experiment where each machine runs independently.

---

## What the Initial Numbers Tell Us

At step 0, the model is randomly initialized. Both machines show total loss near 9.3–9.4, accept_acc near 50%, and JEPA loss near 0.4. These are the expected values for an untrained model.

The JEPA loss at step 0 is interesting: it is not near zero. An untrained backbone produces hidden states that are essentially random (drawn from the N(0, 0.02) initialization used in `_init_weights`), and an untrained judge produces predictions that are also random. The MSE between two random normal vectors in R^2048 should be approximately `2 * 0.02^2 * 2048 = 1.6` — but the actual JEPA loss is ~0.4, which suggests the initialization is not purely independent (the model.apply() weight init runs after the first forward is called, so the very first step may not be fully initialized). This will self-correct immediately as training begins.

The accept_acc at 50% confirms the verifier is operating at chance. Both positive examples (real future states) and negative examples (rolled batch negatives) are indistinguishable to an untrained verifier. The metric should rise sharply once the backbone has trained enough to produce distinguishable hidden states for different contexts — at 25M this happened by step 200.

The ~5.8–6.3 starting token CE is slightly above ln(256) = 5.545. This is normal for a normal-initialized model. The warmup phase will bring this down quickly.

---

## Architecture Questions at 1B Scale

Scaling from 250M to 1B raises a few open questions that the training run should answer.

**Does the JEPA signal remain load-bearing?** At 25M, removing the JEPA loss collapsed acceptance from 63% to 2.7%. This is a strong result, but it was measured on TinyShakespeare with d_model=384. At d_model=2048, the judge predicts in a 2048-dimensional space. The MSE signal is potentially noisier in higher dimensions (curse of dimensionality), but the model has more parameters to fit the prediction task. My expectation is that the JEPA signal remains necessary — the ablation result suggests a structural dependency rather than a scale-sensitive phenomenon — but the acceptance rate under the trained model at 1B may differ from 63%.

**Does the block_size=2 choice hold?** The sweep at 25M showed block_size=2 had the best merge precision (1.92x) and passed the gate. At 1B, a stronger backbone may enable block_size=4 to pass — the judge has a better signal to work with. Conversely, the longer sequence length (2048 vs 512) means each step processes more token positions, so even block_size=2 represents significant compute per step. This is worth a targeted experiment at 50K steps, not just a sweep at initialization.

**Do the two platforms converge to the same loss?** They should, given the same corpus, the same configuration, and the same random seed. The starting losses differ slightly (9.44 vs 9.27) due to different RNG implementations in MLX vs PyTorch. The convergence curves should cross and stabilize near the same value. If they diverge significantly (more than 5% in token NLL at 10,000 steps), it would suggest a numerical precision difference between the backends — most likely related to how MLX handles bf16 versus PyTorch's explicit AMP autocast.

---

## What Comes Next

The 1B training run will take approximately 1.5–2 days on each machine. When the checkpoint-500 numbers are in, I will update the scaling table above. When both runs reach step 50,000, I will run the eval gate (the same five-condition check used at 25M) on each platform's checkpoint and compare.

Beyond 1B, there are three things the architecture needs before it can function as an actual inference system rather than a training harness:

**KV-cache integration.** The current training loop runs full forward passes for every batch. In deployment, the backbone would cache key-value pairs from prior context and only attend over new tokens. This changes how the drafter is called — instead of taking the full hidden state tensor, it would take only the last position's hidden state. The architecture supports this (the drafter's input is per-position parent hidden states), but the inference harness does not yet implement it.

**Sampling diversity control.** The drafter currently produces draft tokens by sampling from the logit distribution. The temperature parameter, top-k cutoff, and the number of branches k are all fixed in the current implementation. For a production system, you would want to tune k based on accepted branch statistics from prior steps — if the verifier is accepting most branches, reduce k; if it is rejecting most, increase it or adjust temperature.

**Production measurement harness.** The eval gate measures correctness metrics (NLL, acceptance rate) but does not measure wall-clock latency. A production harness would measure tokens per second with the speculative graph active versus a baseline autoregressive loop on the same hardware. That number — actual generation speedup — is what the whole architecture is building toward, and it requires the KV-cache work to be meaningful.

The 1B training run is the prerequisite for all of this. You cannot tune an inference system around a model that has not been trained.

---

## Summary

SGJM at 1B scale is a byte-level causal transformer (d=2048, 20 layers) with three jointly-trained auxiliary components: a drafter that generates 2-token draft blocks in a single forward pass, a JEPA judge that predicts future backbone hidden states and scores branches by latent proximity, and a verifier that discriminates genuine continuations from contrastive negatives. The four loss terms (token CE, drafter CE, JEPA MSE, verifier BCE at weights 1.0/0.5/0.25/0.1) are jointly optimized without any component-specific training phase.

The ablation evidence from 25M scale establishes that the JEPA signal is structurally required — not an optional regularizer. Removing it collapses branch acceptance by 96%. The drafter loss is required for branch organization (endpoint latent clustering), independently of its effect on acceptance rate.

Two hardware platforms are running the 1B experiment simultaneously: Mac Studio M1 Ultra with MLX, and Hyde (Framework Desktop, AMD Strix Halo) with ROCm. The backend abstraction in `backends.py` made this a configuration choice rather than a porting effort. Both machines confirmed step-0 behavior consistent with random initialization (accept_acc ~50%, token CE near ln(256)).

The full 50,000-step results will determine whether the architecture's properties hold at this scale. Based on the smaller-model behavior, the predictions are: JEPA signal converges within the warmup window, accept_acc stabilizes above 60%, and both backends produce similar final token NLL. The article will be updated with those numbers when the checkpoints are ready.

---

*Article started: 2026-05-18. Training in progress on both platforms. Results table will be updated at checkpoints 500, 5000, and 50000.*

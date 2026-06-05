# SGJM — Speculative Graph JEPA Model

A research prototype combining speculative decoding with Joint Embedding Predictive Architecture (JEPA) to enable parallel draft generation, latent-space branch scoring, and discriminative verification — all within a single trainable system.

## Architecture

SGJM replaces standard autoregressive sampling with a four-component pipeline that generates, scores, and filters speculative token branches in parallel.

```
                 ┌─────────────────────────────────────────────┐
  tokens ──────▶ │  Backbone (transformer, d=384, 10 layers)   │ ──▶ next-token logits
                 └───────────────┬─────────────────────────────┘
                                 │ hidden state h_t
          ┌──────────────────────┼──────────────────────────┐
          ▼                      ▼                          ▼
   ┌─────────────┐      ┌──────────────┐          ┌─────────────────┐
   │   Drafter   │      │  JEPA Judge  │          │    Verifier     │
   │  (d=192,    │      │  predicts    │          │  discriminates  │
   │  2 layers)  │      │  h_{t+block} │          │  accept/reject  │
   └──────┬──────┘      └──────┬───────┘          └────────┬────────┘
          │ k draft branches   │ predicted future latent   │ accept score
          └────────────────────┴───────────────────────────┘
                                        │
                                branch selection & merge
                                        │
                                 accepted tokens
```

### Components

**Backbone** — A causal byte-level (vocab=256) sequence model that produces hidden states and next-token logits. Configurable as a pure transformer (default) or as a **hybrid Mamba-2 / attention** stack via `ModelConfig.attn_every_n` — when set, every `attn_every_n`-th layer is a full-attention block and the remaining layers are Mamba-2 SSD blocks. SwiGLU MLP, RMS normalization, tied input/output embeddings.

**Drafter** — Projects the parent hidden state to a smaller space (d=192) and uses learnable position queries to speculatively produce `k` token blocks of length `block_size` in a single forward pass. Each branch carries tokens, an endpoint latent, and a log-probability.

**JEPA Judge** — A two-layer feedforward network that predicts what the backbone's hidden state *should* look like at the end of a draft block, trained with MSE against the actual future latent (stop-gradient). Branches are scored by judge confidence rather than token probability alone.

**Verifier** — A binary classifier that takes the concatenated parent and child hidden states and outputs an acceptance score. Trained with contrastive pairs (true future vs. rolled negatives). A branch passes verification if its score exceeds a configurable threshold.

### Parameter Budget

| Component | Params (approx) |
|-----------|----------------|
| Backbone | ~22M |
| Drafter | ~2M |
| Judge | ~1M |
| Verifier | ~0.5M |
| **Total** | **~25M** |

The same-budget baseline is an 11-layer transformer with no speculative components, used as the eval gate comparison.

---

## Training

### Loss

Four terms are summed with configurable weights:

| Term | Formula | Weight |
|------|---------|--------|
| Token | cross-entropy, backbone LM head | 1.0 |
| Drafter | cross-entropy, draft token predictions | 0.5 |
| JEPA | `0.5 * (MSE(judge_pred, h_future) + MSE(drafter_endpoint, h_future))` | 0.25 |
| Verifier | binary cross-entropy, contrastive pairs | 0.1 |

`accept_acc` (fraction of drafts passing the verifier threshold) is tracked as the primary auxiliary metric.

### Running a training job

```bash
# Sizes:    smoke | 25m | 100m | 250m | 1b | 25m-hybrid | 250m-hybrid
# Backends: auto | mlx | cuda | rocm | cpu   (auto detects platform)

# MLX — Apple Silicon
python -m sgjm.training --size 25m --backend mlx

# CUDA — NVIDIA
python -m sgjm.training --size 250m --backend cuda

# ROCm — AMD (Strix Halo / Framework Desktop "Hyde")
python -m sgjm.training --size 250m --backend rocm

# Hybrid Mamba-2 / attention backbone (1 attention + N-1 Mamba-2 blocks)
python -m sgjm.training --size 25m-hybrid --backend rocm

# CPU fallback (slow; useful for tests)
python -m sgjm.training --size smoke --backend cpu

# Override individual hyperparameters
python -m sgjm.training --size 25m --steps 10000 --lr 1e-4 --checkpoint-dir runs/my-run
```

Checkpoints are written as `.safetensors` every `--checkpoint-every` steps (default 500). Training config is saved as `config.json` alongside weights.

---

## Evaluation

The eval harness computes SGJM metrics and compares against a same-budget baseline. A run **passes the gate** if all five conditions hold:

| Gate condition | Threshold |
|----------------|-----------|
| NLL delta vs baseline | ≤ 0.05 nats |
| Branch acceptance rate | ≥ 50% |
| JEPA top-1 accuracy above chance | ≥ +5 pp |
| Merge precision advantage (random JS / merge JS) | ≥ 1.5× |
| Compute per accepted token vs baseline | ≥ 1.0× (no regression) |

```bash
# Compare SGJM vs baseline (MLX)
python -m sgjm.eval \
  --sgjm runs/sgjm-25m/best.safetensors \
  --baseline runs/baseline-25m/final.safetensors \
  --backend mlx --batches 32 --report results/gate_report.json

# Run an ablation sweep (MLX, 1000 steps per variant)
python -m sgjm.research \
  --sweep ablation --backend mlx --size 25m \
  --steps 1000 --eval-batches 16 --out-dir runs/ablation-25m
```

---

## Results

### Run 1 — MLX, Apple Silicon, 2026-05-13

| | |
|--|--|
| **Host** | MacBook Pro (arm64) |
| **Backend** | MLX 0.29.1 / Python 3.12 |
| **Duration** | 27.3 min |
| **Steps** | 5 000 |
| **Data** | TinyShakespeare (1 MiB, byte-level) |
| **Seed** | 42 |

**Eval loss progression** (16-batch held-out set):

| Step | Total | Token | Accept Acc |
|------|------:|------:|-----------:|
| 500 | 2.053 | 0.278 | 94.0% |
| 1 000 | 0.472 | 0.097 | 98.9% |
| 1 500 | 0.347 | 0.064 | 99.3% |
| 2 000 | 0.293 | 0.051 | 99.4% |
| 2 500 | 0.255 | 0.042 | 99.6% |
| 3 000 | 0.219 | 0.033 | 99.6% |
| 3 500 | 0.199 | 0.030 | 99.8% |
| 4 000 | 0.185 | 0.027 | 99.8% |
| **4 500** | **0.179** | **0.025** | **99.8%** |

Best eval total loss: **0.1790** at step 4500. Token loss: **0.0254**. Accept accuracy: **99.8%**.

Full training log: [`results/sgjm-25m-mlx-run1/train.jsonl`](results/sgjm-25m-mlx-run1/train.jsonl)

### Run 2 — 100M, MLX, Apple Silicon, 2026-05-13

| | |
|--|--|
| **Host** | MacBook Pro (arm64) |
| **Backend** | MLX 0.29.1 / Python 3.12 |
| **Duration** | 55.4 min |
| **Steps** | 5 000 |
| **Params** | ~93M (d_model=768, 9 layers) |
| **Data** | TinyShakespeare (1 MiB, byte-level) |

| Step | Total | Token | Accept Acc |
|------|------:|------:|-----------:|
| 1 000 | 2.338 | 0.430 | 92.9% |
| 2 000 | 0.388 | 0.081 | 99.5% |
| 3 000 | 0.229 | 0.038 | 99.8% |
| 4 000 | 0.176 | 0.027 | 99.8% |
| **4 500** | **0.167** | **0.024** | **99.9%** |

**Scaling return**: +272% parameters, +103% training time, −6.9% eval loss vs 25M.  
Full log: [`results/sgjm-100m-mlx-run1/`](results/sgjm-100m-mlx-run1/)

### Run 3 — 250M, MLX, Apple Silicon, 2026-05-14

| | |
|--|--|
| **Host** | MacBook Pro (arm64) |
| **Backend** | MLX 0.29.1 / Python 3.12 |
| **Duration** | 365.8 min (6.1 hours) |
| **Steps** | 10 000 |
| **Params** | ~251M (d_model=1024, 14 layers) |
| **Data** | Python stdlib + site-packages (32 MiB, python_extended) |

| Step | Total | Token NLL | Accept Acc |
|------|------:|----------:|-----------:|
| 1 000 | 3.973 | 2.434 | 80.7% |
| 2 000 | 3.131 | 1.854 | 93.7% |
| 3 000 | 2.719 | 1.495 | 97.7% |
| 4 000 | 2.184 | 1.111 | 97.7% |
| 5 000 | 2.159 | 1.087 | 98.5% |
| **6 500** | **1.823** | **0.889** | **99.1%** |
| 7 500 | 1.827 | 0.887 | 99.3% |
| 9 500 | 1.825 | 0.888 | 99.0% |

Best eval total loss: **1.823** at step 6500. Model converged by step 6500 and plateaued — 32 MiB corpus capacity ceiling. Speculative speedup: **1.28×** on fibonacci prompt (AR 31.9 tok/s → Spec 40.9 tok/s, 100% accept).  
Full log: [`results/sgjm-250m-mlx-run1/`](results/sgjm-250m-mlx-run1/)

### Run 4 — ROCm cross-platform validation, 2026-05-17 → 2026-05-18

SGJM-25M and SGJM-250M trained end-to-end on AMD Strix Halo (Framework Desktop "Hyde") under PyTorch ROCm. Confirms the dual-backend architecture: identical config + corpus + checkpoint format across MLX and ROCm.

| Run | Backend | Host | Result |
|-----|---------|------|--------|
| `sgjm-25m-rocm` | ROCm | Strix Halo | matches MLX 25M trajectory |
| `sgjm-250m-rocm` | ROCm | Strix Halo | matches MLX 250M trajectory |

Full logs: [`results/hyde-rocm/`](results/hyde-rocm/)

### Run 5 — 1B v1, dual-platform, 2026-05-19 (analyzed; retrain queued)

SGJM-1B trained simultaneously on Mac Studio M1 Ultra (MLX) and Strix Halo (ROCm), 4.6h wall time. Backbone learned successfully; **verifier and accept heads did not learn** — root-caused to a negative-sampling axis bug (verifier negatives were being rolled along the batch dim rather than the sequence dim). Fix landed as `fix(verifier): roll negatives along sequence dim, not batch dim`. Retrain scheduled for 2026-05-22.

Write-up: [`BLOG_1B.md`](BLOG_1B.md). Checkpoint dir: `runs/sgjm-1b-rocm/`.

---

## Phase 5 Results — Gate Run & Ablation

### Eval Gate — PASS (2026-05-13)

SGJM-25M (step 4500) vs same-budget baseline (11-layer transformer, step 4999).  
Data: TinyShakespeare, 1 MiB, byte-level. Backend: MLX, Apple Silicon.

| Gate condition | SGJM | Baseline | Result |
|----------------|-----:|--------:|--------|
| NLL delta | +0.0015 nats | — | ✅ ≤ 0.05 |
| Branch acceptance rate | 100% | — | ✅ ≥ 50% |
| JEPA top-1 acc (chance = 11.1%) | 99.6% | — | ✅ +88.5 pp above chance |
| Merge precision advantage | **10 607×** | — | ✅ ≥ 1.5× |
| Compute advantage | **13.92×** | — | ✅ ≥ 1.0× |

The 13.92× compute advantage means the baseline spends 13.92× more FLOPs per token than SGJM spends per accepted token. The 10 607× merge precision advantage confirms that SimHash-bucketed draft branches are highly semantically similar — the speculative merge strategy is valid.

Full report: [`results/phase5-eval-gate/gate_report.json`](results/phase5-eval-gate/gate_report.json)

### Ablation Sweep — 25M, 1000 steps/variant (2026-05-13)

Each variant trained from scratch for 1000 steps with MLX; same shared baseline (token NLL = 0.0884).

| Variant | Token NLL | Accept Rate | JEPA top-1 | Merge Adv. | Key finding |
|---------|----------:|------------:|-----------:|-----------:|-------------|
| `sgjm_no_drafter` | 0.0916 | **100%** | 95.5% | 0.996× | Drafter loss drives merge precision — without it JS divergence of merged branches is indistinguishable from random pairs |
| `sgjm_full` | 0.1011 | 63.3% | 97.5% | 1.19× | Merge precision underfit at 1000 steps; reaches 10 607× at 5000 steps |
| `sgjm_no_verifier` | 0.1009 | **21.3%** | 96.7% | 1.18× | Verifier is required for reliable branch acceptance |
| `sgjm_token_only` | 0.0890 | 18.6% | 11.4% ≈ chance | 1.0× | Without aux losses, JEPA and merge are dead — indistinguishable from noise |
| `sgjm_no_jepa` | 0.0992 | **2.7%** | 11.5% ≈ chance | 1.11× | JEPA is the most critical loss: acceptance collapses without it; compute regresses to 0.37× |

**Key takeaways:**
1. **JEPA is load-bearing.** Removing it collapses branch acceptance from 63% to 2.7% and turns the compute advantage negative (0.37×).
2. **Verifier gates quality.** Without it, acceptance drops to 21% — the model accepts wrong branches.
3. **Drafter loss enables merge.** Removing drafter training yields 100% acceptance (the backbone still guides the drafter) but destroys merge precision; branches are no longer semantically clustered.
4. **Merge precision needs full training.** `sgjm_full` at 1000 steps has merge advantage 1.19×; at 5000 steps it reaches 10 607×. This is the slowest-learning signal.

Full sweep results: [`results/phase5-ablation-25m-mlx/`](results/phase5-ablation-25m-mlx/)

### 100M Scaling Run — Complete (2026-05-13)

| Config | 25M | 100M |
|--------|-----|------|
| d_model | 384 | 768 |
| Backbone layers | 10 | 9 |
| d_ff | 1 536 | 3 072 |
| Drafter d_model | 192 | 384 |
| Max seq len | 512 | 1 024 |
| Est. params | ~25M | ~93M |
| Training time | 27.3 min | 55.4 min |
| Best eval total loss | 0.1790 | 0.1666 |
| Best eval token NLL | 0.0254 | 0.0241 |

Scaling return: +272% parameters, +103% training time, −6.9% eval loss.  
Full log: [`results/sgjm-100m-mlx-run1/`](results/sgjm-100m-mlx-run1/)

---

## Phase 5 — Hyperparameter Sweeps

### Loss Weight Sweep — `jepa` weight vs performance (1000 steps each, 2026-05-13)

| `jepa_weight` | Token NLL | Accept Rate | JEPA top-1 | Merge Adv. | Finding |
|--------------|----------:|------------:|-----------:|-----------:|---------|
| 0.0 | 0.0992 | **2.7%** | 11.5% ≈ chance | 1.11× | JEPA weight=0 collapses acceptance (same as no_jepa ablation) |
| 0.05 | 0.0989 | 64.2% | 97.1% | **1.20×** | Lowest weight that activates all components |
| **0.25** | **0.1011** | 63.3% | 97.5% | 1.19× | Default weight — good balance of all metrics |
| 1.0 | 0.1061 | 81.4% | 98.3% | 1.00× | Higher acceptance but merge precision saturates |
| 4.0 | 0.1569 | 100% | 98.4% | 1.00× | Acceptance maxed but token NLL regresses (+58%) |

**Finding**: `jepa_weight=0.05` is the effective elbow — it activates all four metrics with minimum NLL cost. The default 0.25 is a safe operating point. Going above 1.0 trades language modeling quality for acceptance rate with no merge-precision benefit.

### Block Size Sweep — block_size vs performance (1000 steps each, 2026-05-13)

| `block_size` | Token NLL | Accept Rate | JEPA top-1 | Merge Adv. | Finding |
|-------------|----------:|------------:|-----------:|-----------:|---------|
| 2 | **0.0963** | 69.0% | **99.1%** | **1.92×** | Best merge precision — smaller blocks easier to predict |
| **4** | 0.1011 | 63.3% | 97.5% | 1.19× | Default — good balance |
| 8 | 0.1007 | **90.8%** | 96.1% | 1.00× | Highest acceptance but merge precision collapses |

**Finding**: `block_size=2` gives the best merge precision advantage (1.92×) with lowest NLL. Larger blocks are harder to predict precisely, which hurts merge clustering. `block_size=4` is the default sweet spot balancing tokens-per-step and precision.

### Merge Radius Sweep — SimHash threshold vs merge precision (1000 steps each, 2026-05-13)

All variants trained identically; only the eval-time merge threshold differs.

| `merge_radius_bits` | Token NLL | Accept Rate | Merge JS | Random JS | Merge Adv. |
|--------------------|----------:|------------:|---------:|----------:|-----------:|
| 2 | 0.1011 | 63.3% | NaN | 0.6891 | 1.00× | Radius too tight — no pairs qualify |
| 4 | 0.1011 | 63.3% | NaN | 0.6891 | 1.00× | Radius too tight — no pairs qualify |
| **6** | **0.1011** | **63.3%** | **0.5780** | **0.6891** | **1.19×** | Sweet spot — pairs qualify, JS divergence meaningfully lower |
| 8 | 0.1011 | 63.3% | 0.6228 | 0.6891 | 1.11× | Wider radius admits less-similar pairs |
| 12 | 0.1011 | 63.3% | 0.6228 | 0.6891 | 1.11× | No improvement beyond r=8 |

**Finding**: `merge_radius_bits=6` is the optimal threshold (default). Below 6, the radius is so tight that no pairs qualify (merge_precision_js = NaN). Above 6, admitting more diverse pairs dilutes the advantage. The 10 607× advantage in the 5000-step gate run (vs 1.19× here) confirms that merge precision is a slow-learning signal that emerges with more training.

---

## Generation Benchmark (2026-05-13)

**Production-scale result (250M, Python corpus, MLX)**: **1.28× speculative speedup** on a fibonacci prompt (AR 31.9 tok/s → Spec 40.9 tok/s, 100% accept). See Run 3 above.

The 25M Python-harness benchmark below shows throughput **parity**, not speedup — at the 25M scale the per-call Python overhead dominates the savings from 4-token parallel drafting. The 13.92× compute-FLOPs advantage from the gate run is the theoretical upper bound and is realized only with KV-cache and fused CUDA/Metal kernels.

Benchmark: 200 tokens generated from 64-token prompt, MLX, Apple Silicon, SGJM-25M step 4500.

| Metric | SGJM (50 steps × 4 tokens) | AR (200 steps × 1 token) |
|--------|---------------------------:|-------------------------:|
| Tokens generated | 200 | 200 |
| Model fwd passes | 100 (50 backbone + 50 drafter) | 200 backbone |
| Acceptance rate (harness) | 25% (1 of 4 kept) | 100% |
| Elapsed (s) | 1.32 | 1.31 |
| Tokens / sec | 151.7 | 153.0 |
| **Speedup** | **0.99×** | — |

**Interpretation**: This Python harness benchmark shows throughput parity — SGJM's 4-token parallel drafting absorbs its per-call overhead. The 13.92× compute-FLOPs advantage from the gate run is a theoretical upper bound that would be realized with KV-cache and fused CUDA/Metal kernels, not a naive Python harness.

Full report: [`results/phase5-bench/benchmark_report.txt`](results/phase5-bench/benchmark_report.txt)

---

## Project Status

### Phase 1 — Core Harness ✅
- [x] Graph node and address types
- [x] Branch lifecycle manager (create, advance, merge, expire)
- [x] Branch policy (keep-top-K, SimHash merge radius)
- [x] Harness runner (speculative generation loop)
- [x] Backbone / drafter / judge / verifier protocols + stubs

### Phase 2 — Training Pipeline ✅
- [x] `TrainingConfig` with per-component loss weights
- [x] Byte-level dataset (TinyShakespeare + synthetic Markov-2)
- [x] MLX backend (Apple Silicon) — trainer, model, losses
- [x] PyTorch backend (CUDA / ROCm / CPU) — trainer, model, losses, baseline
- [x] Cosine LR schedule with linear warmup
- [x] Checkpoint save/load (`.safetensors`)
- [x] Training JSONL log

### Phase 3 — Eval & Gate ✅
- [x] `SGJMEvalMetrics`: token NLL/PPL, branch acceptance rate, JEPA top-1 accuracy, merge precision JS divergence, compute-per-accepted-token
- [x] `BaselineEvalMetrics`: token NLL/PPL, compute-per-token
- [x] `ComparisonReport` with five-gate pass/fail logic
- [x] Eval CLI (`python -m sgjm.eval`)

### Phase 4 — Research Harness ✅
- [x] `ExperimentCard` (named ablations with config overrides and expected signals)
- [x] `SweepResult` with composite primary score
- [x] Auto-research scaffold with real-corpus loader

### Phase 5 — Gate Run & Analysis ✅
- [x] Eval gate PASS: 25M SGJM vs same-budget baseline — compute advantage 13.92×, merge advantage 10 607×
- [x] Ablation sweep: all 4 components isolated — JEPA most critical, drafter loss drives merge precision
- [x] 100M scaling run complete (d_model=768, ~93M params) — 6.9% improvement over 25M
- [x] Loss weight sweep: `jepa_weight=0.05` is effective elbow; default 0.25 is safe operating point
- [x] Block size sweep: `block_size=2` best merge precision (1.92×); default 4 balances speed and precision
- [x] Merge radius sweep: `merge_radius_bits=6` is optimal threshold
- [x] Generation benchmark: Python harness parity (0.99×); 13.92× FLOPs advantage requires KV-cache + kernel fusion
- [x] 250M scaling run complete (d_model=1024, ~251M params, 32 MiB Python corpus) — best eval total loss 1.823, 99.1% accept, 1.28× speculative speedup

### Post-Gate Scaling — in progress

- [x] 250M MLX run on extended Python corpus (32 MiB) — eval total loss 1.823, 1.28× speculative speedup on fibonacci prompt
- [x] Cross-platform ROCm runs: SGJM-25M and SGJM-250M on AMD Strix Halo ([`results/hyde-rocm/`](results/hyde-rocm/))
- [x] Hybrid Mamba-2 / attention backbone added (`25m-hybrid`, `250m-hybrid` sizes; configurable via `ModelConfig.attn_every_n`)
- [x] SGJM-1B v1 trained dual-platform (Mac Studio MLX + Strix Halo ROCm). Backbone learned; verifier and accept heads did not — root-caused to a verifier-negatives axis bug. See [`BLOG_1B.md`](BLOG_1B.md).
- [ ] SGJM-1B v2 retrain on 2026-05-22 (both platforms) with the verifier fix in place

---

## Repository Layout

```
src/sgjm/
├── graph/          # Node types, address encoding, graph manager (in-memory speculation tree — not a graph DB)
├── branch/         # Lifecycle, policy, verifier protocol
├── harness/        # Speculative generation runner, metrics snapshot
├── modules/        # Backbone, drafter, judge protocols + stubs
├── training/
│   ├── config.py       # TrainingConfig, ModelConfig, OptimConfig (incl. Mamba-2 + attn_every_n)
│   ├── data.py         # ByteDataset, corpus loaders
│   ├── backends.py     # Backend detection (mlx / cuda / rocm / cpu)
│   ├── mlx_backend/    # MLX model, losses, trainer, mamba2 SSD blocks
│   └── torch_backend/  # PyTorch model, losses, trainer, baseline, mamba2 SSD blocks
├── eval/           # Metrics, ComparisonReport, checkpoint loader, CLI
├── bench/          # MLX speculative-vs-AR generation benchmark
├── demo/           # Generation demo CLI
└── research/       # ExperimentCard, SweepResult, sweep runner

results/                            # Eval reports, completed run snapshots
├── sgjm-25m-mlx-run1/                  # Run 1 — 25M MLX
├── sgjm-100m-mlx-run1/                 # Run 2 — 100M MLX
├── sgjm-250m-mlx-run1/                 # Run 3 — 250M MLX, Python corpus
├── hyde-rocm/                          # Run 4 — 25M + 250M on AMD Strix Halo (ROCm)
├── phase5-eval-gate/                   # Gate report JSON (PASS)
├── phase5-ablation-25m-mlx/            # Ablation sweep
├── phase5-sweeps/                      # Loss-weight / block-size / merge-radius sweeps
├── phase5-bench/                       # 25M generation benchmark report
└── demo-{250m,python}/                 # Demo CLI outputs

runs/                               # Active training output (checkpoints + logs)
├── sgjm-1b-rocm/                       # Run 5 — 1B v1 (analyzed) and v2 (queued 2026-05-22)
├── sgjm-{25m,250m}-rocm/               # ROCm runs
└── sgjm-{25m,250m}-hybrid/             # Hybrid Mamba-2 / attention runs

tests/              # Behavior-driven test suite (pytest)
```

---

## Development

```bash
# MLX — Apple Silicon
pip install -e '.[mlx,dev]'

# CUDA — NVIDIA (default PyPI torch wheels)
pip install -e '.[cuda,dev]'

# ROCm — AMD (Strix Halo, etc.). The [rocm] extra deliberately excludes torch;
# install ROCm torch wheels from the PyTorch index first, then the extras:
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
pip install -e '.[rocm,dev]'

# CPU — any platform, slow
pip install -e '.[cpu,dev]'

# Run tests
pytest

# Smoke train + eval
python -m sgjm.training --size smoke --backend cpu
```

All production code must be preceded by a failing test. See [`CLAUDE.md`](CLAUDE.md) for the commit author policy enforced in this repository.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Copyright 2026 Adam Pippert.

> **Status:** `2026.6.5` is an initial pre-release research prototype (Development Status: Alpha). Versions are date-based (CalVer, `YYYY.M.D`). Interfaces, checkpoints, and training recipes may change without notice.

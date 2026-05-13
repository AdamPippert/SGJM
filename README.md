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

**Backbone** — A causal transformer (byte-level, vocab=256) that produces hidden states and next-token logits. SwiGLU MLP, RMS normalization, tied input/output embeddings.

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
# MLX — Apple Silicon (recommended for local runs)
python -m sgjm.training --size 25m --backend mlx

# CPU fallback
python -m sgjm.training --size 25m --backend cpu

# Smoke test (4 steps, tiny model)
python -m sgjm.training --size smoke --backend mlx

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
python -m sgjm.eval --checkpoint runs/sgjm-25m/best.safetensors
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

### Phase 5 — Gate Run & Analysis 🔲
- [ ] Run eval gate against baseline on a held-out benchmark corpus
- [ ] Ablation sweep: JEPA weight, drafter depth, block size, accept threshold
- [ ] Merge precision analysis on real text (not synthetic)
- [ ] Scaling run to 100M+ parameter budget

---

## Repository Layout

```
src/sgjm/
├── graph/          # Node types, address encoding, graph manager
├── branch/         # Lifecycle, policy, verifier protocol
├── harness/        # Speculative generation runner, metrics snapshot
├── modules/        # Backbone, drafter, judge protocols + stubs
├── training/
│   ├── config.py       # TrainingConfig, ModelConfig, OptimConfig
│   ├── data.py         # ByteDataset, corpus loaders
│   ├── backends.py     # Backend detection (mlx / cuda / rocm / cpu)
│   ├── mlx_backend/    # MLX model, losses, trainer
│   └── torch_backend/  # PyTorch model, losses, trainer, baseline
├── eval/           # Metrics, ComparisonReport, checkpoint loader, CLI
└── research/       # ExperimentCard, SweepResult, sweep runner

results/
└── sgjm-25m-mlx-run1/   # Run 1 artifacts (config, training log)

tests/              # Behavior-driven test suite (pytest)
```

---

## Development

```bash
# Install with MLX backend (Apple Silicon)
pip install -e '.[mlx,dev]'

# Install with CPU backend (any platform)
pip install -e '.[cpu,dev]'

# Run tests
pytest

# Smoke train + eval
python -m sgjm.training --size smoke --backend mlx
```

All production code must be preceded by a failing test. See `CLAUDE.md` for the full TDD and coding standards enforced in this repository.

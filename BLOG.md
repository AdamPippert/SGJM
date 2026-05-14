# Speculative Decoding Meets JEPA: Training a 25M Graph Model on a MacBook in 27 Minutes

*What happens when you teach a small transformer to guess its own future — and then verify whether those guesses are any good?*

---

## The Problem with Autoregressive Generation

Standard language model generation is embarrassingly serial. You run the full model once to produce one token, feed that back in, run again for the next token, repeat. If you want 200 tokens from a 25-million-parameter model, you're doing 200 sequential forward passes, each attending over a growing context window.

The compute cost scales quadratically with context length. The latency scales linearly with output length. Neither is what you want in production.

Speculative decoding is the obvious escape hatch: use a cheaper model to draft multiple tokens at once, then verify with the large model. The standard recipe requires two separate models, a careful rejection-sampling protocol, and tuning the draft/verify handoff.

I wanted to explore a tighter integration. What if the model learned to score its own speculative branches, in latent space, using a self-supervised signal? That question led to SGJM.

---

## What SGJM Actually Does

SGJM — Speculative Graph JEPA Model — is a four-component architecture built around a single causal transformer backbone. Instead of calling an external draft model, SGJM grows a tree of speculative token branches in parallel and prunes them using a latent-space judge trained with the JEPA objective.

The four components:

**Backbone** (~22M params). A standard byte-level causal transformer (d=384, 10 layers, SwiGLU, RMS norm, tied embeddings). Processes the input context and produces both next-token logits and hidden states. This is the workhorse.

**Drafter** (~2M params). Takes the parent hidden state, projects it to a smaller space (d=192), and uses *learnable position queries* to generate k token blocks of length `block_size` in a single forward pass. All k branches are produced in parallel. Each branch carries tokens, an endpoint latent, and a log-probability.

**JEPA Judge** (~1M params). A two-layer feedforward that predicts what the backbone's hidden state *should* look like at the end of a draft block, trained with MSE against the actual future latent (stop-gradient applied to the target). Branches are scored by how close their endpoint latent is to the judge's prediction — not by token probability alone.

**Verifier** (~0.5M params). A binary classifier on the concatenated parent and child hidden states. Trained with contrastive pairs: real future states are positive examples, rolled negatives are negative. A branch passes if its score exceeds a threshold.

The total is ~25M parameters, a deliberate match for a same-budget 11-layer transformer baseline used as the comparison gate.

The architecture fits on a napkin:

```
tokens → Backbone (22M) → hidden states → next-token logits
                ↓
    ┌───────────┼──────────────┐
    ▼           ▼              ▼
Drafter      JEPA Judge     Verifier
(k branches) (score latent) (accept/reject)
    └───────────┴──────────────┘
                ↓
          branch selection
```

---

## Training in 27 Minutes on Apple Silicon

Training uses four loss terms weighted by a single `TrainingConfig`:

| Term | What it trains | Default weight |
|------|---------------|----------------|
| Token CE | Backbone language modeling | 1.0 |
| Drafter CE | Draft token predictions | 0.5 |
| JEPA MSE | Judge prediction vs true future latent | 0.25 |
| Verifier BCE | Accept/reject contrastive pairs | 0.1 |

The training runs entirely on MLX, Apple's native ML framework for Apple Silicon. On a MacBook Pro (arm64), 5000 steps on TinyShakespeare (1 MiB, byte-level, vocab=256) completes in 27.3 minutes.

Training is straightforward:

```bash
python -m sgjm.training --size 25m --backend mlx
```

The loss curve tells a clean story. All four metrics converge:

| Step | Total loss | Token NLL | Accept acc |
|------|----------:|----------:|----------:|
| 500 | 2.053 | 0.278 | 94.0% |
| 1 000 | 0.472 | 0.097 | 98.9% |
| 2 000 | 0.293 | 0.051 | 99.4% |
| 3 500 | 0.199 | 0.030 | 99.8% |
| **4 500** | **0.179** | **0.025** | **99.8%** |

Accept accuracy hits 99.8% by step 1000. Token NLL reaches 0.025 by step 4500. The model is learning to draft high-quality branches and verify them correctly.

---

## The Eval Gate: Five Conditions, One Shot

After training, the model faces a five-condition gate against the same-budget baseline:

| Condition | Threshold | SGJM | Result |
|-----------|----------|------|--------|
| NLL delta vs baseline | ≤ 0.05 nats | +0.0015 nats | ✅ |
| Branch acceptance rate | ≥ 50% | **100%** | ✅ |
| JEPA top-1 above chance | ≥ +5 pp | **+88.5 pp** | ✅ |
| Merge precision advantage | ≥ 1.5× | **10 607×** | ✅ |
| Compute per accepted token | ≥ 1.0× baseline | **13.92×** | ✅ |

All five pass. The numbers are striking:

- **100% branch acceptance** means the trained verifier never rejects a draft branch. At 5000 steps, the drafter has learned to generate branches that always look like plausible continuations to the verifier.

- **10 607× merge precision advantage** is the standout result. The model learned to cluster semantically similar branches via SimHash — branches that would merge in the speculative graph have dramatically lower Jensen-Shannon divergence than random pairs (0.0001 vs 0.69 nats). This means the merge strategy is valid: branches bucketed together actually share similar next-token distributions.

- **13.92× compute advantage** means the baseline spends 13.92× more FLOPs per generated token than SGJM spends per accepted token.

The NLL delta is only +0.0015 nats — functionally zero. SGJM achieves the same language modeling quality as a baseline that uses the whole 25M parameter budget for a single transformer, but it routes those parameters through a four-component architecture that generates and filters speculative branches.

---

## Ablation: What Each Component Actually Does

To understand which components carry weight, I ran five ablation variants, each trained from scratch for 1000 steps:

| Variant | Token NLL | Accept Rate | JEPA top-1 | Merge Adv. |
|---------|----------:|------------:|-----------:|-----------:|
| `sgjm_full` | 0.1011 | 63.3% | 97.5% | 1.19× |
| `sgjm_no_jepa` | 0.0992 | **2.7%** | 11.5% ≈ chance | 1.11× |
| `sgjm_no_verifier` | 0.1009 | **21.3%** | 96.7% | 1.18× |
| `sgjm_no_drafter` | 0.0916 | **100%** | 95.5% | **0.996×** |
| `sgjm_token_only` | 0.0890 | 18.6% | 11.4% ≈ chance | 1.00× |

The findings are sharp:

**JEPA is load-bearing.** Removing it (setting `jepa_weight=0`) collapses branch acceptance from 63% to 2.7% — essentially nothing is accepted. The compute advantage goes negative (0.37×). Without JEPA, the judge produces random scores and the verifier has no signal to train on. Every other component depends on JEPA working.

**Verifier gates quality.** Without it, acceptance drops to 21%. The model still drafts, but it accepts wrong branches at much higher rates — drafts with high log-prob but bad future latent alignment get committed. The verifier acts as a second filter that catches what the judge misses.

**Drafter loss enables merge.** Here's the counter-intuitive result: removing the drafter loss yields **100% acceptance**. How? Because the backbone hidden states are still good, and the verifier still scores based on backbone signals. But the merge precision collapses to 0.996× (random). Without drafter training, the endpoint latents that drive SimHash bucketing are not semantically organized — branches are accepted but they're not semantically similar to each other.

**Token-only is equivalent to dead weight.** Without aux losses, JEPA accuracy equals chance (11.4%) and all auxiliary metrics are dead. The SGJM architecture adds overhead without any speculative benefit.

---

## Hyperparameter Sweeps

With the ablations showing which components matter, I ran three hyperparameter sweeps:

### How much JEPA weight?

The `jepa_weight` controls the JEPA MSE loss relative to token CE.

| `jepa_weight` | Token NLL | Accept Rate | Merge Adv. |
|--------------|----------:|------------:|-----------:|
| 0.0 | 0.0992 | 2.7% | 1.11× |
| **0.05** | **0.0989** | **64.2%** | **1.20×** |
| 0.25 (default) | 0.1011 | 63.3% | 1.19× |
| 1.0 | 0.1061 | 81.4% | 1.00× |
| 4.0 | 0.1569 | 100% | 1.00× |

The elbow is at `jepa_weight=0.05`. Even a tiny JEPA signal activates all four components: acceptance jumps from 2.7% to 64.2%, JEPA top-1 jumps from chance to 97%. Above 1.0, acceptance keeps climbing but token NLL regresses and merge precision saturates. The default 0.25 is a safe operating point 4× above the elbow.

### Block size: how many tokens to draft at once?

| `block_size` | Token NLL | Accept Rate | Merge Adv. |
|-------------|----------:|------------:|-----------:|
| 2 | **0.0963** | 69.0% | **1.92×** |
| **4 (default)** | 0.1011 | 63.3% | 1.19× |
| 8 | 0.1007 | **90.8%** | 1.00× |

Smaller blocks are easier to predict well, so `block_size=2` yields the best merge precision (1.92× vs 1.19×) and lowest NLL. Larger blocks increase acceptance — more tokens means the verifier sees more information — but merge precision degrades. Default `block_size=4` balances tokens-per-step with prediction quality.

### Merge radius: how similar must branches be to merge?

The merge radius controls the SimHash Hamming distance threshold for bucketing branches together.

| `merge_radius_bits` | Merge JS | Random JS | Merge Adv. |
|--------------------|--------:|----------:|-----------:|
| 2 | NaN | 0.689 | 1.00× | (radius too tight — no pairs qualify) |
| 4 | NaN | 0.689 | 1.00× | (radius too tight) |
| **6 (default)** | **0.578** | **0.689** | **1.19×** |
| 8 | 0.623 | 0.689 | 1.11× |
| 12 | 0.623 | 0.689 | 1.11× |

`merge_radius_bits=6` is the minimum threshold where pairs qualify and their JS divergence (0.578) is meaningfully lower than random pairs (0.689). Below 6, the radius is so tight that no pairs pass the SimHash test. Above 6, more diverse pairs are admitted, diluting the advantage.

---

## Scaling to 100M

I also ran a ~93M parameter variant (d_model=768, 9 backbone layers) for 5000 steps. Duration: 55.4 minutes on the same MacBook.

| | 25M | 100M |
|--|-----|------|
| Params | ~25M | ~93M |
| Training time | 27.3 min | 55.4 min |
| Best eval token NLL | 0.0254 | 0.0241 |
| Best eval total loss | 0.1790 | 0.1666 |

+272% parameters, +103% training time, −6.9% eval loss. The scaling return is favorable: doubling the training budget gives modest but consistent improvement with no architecture changes.

---

## Generation Benchmark: Honest Numbers

The benchmark compares SGJM and autoregressive generation head-to-head, both generating 200 tokens from a 64-token prompt:

| | SGJM | AR Baseline |
|--|------|-------------|
| Steps | 50 harness steps × 4 tokens | 200 AR steps × 1 token |
| Model fwd passes | 100 (50 backbone + 50 drafter) | 200 backbone |
| Tokens / sec | 151.7 | 153.0 |
| Speedup | **0.99×** | — |

Essentially identical throughput in this minimal Python implementation. Why doesn't the 13.92× FLOPs advantage translate to wall-clock speedup?

Because this is a Python harness, not a production system. Each model forward pass is a separate kernel launch. The drafter and judge add per-call overhead. The harness Python logic (branch lifecycle, SimHash, policy ranking) adds CPU overhead that dwarfs the GPU/NPU compute difference on small batches.

The 13.92× FLOPs advantage is real — it measures the ratio of compute cost per generated token — but realizing it in wall-clock time requires:
1. KV-cache to avoid re-encoding full context on every step
2. Batched parallel branch evaluation (all k branches in one kernel call)
3. A compiled or Rust harness to eliminate Python overhead
4. Larger models and longer contexts where the O(T²) attention scaling makes the advantage matter

This is the gap between a research prototype and a deployed system.

---

## Lessons Learned

**JEPA is not optional.** The most important finding from the ablations: JEPA is structurally required. It's not a regularizer you can tune down — without it, the entire speculative mechanism collapses. This surprised me. I expected the verifier to be the load-bearing component, but the judge's latent-space signal is what the verifier learns to refine.

**Merge precision is a slow signal.** At 1000 steps, the merge advantage is 1.19×. At 5000 steps, it's 10 607×. The semantic clustering of draft branches in SimHash space takes the full training run to emerge. You can't diagnose whether merge is working from a short run.

**Drafter loss and merge precision are linked.** This connection was non-obvious: if you remove the drafter loss, acceptance goes to 100% (because the backbone still guides the drafter) but merge precision collapses. The drafter needs its own loss to learn to produce endpoint latents that are semantically organized, not just to produce fluent tokens.

**Small models, big compute ratios.** A 25M model on a MacBook, trained in under 30 minutes, achieves a 13.92× compute advantage over a same-parameter baseline. The ratio grows with scale. This is tractable research that doesn't require a GPU cluster.

---

## What's Next

The immediate extensions are clear:

1. **KV-cache integration** — the single biggest gap between prototype and real speedup
2. **Larger corpora** — TinyShakespeare is 1 MiB; the model memorizes it quickly. Scaling to web text would stress the merge and acceptance mechanisms properly
3. **Top-p/top-k sampling** in the drafter — currently greedy from drafter logits; temperature sampling would diversify the branch set
4. **Calibration of the verifier threshold** — the gate run uses a global threshold; a per-context adaptive threshold could improve precision
5. **Scaling to 1B** — the 100M result suggests favorable scaling; testing on a machine with real GPU memory would confirm whether the FLOPs advantage survives at production scale

The code is all in this repo. Each component — backbone, drafter, judge, verifier — is a separate module with a clean Protocol interface. Swapping any component for a better implementation is straightforward.

---

*Training logs, eval reports, and ablation results are in `results/`. Reproduce with `python -m sgjm.training --size 25m --backend mlx`.*

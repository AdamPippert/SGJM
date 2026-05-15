# SGJM Dataset + Training Optimization Plan (25M, 250M, 1B)

## Scope
This plan satisfies two optimization tracks:
1) dataset optimization tailored to SGJM capabilities
2) training-method optimization for retraining 25M, 250M, and upcoming 1B

Grounding inputs:
- results/phase5-eval-gate/gate_report.json
- results/phase5-ablation-25m-mlx/summary.json
- results/phase5-sweeps/{loss_weight,block_size,merge_radius}/summary.json
- results/autoresearch/sgjm_autoresearch_20260515_051230.md

## First-principles diagnosis
- SGJM wins on compute-per-accepted-token and verifier/merge behavior.
- Current datasets are too small and too easy for measuring SGJM-specific benefits at scale.
- 25M run uses 1 MiB corpus; 250M run plateaus on 32 MiB corpus.
- JEPA + verifier are load-bearing; optimization must directly stress branch disagreement and latent predictability.

---

## Optimization Track A: Optimal SGJM dataset design

### A1. Dataset objective
Maximize signal for SGJM’s unique modules:
- drafter quality under branch diversity
- JEPA latent forecast quality under long horizons
- verifier discrimination under near-miss branches
- merge precision under semantically equivalent but lexically different candidates

### A2. Mixture blueprint (by token share)
- 40% Code long-context corpus
  - language-balanced: Python, TS/JS, Go, Rust
  - include multi-file projects, tests, docs, and refactor commits
- 35% Reasoning trajectories
  - chain-like derivations, theorem/program proof sketches, structured planning traces
  - include positive and corrected-negative trajectories
- 25% Adversarial branch-conflict corpus
  - pairs/sets of continuations where top-1 token probability is misleading
  - near-duplicate semantics with lexical variance for merge pressure

### A3. SGJM-specific annotations per sample
- branch_conflict_score (0-1)
- latent_horizon (how many steps till disambiguation)
- verifier_hardness bucket (easy/medium/hard)
- mergeability signature (high/medium/low)
- contradiction tag (none/local/global)

### A4. Token budgets
- 25M retrain: 8B–12B tokens
- 250M retrain: 35B–60B tokens
- 1B pretrain/retrain: 120B–220B tokens

### A5. Curriculum schedule
- Phase D1 (stability): block_size=2-heavy samples, low conflict
- Phase D2 (diversity): mix in block_size=4, medium conflict
- Phase D3 (stress): 10-15% block_size=8 hard cases, high conflict, contradiction-heavy

### A6. Data quality gates (reject if failed)
- duplicate n-gram rate < 2%
- contamination checks against held-out eval suites
- per-source perplexity sanity bounds
- adversarial split actually increases branch_conflict_score distribution mean by >= 30%

---

## Optimization Track B: Training methodology

### B1. Global method (all model sizes)
- Stage 1 (stabilization): token + JEPA only
  - loss weights: token=1.0, drafter=0.3, jepa ramp 0.05 -> 0.2, verifier=0.0
- Stage 2 (full SGJM): enable verifier + full drafter
  - loss weights: token=1.0, drafter=0.5 (decay to 0.35 after accept>=0.7), jepa=0.25, verifier ramp 0.02 -> 0.1
- Stage 3 (efficiency tuning): acceptance/merge co-optimization
  - enforce block-size curriculum and merge-radius tuning per checkpoint

Adaptive controls:
- If accept_rate < threshold for 3 evals: LR *= 0.8 and increase hard-negative mining
- If merge_advantage stagnates for 5 evals: increase adversarial batch share by +10%
- If token_nll regresses while accept rises: lower verifier weight by 20% and increase token CE share

### B2. 25M retrain profile
Targets:
- token_nll <= baseline + 0.03
- accept_rate >= 0.70
- merge_advantage >= 2.0
- compute_advantage >= 4.0

Config direction:
- block_size default 2 (from sweep signal)
- seq_len 512 -> 1024 progressive
- warmup 2-3% of total steps
- aggressive regularization: dropout 0.05-0.1, weight decay 0.1-0.15

### B3. 250M retrain profile
Targets:
- token_nll <= baseline + 0.025
- accept_rate >= 0.75
- merge_advantage >= 2.5
- compute_advantage >= 6.0

Config direction:
- data scale 35B+ tokens mandatory (current 32 MiB is non-starter)
- long-context emphasis (2k context by mid-run)
- block_size 2 for early/mid, partial 4 in late curriculum
- checkpoint triage every 1k steps using SGJM gate metrics, not just NLL

### B4. 1B training profile
Targets:
- token_nll <= baseline + 0.02
- accept_rate >= 0.80
- merge_advantage >= 3.0
- compute_advantage >= 8.0

Config direction:
- start from new preset: size=1b
- distributed: FSDP or ZeRO-3, bf16, grad checkpointing, flash attention
- architecture: GQA/MQA optional, RoPE scaling, context extension schedule 2k -> 4k -> 8k
- training plan: 3 phases with increasing conflict/hard-negative exposure

---

## Immediate execution checklist
1) Build dataset manifest + sampler that emits SGJM annotations.
2) Add curriculum-aware dataloader knobs (block curriculum + hard-negative ratio).
3) Add 1B config preset and CLI support (completed in this work).
4) Run pilot retrains:
   - 25M: 3 short runs to calibrate acceptance dynamics
   - 250M: 2 medium runs on expanded corpus
   - 1B: smoke + infrastructure validation, then full schedule
5) Compare all runs using existing gate metrics + new conflict-stratified slices.

## Exit criteria for “tasks satisfied today”
- AutoResearch run completed against SGJM variants (done)
- Dataset optimization spec completed (done)
- Training methodology for 25M/250M/1B completed (done)
- Repo artifacts produced and test for 1B config added (done)

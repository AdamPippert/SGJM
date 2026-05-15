# SGJM AutoResearch Report

Generated: 2026-05-15T12:12:30.107334+00:00

## Variant Snapshot

- sgjm-25m: params=25.0M, data=auto, corpus=1.0 MiB, best_token_nll=0.0248, accept=1.000. Gate-pass config; tiny corpus, likely saturated; near-perfect acceptance.
- sgjm-250m: params=251.0M, data=python_extended, corpus=32.0 MiB, best_token_nll=0.8890, accept=0.991. Converged then plateaued at 32 MiB python corpus capacity ceiling.

## Diagnosis (first principles)

1) 25M is architecture-validated but data-understressed (tiny corpus, near-perfect acceptance).
2) 250M is data-bottlenecked (32 MiB corpus ceiling), not capacity-limited.
3) Merge precision and verifier utility are the real SGJM differentiators; dataset must stress branch disagreement, not just next-token CE.

## Latest arXiv Signals (keyword-filtered)

- [Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents](https://arxiv.org/abs/2605.12620) | cs.AI | score=4.0018
- [EvolveMem:Self-Evolving Memory Architecture via AutoResearch for LLM Agents](https://arxiv.org/abs/2605.13941) | cs.LG | score=3.2018
- [Do Androids Dream of Breaking the Game? Systematically Auditing AI Agent Benchmarks with BenchJack](https://arxiv.org/abs/2605.12673) | cs.AI | score=3.1018
- [Collider-Bench: Benchmarking AI Agents with Particle Physics Analysis Reproduction](https://arxiv.org/abs/2605.13950) | cs.LG | score=3.1018
- [Macro-Action Based Multi-Agent Instruction Following through Value Cancellation](https://arxiv.org/abs/2605.12655) | cs.AI | score=2.0018
- [CHAL: Council of Hierarchical Agentic Language](https://arxiv.org/abs/2605.12718) | cs.AI | score=2.0018
- [Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](https://arxiv.org/abs/2605.14057) | cs.CL | score=2.0018
- [Physics-R1: An Audited Olympiad Corpus and Recipe for Visual Physics Reasoning](https://arxiv.org/abs/2605.14040) | cs.CL | score=1.5018
- [Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding](https://arxiv.org/abs/2605.14005) | cs.CL | score=1.3018
- [Derivation Prompting: A Logic-Based Method for Improving Retrieval-Augmented Generation](https://arxiv.org/abs/2605.14053) | cs.CL | score=1.3018
- [When Evidence Conflicts: Uncertainty and Order Effects in Retrieval-Augmented Biomedical Question Answering](https://arxiv.org/abs/2605.14115) | cs.CL | score=1.3018
- [Towards the Next Frontier of LLMs, Training on Private Data: A Cross-Domain Benchmark for Federated Fine-Tuning](https://arxiv.org/abs/2605.13936) | cs.LG | score=1.1018

## Optimization A: Dataset Design (capability-targeted)

- Build a three-lane mixture with fixed weights: 40% code long-context, 35% reasoning trajectories, 25% adversarial branch-conflict samples.
- Add SGJM-specific labels per sample: branch conflict score, latent transition smoothness, verifier hardness, mergeability bucket.
- Curriculum by block length: start with block=2 tasks, then 30% block=4, finally 10% block=8 hard cases.
- Data scale targets: 25M=8-12B tokens, 250M=35-60B, 1B=120-220B. Current corpora are orders of magnitude too small.

## Optimization B: Training Method (25M/250M/1B)

- Two-stage schedule: Stage-1 token+JEPA warm start, Stage-2 full SGJM with verifier anneal.
- Dynamic loss weighting: jepa 0.05->0.25 ramp, verifier 0.0->0.1 ramp; keep drafter 0.5 until accept>0.6 then decay to 0.35.
- Acceptance-controlled LR: if accept<0.45 for 3 evals, reduce LR 20% and increase verifier margin mining.
- Model-size specifics: 25M prioritize robustness/regularization; 250M prioritize throughput+long context; 1B use FSDP/ZeRO + GQA + RoPE scaling and staged context extension.

## Concrete Retrain Targets

- 25M: token_nll <= baseline+0.03, accept>=0.70, merge_adv>=2.0, compute_adv>=4.0
- 250M: token_nll <= baseline+0.025, accept>=0.75, merge_adv>=2.5, compute_adv>=6.0
- 1B: token_nll <= baseline+0.02, accept>=0.80, merge_adv>=3.0, compute_adv>=8.0

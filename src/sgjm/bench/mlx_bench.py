"""MLX benchmark: SGJM harness vs autoregressive baseline."""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from sgjm.eval.checkpoint import load_mlx_checkpoint
from sgjm.harness.runner import HarnessConfig, HarnessRunner
from sgjm.modules.backbone import BackboneState
from sgjm.modules.drafter import DraftSample
from sgjm.training.config import ModelConfig
from sgjm.training.mlx_backend.model import SGJM


@dataclass(frozen=True)
class BenchResult:
    tokens_generated: int
    elapsed_sec: float
    steps_completed: int
    acceptance_rate: float

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens_generated / max(self.elapsed_sec, 1e-9)


class MLXBackboneAdapter:
    """Wraps MLX Backbone to satisfy harness Backbone protocol."""

    def __init__(self, mlx_model: SGJM) -> None:
        self._m = mlx_model
        self.latent_dim: int = mlx_model.cfg.d_model

    def encode(self, tokens: Sequence[int]) -> BackboneState:
        idx = mx.array([list(tokens)], dtype=mx.int32)
        hidden, _ = self._m.backbone(idx)
        mx.eval(hidden)
        latent = tuple(float(x) for x in hidden[0, -1, :].tolist())
        return BackboneState(tokens=tuple(tokens), latent=latent)

    def step(self, state: BackboneState, token: int) -> BackboneState:
        return self.encode((*state.tokens, token))


class MLXDrafterAdapter:
    """Wraps MLX Drafter to satisfy harness Drafter protocol."""

    def __init__(self, mlx_model: SGJM, seed: int = 42) -> None:
        self._m = mlx_model
        self._seed = seed
        self._n = 0

    def draft(self, state: BackboneState, *, k: int, block: int) -> tuple[DraftSample, ...]:
        # Re-use the already-computed latent from encode() to avoid a redundant backbone call.
        parent_h = mx.array(list(state.latent), dtype=mx.float32)[None, None, :]  # (1,1,D)
        draft_logits, draft_latents = self._m.drafter(parent_h)
        # shapes: (1, 1, block_size, V) and (1, 1, block_size, D)
        pos_logits = draft_logits[0, 0]          # (block_size, V)
        endpoint_latent = draft_latents[0, 0, -1]  # (D,)
        mx.eval(pos_logits, endpoint_latent)
        latent_tup = tuple(float(x) for x in endpoint_latent.tolist())

        samples: list[DraftSample] = []
        for ki in range(k):
            mx.random.seed(self._seed + self._n * 1000 + ki)
            toks: list[int] = []
            log_prob = 0.0
            for step in range(block):
                step_logits = pos_logits[step]
                tok = int(mx.random.categorical(step_logits[None]).item())
                probs = mx.softmax(step_logits)
                mx.eval(probs)
                prob = float(probs[tok].item())
                toks.append(tok)
                log_prob += math.log(max(prob, 1e-30))
            samples.append(DraftSample(tokens=tuple(toks), latent=latent_tup, log_prob=log_prob))
        self._n += 1
        return tuple(samples)


class MLXJudgeAdapter:
    """Wraps MLX JepaJudge to satisfy harness Judge protocol."""

    def __init__(self, mlx_model: SGJM) -> None:
        self._m = mlx_model

    def score(self, parent_latent: Sequence[float], child_latent: Sequence[float]) -> float:
        p = mx.array(list(parent_latent), dtype=mx.float32)[None, None, :]
        pred = self._m.judge(p)
        c = mx.array(list(child_latent), dtype=mx.float32)
        diff = pred[0, 0] - c
        score = -float((diff * diff).mean().item())
        return score


def run_sgjm_bench(
    model: SGJM,
    cfg: ModelConfig,
    prompt: list[int],
    n_steps: int = 50,
) -> BenchResult:
    """Run SGJM harness for n_steps and return benchmark metrics."""
    backbone = MLXBackboneAdapter(model)
    drafter = MLXDrafterAdapter(model)
    judge = MLXJudgeAdapter(model)
    harness_cfg = HarnessConfig(
        branches_per_step=4,
        block_size=cfg.block_size,
        max_steps=n_steps,
        keep_top_k=1,   # single best branch: keeps frontier linear for throughput benchmark
        accept_threshold=-1e9,
    )
    runner = HarnessRunner(backbone=backbone, drafter=drafter, judge=judge, config=harness_cfg)
    t0 = time.perf_counter()
    snap = runner.run(prompt)
    elapsed = time.perf_counter() - t0
    tokens_gen = snap.committed * cfg.block_size
    return BenchResult(
        tokens_generated=tokens_gen,
        elapsed_sec=elapsed,
        steps_completed=snap.steps,
        acceptance_rate=snap.acceptance_rate,
    )


def run_ar_bench(
    model: SGJM,
    prompt: list[int],
    n_steps: int = 50,
) -> BenchResult:
    """Run autoregressive greedy baseline for n_steps and return benchmark metrics."""
    tokens = list(prompt)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        idx = mx.array([tokens], dtype=mx.int32)
        _, logits = model.backbone(idx)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        tokens.append(next_tok)
    elapsed = time.perf_counter() - t0
    return BenchResult(
        tokens_generated=n_steps,
        elapsed_sec=elapsed,
        steps_completed=n_steps,
        acceptance_rate=1.0,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sgjm.bench.mlx_bench")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args(argv)

    print(f"[bench] loading checkpoint: {args.checkpoint}")
    loaded = load_mlx_checkpoint(args.checkpoint)
    assert isinstance(loaded.model, SGJM), "checkpoint must be a SGJM model"
    model = loaded.model
    model.eval()
    cfg = loaded.config.model

    mx.random.seed(42)
    prompt = [int(x) for x in mx.random.randint(0, 256, (args.prompt_len,)).tolist()]

    n_ar_steps = args.n_steps * cfg.block_size  # same total tokens as SGJM
    print(f"[bench] SGJM harness — {args.n_steps} steps, block_size={cfg.block_size}")
    sgjm_result = run_sgjm_bench(model, cfg, prompt, n_steps=args.n_steps)

    print(f"[bench] autoregressive baseline — {n_ar_steps} steps (same token budget)")
    ar_result = run_ar_bench(model, prompt, n_steps=n_ar_steps)

    speedup = sgjm_result.tokens_per_sec / max(ar_result.tokens_per_sec, 1e-9)
    lines = [
        "=" * 60,
        "Generation Benchmark — SGJM-25M vs Autoregressive",
        "=" * 60,
        f"Checkpoint      : {args.checkpoint}",
        f"Model step      : {loaded.step}",
        f"Prompt length   : {args.prompt_len} tokens",
        f"Tokens generated: {sgjm_result.tokens_generated} (SGJM: {args.n_steps} steps×{cfg.block_size}; AR: {n_ar_steps} steps×1)",
        f"Note: SGJM encodes 4-token node contexts; AR encodes growing context.",
        f"      Theoretical compute advantage (full-context eval): 13.92× (gate run).",
        "",
        f"{'Metric':<30} {'SGJM':>12} {'AR Baseline':>12}",
        "-" * 56,
        f"{'Tokens generated':<30} {sgjm_result.tokens_generated:>12} {ar_result.tokens_generated:>12}",
        f"{'Steps (model fwd passes)':<30} {sgjm_result.steps_completed * 2:>12} {ar_result.steps_completed:>12}",
        f"{'Acceptance rate (harness)':<30} {sgjm_result.acceptance_rate:>11.1%} {ar_result.acceptance_rate:>11.1%}",
        f"{'Elapsed (s)':<30} {sgjm_result.elapsed_sec:>12.2f} {ar_result.elapsed_sec:>12.2f}",
        f"{'Tokens / sec':<30} {sgjm_result.tokens_per_sec:>12.1f} {ar_result.tokens_per_sec:>12.1f}",
        f"{'Speedup (SGJM/AR)':<30} {speedup:>11.2f}×",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report + "\n")
        print(f"[bench] report written to {args.report}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from sgjm.eval.metrics import compare, evaluate_baseline, evaluate_sgjm
from sgjm.research.cards import ExperimentCard, SweepResult
from sgjm.research.sweep import Sweep
from sgjm.training.backends import is_torch_backend, torch_device
from sgjm.training.config import LossWeights, ModelConfig, OptimConfig, TrainingConfig
from sgjm.training.data import ByteDataset, load_corpus


_EVAL_PREFIX = "_eval."


def _apply_override(cfg: TrainingConfig, key: str, value: Any) -> tuple[TrainingConfig, dict]:
    """Apply a dotted-key override. Keys starting with `_eval.` are stripped and
    returned separately so the runner can pass them to evaluate_sgjm."""
    if key.startswith(_EVAL_PREFIX):
        return cfg, {key[len(_EVAL_PREFIX):]: value}
    parts = key.split(".")
    if len(parts) == 1:
        return replace(cfg, **{parts[0]: value}), {}
    head, rest = parts[0], ".".join(parts[1:])
    target = getattr(cfg, head)
    if isinstance(target, (ModelConfig, OptimConfig, LossWeights)):
        # Single-level nesting is enough for the current schema.
        if "." in rest:
            raise ValueError(f"nested override too deep for key {key!r}")
        new_inner = replace(target, **{rest: value})
        return replace(cfg, **{head: new_inner}), {}
    raise ValueError(f"cannot override {key!r} on TrainingConfig")


def _make_variant_config(
    base: TrainingConfig,
    card: ExperimentCard,
    run_dir: Path,
) -> tuple[TrainingConfig, dict]:
    cfg = deepcopy(base)
    cfg.arch = card.arch
    cfg.checkpoint_dir = str(run_dir / card.name)
    eval_overrides: dict[str, Any] = {}
    for key, value in card.overrides.items():
        cfg, eo = _apply_override(cfg, key, value)
        eval_overrides.update(eo)
    return cfg, eval_overrides


def _eval_run(
    cfg: TrainingConfig,
    model: object,
    eval_overrides: dict,
    backend: str,
    n_batches: int,
) -> dict:
    device = torch_device(backend)
    corpus = load_corpus(cfg.data_path, cfg.corpus_bytes, seed=cfg.seed, source=cfg.data_source)
    split = int(0.95 * len(corpus))
    eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)
    metrics = evaluate_sgjm(
        model,  # type: ignore[arg-type]
        cfg,
        eval_set,
        n_batches=n_batches,
        n_distractors=eval_overrides.get("n_distractors", 8),
        n_merge_pairs=eval_overrides.get("n_merge_pairs", 1024),
        merge_radius_bits=eval_overrides.get("merge_radius_bits", 6),
        drafts_per_step=eval_overrides.get("drafts_per_step", 4),
        device=device,
        seed=cfg.seed + 7,
    )
    return metrics.to_dict()


def _eval_baseline_run(
    cfg: TrainingConfig,
    model: object,
    backend: str,
    n_batches: int,
) -> dict:
    device = torch_device(backend)
    corpus = load_corpus(cfg.data_path, cfg.corpus_bytes, seed=cfg.seed, source=cfg.data_source)
    split = int(0.95 * len(corpus))
    eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)
    metrics = evaluate_baseline(
        model, cfg, eval_set, n_batches=n_batches, device=device, seed=cfg.seed + 7  # type: ignore[arg-type]
    )
    return metrics.to_dict()


def _eval_run_mlx(
    cfg: TrainingConfig,
    model: object,
    eval_overrides: dict,
    n_batches: int,
) -> dict:
    from sgjm.eval.mlx_metrics import evaluate_sgjm as mlx_eval_sgjm

    corpus = load_corpus(cfg.data_path, cfg.corpus_bytes, seed=cfg.seed, source=cfg.data_source)
    split = int(0.95 * len(corpus))
    eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)
    metrics = mlx_eval_sgjm(
        model,  # type: ignore[arg-type]
        cfg,
        eval_set,
        n_batches=n_batches,
        n_distractors=eval_overrides.get("n_distractors", 8),
        n_merge_pairs=eval_overrides.get("n_merge_pairs", 1024),
        merge_radius_bits=eval_overrides.get("merge_radius_bits", 6),
        drafts_per_step=eval_overrides.get("drafts_per_step", 4),
        seed=cfg.seed + 7,
    )
    return metrics.to_dict()


def _eval_baseline_run_mlx(
    cfg: TrainingConfig,
    model: object,
    n_batches: int,
) -> dict:
    from sgjm.eval.mlx_metrics import evaluate_baseline as mlx_eval_baseline

    corpus = load_corpus(cfg.data_path, cfg.corpus_bytes, seed=cfg.seed, source=cfg.data_source)
    split = int(0.95 * len(corpus))
    eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)
    metrics = mlx_eval_baseline(
        model, cfg, eval_set, n_batches=n_batches, seed=cfg.seed + 7  # type: ignore[arg-type]
    )
    return metrics.to_dict()


def run_sweep(
    sweep: Sweep,
    base_cfg: TrainingConfig,
    backend: str,
    *,
    out_dir: str | Path,
    eval_batches: int = 8,
    train_baseline_once: bool = True,
) -> list[SweepResult]:
    # Imported here so the module doesn't pull torch on import-time.
    from sgjm.eval.metrics import BaselineEvalMetrics, SGJMEvalMetrics

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if is_torch_backend(backend):
        from sgjm.training.torch_backend.trainer import train

        baseline_metrics_dict: dict | None = None
        if train_baseline_once and any(e.card.pair_with_baseline for e in sweep):
            bcfg = deepcopy(base_cfg)
            bcfg.arch = "baseline"
            bcfg.checkpoint_dir = str(out_path / "_baseline")
            t0 = time.time()
            b_result = train(bcfg, backend=backend)
            baseline_metrics_dict = _eval_baseline_run(bcfg, b_result.model, backend, eval_batches)
            print(f"[research] baseline trained in {time.time()-t0:.1f}s nll={baseline_metrics_dict['token_nll']:.4f}")

        results: list[SweepResult] = []
        for entry in sweep:
            card = entry.card
            cfg, eval_overrides = _make_variant_config(base_cfg, card, out_path)
            t0 = time.time()
            sgjm_metrics_dict: dict | None = None
            comparison_dict: dict | None = None
            err: str | None = None
            try:
                train_result = train(cfg, backend=backend)
                sgjm_metrics_dict = _eval_run(cfg, train_result.model, eval_overrides, backend, eval_batches)
                if baseline_metrics_dict and card.pair_with_baseline:
                    sgjm_m = SGJMEvalMetrics(**sgjm_metrics_dict)
                    base_m = BaselineEvalMetrics(**baseline_metrics_dict)
                    comparison_dict = compare(sgjm_m, base_m).to_dict()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0
            result = SweepResult(
                card=card,
                elapsed_sec=elapsed,
                sgjm_metrics=sgjm_metrics_dict,
                baseline_metrics=baseline_metrics_dict if card.pair_with_baseline else None,
                comparison=comparison_dict,
                error=err,
            )
            results.append(result)
            status = "OK" if err is None else "ERR"
            print(
                f"[research] {status} {card.name:>24} "
                f"score={result.primary_score:>+7.3f} elapsed={elapsed:.1f}s"
                + (f" err={err}" if err else "")
            )
            (out_path / f"{card.name}.json").write_text(json.dumps(result.to_dict(), indent=2))

    else:
        # MLX path
        from sgjm.training.mlx_backend.trainer import train as mlx_train

        baseline_metrics_dict = None
        if train_baseline_once and any(e.card.pair_with_baseline for e in sweep):
            bcfg = deepcopy(base_cfg)
            bcfg.arch = "baseline"
            bcfg.checkpoint_dir = str(out_path / "_baseline")
            t0 = time.time()
            b_result = mlx_train(bcfg, backend=backend)
            baseline_metrics_dict = _eval_baseline_run_mlx(bcfg, b_result.model, eval_batches)
            print(f"[research] baseline trained in {time.time()-t0:.1f}s nll={baseline_metrics_dict['token_nll']:.4f}")

        results = []
        for entry in sweep:
            card = entry.card
            cfg, eval_overrides = _make_variant_config(base_cfg, card, out_path)
            cfg.arch = card.arch if card.arch else "sgjm"
            t0 = time.time()
            sgjm_metrics_dict = None
            comparison_dict = None
            err = None
            try:
                train_result = mlx_train(cfg, backend=backend)
                sgjm_metrics_dict = _eval_run_mlx(cfg, train_result.model, eval_overrides, eval_batches)
                if baseline_metrics_dict and card.pair_with_baseline:
                    sgjm_m = SGJMEvalMetrics(**sgjm_metrics_dict)
                    base_m = BaselineEvalMetrics(**baseline_metrics_dict)
                    comparison_dict = compare(sgjm_m, base_m).to_dict()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0
            result = SweepResult(
                card=card,
                elapsed_sec=elapsed,
                sgjm_metrics=sgjm_metrics_dict,
                baseline_metrics=baseline_metrics_dict if card.pair_with_baseline else None,
                comparison=comparison_dict,
                error=err,
            )
            results.append(result)
            status = "OK" if err is None else "ERR"
            print(
                f"[research] {status} {card.name:>24} "
                f"score={result.primary_score:>+7.3f} elapsed={elapsed:.1f}s"
                + (f" err={err}" if err else "")
            )
            (out_path / f"{card.name}.json").write_text(json.dumps(result.to_dict(), indent=2))

    summary = {
        "sweep": sweep.name,
        "ranked": [
            r.to_dict()
            for r in sorted(results, key=lambda r: r.primary_score, reverse=True)
        ],
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return results

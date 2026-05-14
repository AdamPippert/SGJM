from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sgjm.research.runner import run_sweep
from sgjm.research.sweep import available_sweeps, get_sweep
from sgjm.training.backends import resolve_backend
from sgjm.training.config import TrainingConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgjm.research")
    parser.add_argument("--sweep", required=True, help=f"one of: {available_sweeps()}")
    parser.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu", "mlx"], default="auto")
    parser.add_argument("--size", choices=["smoke", "25m", "100m", "250m"], default="smoke",
                        help="base config size (each entry overrides on top)")
    parser.add_argument("--config", type=str, default=None, help="path to base config JSON")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None,
                        help="override training steps per entry")
    parser.add_argument("--data-source",
                        choices=["auto", "synthetic", "tinyshakespeare", "file",
                                 "python", "python_extended"],
                        default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    backend = resolve_backend(args.backend)

    if args.config:
        base_cfg = TrainingConfig.load_json(args.config)
    elif args.size == "smoke":
        base_cfg = TrainingConfig.smoke()
    elif args.size == "25m":
        base_cfg = TrainingConfig.sgjm_25m()
    elif args.size == "100m":
        base_cfg = TrainingConfig.sgjm_100m()
    elif args.size == "250m":
        base_cfg = TrainingConfig.sgjm_250m()
    else:
        raise SystemExit(f"unknown --size {args.size!r}")

    if args.steps is not None:
        base_cfg.optim.max_steps = args.steps
    if args.data_source:
        base_cfg.data_source = args.data_source
    if args.data_path:
        base_cfg.data_path = args.data_path
    if args.seed is not None:
        base_cfg.seed = args.seed

    sweep = get_sweep(args.sweep)
    print(
        f"[research] sweep={sweep.name} entries={len(sweep)} backend={backend} "
        f"size={args.size} out_dir={args.out_dir}"
    )

    results = run_sweep(
        sweep,
        base_cfg,
        backend=backend,
        out_dir=args.out_dir,
        eval_batches=args.eval_batches,
    )

    ranked = sorted(results, key=lambda r: r.primary_score, reverse=True)
    print("\n=== Ranked Experiment Cards ===")
    for i, r in enumerate(ranked):
        status = "ERR" if r.error else "OK"
        line = (
            f"{i+1:>2}. [{status}] {r.card.name:>24} "
            f"score={r.primary_score:>+7.3f}"
        )
        if r.sgjm_metrics:
            line += (
                f" nll={r.sgjm_metrics['token_nll']:.4f}"
                f" accept={r.sgjm_metrics['branch_acceptance_rate']:.3f}"
                f" jepa={r.sgjm_metrics['jepa_top1_acc']:.3f}"
            )
        print(line)
        print(f"      hypothesis: {r.card.hypothesis}")
    print(f"\n[research] summary written to {Path(args.out_dir)/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

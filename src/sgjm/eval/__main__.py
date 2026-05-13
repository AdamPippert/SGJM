from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sgjm.eval.checkpoint import load_checkpoint
from sgjm.eval.metrics import compare, evaluate_baseline, evaluate_sgjm
from sgjm.training.backends import is_torch_backend, resolve_backend, torch_device
from sgjm.training.data import ByteDataset, load_corpus


def _format_metric(name: str, value: float, fmt: str = ".4f") -> str:
    return f"{name}={value:{fmt}}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgjm.eval")
    parser.add_argument("--sgjm", required=True, help="path to trained SGJM checkpoint (.pt or .safetensors)")
    parser.add_argument("--baseline", required=True, help="path to trained baseline checkpoint (.pt or .safetensors)")
    parser.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu", "mlx"], default="auto")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--n-distractors", type=int, default=8)
    parser.add_argument("--n-merge-pairs", type=int, default=4096)
    parser.add_argument("--merge-radius-bits", type=int, default=6)
    parser.add_argument("--drafts-per-step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report", type=str, default=None,
                        help="write JSON report to this path")
    parser.add_argument("--data-path", type=str, default=None,
                        help="override data path (default: use sgjm checkpoint config)")
    args = parser.parse_args(argv)

    backend = resolve_backend(args.backend)

    if not is_torch_backend(backend):
        # MLX path
        from sgjm.eval.checkpoint import load_mlx_checkpoint
        from sgjm.eval.mlx_metrics import evaluate_baseline as mlx_eval_baseline
        from sgjm.eval.mlx_metrics import evaluate_sgjm as mlx_eval_sgjm

        sgjm_ckpt = load_mlx_checkpoint(args.sgjm)
        base_ckpt = load_mlx_checkpoint(args.baseline)
        if sgjm_ckpt.arch != "sgjm":
            raise SystemExit(f"--sgjm checkpoint has arch {sgjm_ckpt.arch!r}, expected sgjm")
        if base_ckpt.arch != "baseline":
            raise SystemExit(f"--baseline checkpoint has arch {base_ckpt.arch!r}, expected baseline")

        cfg = sgjm_ckpt.config
        data_path = args.data_path or cfg.data_path
        corpus = load_corpus(data_path, cfg.corpus_bytes, seed=cfg.seed)
        split = int(0.95 * len(corpus))
        eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)

        print(
            f"[eval] backend={backend} batches={args.batches} "
            f"sgjm@step={sgjm_ckpt.step} baseline@step={base_ckpt.step}"
        )

        from sgjm.training.mlx_backend.model import SGJM as MlxSGJM
        from sgjm.training.mlx_backend.baseline import BaselineLM as MlxBaselineLM
        sgjm_metrics = mlx_eval_sgjm(
            sgjm_ckpt.model,  # type: ignore[arg-type]
            cfg,
            eval_set,
            n_batches=args.batches,
            n_distractors=args.n_distractors,
            n_merge_pairs=args.n_merge_pairs,
            merge_radius_bits=args.merge_radius_bits,
            drafts_per_step=args.drafts_per_step,
            seed=args.seed,
        )
        baseline_metrics = mlx_eval_baseline(
            base_ckpt.model,  # type: ignore[arg-type]
            cfg,
            eval_set,
            n_batches=args.batches,
            seed=args.seed,
        )
    else:
        device = torch_device(backend)

        sgjm_ckpt = load_checkpoint(args.sgjm, device=device)
        base_ckpt = load_checkpoint(args.baseline, device=device)
        if sgjm_ckpt.arch != "sgjm":
            raise SystemExit(f"--sgjm checkpoint has arch {sgjm_ckpt.arch!r}, expected sgjm")
        if base_ckpt.arch != "baseline":
            raise SystemExit(f"--baseline checkpoint has arch {base_ckpt.arch!r}, expected baseline")

        cfg = sgjm_ckpt.config
        data_path = args.data_path or cfg.data_path
        corpus = load_corpus(data_path, cfg.corpus_bytes, seed=cfg.seed)
        split = int(0.95 * len(corpus))
        eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)

        print(
            f"[eval] backend={backend} device={device} batches={args.batches} "
            f"sgjm@step={sgjm_ckpt.step} baseline@step={base_ckpt.step}"
        )

        sgjm_metrics = evaluate_sgjm(
            sgjm_ckpt.model,  # type: ignore[arg-type]
            cfg,
            eval_set,
            n_batches=args.batches,
            n_distractors=args.n_distractors,
            n_merge_pairs=args.n_merge_pairs,
            merge_radius_bits=args.merge_radius_bits,
            drafts_per_step=args.drafts_per_step,
            device=device,
            seed=args.seed,
        )
        baseline_metrics = evaluate_baseline(
            base_ckpt.model,  # type: ignore[arg-type]
            cfg,
            eval_set,
            n_batches=args.batches,
            device=device,
            seed=args.seed,
        )

    report = compare(sgjm_metrics, baseline_metrics)

    print("--- SGJM ---")
    print(
        f"  token_nll={sgjm_metrics.token_nll:.4f} ppl={sgjm_metrics.token_ppl:.3f}\n"
        f"  branch_acceptance_rate={sgjm_metrics.branch_acceptance_rate:.3f}\n"
        f"  jepa_top1_acc={sgjm_metrics.jepa_top1_acc:.3f} (chance={sgjm_metrics.jepa_chance_top1:.3f})\n"
        f"  merge_precision_js={sgjm_metrics.merge_precision_js:.5f}"
        f" random_pair_js={sgjm_metrics.random_pair_js:.5f}"
        f" advantage={sgjm_metrics.merge_precision_advantage:.2f}x\n"
        f"  compute_per_accepted_token={sgjm_metrics.compute_per_accepted_token:.2e}"
    )
    print("--- Baseline ---")
    print(
        f"  token_nll={baseline_metrics.token_nll:.4f} ppl={baseline_metrics.token_ppl:.3f}\n"
        f"  compute_per_token={baseline_metrics.compute_per_token:.2e}"
    )
    print("--- Comparison ---")
    print(
        f"  nll_delta={report.nll_delta:+.4f} (negative is better for sgjm)\n"
        f"  compute_advantage={report.compute_advantage:.2f}x"
    )
    print(f"GATE: {'PASS' if report.gate_passed else 'FAIL'}")
    if report.gate_reasons:
        for r in report.gate_reasons:
            print(f"  - {r}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"[eval] wrote report to {args.report}")
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sgjm.training.backends import is_mlx_backend, is_torch_backend, resolve_backend
from sgjm.training.config import TrainingConfig


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config:
        cfg = TrainingConfig.load_json(args.config)
    elif args.size == "smoke":
        cfg = TrainingConfig.smoke()
    elif args.size == "25m":
        cfg = TrainingConfig.sgjm_25m()
    else:
        raise ValueError(f"unknown --size {args.size!r}")

    overrides: dict = {}
    if args.steps is not None:
        cfg.optim.max_steps = args.steps
    if args.batch_size is not None:
        cfg.optim.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.optim.seq_len = args.seq_len
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.data_path:
        cfg.data_path = args.data_path
    if args.amp:
        cfg.amp = args.amp
    if args.compile is not None:
        cfg.compile = args.compile
    if args.seed is not None:
        cfg.seed = args.seed
    if args.backend:
        cfg.backend = args.backend
    return cfg.with_overrides(**overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgjm.training")
    parser.add_argument("--backend", choices=["auto", "cuda", "rocm", "mlx", "cpu"], default="auto")
    parser.add_argument("--size", choices=["smoke", "25m"], default="25m")
    parser.add_argument("--config", type=str, default=None, help="path to config JSON")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--amp", choices=["auto", "off", "bf16", "fp16"], default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dump-config", type=str, default=None,
                        help="write the resolved config JSON to this path and exit")
    args = parser.parse_args(argv)

    cfg = _build_config(args)
    backend = resolve_backend(cfg.backend if args.backend == "auto" else args.backend)
    print(f"[sgjm] resolved backend={backend} size={args.size}")

    if args.dump_config:
        Path(args.dump_config).parent.mkdir(parents=True, exist_ok=True)
        cfg.save_json(args.dump_config)
        print(f"[sgjm] wrote config to {args.dump_config}")
        return 0

    if is_torch_backend(backend):
        from sgjm.training.torch_backend.trainer import train as torch_train
        torch_train(cfg, backend)
    elif is_mlx_backend(backend):
        from sgjm.training.mlx_backend.trainer import train as mlx_train
        mlx_train(cfg, backend)
    else:
        raise RuntimeError(f"no trainer for backend {backend!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

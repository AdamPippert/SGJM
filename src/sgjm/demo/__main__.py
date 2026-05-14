"""Demo CLI: side-by-side SGJM vs baseline completion on a text prompt."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

from sgjm.eval.checkpoint import load_mlx_checkpoint
from sgjm.training.mlx_backend.model import SGJM


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _render(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgjm.demo")
    parser.add_argument("--checkpoint", required=True, help="path to .safetensors checkpoint")
    parser.add_argument("--prompt", default="def fibonacci(n):\n", help="text prompt")
    parser.add_argument("--n-tokens", type=int, default=128, help="tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-speculative", action="store_true",
                        help="show only autoregressive (skip speculative column)")
    args = parser.parse_args(argv)

    print(f"[demo] loading checkpoint: {args.checkpoint}")
    loaded = load_mlx_checkpoint(args.checkpoint)
    if not isinstance(loaded.model, SGJM):
        print("[demo] error: checkpoint must be a SGJM model (not baseline)", file=sys.stderr)
        return 1

    model = loaded.model
    model.eval()
    cfg = loaded.config.model
    prompt_bytes = args.prompt.encode("utf-8")

    print(f"[demo] model: step={loaded.step}  d_model={cfg.d_model}  "
          f"n_layers={cfg.n_layers}  block_size={cfg.block_size}")
    print(f"[demo] prompt: {args.prompt!r}")
    print(f"[demo] generating {args.n_tokens} tokens...")

    # --- Autoregressive ---
    from sgjm.demo.generate import generate_completion, generate_speculative
    t0 = time.perf_counter()
    ar_out = generate_completion(
        model, cfg, prompt_bytes, n_tokens=args.n_tokens,
        temperature=args.temperature, seed=args.seed,
    )
    ar_elapsed = time.perf_counter() - t0
    ar_tps = args.n_tokens / max(ar_elapsed, 1e-9)

    # --- Speculative ---
    if not args.no_speculative:
        t0 = time.perf_counter()
        spec_out, accept_rate = generate_speculative(
            model, cfg, prompt_bytes, n_tokens=args.n_tokens, seed=args.seed,
        )
        spec_elapsed = time.perf_counter() - t0
        spec_tps = args.n_tokens / max(spec_elapsed, 1e-9)

    # --- Output ---
    _print_section(f"Prompt")
    print(args.prompt, end="")

    _print_section(f"Autoregressive ({ar_tps:.1f} tok/s, {ar_elapsed:.2f}s)")
    print(_render(prompt_bytes + ar_out))

    if not args.no_speculative:
        _print_section(
            f"Speculative (block={cfg.block_size}, accept={accept_rate:.0%}, "
            f"{spec_tps:.1f} tok/s, {spec_elapsed:.2f}s)"
        )
        print(_render(prompt_bytes + spec_out))

    _print_section("Comparison")
    print(f"{'Method':<22} {'Tokens':>8} {'Time(s)':>8} {'Tok/s':>8}")
    print(f"{'─'*22} {'─'*8} {'─'*8} {'─'*8}")
    print(f"{'Autoregressive':<22} {args.n_tokens:>8} {ar_elapsed:>8.2f} {ar_tps:>8.1f}")
    if not args.no_speculative:
        speedup = spec_tps / max(ar_tps, 1e-9)
        print(f"{'Speculative':<22} {args.n_tokens:>8} {spec_elapsed:>8.2f} "
              f"{spec_tps:>8.1f}  ({speedup:.2f}× speedup, {accept_rate:.0%} accept)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

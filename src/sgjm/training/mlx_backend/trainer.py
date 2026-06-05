from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from sgjm.training.backends import ResolvedBackend
from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset, load_corpus
from sgjm.training.mlx_backend.losses import compute_losses
from sgjm.training.mlx_backend.model import SGJM


@dataclass
class TrainResult:
    model: nn.Module
    final_step: int
    best_eval: float
    checkpoint_path: Path | None


def _cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _to_array(xs: list[list[int]]) -> mx.array:
    return mx.array(xs, dtype=mx.int32)


def evaluate(
    model: SGJM,
    cfg: TrainingConfig,
    dataset: ByteDataset,
    rng: random.Random,
    n_batches: int,
) -> dict[str, float]:
    sums: dict[str, float] = {}
    model.eval()
    for _ in range(n_batches):
        xs, ys = dataset.batch(cfg.optim.batch_size, rng)
        x, y = _to_array(xs), _to_array(ys)
        _, parts = compute_losses(model, x, y, cfg)
        mx.eval(parts)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
    model.train()
    return {k: v / n_batches for k, v in sums.items()}


def _baseline_loss(
    model: nn.Module,
    x: mx.array,
    y: mx.array,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Compute token cross-entropy loss for BaselineLM."""
    _, logits = model(x)
    V = logits.shape[-1]
    token_loss = nn.losses.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1), reduction="mean"
    )
    return token_loss, {"total": token_loss, "token": token_loss}


def _evaluate_baseline(
    model: nn.Module,
    cfg: TrainingConfig,
    dataset: ByteDataset,
    rng: random.Random,
    n_batches: int,
) -> dict[str, float]:
    sums: dict[str, float] = {}
    model.eval()
    for _ in range(n_batches):
        xs, ys = dataset.batch(cfg.optim.batch_size, rng)
        x, y = _to_array(xs), _to_array(ys)
        _, parts = _baseline_loss(model, x, y)
        mx.eval(parts)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
    model.train()
    return {k: v / n_batches for k, v in sums.items()}


def train(
    cfg: TrainingConfig,
    backend: ResolvedBackend,
    progress: callable | None = None,
) -> TrainResult:
    if backend != "mlx":
        raise ValueError(f"mlx trainer cannot run on backend {backend!r}")

    mx.random.seed(cfg.seed)
    train_rng = random.Random(cfg.seed)
    eval_rng = random.Random(cfg.seed + 1)

    corpus = load_corpus(cfg.data_path, cfg.corpus_bytes, seed=cfg.seed, source=cfg.data_source)
    split = int(0.95 * len(corpus))
    train_set = ByteDataset(corpus[:split], cfg.optim.seq_len)
    eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)

    arch = cfg.arch
    if arch == "baseline":
        from sgjm.training.mlx_backend.baseline import BaselineLM
        model: nn.Module = BaselineLM(cfg.model)
        n_params = model.num_parameters()
        print(f"[baseline] backend=mlx params={n_params/1e6:.2f}M")
    else:
        model = SGJM(cfg.model)
        n_params = model.num_parameters()
        print(f"[sgjm] backend=mlx params={n_params/1e6:.2f}M")

    mx.eval(model.parameters())

    optimizer = optim.AdamW(
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )

    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "config.json")
    log_file = (out_dir / "train.jsonl").open("a", buffering=1)

    if arch == "baseline":
        def loss_fn(
            m: nn.Module, x: mx.array, y: mx.array
        ) -> tuple[mx.array, dict[str, mx.array]]:
            return _baseline_loss(m, x, y)
    else:
        def loss_fn(
            m: nn.Module, x: mx.array, y: mx.array
        ) -> tuple[mx.array, dict[str, mx.array]]:
            return compute_losses(m, x, y, cfg)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def _save(step: int, name: str) -> Path:
        path = out_dir / f"{name}.safetensors"
        flat = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(path), flat)
        (out_dir / f"{name}.meta.json").write_text(
            json.dumps({"step": step, "config": cfg.to_dict()}, indent=2)
        )
        return path

    best_eval = float("inf")
    best_path: Path | None = None
    last_path: Path | None = None
    model.train()
    t0 = time.time()

    for step in range(cfg.optim.max_steps):
        lr = _cosine_lr(step, cfg.optim.warmup_steps, cfg.optim.max_steps, cfg.optim.lr)
        optimizer.learning_rate = lr

        xs, ys = train_set.batch(cfg.optim.batch_size, train_rng)
        x, y = _to_array(xs), _to_array(ys)

        (total, parts), grads = loss_and_grad(model, x, y)

        gnorm_val = float("nan")
        bad_grad = False
        if cfg.optim.grad_clip and cfg.optim.grad_clip > 0:
            grads, gnorm = optim.clip_grad_norm(grads, cfg.optim.grad_clip)
            mx.eval(gnorm)
            gnorm_val = float(gnorm)
            bad_grad = not math.isfinite(gnorm_val)

        if not bad_grad:
            optimizer.update(model, grads)

        mx.eval(model.parameters(), optimizer.state, parts)

        if step % cfg.log_every == 0 or step == cfg.optim.max_steps - 1:
            entry: dict[str, object] = {
                "step": step,
                "lr": lr,
                "gnorm": gnorm_val,
                "skip": bad_grad,
                "elapsed": time.time() - t0,
                **{k: float(v) for k, v in parts.items()},
            }
            log_file.write(json.dumps(entry) + "\n")
            if arch == "baseline":
                print(
                    f"[baseline] step={step:>6} lr={lr:.2e} "
                    f"total={float(parts['total']):.4f} "
                    f"tok={float(parts['token']):.4f}"
                )
            else:
                skip_tag = " SKIP" if bad_grad else ""
                print(
                    f"[sgjm] step={step:>6} lr={lr:.2e} gnorm={gnorm_val:.3e}{skip_tag} "
                    f"total={float(parts['total']):.4f} "
                    f"tok={float(parts['token']):.4f} "
                    f"draft={float(parts['drafter']):.4f} "
                    f"jepa={float(parts['jepa']):.4f} "
                    f"ver={float(parts['verifier']):.4f} "
                    f"acc={float(parts['accept_acc']):.3f}"
                )
            if progress is not None:
                progress(step, entry)

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            if arch == "baseline":
                eval_metrics = _evaluate_baseline(model, cfg, eval_set, eval_rng, cfg.optim.eval_batches)
            else:
                eval_metrics = evaluate(model, cfg, eval_set, eval_rng, cfg.optim.eval_batches)
            log_file.write(json.dumps({"step": step, "eval": eval_metrics}) + "\n")
            print(f"[{arch}] eval@{step}: {eval_metrics}")
            if eval_metrics["total"] < best_eval:
                best_eval = eval_metrics["total"]
                best_path = _save(step, "best")

        if cfg.checkpoint_every and step > 0 and step % cfg.checkpoint_every == 0:
            last_path = _save(step, "last")

    last_path = _save(cfg.optim.max_steps - 1, "final")
    log_file.close()
    return TrainResult(
        model=model,
        final_step=cfg.optim.max_steps - 1,
        best_eval=best_eval,
        checkpoint_path=best_path or last_path,
    )


def load_weights(model: SGJM, path: str | Path) -> None:
    weights = mx.load(str(path))
    model.update(tree_unflatten(list(weights.items())))

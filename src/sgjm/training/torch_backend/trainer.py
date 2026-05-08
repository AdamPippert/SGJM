from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from sgjm.training.backends import ResolvedBackend, torch_device
from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset, load_corpus
from sgjm.training.torch_backend.losses import compute_losses
from sgjm.training.torch_backend.model import SGJM


@dataclass
class TrainResult:
    model: SGJM
    final_step: int
    best_eval: float
    checkpoint_path: Path | None


def _amp_dtype(cfg: TrainingConfig, backend: ResolvedBackend) -> torch.dtype | None:
    mode = cfg.amp
    if mode == "off":
        return None
    if backend == "cpu":
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    # auto
    if backend == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if backend == "rocm":
        return torch.bfloat16
    return None


def _cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _to_device(
    xs: list[list[int]],
    ys: list[list[int]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(
    model: SGJM,
    cfg: TrainingConfig,
    dataset: ByteDataset,
    rng: random.Random,
    device: str,
    n_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    for _ in range(n_batches):
        xs, ys = dataset.batch(cfg.optim.batch_size, rng)
        x, y = _to_device(xs, ys, device)
        _, parts = compute_losses(model, (x, y), cfg)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
    if was_training:
        model.train()
    return {k: v / n_batches for k, v in sums.items()}


def train(
    cfg: TrainingConfig,
    backend: ResolvedBackend,
    progress: callable | None = None,
) -> TrainResult:
    if backend not in ("cuda", "rocm", "cpu"):
        raise ValueError(f"torch trainer cannot run on backend {backend!r}")

    device = torch_device(backend)
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    train_rng = random.Random(cfg.seed)
    eval_rng = random.Random(cfg.seed + 1)

    corpus = load_corpus(cfg.data_path, cfg.corpus_bytes, seed=cfg.seed)
    split = int(0.95 * len(corpus))
    train_set = ByteDataset(corpus[:split], cfg.optim.seq_len)
    eval_set = ByteDataset(corpus[split:], cfg.optim.seq_len)

    model = SGJM(cfg.model).to(device)
    breakdown = model.param_breakdown()
    total_params = sum(breakdown.values())
    print(
        f"[sgjm] backend={backend} device={device} "
        f"params={total_params/1e6:.2f}M "
        + " ".join(f"{k}={v/1e6:.2f}M" for k, v in breakdown.items())
    )

    if cfg.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )

    amp_dtype = _amp_dtype(cfg, backend)
    use_grad_scaler = amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "config.json")
    log_path = out_dir / "train.jsonl"
    log_file = log_path.open("a", buffering=1)

    def _save(step: int, name: str) -> Path:
        path = out_dir / f"{name}.pt"
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
            },
            path,
        )
        return path

    best_eval = float("inf")
    best_path: Path | None = None
    last_path: Path | None = None
    model.train()
    t0 = time.time()

    for step in range(cfg.optim.max_steps):
        lr = _cosine_lr(step, cfg.optim.warmup_steps, cfg.optim.max_steps, cfg.optim.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xs, ys = train_set.batch(cfg.optim.batch_size, train_rng)
        x, y = _to_device(xs, ys, device)

        optimizer.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                total, parts = compute_losses(model, (x, y), cfg)
        else:
            total, parts = compute_losses(model, (x, y), cfg)

        if use_grad_scaler:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()

        if step % cfg.log_every == 0 or step == cfg.optim.max_steps - 1:
            entry = {
                "step": step,
                "lr": lr,
                "elapsed": time.time() - t0,
                **{k: float(v) for k, v in parts.items()},
            }
            log_file.write(json.dumps(entry) + "\n")
            print(
                f"[sgjm] step={step:>6} lr={lr:.2e} "
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
            eval_metrics = evaluate(
                model, cfg, eval_set, eval_rng, device, cfg.optim.eval_batches
            )
            entry = {"step": step, "eval": eval_metrics}
            log_file.write(json.dumps(entry) + "\n")
            print(f"[sgjm] eval@{step}: {eval_metrics}")
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

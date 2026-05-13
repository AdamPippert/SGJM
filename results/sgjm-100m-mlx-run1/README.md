# SGJM-100M — MLX Training Run 1

**Date**: 2026-05-13  
**Host**: MacBook Pro (Apple Silicon, arm64)  
**Backend**: MLX 0.29.1 / Python 3.12  
**Seed**: 42

## Model Architecture

| Component | Config |
|-----------|--------|
| Backbone d_model | 768 |
| Backbone layers | 9 |
| Backbone heads | 12 |
| Backbone d_ff | 3 072 |
| Drafter d_model | 384 |
| Drafter layers | 2 |
| Drafter heads | 6 |
| Drafter d_ff | 1 536 |
| Judge hidden | 1 024 |
| Verifier hidden | 512 |
| Vocab size | 256 (byte-level) |
| Max seq len | 1 024 |
| Block size | 4 |
| Tied embeddings | yes |
| **Est. total params** | **~93M** |

## Training Config

| Param | Value |
|-------|-------|
| Steps | 5 000 |
| Batch size | 4 |
| Seq len | 512 |
| LR | 1.5e-4 (cosine decay) |
| Warmup steps | 1 000 |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Optimizer | AdamW (β=0.9, 0.95) |
| Data source | auto (TinyShakespeare) |
| Corpus bytes | 1 MiB |

## Results

**Training duration**: 55.4 minutes  
**Best checkpoint**: step 4500 (`best.safetensors`)

### Training Loss (first/last)

| Metric | Step 0 | Step 4999 |
|--------|--------|-----------|
| Total loss | 9.2877 | 0.1756 |
| Token loss | 6.0675 | 0.0315 |
| Accept accuracy | 50.6% | 99.8% |

### Eval Loss (held-out set, every 500 steps)

| Step | Total | Token | Accept Acc |
|------|-------|-------|------------|
| 500 | 6.9780 | 4.0188 | 85.5% |
| 1 000 | 2.3378 | 0.4299 | 92.9% |
| 1 500 | 1.0645 | 0.1426 | 98.8% |
| 2 000 | 0.3877 | 0.0813 | 99.5% |
| 2 500 | 0.3009 | 0.0603 | 99.3% |
| 3 000 | 0.2294 | 0.0380 | 99.8% |
| 3 500 | 0.1930 | 0.0288 | 99.9% |
| 4 000 | 0.1764 | 0.0270 | 99.8% |
| **4 500** | **0.1666** | **0.0241** | **99.9%** |

Best eval total loss: **0.1666** at step 4500.  
vs 25M best: 0.1790 — **100M improves by 0.0124 nats (6.9%)**.

## Comparison vs 25M

| | 25M | 100M | Delta |
|--|-----|------|-------|
| Params | ~25M | ~93M | +272% |
| Training time | 27.3 min | 55.4 min | +103% |
| Best eval token NLL | 0.0254 | 0.0241 | −5.1% |
| Best eval total loss | 0.1790 | 0.1666 | −6.9% |
| Final accept acc | 99.8% | 99.8% | = |

Scaling from 25M to 100M parameters yields a 6.9% reduction in total eval loss with 2× training time — a favorable scaling return on Apple Silicon.

## Artifacts

| File | Description |
|------|-------------|
| `config.json` | Full resolved TrainingConfig |
| `train.jsonl` | Per-step training log + eval entries |
| `best.safetensors` | Weights at step 4500 — not in git |
| `final.safetensors` | Weights at step 4999 — not in git |

To reproduce: `python -m sgjm.training --size 100m --backend mlx --steps 5000 --seed 42`

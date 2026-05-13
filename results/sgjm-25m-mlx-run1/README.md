# SGJM-25M — MLX Training Run 1

**Date**: 2026-05-13  
**Host**: MacBook Pro (Apple Silicon, arm64)  
**Backend**: MLX 0.29.1 / Python 3.12  
**Seed**: 42

## Model Architecture

| Component | Config |
|-----------|--------|
| Backbone d_model | 384 |
| Backbone layers | 10 |
| Backbone heads | 6 |
| Backbone d_ff | 1536 |
| Drafter d_model | 192 |
| Drafter layers | 2 |
| Drafter heads | 4 |
| Drafter d_ff | 768 |
| Judge hidden | 512 |
| Verifier hidden | 256 |
| Vocab size | 256 (byte-level) |
| Max seq len | 512 |
| Block size | 4 |
| Tied embeddings | yes |

## Training Config

| Param | Value |
|-------|-------|
| Steps | 5000 |
| Batch size | 16 |
| Seq len | 256 |
| LR | 3e-4 (cosine decay) |
| Warmup steps | 200 |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Optimizer | AdamW (β=0.9,0.95) |
| Data source | auto (TinyShakespeare / synthetic) |
| Corpus bytes | 1 MiB |
| AMP | auto |

## Loss Weights

| Component | Weight |
|-----------|--------|
| Token (LM) | 1.0 |
| Drafter | 0.5 |
| JEPA | 0.25 |
| Verifier | 0.1 |

## Results

**Training duration**: 27.3 minutes  
**Best checkpoint**: step 4500 (`best.safetensors`)

### Training Loss (every 25 steps, first/last)

| Metric | Step 0 | Step 4999 |
|--------|--------|-----------|
| Total loss | 9.2762 | 0.1795 |
| Token loss | 6.0495 | 0.0242 |
| Drafter loss | 5.7154 | 0.0328 |
| JEPA loss | 1.1972 | 0.5381 |
| Verifier loss | 0.6972 | 0.0433 |
| Accept accuracy | 52.3% | 98.1% |

### Eval Loss (16-batch held-out set, every 500 steps)

| Step | Total | Token | Accept Acc |
|------|-------|-------|------------|
| 500 | 2.0528 | 0.2775 | 94.0% |
| 1000 | 0.4720 | 0.0972 | 98.9% |
| 1500 | 0.3474 | 0.0641 | 99.3% |
| 2000 | 0.2927 | 0.0512 | 99.4% |
| 2500 | 0.2553 | 0.0422 | 99.6% |
| 3000 | 0.2192 | 0.0332 | 99.6% |
| 3500 | 0.1986 | 0.0295 | 99.8% |
| 4000 | 0.1845 | 0.0265 | 99.8% |
| 4500 | **0.1790** | **0.0254** | **99.8%** |

Best eval total loss: **0.1790** at step 4500.

## Artifacts

| File | Description |
|------|-------------|
| `config.json` | Full resolved TrainingConfig |
| `train.jsonl` | Per-step training log + eval entries |
| `best.safetensors` | Weights at best eval loss (step 4500) — not in git |
| `final.safetensors` | Weights at step 4999 — not in git |

Weights are excluded from the repository (`.gitignore: *.safetensors`).
To reproduce: `python -m sgjm.training --size 25m --backend mlx --seed 42`

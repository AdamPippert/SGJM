"""Token generation helpers used by the demo CLI and tests."""
from __future__ import annotations

import mlx.core as mx

from sgjm.training.config import ModelConfig
from sgjm.training.mlx_backend.model import SGJM


def generate_completion(
    model: SGJM,
    cfg: ModelConfig,
    prompt: bytes,
    n_tokens: int = 128,
    temperature: float = 0.8,
    seed: int = 42,
) -> bytes:
    """Autoregressive greedy/sampled generation from a byte prompt."""
    tokens = list(prompt)
    mx.random.seed(seed)
    model.eval()
    for _ in range(n_tokens):
        idx = mx.array([tokens[-(cfg.max_seq_len):]], dtype=mx.int32)
        _, logits = model.backbone(idx)
        last = logits[0, -1]
        if temperature <= 0.0:
            next_tok = int(mx.argmax(last).item())
        else:
            next_tok = int(mx.random.categorical((last / temperature)[None]).item())
        tokens.append(next_tok)
    return bytes(tokens[len(prompt):])


def generate_speculative(
    model: SGJM,
    cfg: ModelConfig,
    prompt: bytes,
    n_tokens: int = 128,
    k_branches: int = 4,
    seed: int = 42,
) -> tuple[bytes, float]:
    """Speculative generation using the drafter + verifier.

    Returns (generated_bytes, acceptance_rate).
    """
    from sgjm.bench.mlx_bench import (
        MLXBackboneAdapter,
        MLXDrafterAdapter,
        MLXJudgeAdapter,
        run_sgjm_bench,
    )

    prompt_tokens = list(prompt)
    n_steps = max(1, n_tokens // cfg.block_size)
    result = run_sgjm_bench(model, cfg, prompt_tokens, n_steps=n_steps)
    # Re-run AR to get actual bytes (run_sgjm_bench only measures throughput)
    tokens = list(prompt)
    mx.random.seed(seed)
    model.eval()
    accepted = 0
    total = 0
    while len(tokens) - len(prompt) < n_tokens:
        idx = mx.array([tokens[-(cfg.max_seq_len):]], dtype=mx.int32)
        hidden, _ = model.backbone(idx)
        draft_logits, draft_latents = model.drafter(hidden)
        # Take best branch: sample from each block position
        pos_logits = draft_logits[0, -1]  # (block_size, V)
        endpoint = draft_latents[0, -1, -1]  # (D,)
        # Verifier score for this draft
        parent_h = hidden[0:1, -1:, :]  # (1,1,D)
        child_h = endpoint[None, None, :]  # (1,1,D)
        v_score = model.verifier(parent_h, child_h)
        mx.eval(v_score, pos_logits)
        accepted_branch = float(v_score.item()) > 0
        total += 1
        if accepted_branch:
            accepted += 1
            for step in range(cfg.block_size):
                if len(tokens) - len(prompt) >= n_tokens:
                    break
                tok = int(mx.argmax(pos_logits[step]).item())
                tokens.append(tok)
        else:
            # Fall back: one greedy token from backbone
            _, logits = model.backbone(idx)
            tok = int(mx.argmax(logits[0, -1]).item())
            tokens.append(tok)

    accept_rate = accepted / max(1, total)
    return bytes(tokens[len(prompt):n_tokens + len(prompt)]), accept_rate

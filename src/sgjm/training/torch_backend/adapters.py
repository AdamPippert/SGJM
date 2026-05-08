from __future__ import annotations

from dataclasses import dataclass

import torch

from sgjm.modules.backbone import BackboneState
from sgjm.modules.drafter import DraftSample
from sgjm.training.torch_backend.model import SGJM


@dataclass
class TorchBackboneAdapter:
    model: SGJM
    device: str = "cpu"

    @property
    def latent_dim(self) -> int:
        return self.model.cfg.d_model

    @torch.no_grad()
    def encode(self, tokens) -> BackboneState:
        toks = tuple(int(t) for t in tokens)
        if not toks:
            zero = (0.0,) * self.latent_dim
            return BackboneState(tokens=(), latent=zero)
        idx = torch.tensor([list(toks)], dtype=torch.long, device=self.device)
        hidden, _ = self.model.backbone(idx)
        last = hidden[0, -1].float().cpu().tolist()
        return BackboneState(tokens=toks, latent=tuple(last))

    def step(self, state: BackboneState, token: int) -> BackboneState:
        return self.encode(state.tokens + (int(token),))


@dataclass
class TorchDrafterAdapter:
    model: SGJM
    device: str = "cpu"
    temperature: float = 1.0

    @torch.no_grad()
    def draft(self, state: BackboneState, *, k: int, block: int) -> tuple[DraftSample, ...]:
        if block != self.model.cfg.block_size:
            raise ValueError(
                f"drafter trained for block={self.model.cfg.block_size}, requested {block}"
            )
        if not state.tokens:
            return ()
        idx = torch.tensor([list(state.tokens)], dtype=torch.long, device=self.device)
        hidden, _ = self.model.backbone(idx)
        parent = hidden[:, -1:]
        logits, latents = self.model.drafter(parent)
        logits = logits[0, 0].float()
        latents = latents[0, 0].float()
        samples: list[DraftSample] = []
        for _ in range(k):
            tokens: list[int] = []
            log_prob = 0.0
            for t in range(block):
                probs = torch.softmax(logits[t] / max(self.temperature, 1e-6), dim=-1)
                tok = int(torch.multinomial(probs, 1).item())
                tokens.append(tok)
                log_prob += float(torch.log(probs[tok] + 1e-9))
            child_latent = tuple(latents[-1].cpu().tolist())
            samples.append(DraftSample(
                tokens=tuple(tokens),
                latent=child_latent,
                log_prob=log_prob,
            ))
        return tuple(samples)


@dataclass
class TorchJudgeAdapter:
    model: SGJM
    device: str = "cpu"

    @torch.no_grad()
    def score(self, parent_latent, child_latent) -> float:
        parent = torch.tensor(parent_latent, dtype=torch.float32, device=self.device)[None, None]
        child = torch.tensor(child_latent, dtype=torch.float32, device=self.device)[None, None]
        pred = self.model.judge(parent)
        return -float(((pred - child) ** 2).mean())


def bundle_for_harness(
    model: SGJM,
    device: str = "cpu",
    temperature: float = 1.0,
) -> tuple[TorchBackboneAdapter, TorchDrafterAdapter, TorchJudgeAdapter]:
    model.eval()
    return (
        TorchBackboneAdapter(model=model, device=device),
        TorchDrafterAdapter(model=model, device=device, temperature=temperature),
        TorchJudgeAdapter(model=model, device=device),
    )

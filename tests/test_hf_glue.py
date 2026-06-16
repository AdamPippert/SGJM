"""Tests for the HuggingFace publishing glue (sgjm.hf).

Skipped unless the `hf` extra (transformers + safetensors) is installed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from sgjm.hf.configuration_sgjm import MODEL_FIELDS, SGJMConfig  # noqa: E402
from sgjm.hf.convert import convert_checkpoint  # noqa: E402
from sgjm.hf.modeling_sgjm import SGJMForCausalLM  # noqa: E402
from sgjm.training.config import TrainingConfig  # noqa: E402
from sgjm.training.torch_backend.model import SGJM  # noqa: E402

HF_DIR = Path(__file__).resolve().parent.parent / "src" / "sgjm" / "hf"


def _small(**over) -> SGJMConfig:
    base = dict(
        d_model=64, n_layers=4, n_heads=4, d_ff=128, max_seq_len=64, block_size=2,
        drafter_layers=1, drafter_d_model=32, drafter_heads=4, drafter_d_ff=64,
        judge_hidden=64, verifier_hidden=64,
    )
    base.update(over)
    return SGJMConfig(**base)


def test_config_roundtrips_to_model_config():
    cfg = _small(attn_every_n=2, mamba_state_size=16, mamba_head_dim=16)
    mc = cfg.to_model_config()
    for f in MODEL_FIELDS:
        assert getattr(cfg, f) == getattr(mc, f)
    assert cfg.backbone == "hybrid-mamba2-attention"
    assert _small(attn_every_n=0).backbone == "transformer"


@pytest.mark.parametrize("over", [
    {"attn_every_n": 0},                                            # pure transformer
    {"attn_every_n": 2, "mamba_state_size": 16, "mamba_head_dim": 16},  # hybrid
])
def test_forward_loss_and_generate(over):
    torch.manual_seed(0)
    model = SGJMForCausalLM(_small(**over)).eval()
    ids = torch.randint(0, 256, (2, 12))

    out = model(input_ids=ids)
    assert out.logits.shape == (2, 12, 256)

    loss = model(input_ids=ids, labels=ids.clone()).loss
    assert torch.isfinite(loss)

    gen = model.generate(ids[:, :3], max_new_tokens=5, do_sample=False)
    assert gen.shape == (2, 8)


@pytest.mark.parametrize("over", [
    {"attn_every_n": 0},
    {"attn_every_n": 2, "mamba_state_size": 16, "mamba_head_dim": 16},
])
def test_convert_and_reload_is_bit_identical(over):
    torch.manual_seed(0)
    cfg = _small(**over)
    model = SGJMForCausalLM(cfg).eval()
    ids = torch.randint(0, 256, (2, 12))

    tc = TrainingConfig.from_dict(
        {"arch": "sgjm", "model": {f: getattr(cfg, f) for f in MODEL_FIELDS}}
    )
    ref = SGJM(tc.model).eval()
    ref.load_state_dict(model.model.state_dict())

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ckpt = td / "best.pt"
        torch.save({"arch": "sgjm", "config": tc.to_dict(),
                    "model": ref.state_dict(), "step": 1}, ckpt)
        out = td / "repo"
        convert_checkpoint(ckpt, out)
        # publish.py stages these two files into the repo for trust_remote_code
        for name in ("configuration_sgjm.py", "modeling_sgjm.py"):
            (out / name).write_bytes((HF_DIR / name).read_bytes())

        assert AutoConfig.from_pretrained(out, trust_remote_code=True).model_type == "sgjm"
        loaded = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True).eval()

    with torch.no_grad():
        delta = (model(input_ids=ids).logits - loaded(input_ids=ids).logits).abs().max()
    assert float(delta) < 1e-5
    # tie must survive the round-trip
    bb = loaded.model.backbone
    assert bb.lm_head.weight.data_ptr() == bb.tok_emb.weight.data_ptr()

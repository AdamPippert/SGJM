"""Convert a trained SGJM checkpoint into a HuggingFace model directory.

Reads a torch training checkpoint (``{"model": state_dict, "config": ...}``) and
writes ``config.json`` + ``model.safetensors`` with the key layout the
``SGJMForCausalLM`` wrapper expects (everything under the ``model.`` prefix).

    python -m sgjm.hf.convert runs/sgjm-25m/best.pt out/SGJM-25M

The release pipeline (cdr-hf-releases/publish.py --weights) calls this so the HF
repo ends up with weights that load via ``from_pretrained(..., trust_remote_code=True)``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sgjm.hf.configuration_sgjm import SGJMConfig

AUTO_MAP = {
    "AutoConfig": "configuration_sgjm.SGJMConfig",
    "AutoModelForCausalLM": "modeling_sgjm.SGJMForCausalLM",
}


def convert_checkpoint(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    import torch
    from safetensors.torch import save_file

    from sgjm.training.config import TrainingConfig

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise SystemExit(f"{ckpt_path}: not an SGJM training checkpoint "
                         "(expected 'model' and 'config' keys)")
    train_cfg = TrainingConfig.from_dict(ckpt["config"])
    if ckpt.get("arch", train_cfg.arch) != "sgjm":
        raise SystemExit(f"{ckpt_path}: arch is not 'sgjm'; nothing to publish")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Prefix every key to live under SGJMForCausalLM.model, drop the tied head.
    state = {f"model.{k}": v.contiguous() for k, v in ckpt["model"].items()}
    if train_cfg.model.tie_embeddings:
        state.pop("model.backbone.lm_head.weight", None)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})

    cfg = SGJMConfig.from_model_config(train_cfg.model)
    cfg.architectures = ["SGJMForCausalLM"]
    cfg.auto_map = AUTO_MAP
    cfg.save_pretrained(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="path to a torch SGJM checkpoint (.pt)")
    ap.add_argument("out_dir", help="output directory for config.json + model.safetensors")
    args = ap.parse_args()
    dest = convert_checkpoint(args.checkpoint, args.out_dir)
    print(f"wrote {dest}/config.json and {dest}/model.safetensors")


if __name__ == "__main__":
    main()

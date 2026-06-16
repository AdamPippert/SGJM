"""HuggingFace `trust_remote_code` glue for the SGJM family.

The two files published into each HF model repo are `configuration_sgjm.py` and
`modeling_sgjm.py`; `convert.py` turns a training checkpoint into a loadable
HF directory. Importing the model classes requires `transformers` (the `hf`
optional extra); the config alone does too, but nothing here pulls in a backend.
"""
from __future__ import annotations

from sgjm.hf.configuration_sgjm import SGJMConfig

__all__ = ["SGJMConfig", "SGJMForCausalLM", "convert_checkpoint"]


def __getattr__(name: str):
    # Lazy so `import sgjm.hf` works for config-only use without torch loaded.
    if name == "SGJMForCausalLM":
        from sgjm.hf.modeling_sgjm import SGJMForCausalLM

        return SGJMForCausalLM
    if name == "convert_checkpoint":
        from sgjm.hf.convert import convert_checkpoint

        return convert_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

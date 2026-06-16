"""HuggingFace `AutoModelForCausalLM` wrapper for SGJM.

This adapter is deliberately thin: the architecture lives once, in the library
(`sgjm.training.torch_backend.model.SGJM`), and this wrapper exposes its backbone
through the standard `transformers` generation interface. The speculative drafter,
JEPA judge, and verifier are still constructed (so full checkpoints load cleanly)
but are not part of the plain autoregressive `generate` path.

Loading published weights therefore requires the architecture package:

    pip install "sgjm @ git+https://github.com/AdamPippert/SGJM"

This keeps a single source of truth for the model definition — the HuggingFace
repo carries weights + this adapter + a card, not a second copy of the network.
"""
from __future__ import annotations

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_sgjm import SGJMConfig

_INSTALL_HINT = (
    "SGJM checkpoints need the architecture package. Install it with:\n"
    '    pip install "sgjm @ git+https://github.com/AdamPippert/SGJM"'
)


class SGJMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SGJMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_cache_class = False
    # lm_head is tied to the token embedding inside the library model ({target: source}).
    _tied_weights_keys = {"model.backbone.lm_head.weight": "model.backbone.tok_emb.weight"}

    def __init__(self, config: SGJMConfig) -> None:
        super().__init__(config)
        try:
            from sgjm.training.torch_backend.model import SGJM
        except ImportError as exc:  # pragma: no cover - exercised only without the lib
            raise ImportError(_INSTALL_HINT) from exc
        self.model = SGJM(config.to_model_config())
        # Sets up tie tracking / static properties HF needs at load time.
        self.post_init()

    def _init_weights(self, module):
        # Defer to the library's own initializer so the wrapper inits identically.
        from sgjm.training.torch_backend.model import SGJM

        SGJM._init_weights(module)

    # --- embedding plumbing (HF tie/resize hooks) ---------------------------
    def get_input_embeddings(self):
        return self.model.backbone.tok_emb

    def set_input_embeddings(self, value):
        self.model.backbone.tok_emb = value

    def get_output_embeddings(self):
        return self.model.backbone.lm_head

    def set_output_embeddings(self, value):
        self.model.backbone.lm_head = value

    # --- forward / generation ----------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = True if return_dict is None else return_dict
        _hidden, logits = self.model.backbone(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # No KV cache: feed the (truncated) context each step.
        max_len = self.config.max_seq_len
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, -max_len:]
        return {"input_ids": input_ids}

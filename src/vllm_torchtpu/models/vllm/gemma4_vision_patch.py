# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Routes Gemma-4's HuggingFace vision encoder attention through MMEncoderAttention.

In vLLM v0.33.0, ``Gemma4ForConditionalGeneration`` instantiates HuggingFace's
``Gemma4VisionModel`` via ``AutoModel.from_config``, whose ``Gemma4VisionAttention``
dispatches through HuggingFace's ``ALL_ATTENTION_FUNCTIONS`` with a 4D bidirectional
padding mask rather than vLLM's ``MMEncoderAttention``. That bypasses
``--mm-encoder-attn-backend`` and incurs a host-device sync inside
``create_bidirectional_mask``.

TODO(https://github.com/vllm-project/vllm/pull/65431): Remove this patch once
vLLM's native packed ``Gemma4VisionModel`` on ``MMEncoderAttention`` lands and
``vllm-torchtpu`` advances its vLLM pin.
"""

import functools
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_GEMMA4_VISION_MODEL_TYPE = "gemma4_vision"


def _has_gemma4_vision_tower(model_config: "ModelConfig | None") -> bool:
    if model_config is None:
        return False
    hf_config = model_config.hf_config
    vision_config = getattr(hf_config, "vision_config", None)
    if vision_config is None:
        return False
    return vision_config.model_type == _GEMMA4_VISION_MODEL_TYPE


def build_padded_row_cu_seqlens(
    bsz: int,
    seq_len: int,
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build static-shape ``cu_seqlens`` isolating each row's valid patch prefix.

    Gemma-4 vision inputs have shape ``[B, S, D]`` where each row ``b`` holds
    ``valid_b`` contiguous valid patches followed by ``S - valid_b`` padding
    patches (``pixel_position_ids == -1``). Interleaving ``valid_ends`` and
    ``row_ends`` into a fixed-length ``[2 * B + 1]`` ``cu_seqlens`` vector
    isolates every image's valid prefix from both its padding suffix and other
    images with zero dynamic shapes or gather/scatter copies under
    ``torch.compile(fullgraph=True, dynamic=False)``.
    """
    if attention_mask is None:
        cu_seqlens = torch.arange(
            0,
            (bsz + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=device,
        )
        return cu_seqlens, None

    if attention_mask.ndim == 2:
        valid_mask = attention_mask.to(torch.bool)
    elif attention_mask.ndim == 4:
        # 4D bidirectional mask [B, 1, S, S]: bool (True=valid) or additive float
        # (0.0=valid, -inf=masked). Extract the per-key validity from row 0.
        key_mask = attention_mask[:, 0, 0, :]
        valid_mask = (
            key_mask if key_mask.dtype == torch.bool else torch.isfinite(key_mask)
        )
    else:
        raise ValueError(
            f"[gemma4_vision_patch] Expected 2D or 4D attention_mask, "
            f"got shape {tuple(attention_mask.shape)}."
        )

    valid_lens = valid_mask.sum(dim=-1, dtype=torch.int32)
    row_starts = torch.arange(
        0, bsz * seq_len, seq_len, dtype=torch.int32, device=device
    )
    valid_ends = row_starts + valid_lens
    row_ends = row_starts + seq_len
    pairs = torch.stack((valid_ends, row_ends), dim=-1).reshape(-1)
    cu_seqlens = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), pairs),
        dim=0,
    )
    return cu_seqlens, valid_mask


def install_gemma4_vision_patches(modeling_gemma4: Any) -> None:
    """Patch HF ``Gemma4VisionEncoder`` and ``Gemma4VisionAttention`` in place.

    Adapts ``Gemma4VisionEncoder.forward`` and ``Gemma4VisionAttention.forward``
    from ``transformers.models.gemma4.modeling_gemma4`` (``transformers>=5.18.0``)
    to skip ``create_bidirectional_mask`` and ``ALL_ATTENTION_FUNCTIONS`` when
    ``--mm-encoder-attn-backend`` is not ``TORCH_SDPA``.
    """
    attn_cls = modeling_gemma4.Gemma4VisionAttention
    encoder_cls = modeling_gemma4.Gemma4VisionEncoder
    apply_multidim_rope = modeling_gemma4.apply_multidimensional_rope
    base_output_cls = modeling_gemma4.BaseModelOutputWithPast

    if getattr(attn_cls, "_tpu_mm_encoder_attn_patched", False):
        return

    orig_attn_init = attn_cls.__init__
    orig_attn_forward = attn_cls.forward
    orig_encoder_forward = encoder_cls.forward

    @functools.wraps(orig_attn_init)
    def patched_attn_init(self: Any, config: Any, layer_idx: int) -> None:
        orig_attn_init(self, config, layer_idx)
        self._tpu_mm_attn = MMEncoderAttention(
            num_heads=config.num_attention_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=config.num_key_value_heads,
            prefix=f"vision_tower.encoder.layers.{layer_idx}.self_attn.attn",
        )

    @functools.wraps(orig_encoder_forward)
    def patched_encoder_forward(
        self: Any,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if (
            len(self.layers) == 0
            or self.layers[0].self_attn._tpu_mm_attn.attn_backend
            == AttentionBackendEnum.TORCH_SDPA
        ):
            return orig_encoder_forward(
                self,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_position_ids=pixel_position_ids,
                **kwargs,
            )

        bsz, seq_len = inputs_embeds.shape[:2]
        cu_seqlens, valid_mask = build_padded_row_cu_seqlens(
            bsz, seq_len, attention_mask, inputs_embeds.device
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                _tpu_cu_seqlens=cu_seqlens,
                _tpu_valid_mask=valid_mask,
                **kwargs,
            )
        return base_output_cls(last_hidden_state=hidden_states)

    @functools.wraps(orig_attn_forward)
    def patched_attn_forward(
        self: Any,
        hidden_states: torch.Tensor,
        position_embeddings: Any = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cu_seqlens = kwargs.pop("_tpu_cu_seqlens", None)
        valid_mask = kwargs.pop("_tpu_valid_mask", None)
        mm_attn = self._tpu_mm_attn
        if (
            cu_seqlens is None
            or mm_attn.attn_backend == AttentionBackendEnum.TORCH_SDPA
        ):
            return orig_attn_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        bsz, seq_len = hidden_states.shape[:2]
        hidden_shape = (bsz, seq_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query_states = apply_multidim_rope(query_states, cos, sin, position_ids)

        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        key_states = apply_multidim_rope(key_states, cos, sin, position_ids)

        value_states = self.v_norm(self.v_proj(hidden_states).view(hidden_shape))

        total_tokens = bsz * seq_len
        attn_output = mm_attn(
            query_states.reshape(1, total_tokens, -1, self.head_dim),
            key_states.reshape(1, total_tokens, -1, self.head_dim),
            value_states.reshape(1, total_tokens, -1, self.head_dim),
            cu_seqlens=cu_seqlens,
        )
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        if valid_mask is not None:
            attn_output = torch.where(
                valid_mask.unsqueeze(-1),
                attn_output,
                torch.zeros_like(attn_output),
            )
        attn_output = self.o_proj(attn_output.contiguous())
        return attn_output, None

    attn_cls.__init__ = patched_attn_init
    attn_cls.forward = patched_attn_forward
    encoder_cls.forward = patched_encoder_forward
    attn_cls._tpu_mm_encoder_attn_patched = True
    logger.info("Applied TPU patch: Gemma4VisionAttention -> MMEncoderAttention.")


def maybe_patch_gemma4_vision(
    model_config: "ModelConfig | None" = None,
) -> bool | None:
    """Install Gemma-4 vision attention TPU routing when loading a Gemma-4 VLM."""
    if not _has_gemma4_vision_tower(model_config):
        return False

    import transformers.models.gemma4.modeling_gemma4 as modeling_gemma4

    install_gemma4_vision_patches(modeling_gemma4)
    return None

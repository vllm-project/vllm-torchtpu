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
"""Multi-Token Prediction (MTP) draft model for zai-org/GLM-5.3-Flash on TPU.

vLLM builds the draft config itself: for a `glm5_next` target,
`SpeculativeConfig.hf_config_override` rewrites the draft's `hf_config` to
`model_type="glm5_next_mtp"`, `architectures=["Glm5NextMTPModel"]` and
`n_predict = text_config.num_nextn_predict_layers`. This module only provides
the TPU model class that architecture resolves to.

In the released checkpoint, layer 45 (`model.language_model.layers.45.*`) is
the single next-n prediction layer (`num_hidden_layers = 45`,
`num_nextn_predict_layers = 1`). Unlike target layers 0..44, which keep a
4-stream Manifold-Constrained Hyper-Connection (mHC) residual state and
alternate 3:1 between KDA linear attention and MLA, layer 45 has:

  1. Single-stream input fusion:
     - `enorm`: RMSNorm over the token embeddings (zeroed at position 0).
     - `hnorm`: RMSNorm over the target's post-final-norm hidden states.
     - `eh_proj`: unquantized BF16 Linear(2 * hidden -> hidden), listed in
       `quantization_config.modules_to_not_convert`.
  2. A plain single-stream decoder block (`mtp_block`):
     - No `hc_attn_*` / `hc_ffn_*` parameters.
     - `input_layernorm`, NoPE MLA `self_attn` (both `rpa` and `absorbed`
       implementations), `post_attention_layernorm`.
     - `mlp`: `Glm5NextMoE` (288 routed FP8 experts + 1 shared expert,
       `swiglu_limit = 10.0`).
  3. Shared prediction head (`shared_head`):
     - `shared_head.norm` is applied inside `forward`, so the returned hidden
       state feeds both `compute_logits` and the next draft step.
     - `shared_head.head` is a placeholder `ParallelLMHead` that
       `Eagle3Proposer` replaces with the target's `lm_head`
       (`vllm_torchtpu.spec_decode.utils.maybe_share_lm_head`).
"""

import functools
import re
from collections.abc import Iterable, MutableSequence, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import is_pp_missing_parameter, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_collectives_tpu,
    glm53_mhc_tpu,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm5_next.model import (
    PACKED_MODULES_MAPPING,
    Glm5NextMLAAttention,
    Glm5NextMLP,
    Glm5NextMoE,
    load_glm53_checkpoint_tensor,
)
from vllm_torchtpu.spec_decode.mtp_loop_model import MTP_DYNAMIC_ARG_DIMS

logger = init_logger(__name__)


def glm53_num_mtp_layers(*configs: Any) -> int:
    """MTP depth from the first config that sets it, or 1.

    Checks `num_nextn_predict_layers` (the checkpoint's text config) and then
    `n_predict` (set by vLLM on the draft's top-level config). Checkpoints may
    carry `num_nextn_predict_layers: null`, so `getattr(..., 1)` is not enough.
    """
    for cfg in configs:
        for key in ("num_nextn_predict_layers", "n_predict"):
            value = getattr(cfg, key, None)
            if value is not None:
                return int(value)
    return 1


def _draft_model_config(vllm_config: VllmConfig) -> ModelConfig:
    """The draft's `ModelConfig`.

    `Eagle3Proposer` builds the draft under a copy of the target's
    `VllmConfig`, so `vllm_config.model_config` is still the target's; the
    draft's own config lives on the speculative config.
    """
    spec = vllm_config.speculative_config
    if spec is not None and spec.draft_model_config is not None:
        return spec.draft_model_config
    return vllm_config.model_config


def _draft_text_config(draft_config: Any) -> Any:
    text_cfg = getattr(draft_config, "hf_text_config", None)
    if text_cfg is not None:
        return text_cfg
    hf_cfg = getattr(draft_config, "hf_config", draft_config)
    return getattr(hf_cfg, "text_config", hf_cfg)


_MTP_LAYER_IDX_RE = re.compile(r"^model\.layers\.(\d+)\.")
# Tensors of an MTP layer that live on `Glm5NextMultiTokenPredictorLayer`
# itself (or, for `embed_tokens`, on the predictor) rather than in its
# `mtp_block`.
_MTP_TOP_LEVEL_WEIGHTS: tuple[str, ...] = (
    "embed_tokens",
    "enorm",
    "hnorm",
    "eh_proj",
    "shared_head",
)


def _op_tag(eps: float) -> str:
    return f"{eps:.0e}".replace("-", "m").replace("+", "p")


def _fused_eh_norm_fn(
    pos_1d: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Pure-PyTorch MTP input masking, dual RMSNorm and concatenation."""
    mask = (pos_1d != 0).unsqueeze(-1)
    e_masked = torch.where(mask, inputs_embeds, torch.zeros_like(inputs_embeds))
    e_normed = glm53_mhc_tpu._rms_norm_weighted_fn(e_masked, enorm_weight, eps)
    h_normed = glm53_mhc_tpu._rms_norm_weighted_fn(
        previous_hidden_states, hnorm_weight, eps
    )
    return torch.cat([e_normed, h_normed], dim=-1)


def _fused_eh_norm_jax(
    positions: jax.Array,
    inputs_embeds: jax.Array,
    previous_hidden_states: jax.Array,
    enorm_weight: jax.Array,
    hnorm_weight: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    """JAX twin of `_fused_eh_norm_fn` (fp32 norms, same op order)."""
    mask = jnp.expand_dims(positions != 0, axis=-1)
    e_masked = jnp.where(mask, inputs_embeds, jnp.zeros_like(inputs_embeds))
    e_normed = glm53_mhc_tpu._rms_norm_weighted_jax(e_masked, enorm_weight, eps=eps)
    h_normed = glm53_mhc_tpu._rms_norm_weighted_jax(
        previous_hidden_states, hnorm_weight, eps=eps
    )
    return jnp.concatenate([e_normed, h_normed], axis=-1)


def _fake_fused_eh_norm(
    pos: torch.Tensor,
    emb: torch.Tensor,
    prev: torch.Tensor,
    ew: torch.Tensor,
    hw: torch.Tensor,
) -> torch.Tensor:
    return emb.new_empty((*emb.shape[:-1], emb.shape[-1] * 2))


_FUSED_EH_NORM_OPS: dict[float, Any] = {}


def get_fused_eh_norm_op(eps: float) -> Any:
    """Build (or fetch) the fused MTP input-norm `jax_op` for `eps`.

    Call it at model construction: registering a `jax_op` cannot be traced.
    """
    eps = float(eps)
    op = _FUSED_EH_NORM_OPS.get(eps)
    if op is None:
        op = pallas.jax_op(
            f"glm53::mtp_fused_eh_norm_e{_op_tag(eps)}",
            functools.partial(_fused_eh_norm_jax, eps=eps),
        )
        op.register_fake(_fake_fused_eh_norm)
        _FUSED_EH_NORM_OPS[eps] = op
    return op


def fused_eh_norm_tpu(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """`cat([enorm(masked_embeds), hnorm(previous_hidden_states)], dim=-1)`.

    In eager TPU mode this is one `jax_op` dispatch instead of the position
    mask, two RMSNorms and the concatenation. Under `torch.compile` the
    pure-PyTorch ops are traced inline into the surrounding graph.
    """
    pos_1d = positions[-1] if positions.ndim == 2 else positions
    if inputs_embeds.device.type == "tpu" and not torch.compiler.is_compiling():
        return get_fused_eh_norm_op(eps)(
            pos_1d.contiguous(),
            inputs_embeds.contiguous(),
            previous_hidden_states.contiguous(),
            enorm_weight,
            hnorm_weight,
        )
    return _fused_eh_norm_fn(
        pos_1d, inputs_embeds, previous_hidden_states, enorm_weight, hnorm_weight, eps
    )


def _fused_add_rms_norm_fn(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch residual add followed by a weighted fp32 RMSNorm."""
    res = residual + x
    normed = glm53_mhc_tpu._rms_norm_weighted_fn(res, weight, eps)
    return normed, res


def _fused_add_rms_norm_jax(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """JAX twin of `_fused_add_rms_norm_fn`."""
    res = (residual + x).astype(residual.dtype)
    normed = glm53_mhc_tpu._rms_norm_weighted_jax(res, weight, eps=eps)
    return normed, res


def _fake_fused_add_rms_norm(
    x: torch.Tensor, res: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(res), torch.empty_like(res)


_FUSED_ADD_RMS_NORM_OPS: dict[float, Any] = {}


def get_fused_add_rms_norm_op(eps: float) -> Any:
    """Build (or fetch) the fused residual-add + RMSNorm `jax_op` for `eps`.

    Call it at model construction: registering a `jax_op` cannot be traced.
    """
    eps = float(eps)
    op = _FUSED_ADD_RMS_NORM_OPS.get(eps)
    if op is None:
        op = pallas.jax_op(
            f"glm53::mtp_fused_add_rms_norm_e{_op_tag(eps)}",
            functools.partial(_fused_add_rms_norm_jax, eps=eps),
        )
        op.register_fake(_fake_fused_add_rms_norm)
        _FUSED_ADD_RMS_NORM_OPS[eps] = op
    return op


def fused_add_rms_norm_tpu(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(rms_norm(residual + x, weight, eps), residual + x)` in one dispatch."""
    if x.device.type == "tpu" and not torch.compiler.is_compiling():
        return get_fused_add_rms_norm_op(eps)(
            x.contiguous(), residual.contiguous(), weight
        )
    return _fused_add_rms_norm_fn(x, residual, weight, eps)


class Glm5NextMTPSharedHead(nn.Module):
    """MTP final RMSNorm and LM head (found by `iter_mtp_shared_heads`)."""

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        # Placeholder; `Eagle3Proposer` swaps in the target's `lm_head`.
        self.head = ParallelLMHead(
            int(config.vocab_size),
            self.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return glm53_mhc_tpu.rms_norm_weighted(
            hidden_states, self.norm.weight, self.rms_norm_eps
        )


class Glm5NextMTPDecoderLayer(nn.Module):
    """Single-stream GLM-5.3-Flash MTP decoder block (no mHC)."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.rms_norm_eps = float(config.rms_norm_eps)

        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = Glm5NextMLAAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # MTP layers sit past `num_hidden_layers`, outside `mlp_layer_types`;
        # they take the type of the last target layer (sparse in the release).
        mlp_layer_types = getattr(config, "mlp_layer_types", None)
        if mlp_layer_types and layer_idx < len(mlp_layer_types):
            mlp_type = mlp_layer_types[layer_idx]
        else:
            mlp_type = mlp_layer_types[-1] if mlp_layer_types else "sparse"
        n_routed = int(getattr(config, "n_routed_experts", 0) or 0)
        if mlp_type == "dense" or n_routed <= 0:
            self.mlp: nn.Module = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=int(config.intermediate_size),
                swiglu_limit=float(config.swiglu_limit),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention + MLP; returns `(ffn_out, residual)` for the final norm."""
        residual = hidden_states
        norm_attn_in = glm53_mhc_tpu.rms_norm_weighted(
            hidden_states, self.input_layernorm.weight, self.rms_norm_eps
        )
        attn_out = self.self_attn(norm_attn_in, positions)
        norm_ffn_in, residual = fused_add_rms_norm_tpu(
            attn_out,
            residual,
            self.post_attention_layernorm.weight,
            self.rms_norm_eps,
        )
        ffn_out = self.mlp(norm_ffn_in)
        return ffn_out, residual


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    """One MTP stage: `enorm`, `hnorm`, `eh_proj`, decoder block, final norm."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Any,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.rms_norm_eps = float(config.rms_norm_eps)
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.hnorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        # The checkpoint stores `eh_proj` unquantized (`modules_to_not_convert`).
        self.eh_proj = ReplicatedLinear(
            self.hidden_size * 2,
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.eh_proj",
        )
        # The block takes `prefix` itself (e.g. "model.layers.45"), not
        # f"{prefix}.mtp_block", so `modules_to_not_convert` entries such as
        # "model.layers.45.self_attn.kv_b_proj" match `is_layer_skipped`.
        self.mtp_block = Glm5NextMTPDecoderLayer(
            config=config,
            layer_idx=layer_idx,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.shared_head = Glm5NextMTPSharedHead(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_head",
        )
        # Register the custom ops now so a compiled forward finds them cached.
        get_fused_eh_norm_op(self.rms_norm_eps)
        get_fused_add_rms_norm_op(self.rms_norm_eps)
        glm53_mhc_tpu.get_rms_norm_weighted_op(self.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids, spec_step_index
        eh_input = fused_eh_norm_tpu(
            positions=positions,
            inputs_embeds=inputs_embeds,
            previous_hidden_states=previous_hidden_states,
            enorm_weight=self.enorm.weight,
            hnorm_weight=self.hnorm.weight,
            eps=self.rms_norm_eps,
        )
        hidden_states, _ = self.eh_proj(eh_input)
        ffn_out, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states
        )
        # The post-norm state feeds both the draft logits and the next step.
        hidden_states, _ = fused_add_rms_norm_tpu(
            ffn_out,
            residual,
            self.shared_head.norm.weight,
            self.rms_norm_eps,
        )
        return hidden_states, hidden_states


class Glm5NextMultiTokenPredictor(nn.Module):
    """Inner `model` module of `Glm5NextMTP`."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        draft_config = _draft_model_config(vllm_config)
        self.config = _draft_text_config(draft_config)
        self.vocab_size = int(self.config.vocab_size)
        self.hidden_size = int(self.config.hidden_size)
        self.mtp_start_layer_idx = int(self.config.num_hidden_layers)
        self.num_mtp_layers = glm53_num_mtp_layers(self.config, draft_config.hf_config)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(idx): Glm5NextMultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=idx,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        # Plain list for the per-step lookup (no string build + hash).
        self._mtp_layers = list(self.layers.values())
        self.logits_processor = LogitsProcessor(self.vocab_size)
        if not getattr(vllm_config.model_config, "enforce_eager", True):
            glm53_collectives_tpu.prepare()

    def _step_idx(self, spec_step_idx: int) -> int:
        return 0 if self.num_mtp_layers == 1 else spec_step_idx % self.num_mtp_layers

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        step_idx = self._step_idx(spec_step_idx)
        return self._mtp_layers[step_idx](
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=previous_hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_index=step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        mtp_layer = cast(
            Glm5NextMultiTokenPredictorLayer,
            self._mtp_layers[self._step_idx(spec_step_idx)],
        )
        head = mtp_layer.shared_head.head
        if (
            torch.compiler.is_compiling()
            and getattr(head, "tp_size", 1) > 1
            and getattr(self.logits_processor, "head_dtype", None) is None
            and getattr(self.logits_processor, "soft_cap", None) is None
            and getattr(self.logits_processor, "scale", 1.0) == 1.0
        ):
            # In a compiled draft loop the token count is symbolic, and the
            # generic all-gather allocates `(num_tokens * tp_size, vocab_shard)`,
            # which the TPU lowering rejects. `vocab_all_gather` keeps the
            # token axis untouched.
            logits = head.quant_method.apply(head, hidden_states, bias=None)
            logits = glm53_collectives_tpu.vocab_all_gather(logits)
            return logits[..., : self.logits_processor.org_vocab_size]
        return self.logits_processor(head, hidden_states)


@support_torch_compile(dynamic_arg_dims=MTP_DYNAMIC_ARG_DIMS)
class Glm5NextMTP(nn.Module, MixtureOfExperts):
    """vLLM `Glm5NextMTPModel` draft for zai-org/GLM-5.3-Flash on TPU."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    packed_modules_mapping = PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix or "mtp_model"
        self.vllm_config = vllm_config
        self.config = _draft_text_config(_draft_model_config(vllm_config))
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self._init_moe_metadata()

    def _init_moe_metadata(self) -> None:
        # `MixtureOfExperts` contract: EPLB and the expert-parallel runner read
        # these attributes from every model that declares the interface.
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.moe_mlp_layers: list[Glm5NextMoE] = [
            cast(Glm5NextMoE, mtp_layer.mtp_block.mlp)
            for layer in self.model.layers.values()
            if isinstance(
                (
                    mtp_layer := cast(Glm5NextMultiTokenPredictorLayer, layer)
                ).mtp_block.mlp,
                Glm5NextMoE,
            )
        ]
        self.moe_layers: Sequence[Any] = [mlp.experts for mlp in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = int(getattr(self.config, "n_group", 1) or 1)
        self.num_logical_experts = int(getattr(self.config, "n_routed_experts", 0) or 0)
        self.num_physical_experts = self.num_logical_experts
        self.num_local_physical_experts = self.num_logical_experts
        self.num_routed_experts = self.num_logical_experts
        self.num_shared_experts = int(getattr(self.config, "n_shared_experts", 0) or 0)
        self.num_redundant_experts = 0

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del intermediate_tensors
        return self.model(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)

    def process_weights_after_loading(self) -> None:
        """Resolve the fused MoE op and TP collectives before torch.compile."""
        if not getattr(self.vllm_config.model_config, "enforce_eager", True):
            glm53_collectives_tpu.prepare()
        for module in self.modules():
            if isinstance(module, Glm5NextMoE):
                module.__dict__.pop("_fused_moe_args", None)
                module._fused_moe()

    @staticmethod
    def _normalize_mtp_checkpoint_name(
        ckpt_name: str,
        mtp_start_layer_idx: int,
        num_mtp_layers: int,
    ) -> str | None:
        """Map a checkpoint tensor name into `Glm5NextMTP`, or None to skip.

        Keeps only the MTP layers `layers.{start}..{start + n - 1}`. Drops the
        vision tower, `rotary_emb.inv_freq` and the sparse-attention indexer
        tensors (the TPU MLA runs dense attention).
        """
        if ckpt_name.startswith(("model.visual.", "visual.")):
            return None
        name = ckpt_name
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model.") :]
        elif name.startswith("language_model."):
            name = "model." + name[len("language_model.") :]
        elif name.startswith("layers."):
            name = "model." + name

        if "rotary_emb.inv_freq" in name or ".self_attn.indexer." in name:
            return None

        layer_match = _MTP_LAYER_IDX_RE.match(name)
        if layer_match is None:
            return None
        layer_idx = int(layer_match.group(1))
        if not (
            mtp_start_layer_idx <= layer_idx < mtp_start_layer_idx + num_mtp_layers
        ):
            return None

        layer_prefix = f"model.layers.{layer_idx}."
        if ".embed_tokens." in name:
            return name.replace(layer_prefix, "model.", 1)
        if not any(f".{w}." in name for w in _MTP_TOP_LEVEL_WEIGHTS):
            name = name.replace(layer_prefix, f"{layer_prefix}mtp_block.", 1)
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the MTP layers (`layers.{num_hidden_layers}..`) of the checkpoint."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        unexpected: list[str] = []
        mtp_start = self.model.mtp_start_layer_idx
        num_mtp = self.model.num_mtp_layers

        for ckpt_name, loaded_weight in weights:
            name = self._normalize_mtp_checkpoint_name(ckpt_name, mtp_start, num_mtp)
            if name is None:
                continue
            load_glm53_checkpoint_tensor(
                self,
                name,
                ckpt_name,
                loaded_weight,
                params_dict,
                loaded_params,
                unexpected,
            )

        if unexpected:
            logger.warning(
                "GLM-5.3 MTP: %d checkpoint tensors had no matching parameter, e.g. %s",
                len(unexpected),
                unexpected[:20],
            )

        # `embed_tokens` and `shared_head.head` come from the target model
        # (`maybe_share_embeddings` / `maybe_share_lm_head`).
        shared_suffixes = ("embed_tokens.weight", "shared_head.head.weight")
        missing = sorted(
            n
            for n in params_dict
            if n not in loaded_params
            and not n.endswith(shared_suffixes)
            and not (
                n.endswith((".weight_scale", ".weight_scale_inv"))
                and any(
                    cand in loaded_params
                    for cand in (
                        n.removesuffix("_inv"),
                        f"{n.removesuffix('_inv')}_inv",
                        n.replace(".shared_head.head.", "."),
                    )
                )
            )
            and ".shared_head.head." not in n
            and not is_pp_missing_parameter(n, self)
        )
        if missing:
            raise ValueError(
                f"GLM-5.3 MTP: {len(missing)} parameters were not loaded from the "
                f"checkpoint, e.g. {missing[:20]}"
            )
        logger.info(
            "GLM-5.3 MTP: loaded %d draft parameters from layers %d..%d.",
            len(loaded_params),
            mtp_start,
            mtp_start + num_mtp - 1,
        )
        return loaded_params

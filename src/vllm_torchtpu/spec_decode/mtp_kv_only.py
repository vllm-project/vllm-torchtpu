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
"""Pass 1 of a two-pass MTP draft step 0: write the caches, nothing else.

Draft step 0 runs the MTP layer on every row of the target's chunk, but a draft
token is read from one row per request only (the end of its accepted prefix,
`last_token_indices`). Every other row still has to leave its MTP KV entry and
its indexer-K entry behind, because later rows and later steps attend to them,
yet nothing reads its attention output, MoE output or top-k. With
`TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS` the proposer splits a large step 0 in two:

* pass 1 (`MtpKvOnlyModel`, this module) runs only what produces the two cache
  entries -- embedding, enorm/hnorm, eh_proj, input norm, then kv_a + kv norm +
  RoPE into the MLA KV cache and wk + k norm + RoPE into the indexer-K cache --
  for every row, with the target's own step-0 metadata;
* pass 2 is the draft's ordinary step-0 program on the per-request rows alone,
  reading the caches pass 1 just wrote (`Eagle3Proposer._two_pass_step0`).

Pass 1 is `MlaKvWriteOp` (layers/adapter/custom_ops/mla_attention_op.py): the
latents come from the wrapper's and indexer's own projections, and the rows go
through the same cache writers the attention and indexer ops call, with the
same owner masks and slots, so every entry it writes is the entry the full
pass writes. Only the upstream MTP prefix up to the decoder layer's input norm
(`mtp_layer_input`) is mirrored here; tests/layers/adapter/test_mla_kv_only.py
pins it against `DeepSeekMultiTokenPredictorLayer.forward`.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

from typing import Any

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig

# Same dims as the draft's own compile (`MTP_DYNAMIC_ARG_DIMS`): dim 0, the
# token dimension, of every tensor argument.
MTP_KV_ONLY_DYNAMIC_ARG_DIMS: dict[str, int | list[int] | dict[int, str]] = {
    "input_ids": 0,
    "positions": 0,
    "hidden_states": 0,
}


def _single_mtp_layer(mtp: Any) -> Any:
    predictor = getattr(mtp, "model", None)
    layers: Any = getattr(predictor, "layers", None)
    if not isinstance(layers, nn.ModuleDict):
        return None
    if len(layers) != 1:
        return None
    start = getattr(predictor, "mtp_start_layer_idx", None)
    if start is None or str(start) not in layers:
        return None
    return layers[str(start)]


def kv_only_unsupported_reason(mtp: Any) -> str | None:
    """Why pass 1 cannot serve this draft, or None when it can.

    Pass 1 mirrors one specific forward: a single-layer `DeepSeekMTP` whose
    decoder layer runs the TPU sparse (DSA) MLA wrapper with its own indexer.
    Anything else keeps the single full step-0 pass.
    """
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUIndexer,
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    if mtp is None:
        return "no draft model"
    layer = _single_mtp_layer(mtp)
    if layer is None:
        return "the draft is not a single-layer DeepSeek-style MTP"
    if getattr(mtp.model, "embed_tokens", None) is None:
        return "the MTP predictor has no embed_tokens"
    for name in ("enorm", "hnorm", "eh_proj", "mtp_block"):
        if getattr(layer, name, None) is None:
            return f"the MTP layer has no {name}"
    block = layer.mtp_block
    if getattr(block, "input_layernorm", None) is None:
        return "the MTP decoder layer has no input_layernorm"
    wrapper = getattr(getattr(block, "self_attn", None), "mla_attn", None)
    if not isinstance(wrapper, VllmTPUMultiHeadLatentAttentionWrapper):
        return "the MTP attention is not the TPU MLA wrapper"
    if not wrapper.is_sparse:
        return "the MTP attention is not sparse (DSA) MLA"
    if not isinstance(wrapper.indexer, VllmTPUIndexer):
        return (
            "the MTP attention has no indexer of its own "
            f"({type(wrapper.indexer).__name__}); pass 1 must write its K cache"
        )
    if wrapper.kimi_preprocess is not None:
        return "the Kimi fused MLA preprocess has no separate KV path"
    if getattr(wrapper.indexer, "indexer_op", None) is None:
        return "the MTP indexer has no indexer op"
    return _not_the_mirrored_classes(mtp, layer, wrapper)


def kv_only_cache_layer_names(mtp: Any) -> tuple[str, str]:
    """The MLA layer and indexer cache whose step-0 metadata pass 1 reads,
    for a draft `kv_only_unsupported_reason` accepts."""
    layer = _single_mtp_layer(mtp)
    assert layer is not None
    wrapper = layer.mtp_block.self_attn.mla_attn
    return wrapper.mla_attn.layer_name, wrapper.indexer.indexer_op.k_cache.prefix


def _not_the_mirrored_classes(mtp: Any, layer: Any, wrapper: Any):
    """Pass 1 mirrors the upstream MTP prefix (`mtp_layer_input`, pinned by
    tests/layers/adapter/test_mla_kv_only.py) and assumes the full pass runs
    the forwards whose projections and writers `MlaKvWriteOp` shares. A
    patched or overridden forward may compute something else, so refuse it
    rather than drift."""
    from vllm.model_executor.models.deepseek_mtp import (
        DeepSeekMultiTokenPredictor,
        DeepSeekMultiTokenPredictorLayer,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUIndexer,
        VllmTPUMultiHeadLatentAttentionWrapper,
        VllmTPUSparseAttnIndexer,
    )

    impl = getattr(getattr(wrapper, "mla_attn", None), "impl", None)
    if not isinstance(wrapper.indexer.indexer_op, VllmTPUSparseAttnIndexer):
        return "the MTP indexer op is not the TPU sparse attention indexer"
    if not isinstance(impl, PallasMLAttentionBackendImpl):
        return "the MTP attention backend is not the TPU MLA backend"
    methods = (
        (mtp.model, DeepSeekMultiTokenPredictor, "forward"),
        (layer, DeepSeekMultiTokenPredictorLayer, "forward"),
        (layer.mtp_block, DeepseekV2DecoderLayer, "forward"),
        (wrapper, VllmTPUMultiHeadLatentAttentionWrapper, "forward"),
        (wrapper.indexer, VllmTPUIndexer, "forward"),
        (wrapper.indexer.indexer_op, VllmTPUSparseAttnIndexer, "forward_oot"),
        (impl, PallasMLAttentionBackendImpl, "forward"),
    )
    for obj, cls, name in methods:
        if getattr(type(obj), name) is not getattr(cls, name):
            return f"{type(obj).__name__}.{name} is not {cls.__name__}.{name}"
    return None


def mtp_layer_input(
    mtp: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[nn.Module, torch.Tensor]:
    """The MTP decoder layer and what its attention receives: the prefix of
    `DeepSeekMultiTokenPredictor.forward` -> `DeepSeekMultiTokenPredictorLayer
    .forward` -> `DeepseekV2DecoderLayer.forward` (residual None) up to the
    `self_attn` call, at draft step 0."""
    layer = _single_mtp_layer(mtp)
    assert layer is not None
    inputs_embeds = mtp.model.embed_tokens(input_ids)
    # masking inputs at position 0, as not needed by MTP
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = layer.enorm(inputs_embeds)
    previous_hidden_states = layer.hnorm(hidden_states)
    hidden = layer.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
    return layer, layer.mtp_block.input_layernorm(hidden)


@support_torch_compile(dynamic_arg_dims=MTP_KV_ONLY_DYNAMIC_ARG_DIMS)
class MtpKvOnlyModel(nn.Module):
    """Pass 1: writes the MTP layer's KV and indexer-K rows for every token."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: nn.Module,
        mesh: Any = None,
    ) -> None:
        from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
            MlaKvWriteOp,
        )

        # vllm_config is read by the compile decorator (see MtpLoopStepModel).
        super().__init__()
        # Read by the plugin's compile_prefix patch: own compile-cache folder.
        self.prefix = prefix
        reason = kv_only_unsupported_reason(mtp)
        if reason is not None:
            raise ValueError(reason)
        self.mtp = mtp
        layer = _single_mtp_layer(mtp)
        assert layer is not None
        self._write = MlaKvWriteOp(layer.mtp_block.self_attn.mla_attn, mesh)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, attn_input = mtp_layer_input(self.mtp, input_ids, positions, hidden_states)
        return self._write(positions, attn_input)

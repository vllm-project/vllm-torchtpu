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
yet nothing reads its attention output, MoE output or top-k. A two-pass step 0
splits it:

* pass 1 (`MtpKvOnlyModel`, this module) runs only what produces the two cache
  entries -- embedding, enorm/hnorm, eh_proj, input norm, then kv_a + kv norm +
  RoPE into the MLA KV cache and wk + k norm + RoPE into the indexer-K cache --
  for every row, with the chunk's step-0 metadata;
* pass 2 is the draft's ordinary step-0 program on the per-request rows alone,
  reading the caches pass 1 just wrote.

Pass 1 is `MlaKvWriteOp` (layers/adapter/custom_ops/mla_attention_op.py): the
latents come from the wrapper's and indexer's own projections, and the rows go
through the same cache writers the attention and indexer ops call, with the
same owner masks and slots, so every entry it writes is the entry the full
pass writes. Only the upstream MTP prefix up to the decoder layer's input norm
(`mtp_layer_input`) is mirrored here.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

from typing import Any, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
)

# Same dims as the draft's own compile (`MTP_DYNAMIC_ARG_DIMS`): dim 0, the
# token dimension, of every tensor argument.
MTP_KV_ONLY_DYNAMIC_ARG_DIMS: dict[str, int | list[int] | dict[int, str]] = {
    "input_ids": 0,
    "positions": 0,
    "hidden_states": 0,
}


def _mtp_layer(
    predictor: DeepSeekMultiTokenPredictor,
) -> DeepSeekMultiTokenPredictorLayer:
    """The predictor's first (for a supported draft, only) MTP layer."""
    return cast(
        DeepSeekMultiTokenPredictorLayer,
        predictor.layers[str(predictor.mtp_start_layer_idx)],
    )


def kv_only_unsupported_reason(mtp: nn.Module) -> str | None:
    """Why pass 1 cannot serve this draft, or None when it can.

    Pass 1 mirrors one specific forward: a single-layer `DeepSeekMTP` whose
    decoder layer runs the TPU sparse (DSA) MLA wrapper with its own indexer.
    """
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUIndexer,
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    if not isinstance(mtp, DeepSeekMTP):
        return f"the draft is a {type(mtp).__name__}, not a DeepSeekMTP"
    predictor = mtp.model
    if not isinstance(predictor, DeepSeekMultiTokenPredictor):
        return f"the MTP predictor is a {type(predictor).__name__}"
    if predictor.num_mtp_layers != 1 or len(predictor.layers) != 1:
        return f"the draft has {len(predictor.layers)} MTP layers, not one"
    layer = _mtp_layer(predictor)
    if not isinstance(layer, DeepSeekMultiTokenPredictorLayer):
        return f"the MTP layer is a {type(layer).__name__}"
    block = layer.mtp_block
    if not isinstance(block, DeepseekV2DecoderLayer):
        return f"the MTP decoder layer is a {type(block).__name__}"
    if not isinstance(block.self_attn, DeepseekV2MLAAttention):
        return "the MTP attention is not MLA"
    wrapper = block.self_attn.mla_attn
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
    return _not_the_mirrored_forwards(mtp, predictor, layer, wrapper)


def _not_the_mirrored_forwards(
    mtp: DeepSeekMTP,
    predictor: DeepSeekMultiTokenPredictor,
    layer: DeepSeekMultiTokenPredictorLayer,
    wrapper: Any,
) -> str | None:
    """Pass 1 mirrors the upstream MTP prefix (`mtp_layer_input`) and assumes
    the full pass runs the forwards whose projections and writers
    `MlaKvWriteOp` shares. A forward a subclass overrides, or a patch replaces
    on the class itself, may compute something else, so refuse it rather than
    drift. That includes a draft whose own forward evaluates step 0 on
    selected rows: the two-pass step 0 assumes the upstream forward."""
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUIndexer,
        VllmTPUMultiHeadLatentAttentionWrapper,
        VllmTPUSparseAttnIndexer,
    )

    indexer_op = wrapper.indexer.indexer_op
    if not isinstance(indexer_op, VllmTPUSparseAttnIndexer):
        return "the MTP indexer op is not the TPU sparse attention indexer"
    impl = wrapper.mla_attn.impl
    if not isinstance(impl, PallasMLAttentionBackendImpl):
        return "the MTP attention backend is not the TPU MLA backend"
    methods = (
        (mtp, DeepSeekMTP, "forward"),
        (predictor, DeepSeekMultiTokenPredictor, "forward"),
        (layer, DeepSeekMultiTokenPredictorLayer, "forward"),
        (layer.mtp_block, DeepseekV2DecoderLayer, "forward"),
        (wrapper, VllmTPUMultiHeadLatentAttentionWrapper, "forward"),
        (wrapper.indexer, VllmTPUIndexer, "forward"),
        (indexer_op, VllmTPUSparseAttnIndexer, "forward_oot"),
        (impl, PallasMLAttentionBackendImpl, "forward"),
    )
    for obj, cls, name in methods:
        fn = getattr(cls, name)
        if getattr(type(obj), name) is not fn:
            return f"{type(obj).__name__}.{name} is not {cls.__name__}.{name}"
        # A patch (patch_registry) swaps the method on `cls` itself, so the
        # identity above still holds; it is defined somewhere else, though.
        if fn.__module__ != cls.__module__ or fn.__qualname__ != (
            f"{cls.__qualname__}.{name}"
        ):
            return (
                f"{cls.__name__}.{name} is replaced by {fn.__module__}."
                f"{fn.__qualname__}"
            )
    return None


def mtp_layer_input(
    mtp: DeepSeekMTP,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[nn.Module, torch.Tensor]:
    """The MTP decoder layer and what its attention receives at draft step 0.

    A copy of the prefix of `DeepSeekMultiTokenPredictor.forward` ->
    `DeepSeekMultiTokenPredictorLayer.forward` -> `DeepseekV2DecoderLayer
    .forward` (residual None) up to the `self_attn` call, as in vLLM v0.33.0,
    the version `pyproject.toml` pins. `kv_only_unsupported_reason` refuses a
    subclass or patch of those forwards, but it cannot see an upstream change
    to their bodies: on a vLLM pin bump,
    `test_mtp_layer_input_matches_the_upstream_prefix`
    (tests/spec_decode/test_mtp_kv_only.py) is the guard.
    """
    predictor = mtp.model
    layer = _mtp_layer(predictor)
    inputs_embeds = predictor.embed_tokens(input_ids)
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
        mtp: DeepSeekMTP,
        mesh: Any = None,
    ) -> None:
        from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
            MlaKvWriteOp,
            VllmTPUMultiHeadLatentAttentionWrapper,
        )

        # vllm_config is read by the compile decorator (see MtpLoopStepModel).
        super().__init__()
        # Read by the plugin's compile_prefix patch: own compile-cache folder.
        self.prefix = prefix
        reason = kv_only_unsupported_reason(mtp)
        if reason is not None:
            raise NotImplementedError(reason)
        self.mtp = mtp
        wrapper = _mtp_layer(mtp.model).mtp_block.self_attn.mla_attn
        assert isinstance(wrapper, VllmTPUMultiHeadLatentAttentionWrapper)
        self._write = MlaKvWriteOp(wrapper, mesh)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, attn_input = mtp_layer_input(self.mtp, input_ids, positions, hidden_states)
        return self._write(positions, attn_input)

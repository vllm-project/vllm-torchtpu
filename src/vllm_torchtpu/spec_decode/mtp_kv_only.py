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
`TPU_MTP_KV_ONLY_STEP0` the proposer splits a large step 0 that carries a
prefill span in two (`MtpKvOnlyStep0`):

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

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
)

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

logger = init_logger(__name__)

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


def _sizes_at_or_above(sizes: Iterable[int | str], min_tokens: int) -> list[int | str]:
    """Pass 1's compile sizes: the draft's token buckets at or above the
    gate, the only ones `MtpKvOnlyStep0.takes_two_pass` dispatches."""
    # TPUPlatform sets compile_sizes to the token padding buckets (ints).
    kept: list[int | str] = [int(size) for size in sizes if int(size) >= min_tokens]
    return kept


def _layout_unsupported_reason(
    proposer: "Eagle3Proposer", all_attn_layers: dict[str, AttentionLayerBase]
) -> str | None:
    """Why `proposer`'s draft or layout cannot take the two-pass step 0."""
    if proposer.speculative_config.method != "mtp":
        return f"the draft method is {proposer.speculative_config.method!r}"
    if proposer._draft_kv_shared:
        return "the draft reads the target's KV cache"
    # TODO: support PCP: it localizes the gather indices per rank, which
    #   pass 2's row selection does not model yet.
    if proposer.runner._pcp_enabled:
        return "prefill context parallelism is on"
    assert proposer.draft_model is not None
    reason = kv_only_unsupported_reason(proposer.draft_model)
    if reason is not None:
        return reason
    # Pass 1 reads the step-0 metadata of the draft's MLA layer and indexer
    # cache, which the draft's attention layers carry.
    assert proposer._draft_attn_layer_names is not None
    indexer_names, mla_names = proposer._split_dsa_draft_layers(
        all_attn_layers, proposer._draft_attn_layer_names
    )
    if len(mla_names) != 1 or len(indexer_names) != 1:
        return (
            f"the draft registered MLA layers {mla_names} and indexer caches "
            f"{indexer_names}, not one of each"
        )
    return None


class MtpKvOnlyStep0:
    """What `TPU_MTP_KV_ONLY_STEP0` adds to `Eagle3Proposer`.

    Pass 1's compiled program (`model`), the smallest padded step 0 that
    takes the two passes (`min_tokens`), and the copy of the draft's config
    pass 1 compiles under, whose `compile_sizes` keep only the buckets at or
    above that gate. The proposer runs the two passes
    (`Eagle3Proposer._two_pass_step0`).
    """

    def __init__(
        self, model: nn.Module, min_tokens: int, vllm_config: VllmConfig
    ) -> None:
        self.model = model
        self.min_tokens = min_tokens
        self.vllm_config = vllm_config

    @classmethod
    def build(
        cls,
        proposer: "Eagle3Proposer",
        all_attn_layers: dict[str, AttentionLayerBase],
    ) -> "MtpKvOnlyStep0":
        """Pass 1 for `proposer`'s loaded draft.

        The switch is explicit, so a draft or layout that cannot take the
        path raises, and so does a failed build: falling back on one rank
        would leave it on the single pass while its TP peers run the two
        passes, and their collectives would not match.
        """
        runner = proposer.runner
        reason = _layout_unsupported_reason(proposer, all_attn_layers)
        if reason is not None:
            raise NotImplementedError(
                f"TPU_MTP_KV_ONLY_STEP0 is set, but {reason}. Unset it to keep "
                "the single full draft step 0."
            )
        # A step 0 of at most the largest loop bucket is also a loop-step
        # shape: the warmup's single-request specs at those sizes are what
        # warm the draft at p = T, and the two passes would run it at a
        # smaller p instead. Pass 2 at p ~ T would save nearly nothing there
        # anyway.
        min_tokens = proposer._max_loop_bucket() + 1
        largest_bucket = max(runner.num_tokens_paddings)
        if min_tokens > largest_bucket:
            raise ValueError(
                f"TPU_MTP_KV_ONLY_STEP0 is set, but no token bucket reaches its "
                f"{min_tokens}-token gate (the largest is {largest_bucket}). Raise "
                "max_num_batched_tokens above the largest draft loop bucket, or "
                "unset it."
            )
        assert proposer.draft_model is not None
        config = copy.copy(proposer.draft_vllm_config)
        config.compilation_config = copy.copy(config.compilation_config)
        config.compilation_config.compile_sizes = _sizes_at_or_above(
            config.compilation_config.compile_sizes or (), min_tokens
        )
        with (
            set_model_tag("eagle_head"),
            set_vllm_model_wrapper_context(mesh=runner.mesh, vllm_config=config),
            set_current_vllm_config(config),
        ):
            model = MtpKvOnlyModel(
                vllm_config=config,
                prefix="mtp_prefill_kv_only",
                mtp=proposer.draft_model,
                mesh=runner.mesh,
            )
        logger.info(
            "MTP draft step 0 of chunks with >= %d padded tokens and a prefill "
            "span runs a KV-only pass over every row, then the full layer on "
            "one row per request.",
            min_tokens,
        )
        return cls(model, min_tokens, config)

    def takes_two_pass(self, num_tokens: int, carries_prefill: bool) -> bool:
        """Whether a step 0 of `num_tokens` padded tokens takes the two
        passes; verify-only steps keep the single pass."""
        return carries_prefill and num_tokens >= self.min_tokens

    def set_compile_sizes(self, draft_sizes: Iterable[int | str]) -> None:
        """Follow the draft's token buckets (the runner adds pipeline-chunk
        buckets after load), keeping those at or above the gate."""
        self.vllm_config.compilation_config.compile_sizes = _sizes_at_or_above(
            draft_sizes, self.min_tokens
        )

    def precompile_gather_rows(self, proposer: "Eagle3Proposer") -> None:
        """Warm pass 2's input gather (`_draft_gather_rows`) on every (T, p)
        it can take: T a token bucket at or above the gate, p a loop bucket."""
        runner = proposer.runner
        max_loop_bucket = proposer._max_loop_bucket()
        # p is always `_loop_bucket(num_reqs)`, a token padding below the gate.
        idx_sizes = [t for t in runner.num_tokens_paddings if t <= max_loop_bucket]
        hidden_size = proposer._draft_hidden_size()
        with runner._precompile_timed("drafter two-pass gather_rows"):
            for num_tokens in runner.num_tokens_paddings:
                if num_tokens < self.min_tokens:
                    continue
                ids = torch.zeros(num_tokens, dtype=torch.int32).to(runner.device)
                pos = torch.zeros(
                    (3, num_tokens) if runner.uses_mrope else (num_tokens,),
                    dtype=torch.int32,
                ).to(runner.device)
                hidden = torch.zeros(
                    (num_tokens, hidden_size), dtype=runner._hidden_states_dtype
                ).to(runner.device)
                for p in idx_sizes:
                    idx = torch.zeros(p, dtype=torch.int64).to(runner.device)
                    synchronize_tensors(
                        proposer._draft_gather_rows(ids, pos, hidden, idx)
                    )

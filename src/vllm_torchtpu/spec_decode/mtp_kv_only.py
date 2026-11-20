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
pass writes. Pass 1 runs the draft's own upstream predictor and MTP layer
forwards with a stand-in for the decoder layer (`kv_only_predictor`), so one
code path computes the layer input for both passes; only the decoder layer's
input norm before its attention is mirrored here.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
from jax.experimental.pallas import tpu as pltpu
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

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.tpu_info import get_chip_version
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


def mtp_kv_only_step0_config_reason(
    vllm_config: VllmConfig, pcp_enabled: bool
) -> str | None:
    """Why this configuration keeps the MTP draft's single step-0 pass
    (TPU_MTP_KV_ONLY_STEP0), or None when the draft may take the two passes.

    Decided from the configuration alone, before any weights load, so every
    process and pipeline stage agrees. Read by
    `TpuPlatform.check_and_update_config` (the startup log), by the proposer
    at load (whether to build the two passes) and by
    `TPUModelRunner._warmup_mtp_prefill` (whether to add the two-pass warmup
    step on every pipeline stage). The draft's own structure (one DSA MTP
    layer, one MLA layer and one indexer cache) is checked when it loads
    (`_layout_unsupported_reason`).
    """
    if not envs.TPU_MTP_KV_ONLY_STEP0:
        return "TPU_MTP_KV_ONLY_STEP0 is 0"
    spec = vllm_config.speculative_config
    method = spec.method if spec is not None else None
    if method != "mtp":
        return f"the draft method is {method!r}"
    # TODO: support PCP: it localizes the gather indices per rank, which
    #   pass 2's row selection does not model yet.
    if pcp_enabled:
        return "prefill context parallelism is on"
    return None


def kv_only_unsupported_reason(mtp: nn.Module) -> str | None:
    """Why pass 1 cannot serve this draft, or None when it can.

    Pass 1 serves one specific draft: a single-layer `DeepSeekMTP` whose
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
    """Pass 1 runs the upstream predictor and MTP layer forwards
    (`kv_only_predictor`) and assumes the full pass runs the forwards whose
    projections and writers `MlaKvWriteOp` shares. A forward a subclass
    overrides, or a patch replaces on the class itself, may compute something
    else, so refuse it rather than drift. That includes a draft whose own
    forward evaluates step 0 on selected rows: the two-pass step 0 assumes the
    upstream forward."""
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


class _KvOnlyDecoderLayer(nn.Module):
    """Pass 1's stand-in for the MTP decoder layer (`mtp_block`).

    It hands what the layer's attention would receive to `write` and skips
    the attention, the MoE and the rest of the layer. With residual None,
    `DeepseekV2DecoderLayer.forward` runs only `input_layernorm` before
    `self_attn` (vLLM v0.33.0, the version `pyproject.toml` pins); on a pin
    bump, `test_kv_only_stand_in_sees_the_attention_input`
    (tests/spec_decode/test_mtp_kv_only.py) is the guard.
    """

    def __init__(self, block: DeepseekV2DecoderLayer, write: Any) -> None:
        super().__init__()
        self.input_layernorm = block.input_layernorm
        # Read by the MTP layer's forward after the decoder layer, which
        # all-gathers a sequence-parallel MoE's token shards. This stand-in
        # returns every row unsharded, so there is nothing to gather.
        self.use_sequence_parallel_moe = False
        self._write = write

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert residual is None
        self._write(positions, self.input_layernorm(hidden_states))
        # (hidden_states, residual) as the decoder layer returns them; the
        # MTP layer's tail runs on them, and pass 1 keeps one row of it.
        return hidden_states, hidden_states


def _with_child(module: nn.Module, name: str, child: nn.Module) -> nn.Module:
    """A shallow copy of `module` whose child `name` is `child`; the
    parameters and every other child are shared with `module`."""
    copied = copy.copy(module)
    copied.__dict__["_modules"] = {**module._modules, name: child}
    return copied


def kv_only_predictor(mtp: DeepSeekMTP, write: Any) -> nn.Module:
    """`mtp`'s predictor with `_KvOnlyDecoderLayer` in place of its decoder
    layer. Its upstream forwards (embedding, the position-0 mask, enorm,
    hnorm, eh_proj) compute the layer input for pass 1 as they do for the
    full pass; `mtp` itself is unchanged."""
    predictor = mtp.model
    layer = _mtp_layer(predictor)
    kv_layer = _with_child(
        layer, "mtp_block", _KvOnlyDecoderLayer(layer.mtp_block, write)
    )
    layers = nn.ModuleDict({str(predictor.mtp_start_layer_idx): kv_layer})
    return _with_child(predictor, "layers", layers)


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
        wrapper = _mtp_layer(mtp.model).mtp_block.self_attn.mla_attn
        assert isinstance(wrapper, VllmTPUMultiHeadLatentAttentionWrapper)
        self._write = MlaKvWriteOp(wrapper, mesh)
        self.predictor = kv_only_predictor(mtp, self._write)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _ = self.predictor(input_ids, positions, hidden_states)
        # The writes are what pass 1 is for; one row keeps the program's
        # output small, and the rest of the layer's tail is never computed.
        return hidden[:1]


# The smallest padded step 0 (T tokens) that takes the two passes, per loop
# bucket p (the token bucket its requests pad to). Measured on TPU v7x with a
# one-layer GLM-5.2-dims draft (TP8, fp8 KV cache, K = 3) at an 8k-token
# context, on steps with requests decoding while one more prefills (and, at
# p = 16, one request prefilling alone): from each entry up, the two passes
# were faster in every measured step. For p >= 32 the entry is the smallest T
# measured, where they were already faster. At a 64k context the crossover was
# no higher. A loop bucket above 256 keeps p = 256's T/p (extrapolated).
_TWO_PASS_MIN_TOKENS: dict[int, int] = {16: 256, 32: 256, 64: 256, 128: 512, 256: 1024}


def two_pass_min_tokens(loop_bucket: int) -> int:
    """The smallest padded step 0 that takes the two passes when its requests
    pad to `loop_bucket`.

    A bucket the table does not list takes the next larger entry's threshold
    (a smaller p only makes pass 2 cheaper), or, above the largest, that
    entry's T/p.
    """
    table = _TWO_PASS_MIN_TOKENS
    if loop_bucket in table:
        return table[loop_bucket]
    above = [p for p in table if p > loop_bucket]
    if above:
        p = min(above)
        min_tokens = table[p]
    else:
        p = max(table)
        min_tokens = -(-table[p] * loop_bucket // p)
    logger.info_once(
        "MTP two-pass draft step 0: no measured threshold for loop bucket %d; "
        "taking the two passes from %d padded tokens (from p = %d's entry).",
        loop_bucket,
        min_tokens,
        p,
    )
    return min_tokens


def _sizes_at_or_above(sizes: Iterable[int | str], min_tokens: int) -> list[int | str]:
    """Pass 1's compile sizes: the draft's token buckets at or above the
    gate, the only ones `MtpKvOnlyStep0.takes_two_pass` dispatches."""
    # TPUPlatform sets compile_sizes to the token padding buckets (ints).
    kept: list[int | str] = [int(size) for size in sizes if int(size) >= min_tokens]
    return kept


def _layout_unsupported_reason(
    proposer: "Eagle3Proposer", all_attn_layers: dict[str, AttentionLayerBase]
) -> str | None:
    """Why `proposer`'s loaded draft cannot take the two-pass step 0 on this
    chip. The configuration (draft method, PCP) is checked before any weights
    load, in `TpuPlatform.check_and_update_config`."""
    if proposer._draft_kv_shared:
        return "the draft reads the target's KV cache"
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
    chip = get_chip_version()
    if chip is not pltpu.ChipVersion.TPU_7X:
        return f"its dispatch table is measured on TPU7x, not {chip.value}"
    return None


class MtpKvOnlyStep0:
    """What `TPU_MTP_KV_ONLY_STEP0` adds to `Eagle3Proposer`.

    Pass 1's compiled program (`model`), the smallest padded step 0 that can
    take the two passes at any loop bucket the draft reaches (`min_tokens`),
    and the copy of the draft's config pass 1 compiles under, whose
    `compile_sizes` keep only the buckets at or above it. A step 0 takes the
    two passes where `two_pass_min_tokens` says they beat the single pass
    (`takes_two_pass`). The proposer runs them
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
    ) -> "MtpKvOnlyStep0 | None":
        """Pass 1 for `proposer`'s loaded draft, or None when the draft or its
        layout cannot take the two passes, or no token bucket would.

        That decision depends only on the configuration and the draft's
        structure, so every rank makes the same one. A build that fails
        raises instead: falling back on one rank would leave it on the single
        pass while its TP peers run the two passes, and their collectives
        would not match.
        """
        runner = proposer.runner
        reason = _layout_unsupported_reason(proposer, all_attn_layers)
        if reason is not None:
            logger.info("MTP draft step 0 keeps the single pass: %s.", reason)
            return None
        loop_buckets = [
            p for p in runner.num_tokens_paddings if p <= proposer._max_loop_bucket()
        ]
        min_tokens = min(two_pass_min_tokens(p) for p in loop_buckets)
        largest_bucket = max(runner.num_tokens_paddings)
        if min_tokens > largest_bucket:
            logger.info(
                "MTP draft step 0 keeps the single pass: no token bucket reaches "
                "%d tokens (the largest is %d).",
                min_tokens,
                largest_bucket,
            )
            return None
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
            "MTP draft step 0 of a chunk with a prefill span runs a KV-only pass "
            "over every row, then the full layer on one row per request, from "
            "%d padded tokens (more as the request count grows).",
            min_tokens,
        )
        return cls(model, min_tokens, config)

    def takes_two_pass(
        self, num_tokens: int, loop_bucket: int, carries_prefill: bool
    ) -> bool:
        """Whether a step 0 of `num_tokens` padded tokens, whose requests pad
        to `loop_bucket`, takes the two passes; verify-only steps keep the
        single pass."""
        return carries_prefill and num_tokens >= max(
            self.min_tokens, two_pass_min_tokens(loop_bucket)
        )

    def set_compile_sizes(self, draft_sizes: Iterable[int | str]) -> None:
        """Follow the draft's token buckets (the runner adds pipeline-chunk
        buckets after load), keeping those at or above the gate."""
        self.vllm_config.compilation_config.compile_sizes = _sizes_at_or_above(
            draft_sizes, self.min_tokens
        )

    def missing_buckets(self, target_buckets: Iterable[int]) -> list[int]:
        """Target token buckets a step 0 can take the two passes at that the
        KV-only pass has no compiled graph for."""
        compiled = {
            int(size)
            for size in self.vllm_config.compilation_config.compile_sizes or ()
        }
        return sorted(
            int(t)
            for t in target_buckets
            if int(t) >= self.min_tokens and int(t) not in compiled
        )

    def precompile_gather_rows(self, proposer: "Eagle3Proposer") -> None:
        """Warm pass 2's input gather (`_draft_gather_rows`) on every (T, p)
        the two passes take: T a token bucket, p a loop bucket, and
        `takes_two_pass` true for them."""
        runner = proposer.runner
        max_loop_bucket = proposer._max_loop_bucket()
        # p is always `_loop_bucket(num_reqs)`, a token padding.
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
                    if not self.takes_two_pass(num_tokens, p, carries_prefill=True):
                        continue
                    idx = torch.zeros(p, dtype=torch.int64).to(runner.device)
                    synchronize_tensors(
                        proposer._draft_gather_rows(ids, pos, hidden, idx)
                    )

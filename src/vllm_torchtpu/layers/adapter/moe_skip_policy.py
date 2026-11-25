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
"""Padded-token skipping policy for BF16 MoE ("MoE skip" in names here).

With ``TPU_MOE_SKIP_PADDED_TOKENS=1``, ``VllmUnquantizedFusedMoEMethod`` drops
padded rows inside ``fused_moe_gmm`` by row index: it passes this rank's token
padding mask (``TokenPaddingState.local_padding_mask``) as ``row_info``. Real
rows keep the stock dispatch, including routes whose routing weight is exactly 0.

Each BF16 MoE layer resolves the policy from the ``VllmConfig`` it is built
under, so the policy needs no runner state. A draft model is built under a copy
of the target's config, which keeps ``speculative_config``, so its layers take
the speculative-decoding fallback like the target's. Unsupported configurations
(expert parallelism, DP / PP / PCP / DCP > 1, the pipelined MoE collective,
speculative decoding, LoRA, quantized models) and a layer built without a
current ``VllmConfig`` run the stock path, and the layer logs the reason.

The remaining fallbacks, the cost on full buckets and the conditions for turning
skipping on by default are tracked in
https://github.com/vllm-project/vllm-torchtpu/issues/3777.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
)
from vllm_torchtpu.utils import get_dp_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class Bf16MoeSkipPolicy:
    """Whether BF16 MoE layers skip padded tokens, and why not if disabled."""

    enabled: bool
    fallback_reason: str | None = None


def resolve_bf16_moe_skip_policy(
    vllm_config: VllmConfig | None,
) -> Bf16MoeSkipPolicy:
    """Resolve the policy from ``TPU_MOE_SKIP_PADDED_TOKENS`` and the config
    a BF16 MoE layer is built under (``None``: no current config)."""
    if not envs.TPU_MOE_SKIP_PADDED_TOKENS:
        return Bf16MoeSkipPolicy(enabled=False)
    reason = _fallback_reason(vllm_config)
    if reason is not None:
        return Bf16MoeSkipPolicy(enabled=False, fallback_reason=reason)
    return Bf16MoeSkipPolicy(enabled=True)


def _fallback_reason(vllm_config: VllmConfig | None) -> str | None:
    if vllm_config is None:
        return "no current VllmConfig"
    # Imported here: the module defining it imports this one.
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedConfig,
    )

    if not isinstance(vllm_config.quant_config, VllmUnquantizedConfig):
        return "quantized model"
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_expert_parallel:
        return "expert parallelism"
    for name, size in (
        # For DP without EP the worker sets data_parallel_size to 1;
        # get_dp_size also reads TORCH_TPU_DP_SIZE.
        ("data_parallel_size", get_dp_size(parallel_config)),
        ("pipeline_parallel_size", parallel_config.pipeline_parallel_size),
        (
            "prefill_context_parallel_size",
            parallel_config.prefill_context_parallel_size,
        ),
        (
            "decode_context_parallel_size",
            parallel_config.decode_context_parallel_size,
        ),
    ):
        if size > 1:
            return f"{name} > 1"
    if enable_pipelined_collective_and_compute():
        return "pipelined MoE collective (TPU_MOE_COLLECTION_CHUNK_SIZE)"
    if vllm_config.speculative_config is not None:
        return "speculative decoding"
    if vllm_config.lora_config is not None:
        return "LoRA"
    return None

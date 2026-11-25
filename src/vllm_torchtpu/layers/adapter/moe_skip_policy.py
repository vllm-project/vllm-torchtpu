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
padded rows inside ``fused_moe_gmm`` by row index: it passes the valid-token
counts that ``TokenPaddingState`` tracks as ``row_info`` (a 1-element tensor
for TP / TP+EP / uncoupled DP, or per-rank counts gathered across DP for DP+EP
and chunked ``pipelined_fused_moe_gmm``). Real rows keep the stock dispatch,
including routes whose routing weight is exactly 0.

``TPUModelRunner`` resolves the policy once, before it builds the target model
and the drafter, and publishes it with ``bf16_policy_scope`` so the MoE layers,
the GMM kernel prebuild and ``TokenPaddingState`` use the same mode. A layer
built outside that scope runs the stock path. Unsupported configurations
(PP / PCP / DCP > 1, the fused MoE decode kernels, speculative decoding, LoRA,
quantized models) also run the stock path and allocate no valid-token count;
the BF16 MoE layers log the reason. Uncoupled DP without EP is collapsed to
``data_parallel_size=1`` per engine by ``TPUWorker.init_device`` and runs like
TP per engine.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_active_bf16_policy: Bf16MoeSkipPolicy | None = None


@dataclass(frozen=True)
class Bf16MoeSkipPolicy:
    """Whether BF16 MoE layers skip padded tokens, and why not if disabled."""

    enabled: bool
    fallback_reason: str | None = None


@contextlib.contextmanager
def bf16_policy_scope(policy: Bf16MoeSkipPolicy) -> Iterator[None]:
    """Publish the runner's policy to the BF16 MoE layers built in the scope."""
    global _active_bf16_policy
    prev = _active_bf16_policy
    _active_bf16_policy = policy
    try:
        yield
    finally:
        _active_bf16_policy = prev


def get_active_bf16_policy() -> Bf16MoeSkipPolicy | None:
    """Return the policy published by ``bf16_policy_scope``, if any."""
    return _active_bf16_policy


def resolve_bf16_moe_skip_policy(
    *,
    skip_flag: bool,
    parallel_config: ParallelConfig,
    pipelined_collective: bool,
    sequence_parallel: bool,
    fused_decode_kernel: bool,
    has_speculative: bool,
    has_lora: bool,
    is_unquantized_model: bool,
) -> Bf16MoeSkipPolicy:
    """Resolve the policy before any MoE layer, kernel prebuild or trace."""
    if not skip_flag:
        return Bf16MoeSkipPolicy(enabled=False)

    fallback_reason: str | None = None
    if not is_unquantized_model:
        fallback_reason = "quantized model"
    else:
        for name, size in (
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
                fallback_reason = f"{name} > 1"
                break
    if fallback_reason is None and sequence_parallel and pipelined_collective:
        fallback_reason = (
            "sequence-parallel MoE with pipelined collective "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE)"
        )
    if fallback_reason is None and fused_decode_kernel:
        fallback_reason = "fused MoE decode kernels"
    if fallback_reason is None and has_speculative:
        fallback_reason = "speculative decoding"
    if fallback_reason is None and has_lora:
        fallback_reason = "LoRA"

    if fallback_reason is not None:
        return Bf16MoeSkipPolicy(enabled=False, fallback_reason=fallback_reason)
    return Bf16MoeSkipPolicy(enabled=True)

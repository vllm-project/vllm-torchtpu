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
"""Effective BF16 MoeSkip policy.

With ``TPU_MOE_SKIP_PADDED_TOKENS=1`` on a supported BF16 topology,
``VllmUnquantizedFusedMoEMethod`` drops padded rows inside ``fused_moe_gmm``
using a 1-element local valid-row count (``padding_rows="count"``) rather than
zeroing routing weights ahead of the kernel. Real rows keep stock dispatch
behavior even when a routing weight is zero.

Capability gating runs BEFORE model loading / MoE kernel prebuild / first
backbone tracing so ``TPUModelRunner``, ``TokenPaddingState`` and
``VllmUnquantizedFusedMoEMethod`` share one effective mode:

- Supported BF16 topology:
  ``skip_padded_tokens=True, padding_rows="count", row_info=<local_valid_tokens>``
- ``flag-off``, EP fallback, or unsupported topologies (DP > 1, PP > 1,
  PCP > 1, DCP > 1, pipelined MoE collective, fused MoE decode kernels,
  speculative decoding / drafter, LoRA, non-BF16 / mixed MoE):
  ``skip_padded_tokens=False, padding_rows=None, row_info=None`` (and no
  ``local_valid_tokens`` allocation).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
)

_active_bf16_policy: Bf16MoeSkipPolicy | None = None


@dataclass(frozen=True)
class Bf16MoeSkipPolicy:
    """Effective BF16 MoeSkip mode shared across runner and MoE method."""

    enabled: bool
    fallback_reason: str | None = None

    @property
    def padding_rows(self) -> str | None:
        return "count" if self.enabled else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "padding_rows": self.padding_rows,
            "fallback_reason": self.fallback_reason,
        }


@contextlib.contextmanager
def bf16_policy_scope(policy: Bf16MoeSkipPolicy) -> Iterator[None]:
    """Publish the runner-resolved BF16 MoeSkip policy while loading layers."""
    global _active_bf16_policy
    prev = _active_bf16_policy
    _active_bf16_policy = policy
    try:
        yield
    finally:
        _active_bf16_policy = prev


def get_active_bf16_policy() -> Bf16MoeSkipPolicy | None:
    """Return the active runner-resolved BF16 MoeSkip policy, if any."""
    return _active_bf16_policy


def bf16_layer_unsupported_reason(moe: Any) -> str | None:
    """Per-layer topology check for BF16 row-count MoeSkip (defense-in-depth
    and standalone unit-test fallback)."""
    par = getattr(moe, "moe_parallel_config", None)
    if par is not None:
        if getattr(par, "use_ep", False):
            return "expert parallelism"
        for attr, label in (
            ("dp_size", "data_parallel_size"),
            ("pcp_size", "prefill_context_parallel_size"),
            ("dcp_size", "decode_context_parallel_size"),
            ("pp_size", "pipeline_parallel_size"),
        ):
            if getattr(par, attr, 1) > 1:
                return f"{label} > 1"
    if enable_pipelined_collective_and_compute():
        return "pipelined MoE collective (TPU_MOE_COLLECTION_CHUNK_SIZE)"
    if envs.USE_MOE_TP_DECODE_KERNEL or envs.FUSED_MOE_DECODE_KERNEL:
        return "fused MoE decode kernels"
    return None


def resolve_bf16_moe_skip_policy(
    *,
    skip_flag: bool,
    parallel_config: Any,
    pipelined_collective: bool,
    fused_decode_kernel: bool,
    has_speculative: bool,
    has_lora: bool,
    is_unquantized_model: bool,
) -> Bf16MoeSkipPolicy:
    """Resolve the shared BF16 MoeSkip effective mode before model loading.

    Unsupported configurations cleanly fall back to the stock contract
    ``(skip_padded_tokens=False, padding_rows=None, row_info=None)`` before MoE
    kernel prebuild and before the first backbone trace.
    """
    if not skip_flag:
        return Bf16MoeSkipPolicy(enabled=False)

    fallback_reason: str | None = None
    if not is_unquantized_model:
        fallback_reason = "non-unquantized quantization config"
    elif getattr(parallel_config, "enable_expert_parallel", False):
        fallback_reason = "expert parallelism"
    else:
        for name in (
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel_config, name, 1) > 1:
                fallback_reason = f"{name} > 1"
                break
    if fallback_reason is None and pipelined_collective:
        fallback_reason = "pipelined MoE collective (TPU_MOE_COLLECTION_CHUNK_SIZE)"
    if fallback_reason is None and fused_decode_kernel:
        fallback_reason = "fused MoE decode kernels"
    if fallback_reason is None and has_speculative:
        fallback_reason = "speculative decoding"
    if fallback_reason is None and has_lora:
        fallback_reason = "LoRA"

    if fallback_reason is not None:
        return Bf16MoeSkipPolicy(enabled=False, fallback_reason=fallback_reason)
    return Bf16MoeSkipPolicy(enabled=True)


def resolve_standalone_bf16_moe_skip_policy(moe: Any) -> Bf16MoeSkipPolicy:
    """Resolve BF16 MoeSkip policy when VllmUnquantizedFusedMoEMethod is
    constructed outside TPUModelRunner.load_model()."""
    if not envs.TPU_MOE_SKIP_PADDED_TOKENS:
        return Bf16MoeSkipPolicy(enabled=False)
    reason = bf16_layer_unsupported_reason(moe)
    if reason is not None:
        return Bf16MoeSkipPolicy(enabled=False, fallback_reason=reason)
    return Bf16MoeSkipPolicy(enabled=True)

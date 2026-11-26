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
"""Planning and precompilation helpers for decode-only batched RPA graphs."""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import vllm.envs as vllm_envs
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

DECODE_ONLY_CACHE_TAG = "decode_only"
DECODE_ONLY_PREFIX_SUFFIX = f"_{DECODE_ONLY_CACHE_TAG}"


def _agree_across_pp(has_local_buckets: bool, pipeline_parallel_size: int) -> bool:
    """Ensure every pipeline stage makes the same decode-only graph decision."""
    if pipeline_parallel_size <= 1:
        return has_local_buckets
    from vllm.distributed.parallel_state import get_pp_group

    pp_group = get_pp_group()
    agreed = torch.tensor([int(has_local_buckets)], dtype=torch.int64, device="cpu")
    torch.distributed.all_reduce(
        agreed, op=torch.distributed.ReduceOp.MIN, group=pp_group.cpu_group
    )
    return bool(agreed.item())


def resolve_decode_only_buckets(
    *,
    vllm_config: VllmConfig,
    token_paddings: Sequence[int],
    num_reqs_max_model_len: int,
    num_reqs_most_model_len: int | None,
    reorder_batch_threshold: int | None,
    enforce_eager: bool,
    has_pcp: bool,
    has_kv_cache_bundle: bool,
    pipeline_parallel_size: int = 1,
) -> list[int]:
    """Return token buckets to compile with a decode-only (no-RPAm) graph.

    Logs a warning once whenever ``TPU_RPA_DECODE_ONLY_GRAPH=1`` self-disables
    for an unsupported configuration, and raises when incompatible with mega
    AOT artifacts. Under pipeline parallelism, all stages must agree so
    hand-off warmup runs the same sequence on every stage.
    """
    if not envs.TPU_RPA_DECODE_ONLY_GRAPH:
        return []
    if vllm_envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        raise ValueError(
            "TPU_RPA_DECODE_ONLY_GRAPH=1 is incompatible with "
            "VLLM_USE_MEGA_AOT_ARTIFACT=1. Set VLLM_USE_MEGA_AOT_ARTIFACT=0 "
            "(the TPU default)."
        )

    disable_reason: str | None = None
    if enforce_eager:
        disable_reason = "enforce_eager is enabled"
    elif envs.TPU_KERNEL_ITER_MODE:
        disable_reason = "TPU_KERNEL_ITER_MODE=1 splits RPA ops out of compiled graphs"
    elif has_pcp:
        disable_reason = "prefill context parallelism uses the PCP streaming kernel"
    elif has_kv_cache_bundle:
        disable_reason = "block-major KV bundle uses the bundled RPA v3 path"
    elif not token_paddings:
        disable_reason = "no backbone token buckets configured"
    else:
        layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        has_eligible_layer = any(
            isinstance(
                (impl := getattr(layer, "impl", None)),
                PallasAttentionBackendImpl,
            )
            and impl.has_rpa_decode_only_kernel()
            for layer in layers.values()
        )
        if not has_eligible_layer:
            disable_reason = (
                "no attention layer pre-built a decode-only batched RPA kernel"
            )

    if disable_reason is not None:
        logger.warning_once("TPU_RPA_DECODE_ONLY_GRAPH=1 disabled: %s.", disable_reason)
        _agree_across_pp(False, pipeline_parallel_size)
        return []

    if not _agree_across_pp(True, pipeline_parallel_size):
        logger.warning_once(
            "TPU_RPA_DECODE_ONLY_GRAPH=1 disabled: not all pipeline stages have "
            "eligible batched RPA layers."
        )
        return []

    max_reqs = max(num_reqs_max_model_len, num_reqs_most_model_len or 0)
    max_decode_tokens = max(1, max_reqs * (reorder_batch_threshold or 1))
    # Round up to the covering bucket so steps whose active decode token count
    # falls between two buckets (e.g. 65..100 padding to 128) still hit the
    # decode-only graph.
    paddings = list(token_paddings)
    capped = min(max_decode_tokens, paddings[-1])
    top_bucket = paddings[bisect.bisect_left(paddings, capped)]
    return [n for n in paddings if n <= top_bucket]


def precompile_decode_only_overlay(
    *,
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    decode_buckets: Sequence[int],
    trace_first: bool,
    run_pass: Callable[..., shape_variants.BucketExecutables],
) -> dict[tuple[str, int, int], tuple[Callable[..., Any], tuple]]:
    """Compile the decode-only variant pass and restore the base compile wrapper."""
    shape_variants.retrace(model, vllm_config, prefix_suffix=DECODE_ONLY_PREFIX_SUFFIX)
    decode_session = run_pass(
        decode_buckets,
        trace_first=trace_first,
        allowed_sizes=frozenset(decode_buckets),
        cache_tag=DECODE_ONLY_CACHE_TAG,
        rpa_decode_only=True,
    )
    decode_runnables = shape_variants.collect_variant_runnables(
        decode_session, prefix_suffix=DECODE_ONLY_PREFIX_SUFFIX
    )
    shape_variants.retrace(model, vllm_config)
    return decode_runnables

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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import vllm.envs as vllm_envs
from vllm.config import get_layers_from_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers.attention import Attention

from vllm_torchtpu import envs, utils
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

DECODE_ONLY_CACHE_TAG = "decode_only"
DECODE_ONLY_PREFIX_SUFFIX = f"_{DECODE_ONLY_CACHE_TAG}"


def _min_top_bucket_across_pp(top_bucket: int, pipeline_parallel_size: int) -> int:
    """Agree on a single covering bucket (0 = disabled) across all PP stages."""
    if pipeline_parallel_size <= 1:
        return top_bucket
    return utils.min_over_group(top_bucket, parallel_state.get_pp_group().cpu_group)


def resolve_decode_only_buckets(
    *,
    vllm_config: VllmConfig,
    token_paddings: Sequence[int],
    pad_token_len: Callable[[list[int], int], int],
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
    AOT artifacts. Under pipeline parallelism, all stages agree on the exact
    covering bucket (0 disables) via a single MIN reduction so hand-off warmup
    runs the same bucket list on every stage.
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
        layers = get_layers_from_vllm_config(vllm_config, Attention)
        has_eligible_layer = any(
            isinstance(layer.impl, PallasAttentionBackendImpl)
            and layer.impl.has_rpa_decode_only_kernel()
            for layer in layers.values()
        )
        if not has_eligible_layer:
            disable_reason = (
                "no attention layer pre-built a decode-only batched RPA kernel"
            )

    paddings = list(token_paddings)
    if disable_reason is not None:
        logger.warning_once("TPU_RPA_DECODE_ONLY_GRAPH=1 disabled: %s.", disable_reason)
        local_top_bucket = 0
    else:
        max_reqs = max(num_reqs_max_model_len, num_reqs_most_model_len or 0)
        max_decode_tokens = max(1, max_reqs * (reorder_batch_threshold or 1))
        # Round up to the covering bucket so steps whose active decode token
        # count falls between two buckets (e.g. 65..100 padding to 128) still
        # hit the decode-only graph.
        local_top_bucket = pad_token_len(paddings, min(max_decode_tokens, paddings[-1]))

    top_bucket = _min_top_bucket_across_pp(local_top_bucket, pipeline_parallel_size)
    if top_bucket == 0:
        if local_top_bucket > 0:
            logger.warning_once(
                "TPU_RPA_DECODE_ONLY_GRAPH=1 disabled: not all pipeline stages "
                "have eligible batched RPA layers."
            )
        return []
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
        prefix_suffix=DECODE_ONLY_PREFIX_SUFFIX,
        rpa_decode_only=True,
    )
    decode_runnables = shape_variants.collect_variant_runnables(
        decode_session, prefix_suffix=DECODE_ONLY_PREFIX_SUFFIX
    )
    shape_variants.retrace(model, vllm_config)
    return decode_runnables

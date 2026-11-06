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

"""QSA decode output projection with budgeted BF16 weight prefetch."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import _request_bounds
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import qsa_attention
from vllm_torchtpu.kernels.sparse_gqa_decode import schedule
from vllm_torchtpu.kernels.sparse_gqa_decode.metadata import virtual_page_metadata
from vllm_torchtpu.kernels.sparse_gqa_decode.projection_kernel import (
    decode_projected_kernel,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.projection_planning import (
    plan_projected_decode,
    projection_width,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.query_layout import query_offsets


def projected_attention(
    query: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    starts: jax.Array,
    weight: jax.Array,
    residual: jax.Array,
    hc_gate: jax.Array,
    *,
    request_distribution: jax.Array,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
    original_weight: jax.Array | None = None,
    hc_token_tile: int = 0,
    hc_min_tokens: int = 1024,
    query_prepared: bool = False,
    prepared_cu: jax.Array | None = None,
    sc_min_context: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return projected HC output and the unchanged cache owner."""
    from vllm_torchtpu.kernels.sparse_gqa_decode.query_layout import planning_query

    width = projection_width(query, weight, residual, hc_gate)
    launch = plan_projected_decode(
        planning_query(query, query_prepared),
        gate,
        blocks,
        lengths,
        table,
        cache,
        weight,
        hc_gate,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    padded_width = weight.shape[0] * weight.shape[2]
    supported = len(query) <= len(starts) - 1 and launch is not None

    def fallback():
        prefill_options: dict[str, Any] = dict(sc_min_context=sc_min_context)
        attended, _ = qsa_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            request_distribution=request_distribution,
            query_prepared=query_prepared,
            prepared_cu=prepared_cu,
            kv_heads=kv_heads,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
            **prefill_options,
        )
        matrix = (
            original_weight
            if original_weight is not None
            else weight.transpose(1, 0, 2).reshape(weight.shape[1], -1)[:, :width]
        )
        projected = jnp.matmul(attended.reshape(len(query), -1), matrix).astype(
            jnp.bfloat16
        )
        if (
            hc_token_tile > 0
            and len(query) >= max(hc_token_tile, hc_min_tokens)
            and len(query) % hc_token_tile == 0
        ):
            from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn

            output = gated_residual_fn(hc_gate.shape[-1], width, hc_token_tile)(
                residual, projected, hc_gate
            )
            return output
        output = jnp.concatenate(
            [
                residual[:, branch * width : (branch + 1) * width].astype(jnp.float32)
                + projected.astype(jnp.float32) * hc_gate[:, branch : branch + 1]
                for branch in range(hc_gate.shape[-1])
            ],
            axis=-1,
        ).astype(residual.dtype)
        return output

    def prefetch():
        assert launch is not None
        batch, heads, dim = query.shape
        cfgs = launch.config_for_batch(batch)
        cfgs = dataclasses.replace(
            cfgs,
            # ServingConfigs annotates integer scales but RPA supports float KV scales.
            # pyrefly: ignore[bad-argument-type]
            serve=dataclasses.replace(cfgs.serve, scale_k=k_scale, scale_v=v_scale),
        )
        from vllm_torchtpu.kernels.sparse_gqa_decode.query_layout import prepare_query

        pages, kv_lens = virtual_page_metadata(
            blocks, lengths, table, ratio=ratio, page_size=cache.shape[1]
        )
        if query_prepared:
            q = prepare_query(query, lengths, kv_heads)
            q = jnp.where((lengths > 0)[None, :, None, None, None], q, 0)
        else:
            q = jnp.where((lengths > 0)[:, None, None], query, 0).astype(jnp.float32)
            q = q.reshape(batch, kv_heads, heads // kv_heads, 1, dim).swapaxes(0, 1)
        starts = query_offsets(batch, prepared_cu)
        q_lens = (kv_lens > 0).astype(jnp.int32)
        metadata = schedule.generate(starts, kv_lens, pages, cfgs)
        _, output = decode_projected_kernel(
            cu_q_lens=starts,
            compressed_query_offsets=kv_lens - q_lens,
            q_lens=q_lens,
            metadata=metadata,
            query=q,
            unused_new_kv=jnp.zeros((batch, *cache.shape[2:]), cache.dtype),
            cache=cache.reshape(-1, ratio, *cache.shape[2:]),
            gate=gate,
            output_weight=weight,
            residual=jnp.pad(
                residual.reshape(batch, hc_gate.shape[1], width),
                ((0, 0), (0, 0), (0, padded_width - width)),
            ).reshape(batch, -1),
            hc_gate=hc_gate,
            cfgs=cfgs,
            policy=launch.policy,
        )
        return output.reshape(batch, hc_gate.shape[1], padded_width)[
            :, :, :width
        ].reshape(residual.shape)

    if not supported:
        return fallback(), cache
    prefix_tokens, active_tokens = _request_bounds(
        starts, request_distribution, len(query)
    )
    # Mixed/prefill keeps the upstream segment-specific gating and ownership.
    output = jax.lax.cond(prefix_tokens == active_tokens, prefetch, fallback)
    return output, cache

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
"""Fuse CSA sequence lookup, page lookup, padding scatter, and row lengths.

This preserves sparse_mla's address and int32 padding-scatter semantics. It
changes neither the compressed KV representation nor attention arithmetic.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention.tuned_block_sizes import (
    FUSED_CSA_METADATA_BUCKETS,
)
from vllm_torchtpu.utils import get_device_name


def is_supported(
    topk: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, *, page_size: int
) -> bool:
    """Use only measured small-query buckets; retain the bulk path otherwise."""
    if topk.shape[0] not in (1, 2, 4):
        return False
    max_seqs = cu_q_lens.shape[0] - 1
    key = (
        get_device_name(),
        topk.shape[1],
        max_seqs,
        page_indices.size // max_seqs,
        page_size,
    )
    return topk.shape[0] in FUSED_CSA_METADATA_BUCKETS.get(key, ())


def _sequence_id(token, cu):
    # upper_bound(cu[1:], token), clamped to the final sequence, also matches
    # jnp.repeat's final-element fill for the padded token suffix.
    def condition(state):
        lo, hi = state
        return lo < hi

    def body(state):
        lo, hi = state
        mid = (lo + hi) // 2
        before = cu[mid + 1] <= token
        return jnp.where(before, mid + 1, lo), jnp.where(before, hi, mid)

    lo, _ = lax.while_loop(condition, body, (jnp.int32(0), jnp.int32(cu.shape[0] - 1)))
    return jnp.minimum(lo, cu.shape[0] - 2)


def map_indices(top, pages, token_base, *, page_size, num_pages, pages_per_seq):
    """Shared exact page lookup + int32 padding scatter + valid count."""
    lanes = pltpu.get_tpu_info().num_lanes
    logical = jnp.clip(top // page_size, 0, pages_per_seq - 1)
    physical = jnp.zeros_like(top)
    for c in range(pl.cdiv(pages_per_seq, lanes)):
        window = jnp.broadcast_to(
            pages[:, c * lanes : (c + 1) * lanes], (top.shape[0], lanes)
        )
        local = logical - c * lanes
        gathered = jnp.take_along_axis(window, jnp.clip(local, 0, lanes - 1), axis=1)
        physical = jnp.where((local >= 0) & (local < lanes), gathered, physical)
    # Preserve the original int32 wraparound in the padding scatter.
    flat = token_base * top.shape[1] + lax.broadcasted_iota(jnp.int32, top.shape, 1)
    padding = top == -1
    physical = jnp.where(padding, (flat * 104729) % num_pages, physical)
    offset = jnp.where(padding, (flat * 15485863) % page_size, top % page_size)
    return physical * page_size + offset, jnp.sum(
        top != -1, axis=1, keepdims=True
    ).astype(jnp.int32)


def _kernel(
    cu_ref,
    top_ref,
    pages_ref,
    indices_ref,
    lengths_ref,
    *,
    page_size,
    num_pages,
    pages_per_seq,
):
    info = pltpu.get_tpu_info()
    sublanes = info.num_sublanes
    # A full sublane group keeps JAX's gather in take_along_axis form; a
    # one-row gather is canonicalized to a different unsupported gather.
    top = jnp.broadcast_to(top_ref[...], (sublanes, top_ref.shape[-1]))
    indices, lengths = map_indices(
        top,
        pages_ref[...],
        pl.program_id(0),
        page_size=page_size,
        num_pages=num_pages,
        pages_per_seq=pages_per_seq,
    )
    indices_ref[...] = indices[:1]
    lengths_ref[...] = lengths[:1]


def fused_metadata(
    topk: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    *,
    page_size: int,
    num_pages: int,
) -> tuple[jax.Array, jax.Array]:
    """Return flat physical KV indices and valid top-k counts, both int32.

    Cumulative lengths must be nondecreasing and start at zero. Top-k uses
    -1 padding; downstream SparseMLA requires valid entries to form a prefix.
    No cache is read here. Empty sequences and padded query suffixes retain
    jnp.repeat's semantics, including filling with the final sequence ID.
    """
    assert topk.ndim == 2 and topk.dtype == jnp.int32
    assert page_indices.ndim == 1 and page_indices.dtype == jnp.int32
    assert cu_q_lens.ndim == 1 and cu_q_lens.dtype == jnp.int32
    assert cu_q_lens.shape[0] >= 2
    assert page_size > 0 and num_pages > 0
    tokens, k = topk.shape
    max_seqs = cu_q_lens.shape[0] - 1
    assert page_indices.size > 0 and page_indices.size % max_seqs == 0
    pages_per_seq = page_indices.size // max_seqs
    lanes = pltpu.get_tpu_info().num_lanes
    padded = pl.cdiv(pages_per_seq, lanes) * lanes
    pages = jnp.pad(
        page_indices.reshape(max_seqs, pages_per_seq),
        ((0, 0), (0, padded - pages_per_seq)),
    )
    indices, lengths = pl.pallas_call(
        functools.partial(
            _kernel,
            page_size=page_size,
            num_pages=num_pages,
            pages_per_seq=pages_per_seq,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((None, 1, k), lambda t, cu: (t, 0, 0)),
                pl.BlockSpec(
                    (None, 1, padded), lambda t, cu: (_sequence_id(t, cu), 0, 0)
                ),
            ],
            out_specs=[
                pl.BlockSpec((None, 1, k), lambda t, cu: (t, 0, 0)),
                pl.BlockSpec((None, 1, 1), lambda t, cu: (t, 0, 0)),
            ],
            grid=(tokens,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((tokens, 1, k), jnp.int32),
            jax.ShapeDtypeStruct((tokens, 1, 1), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="CSA_fused_metadata",
    )(cu_q_lens, topk.reshape(tokens, 1, k), pages.reshape(max_seqs, 1, padded))
    return indices.reshape(-1), lengths.reshape(-1)

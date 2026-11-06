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
"""Lazy TC/SC dispatch with one prepared metadata bundle per TC branch."""

import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.sparse_gqa_prefill import owner_window
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import CostModel, build
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    UNION_WORD_BATCH_SIZE,
)


def _selected_tiles(blocks, lengths, *, ratio, groups, kv_tile):
    valid, tail, tail_live = tc._selection_validity(
        blocks, lengths, ratio=ratio, groups=groups
    )
    ids = jnp.concatenate((blocks, tail[:, None]), axis=1)
    live = jnp.concatenate((valid, tail_live[:, None]), axis=1)
    return ids // (kv_tile // ratio), live


def span_upper(blocks, lengths, *, ratio, groups, kv_tile, query_block):
    tiles, live = _selected_tiles(
        blocks, lengths, ratio=ratio, groups=groups, kv_tile=kv_tile
    )
    shape = (-1, query_block, tiles.shape[1])
    lo = jnp.where(live, tiles, jnp.iinfo(jnp.int32).max).reshape(shape).min((1, 2))
    hi = jnp.where(live, tiles, -1).reshape(shape).max((1, 2))
    return jnp.where(hi >= 0, hi - lo + 1, 0)


def sampled_lower(blocks, lengths, *, ratio, groups, kv_tile, query_block):
    # A subset union is a lower bound even for duplicate or invalid IDs.
    ids = blocks.reshape(-1, query_block, blocks.shape[1])[:, ::32]
    ls = lengths.reshape(-1, query_block)[:, ::32]
    tiles, live = _selected_tiles(
        ids.reshape(-1, ids.shape[-1]),
        ls.reshape(-1),
        ratio=ratio,
        groups=groups,
        kv_tile=kv_tile,
    )
    ntiles = (groups * ratio + kv_tile - 1) // kv_tile

    word_bits = jnp.iinfo(jnp.uint32).bits
    words = (tiles // word_bits).reshape(*ids.shape[:2], -1)
    bits = jnp.where(
        live,
        jnp.uint32(1) << (tiles % word_bits).astype(jnp.uint32),
        jnp.uint32(0),
    ).reshape(words.shape)

    def one(word):
        masks = jnp.where(words == word, bits, jnp.uint32(0))
        union = lax.reduce(masks, jnp.uint32(0), lax.bitwise_or, (1, 2))
        return lax.population_count(union).astype(jnp.int32)

    return lax.map(
        one,
        jnp.arange((ntiles + word_bits - 1) // word_bits),
        batch_size=UNION_WORD_BATCH_SIZE,
    ).sum(axis=0)


def bucket_upper(blocks, lengths, *, ratio, groups, kv_tile, query_block):
    tiles, live = _selected_tiles(
        blocks, lengths, ratio=ratio, groups=groups, kv_tile=kv_tile
    )
    ntiles = (groups * ratio + kv_tile - 1) // kv_tile
    width = (ntiles + 31) // 32
    bucket = jnp.where(live, tiles // width, 0).astype(jnp.uint32)
    bits = jnp.where(live, jnp.uint32(1) << bucket, jnp.uint32(0))
    union = lax.reduce(
        bits.reshape(-1, query_block, bits.shape[1]),
        jnp.uint32(0),
        lax.bitwise_or,
        (1, 2),
    )
    return jnp.minimum(lax.population_count(union) * width, ntiles)


def prepared_membership_is_unique(prepared, blocks, lengths, *, ratio, groups):
    valid, _, tail_live = tc._selection_validity(
        blocks, lengths, ratio=ratio, groups=groups
    )
    query_block = len(lengths) // prepared.native_bits.shape[0]
    bits = lax.bitcast_convert_type(prepared.native_bits, jnp.uint32)
    members = lax.population_count(bits).sum(axis=1)[:, :query_block].reshape(-1)
    expected = valid.sum(axis=1) + tail_live.astype(jnp.int32)
    return jnp.all(members == expected)


def sc_prefix_certificates(blocks, lengths, *, ratio):
    complete = jnp.minimum(jnp.maximum(lengths, 0) // ratio, blocks.shape[1])
    meaningful = jnp.arange(blocks.shape[1])[None, :] < complete[:, None]
    valid_prefix = jnp.all(
        jnp.where(
            meaningful, (blocks >= 0) & (blocks < lengths[:, None] // ratio), True
        )
    )
    unique_order = jnp.all(
        jnp.where(meaningful[:, 1:], blocks[:, 1:] > blocks[:, :-1], True)
    )
    return valid_prefix, unique_order


def _context_prefers_sc(lengths, starts, threshold: int | None):
    if threshold is None or threshold > jnp.iinfo(lengths.dtype).max:
        return jnp.bool_(False)
    rows = jnp.arange(len(lengths), dtype=starts.dtype)
    live = (lengths > 0) & (rows >= starts[0]) & (rows < starts[-1])
    maximum = jnp.max(jnp.where(live, lengths, 0), initial=0)
    return maximum >= threshold


def select_prefill(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    cost_model: CostModel | None = None,
    sc_min_context: int | None = None,
    kv_heads: int,
    ratio: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Return attention and route: 0 cheap TC, 1 prepared TC, 2/3 SC, 4 safe TC.

    Work estimates derive from the live tensor geometry and hardware. Other
    request schedules and scales retain their original TC semantics.
    """
    query_block, kv_tile = 256, 256

    def original(_):
        from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import _prefill_attention_tc

        result, _ = _prefill_attention_tc(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return result, jnp.int32(4)

    if (
        len(query) % query_block
        or type(k_scale) not in (int, float)
        or type(v_scale) not in (int, float)
        or k_scale != 1.0
        or v_scale != 1.0
        or not owner_window.is_supported_by_owner_window(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
        )
    ):
        return original(None)

    if cost_model is None:
        cost_model = build(
            query, blocks, table, cache, starts, kv_heads=kv_heads, ratio=ratio
        )

    page_size = cache.shape[1]
    real = jnp.diff(starts)
    eligible = (
        (starts[0] == 0)
        & (starts[-1] == len(query))
        & (jnp.sum(real > 0) == 1)
        & jnp.all(real >= 0)
        & jnp.all((lengths >= 0) & (lengths <= table.shape[1] * page_size))
    )
    selected = (
        jnp.minimum(jnp.maximum(lengths, 0) // ratio, blocks.shape[1]) * ratio
        + jnp.maximum(lengths, 0) % ratio
    )
    pad = (-len(query)) % 8
    selected = jnp.pad(selected, (0, pad))
    sc_tiles = jnp.maximum(
        1, (selected.reshape(-1, 8).max(1) + kv_tile - 1) // kv_tile
    ).sum()
    written_pages = (jnp.maximum(lengths.max(), 0) + page_size - 1) // page_size
    sc_ms = cost_model.sc_ms(sc_tiles, written_pages)

    prefer_sc = _context_prefers_sc(lengths, starts, sc_min_context)

    def choose(_):
        def in_extent(tab):
            groups = tab.shape[1] * page_size // ratio
            ntiles = (groups * ratio + kv_tile - 1) // kv_tile
            words = tc._metadata_native_align(ntiles * (kv_tile // (ratio * 32)))

            def estimate(visits):
                return cost_model.tc_ms(words, visits)

            def consume(prepared, route):
                raw = tc._prefill(
                    query,
                    blocks,
                    lengths,
                    tab,
                    cache,
                    None,
                    ratio=ratio,
                    kv_heads=kv_heads,
                    query_block=query_block,
                    kv_tile=kv_tile,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    bf16_io=True,
                    prepared=prepared,
                    token_minor_query=True,
                )
                return tc._gate_output(raw, gate, lengths), jnp.int32(route)

            def cheap_tc(_):
                return consume(None, 0)

            def sc_intent(route):
                return jnp.zeros_like(query, dtype=jnp.bfloat16), jnp.int32(route)

            kwargs = dict(
                ratio=ratio, groups=groups, kv_tile=kv_tile, query_block=query_block
            )

            def exact(_):
                prepared = tc._prepare_prefill_metadata(
                    blocks,
                    lengths,
                    tab,
                    ratio=ratio,
                    query_block=query_block,
                    kv_tile=kv_tile,
                    physical_page_size=page_size,
                    cache_dtype=cache.dtype,
                )

                prefer_tc = estimate(prepared.active_counts.sum()) <= sc_ms
                if sc_min_context is not None:
                    prefer_tc = prefer_tc & ~prefer_sc
                return lax.cond(
                    prefer_tc,
                    lambda _: consume(prepared, 1),
                    lambda _: certify_sc(prepared, 3),
                    None,
                )

            def certify_sc(prepared, route):
                valid_prefix, unique_order = sc_prefix_certificates(
                    blocks, lengths, ratio=ratio
                )

                def unordered(_):
                    if prepared is None:
                        return exact(None)
                    return lax.cond(
                        prepared_membership_is_unique(
                            prepared, blocks, lengths, ratio=ratio, groups=groups
                        ),
                        lambda _: sc_intent(route),
                        lambda _: consume(prepared, 4),
                        None,
                    )

                return lax.cond(
                    valid_prefix,
                    lambda _: lax.cond(
                        unique_order, lambda _: sc_intent(route), unordered, None
                    ),
                    lambda _: consume(prepared, 4),
                    None,
                )

            def coarse(_):
                return lax.cond(
                    estimate(bucket_upper(blocks, lengths, **kwargs).sum()) <= sc_ms,
                    cheap_tc,
                    exact,
                    None,
                )

            def sampled(_):
                return lax.cond(
                    estimate(sampled_lower(blocks, lengths, **kwargs).sum()) > sc_ms,
                    lambda _: certify_sc(None, 2),
                    coarse,
                    None,
                )

            def spans(_):
                return lax.cond(
                    estimate(span_upper(blocks, lengths, **kwargs).sum()) <= sc_ms,
                    cheap_tc,
                    sampled,
                    None,
                )

            def ordered(_):
                return lax.cond(
                    estimate(len(query) // query_block * ntiles) <= sc_ms,
                    cheap_tc,
                    spans,
                    None,
                )

            if sc_min_context is None:
                return ordered(None)
            return lax.cond(prefer_sc, lambda _: certify_sc(None, 2), ordered, None)

        return tc._metadata_prefix_dispatch(
            in_extent,
            blocks,
            lengths,
            table,
            tokens=len(query),
            query_block=query_block,
            ratio=ratio,
            page_size=page_size,
            kv_tile=kv_tile,
        )

    chosen, route = lax.cond(eligible, choose, original, None)

    def run_sc(_):
        return owner_window.owner_window_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
            workspace_bytes=None,
        )

    output = lax.cond((route == 2) | (route == 3), run_sc, lambda _: chosen, None)
    return output, route

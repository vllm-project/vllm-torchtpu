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
"""QSA cache writes after a Qwen3.8 whole-step megakernel call.

The whole-step kernel attends each QSA layer's own step tokens from VMEM and writes no
cache. Afterwards, in the same JAX callable, every QSA layer's step tokens are written
through the stock writer sequence (`pooled_qsa.qsa_cache_writes`, the code
`_pooled_qsa` runs) from the kernel's k / v / raw-indexer-key projections, so the pool
holds the bytes the stock layers would have left for the same projections: main K/V
(FP8-quantised, paired head slot), the completed groups' compressed keys (packed slot)
and the ring rows + tags, verify drafts included.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
    compressed_pairs,
    main_slot_rows,
    qsa_cache_writes,
    write_compressed,
    write_compressed_pairs,
    write_compressed_pairs_unique,
    write_main_cache,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    _indexer_updates,
    _token_metadata,
    produce_kv,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import pack_rows
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import (
    read_ring_and_tags,
    write_ring_and_tags,
)


@dataclasses.dataclass(frozen=True)
class QSALayerCache:
    """Static per-QSA-layer cache addressing (`Qwen4ExpQSA.bind_pool` keywords)."""

    compressed_slot: int = 0
    main_slot: int = -1
    k_scale: float = 1.0
    v_scale: float = 1.0
    # The main cache group (`main_cache_name`): paired layers (main_slot >= 0)
    # sharing it write the two head pairs of the same rows.
    main_group: str = ""


def wholestep_qsa_writes(
    pool: Any,
    rows: list[tuple[Any, Any, Any]],
    norms: list[tuple[Any, Any]],
    tables: list[tuple[Any, Any, Any]],
    layers: list[QSALayerCache],
    positions: Any,
    trig: Any,
    starts: Any,
    context: Any,
    *,
    compress_ratio: int,
    norm_eps: float,
    rotary_dim: int,
    max_query_len: int,
    num_requests: int | None = None,
    batched: bool = True,
    dcp: tuple[Any, int] | None = None,
    compressed_dcp: tuple[Any, int] | None = None,
) -> Any:
    """Write every QSA layer's step tokens, in layer order; returns the pool.

    rows[i]: layer i's `(key, value, index_k)` projections `[T, kv*D]`, `[T, kv*D]`,
      `[T, index_dim]` (the raw indexer key: `index_qk[:, -index_dim:]`).
    norms[i]: `(k_norm, index_k_norm)` weights. tables[i]: the runner's flat
      `(ring_ids, main_table, compressed_table)` of layer i's cache groups.
    num_requests: the kernel's static request count (`M // T`). Requests past it are
    idle
      padding (the driver calls the kernel only for steps that fit its bucket), so they
      are
      cropped before the writers rather than read, merged and written back as no-ops.
    batched: one ring read / write and one (vmapped) indexer update for all layers, the
      token metadata once (`_batched_writes`); same bytes as the per-layer stock sequence.
      The fused-writer path (`VLLM_TPU_QSA_FUSED_KV_WRITE`) keeps the per-layer sequence.
    dcp: `(rank int32 [1], size)` under VLLM_TPU_QSA_MAIN_KV_DCP: main K/V rows go only to
      their owner rank (`qsa_cache_writes`' rule); compressed keys and rings stay replicated
      unless `compressed_dcp` (VLLM_TPU_QSA_COMPRESSED_DCP, the same pair): then each rank writes
      only its owned groups' compressed keys.
    """
    n = len(context)
    nreq = n if num_requests is None else min(n, num_requests)
    context, starts = context[:nreq], starts[: nreq + 1]
    tables = [
        (r.reshape(n, -1)[:nreq, 0], m.reshape(n, -1)[:nreq], c.reshape(n, -1)[:nreq])
        for r, m, c in tables
    ]
    if batched and not envs.VLLM_TPU_QSA_FUSED_KV_WRITE:
        if _stacked_ok(pool, layers, len(positions)):
            return _stacked_writes(
                pool, rows, norms, tables, layers, positions, trig, starts, context,
                compress_ratio=compress_ratio, norm_eps=norm_eps, rotary_dim=rotary_dim,
                dcp=dcp, compressed_dcp=compressed_dcp,
            )
        return _batched_writes(
            pool, rows, norms, tables, layers, positions, trig, starts, context,
            compress_ratio=compress_ratio, norm_eps=norm_eps, rotary_dim=rotary_dim, dcp=dcp,
            compressed_dcp=compressed_dcp,
        )
    for (key, value, index_k), (nk, nik), (ring, main, comp), lay in zip(
        rows, norms, tables, layers, strict=True
    ):
        k, v = produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim)
        pool, _, _, _ = qsa_cache_writes(
            pool,
            k,
            v,
            index_k,
            positions,
            nik,
            trig,
            starts,
            context,
            ring,
            main,
            comp,
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            rotary_dim=rotary_dim,
            k_scale=lay.k_scale,
            v_scale=lay.v_scale,
            compressed_slot=lay.compressed_slot,
            main_slot=lay.main_slot,
            max_query_len=max_query_len,
            dcp=dcp,
            compressed_dcp=compressed_dcp,
        )
    return pool


def _batched_writes(pool, rows, norms, tables, layers, positions, trig, starts, context, *,
                    compress_ratio, norm_eps, rotary_dim, dcp=None, compressed_dcp=None):
    """`qsa_cache_writes`' unfused sequence for all layers at once.

    Each layer's main K/V and compressed keys still go through their own writer (paired
    /
    packed slots of shared pages need the stock writers' read-modify-write), but the
    ring
    rows (one page per layer and request, disjoint) are read and written by one call
    each,
    and the indexer update runs once over a stacked layer axis. No layer reads what
    another
    layer writes in a step, so the pool ends with the stock per-layer bytes.
    """
    nreq, L = len(context), len(layers)
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(nreq, dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(len(positions), starts, context, slots)
    fp8 = pool.dtype == jnp.float8_e4m3fn
    main_valid, span = valid, pool.shape[1]
    if dcp is not None:  # `qsa_cache_writes`' owner rule
        rank, size = dcp
        span = size * pool.shape[1]
        owner = (logical // pool.shape[1]) % size
        main_valid = valid & (owner == rank.reshape(()).astype(owner.dtype))
    for (key, value, _), (nk, _), (_, main, _), lay in zip(rows, norms, tables, layers, strict=True):
        k, v = produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim)
        page = jnp.where(main_valid, main[req, logical // span], len(pool))
        packed = pack_rows(k, v, fp8=fp8, k_scale=lay.k_scale, v_scale=lay.v_scale)
        slot_kw = {"slot": lay.main_slot} if lay.main_slot >= 0 else {}
        pool = write_main_cache(pool, packed, page, logical, main_valid, **slot_kw)
    raw = jnp.stack([r[2] for r in rows])
    ring_rows = math.ceil(compress_ratio / 8) * 8
    ids = jnp.concatenate([t[0] for t in tables])
    ring, tags = read_ring_and_tags(pool, ids, ring_rows * raw.shape[-1], ring_rows * 2)
    ring = ring.reshape(L, nreq, ring_rows, -1)
    tags = tags.reshape(L, nreq, ring_rows, 2)
    update = jax.vmap(
        lambda raw, ring, tags, nik: _indexer_updates(
            raw,
            positions,
            ring,
            tags,
            nik,
            trig,
            starts,
            context,
            slots,
            compress_ratio,
            norm_eps,
            rotary_dim,
        )
    )
    completed, complete, _, _, ring, tags = update(
        raw, ring, tags, jnp.stack([n[1] for n in norms])
    )
    for o, ((_, _, comp), lay) in enumerate(zip(tables, layers, strict=True)):
        pool = write_compressed(pool, completed[o], complete[o], req, logical, starts, context,
                                comp, compress_ratio, lay.compressed_slot,
                                **({"dcp": compressed_dcp} if compressed_dcp is not None else {}))
    ids = jnp.where(jnp.tile(active, L), ids, 0)
    return write_ring_and_tags(
        pool, ring.reshape(L * nreq, -1), tags.reshape(L * nreq, -1), ids
    )


def _stacked_ok(pool, layers, count: int) -> bool:
    """`_stacked_writes` gives the stock bytes for this step (else `_batched_writes`)."""
    if not envs.VLLM_TPU_MK_STACKED_QSA_WRITES or count % 128 == 0:
        return False  # 128-token tiles: the stock writers take their tiled kernels
    if (envs.VLLM_TPU_QSA_COMPRESSED_RUN_WRITE
            and count >= envs.VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS):
        return False
    paired = [lay.main_slot >= 0 for lay in layers]
    if any(paired):
        # Paired rows are merged per main group; every layer must name its group.
        return all(paired) and all(lay.main_group for lay in layers) and (
            pool.dtype == jnp.float8_e4m3fn and pool.shape[2] == 1
        )
    return True


def _stacked_writes(pool, rows, norms, tables, layers, positions, trig, starts, context, *,
                    compress_ratio, norm_eps, rotary_dim, dcp=None, compressed_dcp=None):
    """`_batched_writes` with every per-layer writer stacked over the layer axis.

    For decode-sized steps (not a multiple of 128 tokens) the stock main K/V writer is
    a gather / merge / scatter and the compressed-key writer one ordered per-pair
    kernel, so per layer it is a few dozen tiny ops. Here K/V production, packing,
    pair metadata and values run once over a stacked layer axis; the main rows go out
    in one gather and one scatter and the compressed keys in one per-pair call over
    all layers in layer order. Paired layers of one main group (two layers' head
    pairs in the same rows) are merged in layer order before the scatter, which is
    what the stock sequence leaves; distinct groups touch distinct pages.
    """
    nreq, L = len(context), len(layers)
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(nreq, dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(len(positions), starts, context, slots)
    fp8 = pool.dtype == jnp.float8_e4m3fn
    main_valid, span = valid, pool.shape[1]
    if dcp is not None:  # `qsa_cache_writes`' owner rule
        rank, size = dcp
        span = size * pool.shape[1]
        owner = (logical // pool.shape[1]) % size
        main_valid = valid & (owner == rank.reshape(()).astype(owner.dtype))
    offsets = logical % pool.shape[1]

    keys = jnp.stack([r[0] for r in rows])
    values = jnp.stack([r[1] for r in rows])
    k, v = jax.vmap(
        lambda key, value, nk: produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim)
    )(keys, values, jnp.stack([n[0] for n in norms]))
    if fp8:
        scales = jnp.asarray([[lay.k_scale, lay.v_scale] for lay in layers], jnp.float32)
        packed = jax.vmap(lambda k, v, s: pack_rows(k, v, fp8=True, k_scale=s[0], v_scale=s[1]))(
            k, v, scales
        )
    else:
        packed = jax.vmap(lambda k, v: pack_rows(k, v, fp8=False))(k, v)
    if layers[0].main_slot >= 0:
        groups: dict[str, list[int]] = {}
        for i, lay in enumerate(layers):
            groups.setdefault(lay.main_group, []).append(i)
        lead = [g[0] for g in groups.values()]
        main = jnp.stack([tables[i][1] for i in lead])
        page = jnp.where(main_valid[None], main[:, req, logical // span], len(pool))
        old = pool[jnp.minimum(page, len(pool) - 1), offsets[None]]
        packing = pool.shape[-2]
        merged = []
        for g, members in enumerate(groups.values()):
            row = old[g]
            for i in members:
                mine = (jnp.arange(packing) // 2 == layers[i].main_slot)[:, None]
                row = jnp.where(mine, main_slot_rows(packed[i], layers[i].main_slot), row)
            merged.append(row)
        pool = pool.at[page, offsets[None]].set(jnp.stack(merged), mode="drop")
    else:
        main = jnp.stack([t[1] for t in tables])
        page = jnp.where(main_valid[None], main[:, req, logical // span], len(pool))
        pool = pool.at[page, offsets[None]].set(packed, mode="drop")

    raw = jnp.stack([r[2] for r in rows])
    ring_rows = math.ceil(compress_ratio / 8) * 8
    ids = jnp.concatenate([t[0] for t in tables])
    ring, tags = read_ring_and_tags(pool, ids, ring_rows * raw.shape[-1], ring_rows * 2)
    ring = ring.reshape(L, nreq, ring_rows, -1)
    tags = tags.reshape(L, nreq, ring_rows, 2)
    update = jax.vmap(
        lambda raw, ring, tags, nik: _indexer_updates(
            raw, positions, ring, tags, nik, trig, starts, context, slots,
            compress_ratio, norm_eps, rotary_dim,
        )
    )
    completed, complete, _, _, ring, tags = update(
        raw, ring, tags, jnp.stack([n[1] for n in norms])
    )
    dcp_kw = {"dcp": compressed_dcp} if compressed_dcp is not None else {}
    pair, write, mask, vals = jax.vmap(
        lambda keys, done, table: compressed_pairs(
            pool, keys, done, req, logical, starts, context, table, compress_ratio, 0, **dcp_kw
        )
    )(completed, complete, jnp.stack([t[2] for t in tables]))
    layer_pairs = pool.shape[1] // (2 * compress_ratio)
    pairs_per_page = math.prod(pool.shape[1:-1]) // (4 if fp8 else 2)
    for lay in layers:  # `compressed_pairs`' slot bound
        assert (lay.compressed_slot + 1) * layer_pairs <= pairs_per_page, (
            "packed compressed-key slot exceeds its pool page"
        )
    shift = jnp.asarray([lay.compressed_slot * layer_pairs for lay in layers], pair.dtype)
    pair = pair + shift[:, None]
    writer = (write_compressed_pairs_unique if envs.VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE
              else write_compressed_pairs)
    pool = writer(
        pool, pair.reshape(-1), write.reshape(-1), mask.reshape(-1, *mask.shape[2:]),
        vals.reshape(-1, *vals.shape[2:]),
    )
    ids = jnp.where(jnp.tile(active, L), ids, 0)
    return write_ring_and_tags(
        pool, ring.reshape(L * nreq, -1), tags.reshape(L * nreq, -1), ids
    )

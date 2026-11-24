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

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
    qsa_cache_writes,
    write_compressed,
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
    compressed_dcp: tuple[Any, int] | None = None, kernel_values: Any = None,  # q38 L2 WFOLD / L1-KVW (end of module)
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
      `VLLM_TPU_QSA_FUSED_KV_WRITE` (D73) does not change this choice (q38 L2 D73 gate): D73's
      per-layer fused writer is for stock steps only; on the whole step it regressed at 64 rows
      (MM, +0.095 ms), so `batched` always takes `_batched_writes`. `batched=False` keeps the
      per-layer stock sequence (where `qsa_cache_writes` still honours the env).
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
    if batched or _kvw_unbatched(kernel_values):  # q38 L2 D73 gate: D14 ignores VLLM_TPU_QSA_FUSED_KV_WRITE (stock steps only); q38 L1-KVW
        return _kvw_writer(nreq, kernel_values, _batched_writes)(  # q38 L1-KVW: `_batched_writes` unless VLLM_TPU_MK_QSA_WRITE_ALL
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


# q38 L1-KVW begin
# KVW-ALL (q38 L2 2e68a7f15 / d1b5247ca) and QSA-WRITE-FOLD stage A (q38 L2 094e7d381 / 6150b16c9), wired into the
# composed whole step by q38 L1-V6 (STACK-D v6). VLLM_TPU_MK_QSA_WRITE_ALL=1: the batched whole-step writes go through
# `qwen38_qsa_write_all.batched_writes_all` (every QSA layer's main K/V rows and compressed pairs in one Pallas call; same
# arguments, same pool bytes as `_batched_writes`). With VLLM_TPU_MK_QSA_WRITE_KVAL=1 as well, the whole-step call hands in
# `kernel_values` (the kernel's own key rows, completed-group keys and rings, `qwen38_qsa_write_all.KernelValues`, for the
# kernel's M // T requests), cropped here to the step's requests. Both flags off and no kernel values: `_batched_writes`,
# exactly as before. Appended at the end of the module, with two existing lines of `wholestep_qsa_writes` extended in
# place (the signature's last parameter line and the batched dispatch), so the flag-off trace keeps its source lines.
KVW_TRACES: dict = {}


def _kvw_unbatched(kernel_values: Any) -> bool:
    """False; raises when the kernel's values reach the per-layer (unbatched) sequence, which cannot take them."""
    if kernel_values is not None:
        raise ValueError("kernel_values needs the batched whole-step writes")
    return False


def _kvw_writer(nreq: int, kernel_values: Any, default: Any) -> Any:
    """`default` (`_batched_writes`) with VLLM_TPU_MK_QSA_WRITE_ALL off and no kernel values; else
    `batched_writes_all`, given the kernel's values cropped to the step's `nreq` requests when there are any. One
    trace-time record (and log line) per kernel-values mode."""
    from vllm_torchtpu import envs

    if kernel_values is None and not envs.VLLM_TPU_MK_QSA_WRITE_ALL:
        return default
    import functools

    from vllm_torchtpu.megakernel_runtime.qwen38_qsa_write_all import batched_writes_all

    kval = kernel_values is not None
    key = "kval" if kval else "all"
    KVW_TRACES[key] = KVW_TRACES.get(key, 0) + 1
    if KVW_TRACES[key] == 1:
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 L1-KVW wholestep_qsa_writes traced: all-layer writer (batched_writes_all), kernel values %s, %d requests",
            "on" if kval else "off", nreq,
        )
    if kval:
        kernel_values = kernel_values._replace(ck=kernel_values.ck[:, :nreq], ring=kernel_values.ring[:, :nreq])
    return functools.partial(batched_writes_all, kernel_values=kernel_values)
# q38 L1-KVW end

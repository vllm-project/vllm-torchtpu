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
"""q38 L2 KVW-ALL: one all-layer post-kernel QSA cache writer for the Qwen3.8 whole step.

A drop-in for `qwen38_wholestep_cache._batched_writes` (same arguments, same pool bytes) behind
VLLM_TPU_MK_QSA_WRITE_ALL (default off, hashed into the TPU compile key). The switch in
`wholestep_qsa_writes` is L1's wiring; this module does not touch it.

Today's batched path batches the ring read / write and the indexer update, but every QSA layer still
runs two serial writers: the main K/V writer (at decode sizes an XLA scatter, or the paired gather +
scatter) and the compressed-key `per_pair_kernel` (one grid step per token, a waited read and a
waited write). Here the values come from the same XLA code (`produce_kv`, `pack_rows`,
`main_slot_rows`, the vmapped `_indexer_updates`, `compressed_pairs`) and ONE Pallas call writes every
layer's main rows and compressed pairs:

1. every main write that needs no old bytes (one DMA per request run of `T` page-contiguous rows,
   else one per row), every partial main-row read and every partial compressed-pair read are issued
   before the first wait;
2. the reads are waited and merged in VMEM (main rows by a byte mask, compressed pairs by BF16 half);
3. every merged write is issued, then every write is waited.

Rings + tags keep the one batched `write_ring_and_tags` call (the fused KVF writer keeps them out of
its kernel too: the served-hang stopgap in `pooled_qsa.qsa_cache_writes`). `ring_tags` lets the caller
hand in the ring / tag read it already made for the kernel inputs instead of a second read.

This is MM's lean fused KV writer pattern (q38mm DSTOCK-B, `fused_kv_write._lean_layer_write`:
compacted SMEM work lists, every read in flight before the first wait, one vector merge, every write
in flight), extended over the layer axis. Credit: MM.

Exactness (same bytes as `_batched_writes`):
- Paired main pages (`main_slot` >= 0) hold two layers' head pairs in one row. Rows that several
  layers write for the same token are folded in XLA, in layer order, with the stock merge
  (`jnp.where` over the packing axis, as `_write_main_cache_paired`), into one new row plus a byte
  mask; only the last writer of the row writes, and it reads the old row only if the mask does not
  cover every byte (the lone layer of a pair). The kernel's merge is `(old & ~mask) | (new & mask)`
  on 32-bit views of the row, the mask in the row's own layout, so it is a byte select.
- Compressed pairs: `compressed_pairs` names one writer per pair; partial pairs (one BF16 half this
  step) are read and merged by half, full pairs are written without a read (`per_pair_kernel`).
- Well-formed steps only, as the stock writers: valid destinations are unique within a layer (the
  XLA scatter this replaces is unordered for duplicates), and two layers share a main row only for
  the same token (paired pages share the block table; each page belongs to one request and column).

q38 L2 WFOLD (QSA-WRITE-FOLD stage A), VLLM_TPU_MK_QSA_WRITE_KVAL (effective only with
VLLM_TPU_MK_QSA_WRITE_ALL): `kernel_values` hands in the whole-step kernel's own cache values instead of
recomputing them here. The kernel already holds, per QSA layer, every row's normed + roped key (`qf_kv`,
emitted as `qsa_kv_new[..., :D]`), and per request the compressed key of the group the step completes
(`qsa_ck_new`) and the raw index-key ring after the step (`qsa_ring_new`). With them the writer drops
`produce_kv` (the key's gemma norm + RoPE) and the vmapped `_indexer_updates` (the group mean, norm and
index RoPE, the per-token member gathers, the ring merge); it keeps the FP8 packing (`pack_rows`), the
write addressing (`main_plan` / `pair_plan`), the ring / tag read and write, and the stock tag rule.
Preconditions, the kernel's own (`qwen38_wholestep_inputs.qsa_step_inputs`: every live request has its
`T` verify rows): every live request has exactly `T` rows and completes at most one compressed group in
the step (T <= compress_ratio; the whole step: T = 4 = ratio). Idle requests' kernel values are never
used (their rows are invalid; their ring / tags are written back as read, as the stock scatter leaves
them).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs, hxsmall
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compressed_pairs, main_slot_rows
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

KERNEL_NAME = "qwen38_qsa_write_all"
# DMA semaphores: direct main rows, main runs, main reads, merged main writes, pair reads, pair writes.
_MAIN_ROW, _MAIN_RUN, _MAIN_READ, _MAIN_MERGED, _PAIR_READ, _PAIR_WRITE = range(6)


def qsa_write_all_enabled() -> bool:
    """VLLM_TPU_MK_QSA_WRITE_ALL: the whole step's QSA cache writes through `batched_writes_all`.

    Registered in envs (hashed), read at trace time by the caller's dispatch."""
    return bool(envs.VLLM_TPU_MK_QSA_WRITE_ALL)


def qsa_write_kval_enabled() -> bool:
    """q38 L2 WFOLD: VLLM_TPU_MK_QSA_WRITE_KVAL, effective only with VLLM_TPU_MK_QSA_WRITE_ALL.

    Both on: the whole-step call returns the kernel's QSA cache values and the writes go through
    `batched_writes_all(kernel_values=...)`. Registered in envs (hashed), read when the call is built."""
    return bool(envs.VLLM_TPU_MK_QSA_WRITE_ALL) and bool(envs.VLLM_TPU_MK_QSA_WRITE_KVAL)


class KernelValues(NamedTuple):
    """q38 L2 WFOLD: the whole-step kernel's own QSA cache values for one step, on one rank.

    keys [L, T, kv * D]: every QSA layer's normed + roped key rows (`qsa_kv_new[o, :T, :D]`);
    ck   [L, R, index_dim]: per request, the key of the compressed group the step completes
         (`qsa_ck_new[o, :R, 0]`);
    ring [L, R, rows, index_dim]: per request, the raw index-key ring after the step (`qsa_ring_new`).
    The values stay `rows[i][1]` (already the kernel's `qsa_kv_new[o, :T, D:]`)."""

    keys: Any
    ck: Any
    ring: Any


def kernel_values_from_outputs(kv: Any, ck_new: Any, ring_new: Any, tokens: int) -> KernelValues:
    """`KernelValues` from the whole step's per-rank outputs: `kv` [L, M, 2D] (`qsa_kv_new`), `ck_new`
    [L, M // T, 1, index_dim] (`qsa_ck_new`), `ring_new` [L, M // T, 8, index_dim] (`qsa_ring_new`); the
    key rows are cropped to the step's `tokens` rows (requests are cropped by `wholestep_qsa_writes`)."""
    hd = kv.shape[-1] // 2
    return KernelValues(kv[:, :tokens, :hd], ck_new[:, :, 0], ring_new)


def _kernel_kv(key_rows: Any, value: Any, head_dim: int) -> tuple[Any, Any]:
    """`produce_kv`'s output shapes from the kernel's key rows (the values are already the kernel's)."""
    return (key_rows.reshape(len(key_rows), key_rows.shape[1] // head_dim, head_dim),
            value.reshape(len(value), value.shape[1] // head_dim, head_dim))


def check_kernel_values(kernel_values: KernelValues, rows: list[Any], tokens: int, requests: int,
                        ring_rows: int, compress_ratio: int) -> None:
    """Raise unless `kernel_values` has the step's shapes and the step its preconditions (no silent fallback)."""
    L = len(rows)
    keys, ck, ring = kernel_values
    want = {"keys": (L, tokens, rows[0][0].shape[-1]), "ck": (L, requests, rows[0][2].shape[-1]),
            "ring": (L, requests, ring_rows, rows[0][2].shape[-1])}
    for name, x in zip(("keys", "ck", "ring"), (keys, ck, ring)):
        if tuple(x.shape) != want[name]:
            raise ValueError(f"{KERNEL_NAME}: kernel_values.{name} {tuple(x.shape)} != {want[name]}")
    if keys.dtype != rows[0][1].dtype:
        raise ValueError(f"{KERNEL_NAME}: kernel key rows {keys.dtype} != value rows {rows[0][1].dtype}")
    if tokens > requests * compress_ratio:
        raise ValueError(f"{KERNEL_NAME}: kernel values cover one completed group per request: {tokens} rows > "
                         f"{requests} requests x ratio {compress_ratio}")


def kernel_indexer_updates(kernel_values: KernelValues, ring: Any, tags: Any, positions: Any, starts: Any,
                           context: Any, slots: Any, logical: Any, valid: Any, req: Any, *, compress_ratio: int):
    """`_indexer_updates`' `(completed, complete, ring, tags)` (vmapped over the layers) from the kernel's values.

    completed [L, T, index_dim]: every token carries its request's completed-group key; `compressed_pairs` reads
      only the group-end tokens of the step (one per live request: precondition), so the others never matter.
    complete [L, T]: the stock rule. ring / tags [L, R, rows, ...]: live requests take the kernel's ring and the
      stock tag rule; idle ones keep the rows as read (the stock scatter drops them)."""
    L, count, rows = ring.shape[0], len(positions), ring.shape[2]
    completed = kernel_values.ck[:, req]
    complete = jnp.broadcast_to(valid & (logical % compress_ratio == compress_ratio - 1), (L, count))
    # The stock tag rule (`_indexer_updates`): lane l keeps the newest position = l (mod rows) below the end.
    lengths = starts[1:] - starts[:-1]
    end = context + lengths
    lanes = jnp.arange(rows)[None]
    last = end[:, None] - 1 - ((end[:, None] - 1 - lanes) % rows)
    local = starts[:-1, None] + last - context[:, None]
    new_lane = (last >= context[:, None]) & (last < end[:, None])
    pp = positions[jnp.clip(local, 0, count - 1)]
    tag_pairs = jnp.stack([pp, jnp.zeros_like(pp)], -1)
    old_tags = jnp.where((context > 0)[None, :, None, None], tags, -1)
    tt = jnp.where(new_lane[None, :, :, None], tag_pairs[None], old_tags).astype(tags.dtype)
    live = ((lengths > 0) & (slots >= 0))[None, :, None, None]
    return (completed, complete, jnp.where(live, kernel_values.ring.astype(ring.dtype), ring),
            jnp.where(live, tt, tags))


def check_supported(pool: Any, layers: list[Any]) -> None:
    """Raise unless the pool geometry is the one this writer covers (no silent fallback)."""
    packing = pool.shape[-2] if pool.ndim == 5 else None
    fp8 = pool.dtype == jnp.float8_e4m3fn
    if pool.ndim != 5 or pool.shape[2] != 1:
        raise ValueError(f"{KERNEL_NAME}: needs a [pages, rows, 1, packing, lanes] pool, got {pool.shape}")
    if not ((fp8 and packing == 4) or (pool.dtype == jnp.bfloat16 and packing == 2)):
        raise ValueError(f"{KERNEL_NAME}: needs FP8 x 4 or BF16 x 2 carriers, got {pool.dtype} x {packing}")
    for lay in layers:
        if lay.main_slot >= 0 and not (fp8 and lay.main_slot < packing // 2):
            raise ValueError(f"{KERNEL_NAME}: paired main slot {lay.main_slot} needs an FP8 x 4 pool")


def _run_length(tokens: int, requests: int) -> int:
    """Rows per request when the step is uniform (the whole step's verify windows), else 1."""
    if requests and tokens % requests == 0 and tokens // requests > 1:
        return tokens // requests
    return 1


def main_plan(pool, rows, norms, tables, layers, positions, trig, starts, req, logical, main_valid, span,
              *, norm_eps, rotary_dim, keys=None):
    """Main K/V rows of every layer as one write list (layer-major entries `l * T + t`).

    keys: optional [L, T, kv * D] key rows (q38 L2 WFOLD: the kernel's `KernelValues.keys`) instead of
    `produce_kv` (gemma norm + RoPE of `rows[i][0]`).

    Returns `new` [L*T, 1, packing, lanes] (pool dtype, each row already merged with the earlier
    layers that write it), `take` (same shape, unsigned, all-ones bytes = written) or None when no
    layer is paired, `page` / `offset` / `mode` [L*T] (mode 0 skip, 1 write, 2 read-merge-write), and
    the request runs (`run_page`, `run_offset`, `run_src`, `run_live` [L*R]) with their static `run`.
    """
    L, T = len(layers), len(positions)
    R = len(starts) - 1
    fp8 = pool.dtype == jnp.float8_e4m3fn
    n_pages, page_rows, packing = pool.shape[0], pool.shape[1], pool.shape[-2]
    paired = any(lay.main_slot >= 0 for lay in layers)
    offset = (logical % page_rows).astype(jnp.int32)
    news, mines = [], []
    for l, ((key, value, _), (nk, _), lay) in enumerate(zip(rows, norms, layers, strict=True)):
        if keys is None:
            k, v = produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim)
        else:  # q38 L2 WFOLD: the kernel's own key rows
            k, v = _kernel_kv(keys[l], value, nk.shape[0])
        packed = pack_rows(k, v, fp8=fp8, k_scale=lay.k_scale, v_scale=lay.v_scale)
        if lay.main_slot >= 0:
            packed = main_slot_rows(packed, lay.main_slot)
            mine = jnp.arange(packing) // 2 == lay.main_slot
        else:
            mine = jnp.ones((packing,), jnp.bool_)
        news.append(packed.astype(pool.dtype))
        mines.append(mine)
    # Every layer's destination page [L, T]: one gather when the tables share a shape.
    mains, col = [t[1] for t in tables], logical // span
    if len({m.shape for m in mains}) == 1:
        page = jnp.stack(mains)[:, req, col].astype(jnp.int32)
    else:
        page = jnp.stack([m[req, col] for m in mains]).astype(jnp.int32)
    live = main_valid[None] & (page >= 0) & (page < n_pages)
    page = jnp.where(live, page, 0)
    # Rows that several layers write for the same token (paired pages): only the last writer writes.
    same = live[:, None] & live[None, :] & (page[:, None] == page[None, :])  # [L, L, T]
    after = jnp.triu(jnp.ones((L, L), jnp.bool_), k=1)  # after[p, l]: layer l writes after layer p
    later = jnp.any(same & after[:, :, None], axis=1)  # [L, T]
    if paired:
        # Fold each such row in layer order with the stock merge (`jnp.where` over the packing axis).
        acc_new, acc_take = [], []
        for l in range(L):
            base_new = jnp.zeros_like(news[l])
            base_take = jnp.zeros((T, packing), jnp.bool_)
            for p in range(l):
                base_new = jnp.where(same[p, l][:, None, None, None], acc_new[p], base_new)
                base_take = jnp.where(same[p, l][:, None], acc_take[p], base_take)
            acc_new.append(jnp.where(mines[l][None, None, :, None], news[l], base_new))
            acc_take.append(mines[l][None, :] | base_take)
        take_rows = jnp.stack(acc_take)  # [L, T, packing]
        full = jnp.all(take_rows, axis=-1)
        new = jnp.concatenate(acc_new)
        udt = jnp.uint8 if jnp.dtype(pool.dtype).itemsize == 1 else jnp.uint16
        ones = jnp.asarray(jnp.iinfo(udt).max, udt)
        mask = take_rows.reshape(L * T, packing)[:, None, :, None]
        take = jnp.broadcast_to(jnp.where(mask, ones, jnp.zeros((), udt)), new.shape)
    else:  # whole rows: the last writer's row is the row
        full = jnp.ones((L, T), jnp.bool_)
        new = jnp.concatenate(news)
        take = None
    mode = jnp.where(live & ~later, jnp.where(full, 1, 2), 0).astype(jnp.int32)  # [L, T]
    # Request runs: a request's `run` rows on one page at consecutive offsets, all plain writes, move
    # as one DMA; their rows leave the per-row list.
    run = _run_length(T, R)
    if run > 1:
        span_idx = jnp.arange(run, dtype=jnp.int32)
        lens = starts[1:] - starts[:-1]
        first = jnp.clip(starts[:-1], 0, T - run).astype(jnp.int32)
        members = first[:, None] + span_idx[None]  # [R, run]
        uniform = ((lens == run) & (starts[:-1] + run <= T)
                   & jnp.all(offset[members] == offset[first][:, None] + span_idx[None], axis=1))
        ok = (uniform[None] & jnp.all(mode[:, members] == 1, axis=2)
              & jnp.all(page[:, members] == page[:, first][:, :, None], axis=2))  # [L, R]
        token = jnp.arange(T, dtype=jnp.int32)
        in_run = (token[:, None] >= first[None]) & (token[:, None] < first[None] + run)  # [T, R]
        mode = jnp.where(jnp.any(ok[:, None, :] & in_run[None], axis=2), 0, mode)
        runs = dict(run_page=page[:, first].reshape(-1), run_offset=jnp.tile(offset[first], L),
                    run_src=(jnp.arange(L, dtype=jnp.int32)[:, None] * T + first[None]).reshape(-1),
                    run_live=ok.astype(jnp.int32).reshape(-1))
    else:
        z = jnp.zeros((1,), jnp.int32)
        runs = dict(run_page=z, run_offset=z, run_src=z, run_live=z)
    return dict(new=new, take=take, page=page.reshape(-1), offset=jnp.tile(offset, L), mode=mode.reshape(-1),
                run=run, **runs)


def pair_plan(pool, completed, complete, tables, layers, req, logical, starts, context, *, compress_ratio,
              compressed_dcp=None):
    """Compressed pairs of every layer (`compressed_pairs`, as `write_compressed` calls it) as one list.

    Returns `page` / `offset` / `mode` [L*T] (mode 0 skip, 1 write both halves, 2 read, merge one
    half, write), `values` [L*T, 2, lanes] BF16 and `take` [L*T, 2, lanes] int32 (1 = half from
    `values`)."""
    page_rows, width = pool.shape[1], pool.shape[-1]
    pages, offsets, modes, values, takes = [], [], [], [], []
    for o, ((_, _, comp), lay) in enumerate(zip(tables, layers, strict=True)):
        pair, write, mask, vals = compressed_pairs(
            pool, completed[o], complete[o], req, logical, starts, context, comp, compress_ratio,
            lay.compressed_slot, **({"dcp": compressed_dcp} if compressed_dcp is not None else {}),
        )
        full = mask[:, 0] & mask[:, 1]
        mode = jnp.where(write, jnp.where(full, 1, 2), 0).astype(jnp.int32)
        safe = jnp.where(write, pair, 0).astype(jnp.int32)
        pages.append(safe // page_rows)
        offsets.append(safe % page_rows)
        modes.append(mode)
        values.append(jnp.pad(vals.astype(jnp.bfloat16), ((0, 0), (0, 0), (0, width - vals.shape[-1]))))
        take = (mask | (mode == 1)[:, None]).astype(jnp.int32)
        takes.append(jnp.broadcast_to(take[:, :, None], (len(take), 2, width)))
    return dict(page=jnp.concatenate(pages), offset=jnp.concatenate(offsets), mode=jnp.concatenate(modes),
                values=jnp.concatenate(values), take=jnp.concatenate(takes))


def write_all_call(pool, mp, cp):
    """The one Pallas call: every layer's main rows (`main_plan`) and compressed pairs (`pair_plan`)."""
    E, Ec = mp["new"].shape[0], cp["values"].shape[0]
    n_runs, run = mp["run_page"].shape[0], mp["run"]
    packing, width = pool.shape[-2], pool.shape[-1]
    merge = mp["take"] is not None

    def kernel(m_page, m_off, m_mode, r_page, r_off, r_src, r_live, c_page, c_off, c_mode, *rest):
        if merge:
            (new_ref, take_ref, cval_ref, ctake_ref, _old, pool_ref,
             mbuf, cbuf, sems, l_row, l_run, l_read, l_cread, l_cwrite) = rest
        else:
            (new_ref, cval_ref, ctake_ref, _old, pool_ref,
             cbuf, sems, l_row, l_run, l_cread, l_cwrite) = rest
            take_ref = mbuf = l_read = None

        def row_write(i, merged):
            src = mbuf if merged else new_ref
            return pltpu.make_async_copy(
                src.at[pl.ds(i, 1)], pool_ref.at[m_page[i], pl.ds(m_off[i], 1)],
                sems.at[_MAIN_MERGED if merged else _MAIN_ROW],
            )

        def run_write(k):
            return pltpu.make_async_copy(
                new_ref.at[pl.ds(r_src[k], run)], pool_ref.at[r_page[k], pl.ds(r_off[k], run)],
                sems.at[_MAIN_RUN],
            )

        def row_read(i):
            return pltpu.make_async_copy(
                pool_ref.at[m_page[i], pl.ds(m_off[i], 1)], mbuf.at[pl.ds(i, 1)], sems.at[_MAIN_READ]
            )

        def pair_copy(i, read):
            stored = pool_ref.at[c_page[i], c_off[i], 0]
            staged = cbuf.at[i]
            src, dst = (stored, staged) if read else (staged, stored)
            return pltpu.make_async_copy(src, dst, sems.at[_PAIR_READ if read else _PAIR_WRITE])

        one, zero = jnp.int32(1), jnp.int32(0)

        # Work lists, branch-free (store the index at the count, advance the count by the predicate).
        def main_lists(i, c):
            n_row, n_read = c
            mode = m_mode[i]
            l_row[n_row] = i
            n_row = n_row + jnp.where(mode == 1, one, zero)
            if merge:
                l_read[n_read] = i
                n_read = n_read + jnp.where(mode == 2, one, zero)
            return n_row, n_read

        def run_list(k, n):
            l_run[n] = k
            return n + jnp.where(r_live[k] != 0, one, zero)

        def pair_lists(i, c):
            n_read, n_write = c
            mode = c_mode[i]
            l_cread[n_read] = i
            l_cwrite[n_write] = i
            return (n_read + jnp.where(mode == 2, one, zero),
                    n_write + jnp.where(mode != 0, one, zero))

        n_row, n_read = jax.lax.fori_loop(0, E, main_lists, (zero, zero))
        n_run = jax.lax.fori_loop(0, n_runs, run_list, zero)
        n_cread, n_cwrite = jax.lax.fori_loop(0, Ec, pair_lists, (zero, zero))

        def over(n, lst, action):
            def body(k, carry):
                action(lst[k])
                return carry

            jax.lax.fori_loop(0, n, body, 0)

        # Phase 1: every write that needs no old bytes and every read, in flight.
        over(n_run, l_run, lambda k: run_write(k).start())
        over(n_row, l_row, lambda i: row_write(i, False).start())
        if merge:
            over(n_read, l_read, lambda i: row_read(i).start())
        over(n_cread, l_cread, lambda i: pair_copy(i, True).start())

        # Phase 2: wait the reads, merge in VMEM.
        if merge:
            over(n_read, l_read, lambda i: row_read(i).wait())
            old = mbuf.bitcast(jnp.uint32).reshape(-1, width)
            new = new_ref.bitcast(jnp.uint32).reshape(-1, width)
            take = take_ref.bitcast(jnp.uint32).reshape(-1, width)
            t = take[...]
            old[...] = (old[...] & ~t) | (new[...] & t)
        over(n_cread, l_cread, lambda i: pair_copy(i, True).wait())
        bf = cbuf.bitcast(jnp.bfloat16) if pool.dtype != jnp.bfloat16 else cbuf
        bf[...] = jnp.where(ctake_ref[...] != 0, cval_ref[...], bf[...])

        # Phase 3: every merged write in flight, then every write waited.
        if merge:
            over(n_read, l_read, lambda i: row_write(i, True).start())
        over(n_cwrite, l_cwrite, lambda i: pair_copy(i, False).start())
        over(n_run, l_run, lambda k: run_write(k).wait())
        over(n_row, l_row, lambda i: row_write(i, False).wait())
        if merge:
            over(n_read, l_read, lambda i: row_write(i, True).wait())
        over(n_cwrite, l_cwrite, lambda i: pair_copy(i, False).wait())

    def full(shape):
        return pl.BlockSpec(shape, lambda i, *_: (0,) * len(shape))

    vmem_in = [mp["new"]] + ([mp["take"]] if merge else []) + [cp["values"], cp["take"]]
    scratch = (
        ([pltpu.VMEM(mp["new"].shape, pool.dtype)] if merge else [])
        + [pltpu.VMEM((Ec, packing, width), pool.dtype), pltpu.SemaphoreType.DMA((6,)),
           pltpu.SMEM((E,), jnp.int32), pltpu.SMEM((n_runs,), jnp.int32)]
        + ([pltpu.SMEM((E,), jnp.int32)] if merge else [])
        + [pltpu.SMEM((Ec,), jnp.int32), pltpu.SMEM((Ec,), jnp.int32)]
    )
    scalars = [mp["page"], mp["offset"], mp["mode"], mp["run_page"], mp["run_offset"], mp["run_src"],
               mp["run_live"], cp["page"], cp["offset"], cp["mode"]]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(1,),
            in_specs=[full(x.shape) for x in vmem_in] + [pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch,
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={len(scalars) + len(vmem_in): 0},
        name=KERNEL_NAME,
    )(*(x.astype(jnp.int32) for x in scalars), *vmem_in, pool)


def batched_writes_all(pool, rows, norms, tables, layers, positions, trig, starts, context, *,
                       compress_ratio, norm_eps, rotary_dim, dcp=None, compressed_dcp=None,
                       ring_tags=None, kernel_values=None):
    """`_batched_writes` with every layer's main K/V and compressed pairs in one Pallas call.

    Same arguments (tables already cropped to the step's requests, ring ids column 0), same pool
    bytes. `ring_tags`: optional `(ring, tags)` from `read_ring_and_tags(pool, ids, ...)` over the
    concatenated per-layer ring ids, read from this pool or from any earlier pool whose ring pages
    hold the same bytes (the whole-step kernel input's read: the kernel and the GDN checkpoint
    scatter write no ring page); None reads them here, as `_batched_writes` does.
    `kernel_values`: optional `KernelValues` (q38 L2 WFOLD, VLLM_TPU_MK_QSA_WRITE_KVAL): the kernel's own key
    rows, completed-group keys and rings instead of `produce_kv` and the vmapped `_indexer_updates` (same
    bytes when they equal what those recompute from the kernel's projections; preconditions in the module
    docstring). None recomputes them, as `_batched_writes` does.
    """
    check_supported(pool, layers)
    nreq, L = len(context), len(layers)
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(nreq, dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(len(positions), starts, context, slots)
    main_valid, span = valid, pool.shape[1]
    if dcp is not None:  # `qsa_cache_writes`' owner rule
        rank, size = dcp
        span = size * pool.shape[1]
        owner = (logical // pool.shape[1]) % size
        main_valid = valid & (owner == rank.reshape(()).astype(owner.dtype))
    if kernel_values is not None:  # q38 L2 WFOLD
        check_kernel_values(kernel_values, rows, len(positions), nreq, math.ceil(compress_ratio / 8) * 8,
                            compress_ratio)
        return _writes_from_kernel_values(
            pool, rows, norms, tables, layers, positions, trig, starts, context, req, logical, valid, main_valid,
            span, slots, active, compress_ratio=compress_ratio, norm_eps=norm_eps, rotary_dim=rotary_dim,
            compressed_dcp=compressed_dcp, ring_tags=ring_tags, kernel_values=kernel_values)
    mp = main_plan(pool, rows, norms, tables, layers, positions, trig, starts, req, logical, main_valid,
                   span, norm_eps=norm_eps, rotary_dim=rotary_dim)
    raw = jnp.stack([r[2] for r in rows])
    ring_rows = math.ceil(compress_ratio / 8) * 8
    ids = jnp.concatenate([t[0] for t in tables])
    if ring_tags is None:
        ring, tags = read_ring_and_tags(pool, ids, ring_rows * raw.shape[-1], ring_rows * 2)
    else:
        ring, tags = ring_tags
    ring = ring.reshape(L, nreq, ring_rows, -1)
    tags = tags.reshape(L, nreq, ring_rows, 2)
    update = jax.vmap(
        lambda raw, ring, tags, nik: _indexer_updates(
            raw, positions, ring, tags, nik, trig, starts, context, slots, compress_ratio, norm_eps,
            rotary_dim,
        )
    )
    completed, complete, _, _, ring, tags = update(raw, ring, tags, jnp.stack([n[1] for n in norms]))
    cp = pair_plan(pool, completed, complete, tables, layers, req, logical, starts, context,
                   compress_ratio=compress_ratio, compressed_dcp=compressed_dcp)
    pool = write_all_call(pool, mp, cp)
    ids = jnp.where(jnp.tile(active, L), ids, 0)
    return write_ring_and_tags(pool, ring.reshape(L * nreq, -1), tags.reshape(L * nreq, -1), ids)


def _writes_from_kernel_values(pool, rows, norms, tables, layers, positions, trig, starts, context, req, logical,
                               valid, main_valid, span, slots, active, *, compress_ratio, norm_eps, rotary_dim,
                               compressed_dcp, ring_tags, kernel_values):
    """q38 L2 WFOLD: `batched_writes_all`'s body with the kernel's values (no `produce_kv`, no `_indexer_updates`).

    A separate body so the flag-off trace of `batched_writes_all` is untouched."""
    nreq, L = len(context), len(layers)
    mp = main_plan(pool, rows, norms, tables, layers, positions, trig, starts, req, logical, main_valid,
                   span, norm_eps=norm_eps, rotary_dim=rotary_dim, keys=kernel_values.keys)
    width = rows[0][2].shape[-1]
    ring_rows = math.ceil(compress_ratio / 8) * 8
    ids = jnp.concatenate([t[0] for t in tables])
    hxs = hxsmall.on("VLLM_TPU_HXS_GLUE_TRIM")  # q38 HXSMALL P1d: ring read / write ids padded to whole DMA batches
    if ring_tags is None:
        ring, tags = (hxsmall.read_ring_and_tags_padded if hxs else read_ring_and_tags)(
            pool, ids, ring_rows * width, ring_rows * 2)
    else:
        ring, tags = ring_tags
    ring = ring.reshape(L, nreq, ring_rows, -1)
    tags = tags.reshape(L, nreq, ring_rows, 2)
    completed, complete, ring, tags = kernel_indexer_updates(
        kernel_values, ring, tags, positions, starts, context, slots, logical, valid, req,
        compress_ratio=compress_ratio)
    cp = pair_plan(pool, completed, complete, tables, layers, req, logical, starts, context,
                   compress_ratio=compress_ratio, compressed_dcp=compressed_dcp)
    pool = write_all_call(pool, mp, cp)
    ids = jnp.where(jnp.tile(active, L), ids, 0)
    if hxs:
        hxsmall.engage("VLLM_TPU_HXS_GLUE_TRIM", "QSA writer: ring read / write of %d ids padded to %d"
                       % (L * nreq, hxsmall.ring_tag_batch_pad(L * nreq)))
    return (hxsmall.write_ring_and_tags_padded if hxs else write_ring_and_tags)(
        pool, ring.reshape(L * nreq, -1), tags.reshape(L * nreq, -1), ids)


__all__ = [
    "KERNEL_NAME",
    "KernelValues",
    "batched_writes_all",
    "check_kernel_values",
    "check_supported",
    "kernel_indexer_updates",
    "kernel_values_from_outputs",
    "main_plan",
    "pair_plan",
    "qsa_write_all_enabled",
    "qsa_write_kval_enabled",
    "write_all_call",
]

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
"""Pallas writer for the 4-way sublane-packed indexer K cache.

The writer has one job: `scatter_rows(cache_kv, src, dst_rows)` writes row `i`
of `src` into flat cache row `dst_rows[i]` (`page * page_size + slot`), and
skips rows whose destination is negative or out of range. It knows nothing
about requests, page tables, ranks, or DCP; all of that lives in the plain-JAX
address layer (`kv_cache_utils`), exactly like the upstream vllm-torchtpu
sparse-MLA writer (`get_dst_rows` -> `scatter.scatter` -> `scatter_rows`).

Why this is not one DMA per token (unlike the sparse-MLA writer): both indexer
cache layouts pack 4 tokens into every int32 word, so a single token is not
DMA-addressable.

  - `HEAD_ALONG_SUBLANE` (`[num_pages, page_size // 4, 4, lkv_dim]`): a token
    is one byte-sublane of a 4-token word row (a *quad*).
  - `SEQ_ALONG_LANE` (`[num_pages, head_dim // 4 + 1, 4, page_size]`): a token
    is one lane of a `(head_dim // 4 + 1, 4, 128)` tile.

So the kernel works in *granules*, the smallest DMA-addressable unit: a quad
(`HAS`, 4 rows) or a 128-lane tile (`SAL`). Two steps:

 1. `_find_runs` walks `dst_rows` once on the scalar core. It compacts the
    writable tokens in place (dropping skipped ones) and records *runs*:
    consecutive compacted tokens that land in consecutive rows of one granule.
    Compaction is what keeps a DCP rank's interleaved chunks in one run per
    granule instead of several runs racing on it.
 2. `_write_runs` writes each run with one read-modify-write of its granule:
    read the granule into VMEM (skipped when the run fills it), blend the run's
    tokens in with one roll + select, and write the granule back. It does this
    in waves of up to `_MAX_WAVE` runs; every wave reads all its partial
    granules, waits for them, blends, writes, and waits for all writes.

Both steps run in the kernel. Building the run table in XLA instead
(`searchsorted` + gathers) measured ~120 µs at 1024 tokens on TPU7x; the
scalar walk costs ~13 µs there and ~1 µs for a 16-token decode step. Writing
page-long runs instead (partial-granule read-modify-write at each end + one
DMA for the whole granules between) was ~5-10 µs faster only for 1024-token
`HAS` prefill, ~2x slower for decode batches of 128+, and ~150 lines longer.

Contract (as with the XLA scatter it replaces): two tokens must not target the
same row, and two runs must not share a granule (tokens bound for one granule
must be adjacent in `src` once skipped tokens are dropped). Both hold whenever
each writable page belongs to at most one sequence per step, which the vLLM
block allocator guarantees. Violations are undefined behaviour.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    config,
    kv_cache_utils,
)

KVLayout = config.KVLayout

# Tokens packed into one int32 word of the cache (fp8/u8 rows).
_KV_PACKING = 4
# Lane width of a TensorCore tile.
_LANES = 128
# Runs (granules) per wave. Caps the read-modify-write scratch at 128 granules
# (`SAL`: `128 * 33 * 4 * 128 B = 2.16 MB`; `HAS`: `128 * 4 * 256 B`).
_MAX_WAVE = 128
# VMEM budget for inputs + scratch; larger calls fail at trace time.
_MAX_SAFE_VMEM_BYTES = 24 * 1024 * 1024
# Tokens per call. SMEM holds `dst_rows` and the run table (`12 B` per token),
# so this keeps it under 600 KB.
_MAX_TOKENS = 48 * 1024


def _align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def cache_page_size(cache_kv: jax.Array, kv_layout: KVLayout | str | None) -> int:
    """Validates `cache_kv` against `kv_layout` and returns its page size."""
    layout = KVLayout.parse(kv_layout)
    if cache_kv.ndim != 4:
        raise ValueError(
            f"Expected 4-D cache_kv, got {cache_kv.ndim}-D shape {cache_kv.shape}."
        )
    if cache_kv.shape[2] != _KV_PACKING:
        raise ValueError(
            f"Expected kv_packing={_KV_PACKING}, got cache_kv shape {cache_kv.shape}."
        )
    if layout == KVLayout.SEQ_ALONG_LANE:
        _, sublane_groups, _, page_size = cache_kv.shape
        if sublane_groups >= 64 and sublane_groups % _KV_PACKING == 0:
            raise ValueError(
                f"cache_kv shape {cache_kv.shape} looks like HEAD_ALONG_SUBLANE"
                f" (dim 1 == {sublane_groups}) but kv_layout=SEQ_ALONG_LANE. Pass"
                " kv_layout='head_along_sublane'."
            )
        if page_size % _LANES:
            raise ValueError(
                "SEQ_ALONG_LANE requires page_size to be a positive multiple of"
                f" {_LANES}, got {page_size=}."
            )
        return page_size
    _, quads_per_page, _, lkv_dim = cache_kv.shape
    page_size = quads_per_page * _KV_PACKING
    if page_size % _LANES or lkv_dim % _LANES:
        raise ValueError(
            f"cache_kv shape {cache_kv.shape} is invalid for HEAD_ALONG_SUBLANE"
            f" (page_size={page_size}, lkv_dim={lkv_dim}; both must be multiples"
            f" of {_LANES}). If this is a SEQ_ALONG_LANE cache, pass"
            " kv_layout='seq_along_lane'."
        )
    return page_size


def _format_rows(src: jax.Array, cache_kv: jax.Array, layout: KVLayout) -> jax.Array:
    """Pads or truncates `src` rows to the cache's per-token record width."""
    if layout == KVLayout.SEQ_ALONG_LANE:
        sublane_groups = cache_kv.shape[1]
        head_dim = (sublane_groups - 1) * _KV_PACKING
        rec_dim = sublane_groups * _KV_PACKING
        width = src.shape[-1]
        if width == rec_dim:
            return src
        if width <= head_dim + 1:
            return jnp.pad(src, ((0, 0), (0, rec_dim - width)))
        # `[head_dim fp8 | 4-byte scale | ...]`: keep head + first scale byte.
        return jnp.pad(src[:, : head_dim + 1], ((0, 0), (0, _KV_PACKING - 1)))
    lkv_dim = cache_kv.shape[-1]
    if src.shape[-1] < lkv_dim:
        return jnp.pad(src, ((0, 0), (0, lkv_dim - src.shape[-1])))
    return src[:, :lkv_dim]


def _blend_token_into_quad(
    cur_quad_i32: jax.Array,
    src_quad_i32: jax.Array,
    src_l: jax.Array,
    dst_l: jax.Array,
) -> jax.Array:
    """Returns `cur_quad_i32` with row `dst_l` set to `src_quad_i32[src_l]`."""
    kv_packing, lkv_dim = cur_quad_i32.shape
    row_iota = lax.broadcasted_iota(jnp.int32, (kv_packing, lkv_dim), 0)
    # Values are u8 widened to i32 (non-negative), so a masked max selects a row.
    tok_val_i32 = jnp.max(
        jnp.where(row_iota == src_l, src_quad_i32, 0), axis=0, keepdims=True
    )
    return jnp.where(
        row_iota == dst_l,
        jnp.broadcast_to(tok_val_i32, (kv_packing, lkv_dim)),
        cur_quad_i32,
    )


class _HasQuads:
    """`HEAD_ALONG_SUBLANE` granule ops: a granule is one 4-token quad.

    The cache is viewed as flat quads `[num_pages * page_size // 4, 4, lkv_dim]`,
    so cache row `r` is sublane `r & 3` of quad `r >> 2`, and the staged source
    (`k_ref`, `[num_quads, 4, lkv_dim]`) is indexed the same way by token.
    """

    granule = _KV_PACKING

    def __init__(self, k_ref, cache_ref, rmw_ref):
        num_pages, quads_per_page, _, lkv_dim = cache_ref.shape
        self.cache = cache_ref.reshape(num_pages * quads_per_page, _KV_PACKING, lkv_dim)
        self.k = k_ref
        self.rmw = rmw_ref
        self.num_src_quads = k_ref.shape[0]
        self.lkv_dim = lkv_dim

    def granule_copy(self, row, rmw_slot, sem, *, to_hbm):
        hbm = self.cache.at[pl.ds(row >> 2, 1)]
        vmem = self.rmw.at[pl.ds(rmw_slot, 1)]
        src, dst = (vmem, hbm) if to_hbm else (hbm, vmem)
        return pltpu.make_async_copy(src, dst, sem)

    def blend(self, rmw_slot, row, src_tok, n):
        """Overwrites rows `[row, row + n)` (one quad) of RMW quad `rmw_slot`."""
        # The `n <= 4` source tokens span at most two staged quads; roll the pair
        # along sublanes so token `src_tok` lands on row `row & 3`.
        q0 = src_tok >> 2
        q1 = jnp.minimum(q0 + 1, self.num_src_quads - 1)
        pair = jnp.concatenate(
            [self.k[q0].astype(jnp.int32), self.k[q1].astype(jnp.int32)], axis=0
        )
        first = row & 3
        src = pltpu.roll(pair, shift=(first - (src_tok & 3)) & 7, axis=0)
        d = lax.broadcasted_iota(jnp.int32, (_KV_PACKING, self.lkv_dim), 0)
        quad = jnp.where(
            (d >= first) & (d < first + n),
            src[:_KV_PACKING],
            self.rmw[rmw_slot].astype(jnp.int32),
        )
        self.rmw[rmw_slot] = quad.astype(self.rmw.dtype)

    def move_token(self, tok, dst_tok):
        """Copies staged token `tok` to staged position `dst_tok` (`<= tok`)."""
        self.k[dst_tok >> 2] = _blend_token_into_quad(
            self.k[dst_tok >> 2].astype(jnp.int32),
            self.k[tok >> 2].astype(jnp.int32),
            tok & 3,
            dst_tok & 3,
        ).astype(self.k.dtype)


class _SalTiles:
    """`SEQ_ALONG_LANE` granule ops: a granule is one 128-lane tile.

    The staged source (`k_ref`, `[sublane_groups, 4, padded_tokens]`) is already
    transposed to the cache layout, so token `t` is lane `t` of the source and
    cache row `page * page_size + slot` is lane `slot` of page `page`.
    """

    granule = _LANES

    def __init__(self, k_ref, cache_ref, rmw_ref, page_size):
        _, self.sublane_groups, _, _ = cache_ref.shape
        self.cache = cache_ref
        self.k = k_ref
        self.rmw = rmw_ref
        self.page_size = page_size
        self.padded_tokens = k_ref.shape[2]
        self.rec_dim = self.sublane_groups * _KV_PACKING

    def _cache_lanes(self, row, n):
        """Lanes `[row, row + n)` of the cache; `row` is 128-aligned."""
        slot = pl.multiple_of(row % self.page_size, _LANES)
        return self.cache.at[row // self.page_size, :, :, pl.ds(slot, n)]

    def _src_tile(self, src_ref, tok_aligned):
        off = pl.multiple_of(
            jnp.clip(tok_aligned, 0, self.padded_tokens - _LANES), _LANES
        )
        return pltpu.bitcast(
            src_ref[:, :, pl.ds(off, _LANES)].reshape(self.rec_dim, _LANES),
            jnp.uint32,
        )

    def granule_copy(self, row, rmw_slot, sem, *, to_hbm):
        hbm = self._cache_lanes((row >> 7) << 7, _LANES)
        vmem = self.rmw.at[rmw_slot]
        src, dst = (vmem, hbm) if to_hbm else (hbm, vmem)
        return pltpu.make_async_copy(src, dst, sem)

    def blend(self, rmw_slot, row, src_tok, n):
        """Overwrites lanes `[row, row + n) & 127` of RMW tile `rmw_slot`."""
        dst_start = row & 127
        src_start = src_tok & 127
        tok_aligned = (src_tok >> 7) << 7
        delta = (dst_start - src_start) & 127
        # The `n <= 128` source lanes span at most two source tiles.
        a = pltpu.roll(self._src_tile(self.k, tok_aligned), shift=delta, axis=1)
        b = pltpu.roll(
            self._src_tile(self.k, tok_aligned + _LANES), shift=delta, axis=1
        )
        lane = lax.broadcasted_iota(jnp.int32, (self.sublane_groups, _LANES), 1)
        src = jnp.where(lane < dst_start + (_LANES - src_start), a, b)
        dst = pltpu.bitcast(
            self.rmw[rmw_slot].reshape(self.rec_dim, _LANES), jnp.uint32
        )
        blended = jnp.where((lane >= dst_start) & (lane < dst_start + n), src, dst)
        self.rmw[rmw_slot] = pltpu.bitcast(blended, jnp.uint8).reshape(
            self.sublane_groups, _KV_PACKING, _LANES
        )

    def move_token(self, tok, dst_tok):
        """Copies staged token (lane) `tok` to lane `dst_tok` (`<= tok`)."""
        src = pltpu.roll(
            self._src_tile(self.k, (tok >> 7) << 7),
            shift=(dst_tok - tok) & 127,
            axis=1,
        )
        dst_off = pl.multiple_of((dst_tok >> 7) << 7, _LANES)
        lane = lax.broadcasted_iota(jnp.int32, (self.sublane_groups, _LANES), 1)
        out = jnp.where(lane == (dst_tok & 127), src, self._src_tile(self.k, dst_off))
        self.k[:, :, pl.ds(dst_off, _LANES)] = pltpu.bitcast(out, jnp.uint8).reshape(
            self.sublane_groups, _KV_PACKING, _LANES
        )


def _find_runs(
    ops: _HasQuads | _SalTiles,
    dst_ref: Any,
    run_src_ref: Any,
    run_dst_ref: Any,
    *,
    num_rows: int,
) -> jax.Array:
    """Walks `dst_rows` once, compacting writable tokens and recording runs.

    Writable token `t` moves to staged position `num_valid` (the number of
    writable tokens before it); that only happens after a skipped token, so the
    common non-DCP step moves nothing. Run `r` covers compacted tokens
    `[run_src[r], run_src[r + 1])` and starts at cache row `run_dst[r]`; a run
    ends where the next row is not consecutive or starts a new granule, so every
    run lies in one granule.

    Args:
      ops: Granule ops holding the staged source.
      dst_ref: `[T]` SMEM flat destination rows.
      run_src_ref: `[T + 1]` SMEM output: first compacted token of each run, plus
        a `num_valid` sentinel after the last run.
      run_dst_ref: `[T + 1]` SMEM output: first cache row of each run.
      num_rows: `num_pages * page_size`.

    Returns:
      The number of runs.
    """

    def _step(tok, carry):
        num_runs, num_valid, next_row = carry
        row = dst_ref[tok]
        valid = (row >= 0) & (row < num_rows)
        starts = valid & ((row != next_row) | (row % ops.granule == 0))
        # Branch-free: slot `num_runs` is only kept if this token starts a run.
        run_src_ref[num_runs] = num_valid
        run_dst_ref[num_runs] = row

        @pl.when(valid & (num_valid != tok))
        def _():
            ops.move_token(tok, num_valid)

        return (
            num_runs + starts.astype(jnp.int32),
            num_valid + valid.astype(jnp.int32),
            jnp.where(valid, row + 1, next_row),
        )

    zero = jnp.int32(0)
    num_runs, num_valid, _ = lax.fori_loop(
        0, dst_ref.shape[0], _step, (zero, zero, jnp.int32(-1)), unroll=8
    )
    run_src_ref[num_runs] = num_valid
    return num_runs


def _write_runs(
    ops: _HasQuads | _SalTiles,
    num_runs: jax.Array,
    run_src_ref: Any,
    run_dst_ref: Any,
    rd_sem: Any,
    wr_sem: Any,
    *,
    wave: int,
) -> None:
    """Writes every run (each inside one granule), `wave` runs at a time.

    Run `i` of a wave uses RMW slot `i`: a partial granule is read into it first
    (a full one is not), then the run's tokens are blended in and the slot is
    written back with one DMA.

    Semaphore rule: all reads of a wave share `rd_sem` and are all waited before
    any slot is consumed; all writes share `wr_sem` and are all waited before
    the next wave reuses the slots. A DMA wait counts bytes rather than naming a
    descriptor, so this "wait for everything, then consume" order is what makes
    sharing one semaphore safe.
    """
    g = ops.granule

    def _wave(w, _):
        first = w * wave
        count = jnp.minimum(wave, num_runs - first)

        def _for_each(fn):
            def _body(i, _):
                src = run_src_ref[first + i]
                fn(i, src, run_dst_ref[first + i], run_src_ref[first + i + 1] - src)
                return None

            lax.fori_loop(0, count, _body, None)

        def _reads(action):
            def _fn(i, src, row, n):
                del src

                @pl.when(n < g)
                def _():
                    action(ops.granule_copy(row, i, rd_sem, to_hbm=False))

            return _fn

        def _blend_and_start_write(i, src, row, n):
            ops.blend(i, row, src, n)
            ops.granule_copy(row, i, wr_sem, to_hbm=True).start()

        def _wait_write(i, src, row, n):
            del src, n
            ops.granule_copy(row, i, wr_sem, to_hbm=True).wait()

        _for_each(_reads(lambda c: c.start()))
        _for_each(_reads(lambda c: c.wait()))
        _for_each(_blend_and_start_write)
        _for_each(_wait_write)
        return None

    lax.fori_loop(0, (num_runs + wave - 1) // wave, _wave, None)


def _has_kernel(
    dst_ref,
    k_ref,
    cache_in_ref,
    cache_ref,
    run_src_ref,
    run_dst_ref,
    rmw_ref,
    sems,
    *,
    page_size: int,
    wave: int,
) -> None:
    del cache_in_ref
    ops = _HasQuads(k_ref, cache_ref, rmw_ref)
    num_runs = _find_runs(
        ops,
        dst_ref,
        run_src_ref,
        run_dst_ref,
        num_rows=cache_ref.shape[0] * page_size,
    )
    _write_runs(
        ops,
        num_runs,
        run_src_ref,
        run_dst_ref,
        sems.at[0],
        sems.at[1],
        wave=wave,
    )


def _sal_kernel(
    dst_ref,
    k_ref,
    cache_in_ref,
    cache_ref,
    run_src_ref,
    run_dst_ref,
    k_sal_ref,
    rmw_ref,
    sems,
    *,
    page_size: int,
    transpose_groups: int,
    wave: int,
) -> None:
    del cache_in_ref
    sublane_groups = cache_ref.shape[1]

    # Transpose the token-major `(128, W)` blocks into the cache layout
    # `(sublane_groups, 4, 128)`, so every later step is lane-indexed by token.
    def _transpose(b, _):
        blk = jnp.swapaxes(k_ref[b], 0, 1).reshape(
            transpose_groups, _KV_PACKING, _LANES
        )[:sublane_groups]
        k_sal_ref[:, :, pl.ds(pl.multiple_of(b * _LANES, _LANES), _LANES)] = blk
        return None

    lax.fori_loop(0, k_ref.shape[0], _transpose, None)
    ops = _SalTiles(k_sal_ref, cache_ref, rmw_ref, page_size)
    num_runs = _find_runs(
        ops,
        dst_ref,
        run_src_ref,
        run_dst_ref,
        num_rows=cache_ref.shape[0] * page_size,
    )
    _write_runs(
        ops,
        num_runs,
        run_src_ref,
        run_dst_ref,
        sems.at[0],
        sems.at[1],
        wave=wave,
    )


def _check_vmem(vmem_bytes: int, per_token_bytes: int, fixed_bytes: int, name):
    if vmem_bytes > _MAX_SAFE_VMEM_BYTES:
        max_tokens = (_MAX_SAFE_VMEM_BYTES - fixed_bytes) // per_token_bytes
        raise ValueError(
            f"scatter_rows({name}) needs {vmem_bytes / 2**20:.2f} MiB of VMEM,"
            f" exceeding the safe limit of {_MAX_SAFE_VMEM_BYTES / 2**20:.2f} MiB"
            f" (max supported tokens per call: ~{max_tokens})."
        )


def _run_table_shapes(num_tokens: int) -> list[Any]:
    """SMEM `run_src` / `run_dst` scratch (one slot per token + a sentinel).

    With `dst_rows` that is `12 * num_tokens` bytes of SMEM (see `_MAX_TOKENS`).
    """
    return [pltpu.SMEM((num_tokens + 1,), jnp.int32)] * 2


def _has_launch(cache_kv: jax.Array, rows: jax.Array, *, page_size: int, wave: int):
    """`HEAD_ALONG_SUBLANE`: `(staged source, kernel, scratch shapes)`."""
    num_tokens, lkv_dim = rows.shape
    padded_tokens = _align_to(num_tokens, _KV_PACKING)
    rmw_bytes = wave * _KV_PACKING * lkv_dim
    # Staged source + the RMW quads.
    _check_vmem(
        padded_tokens * lkv_dim + rmw_bytes,
        lkv_dim,
        rmw_bytes,
        "HEAD_ALONG_SUBLANE",
    )
    k_in = jnp.pad(rows, ((0, padded_tokens - num_tokens), (0, 0))).reshape(
        padded_tokens // _KV_PACKING, _KV_PACKING, lkv_dim
    )
    scratch_shapes = _run_table_shapes(num_tokens) + [
        pltpu.VMEM((wave, _KV_PACKING, lkv_dim), cache_kv.dtype),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    kernel = functools.partial(_has_kernel, page_size=page_size, wave=wave)
    return k_in, kernel, scratch_shapes


def _sal_launch(cache_kv: jax.Array, rows: jax.Array, *, page_size: int, wave: int):
    """`SEQ_ALONG_LANE`: `(staged source, kernel, scratch shapes)`."""
    num_tokens, rec_dim = rows.shape
    sublane_groups = rec_dim // _KV_PACKING
    transpose_width = _align_to(rec_dim, _LANES)
    padded_tokens = _align_to(num_tokens, _LANES)
    rmw_bytes = wave * rec_dim * _LANES
    per_token_bytes = transpose_width + rec_dim
    # Token-major staged source + transposed copy + RMW tiles.
    _check_vmem(
        padded_tokens * per_token_bytes + rmw_bytes,
        per_token_bytes,
        rmw_bytes,
        "SEQ_ALONG_LANE",
    )
    k_in = jnp.pad(
        rows, ((0, padded_tokens - num_tokens), (0, transpose_width - rec_dim))
    ).reshape(padded_tokens // _LANES, _LANES, transpose_width)
    kernel = functools.partial(
        _sal_kernel,
        page_size=page_size,
        transpose_groups=transpose_width // _KV_PACKING,
        wave=wave,
    )
    tokens_shape = (sublane_groups, _KV_PACKING, padded_tokens)
    scratch_shapes = _run_table_shapes(num_tokens) + [
        pltpu.VMEM(tokens_shape, cache_kv.dtype),  # transposed source
        pltpu.VMEM((wave, sublane_groups, _KV_PACKING, _LANES), cache_kv.dtype),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    return k_in, kernel, scratch_shapes


def scatter_rows(
    cache_kv: jax.Array,
    src: jax.Array,
    dst_rows: jax.Array,
    kv_layout: KVLayout | str | None = KVLayout.HEAD_ALONG_SUBLANE,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Writes `src[i]` into flat cache row `dst_rows[i]` of `cache_kv` in place.

    Args:
      cache_kv: 4-D paged indexer K cache, `HEAD_ALONG_SUBLANE` `[num_pages,
        page_size // 4, 4, lkv_dim]` or `SEQ_ALONG_LANE` `[num_pages, head_dim //
        4 + 1, 4, page_size]`. Donate it: the update aliases it in HBM.
      src: `[num_tokens, width]` u8 token records (padded/truncated to the cache's
        record width).
      dst_rows: `[num_tokens]` int32 flat rows `page * page_size + slot`; negative
        or `>= num_pages * page_size` means "do not write this token".
      kv_layout: Cache layout.
      interpret: When True (or when `jax.default_backend() != "tpu"`), falls back
        to `scatter_rows_reference` after argument and VMEM validation.

    Returns:
      The updated `cache_kv`.
    """
    layout = KVLayout.parse(kv_layout)
    page_size = cache_page_size(cache_kv, layout)
    num_tokens = src.shape[0]
    if dst_rows.shape != (num_tokens,):
        raise ValueError(
            f"dst_rows shape {dst_rows.shape} must be ({num_tokens},) to match"
            f" src shape {src.shape}."
        )
    if num_tokens == 0:
        return cache_kv
    if num_tokens > _MAX_TOKENS:
        raise ValueError(
            f"scatter_rows supports at most {_MAX_TOKENS} tokens per call (SMEM"
            f" run table), got {num_tokens}."
        )

    launch = _sal_launch if layout == KVLayout.SEQ_ALONG_LANE else _has_launch
    k_in, kernel, scratch_shapes = launch(
        cache_kv,
        _format_rows(src, cache_kv, layout),
        page_size=page_size,
        wave=min(num_tokens, _MAX_WAVE),
    )
    if interpret or jax.default_backend() != "tpu":
        return scatter_rows_reference(cache_kv, src, dst_rows, layout)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(cache_kv.shape, cache_kv.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(
                    k_in.shape,
                    lambda *_: (0,) * k_in.ndim,
                    memory_space=pltpu.VMEM,
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        input_output_aliases={2: 0},
        name="indexer_k_scatter_rows",
    )(dst_rows.astype(jnp.int32), k_in, cache_kv)


def coalesced_kv_append(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_layout: KVLayout | str | None = KVLayout.HEAD_ALONG_SUBLANE,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Appends this step's K rows into an unsharded paged indexer cache.

    Args:
      cache_kv: 4-D paged indexer K cache (see `scatter_rows`).
      k_packed: `[num_tokens, width]` u8 K records in request-major order.
      seq_lens: `[num_seqs]` KV lengths including this step's tokens.
      page_indices: `[num_seqs, pages_per_seq]` (or flattened) page table.
      cu_q_lens: `[num_seqs + 1]` cumulative new-token counts.
      kv_layout: Cache layout.
      interpret: Forwarded to `scatter_rows`.

    Returns:
      The updated `cache_kv`.
    """
    page_size = cache_page_size(cache_kv, kv_layout)
    page, slot = kv_cache_utils.get_page_and_slot(
        k_packed.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        page_size=page_size,
        num_pages=cache_kv.shape[0],
    )
    return scatter_rows(
        cache_kv,
        k_packed,
        kv_cache_utils.get_dst_rows(page, slot, page_size),
        kv_layout,
        interpret=interpret,
    )


def coalesced_kv_append_dcp(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dcp_rank: jax.Array | int,
    *,
    dcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout | str | None = KVLayout.HEAD_ALONG_SUBLANE,
    interpret: bool = False,
) -> jax.Array:
    """DCP twin of `coalesced_kv_append`: writes the rows this rank owns.

    Call it from inside a `shard_map` with this rank's cache shard; `dcp_rank`
    is traced data, so one compiled program serves every rank. `k_packed` and
    the sequence metadata are replicated; only the address mask differs per rank
    (`kv_cache_utils.get_dcp_page_and_slot`), so no collective is needed.

    Args:
      cache_kv: This rank's 4-D cache shard (see `scatter_rows`).
      k_packed: `[num_tokens, width]` u8 K records, identical on every rank.
      seq_lens: `[num_seqs]` global KV lengths including this step's tokens.
      page_indices: `[num_seqs, pages_per_seq]` (or flattened) virtual page ids.
      cu_q_lens: `[num_seqs + 1]` cumulative new-token counts.
      dcp_rank: This shard's DCP index (int or traced scalar).
      dcp_size: Number of DCP ranks.
      interleave_size: Consecutive positions per rank per round-robin step.
      kv_layout: Cache layout.
      interpret: Forwarded to `scatter_rows`.

    Returns:
      The updated cache shard.
    """
    page_size = cache_page_size(cache_kv, kv_layout)
    page, slot = kv_cache_utils.get_dcp_page_and_slot(
        k_packed.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        dcp_rank,
        page_size=page_size,
        num_pages=cache_kv.shape[0],
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    return scatter_rows(
        cache_kv,
        k_packed,
        kv_cache_utils.get_dst_rows(page, slot, page_size),
        kv_layout,
        interpret=interpret,
    )


def scatter_rows_reference(
    cache_kv: jax.Array,
    src: jax.Array,
    dst_rows: jax.Array,
    kv_layout: KVLayout | str | None = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    """Pure-XLA `scatter_rows` (same arguments and contract); the test golden."""
    layout = KVLayout.parse(kv_layout)
    page_size = cache_page_size(cache_kv, layout)
    num_rows = cache_kv.shape[0] * page_size
    rows = _format_rows(src, cache_kv, layout)
    valid = (dst_rows >= 0) & (dst_rows < num_rows)
    if layout == KVLayout.SEQ_ALONG_LANE:
        page = jnp.where(valid, dst_rows // page_size, kv_cache_utils.OOB_PAGE)
        slot = jnp.where(valid, dst_rows % page_size, 0)
        return cache_kv.at[page, :, :, slot].set(
            rows.reshape(-1, cache_kv.shape[1], _KV_PACKING), mode="drop"
        )
    # `[pages, page_size // 4, 4, lkv_dim]` is row-major in `page * page_size +
    # slot`, so it is a plain `[num_rows, lkv_dim]` table.
    flat = cache_kv.reshape(num_rows, cache_kv.shape[-1])
    flat = flat.at[jnp.where(valid, dst_rows, num_rows)].set(rows, mode="drop")
    return flat.reshape(cache_kv.shape)

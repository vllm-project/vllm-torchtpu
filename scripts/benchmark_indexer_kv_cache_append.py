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
"""Microbenchmark for the indexer K-cache writer (`kv_cache_append`).

Combines two benchmarks:
  1. `append`: Per-call slope `(t(36 calls) - t(4 calls)) / 32` of the append
     alone across decode, prefill, mixed serving, and AgentX DCP16 shapes for
     both `head_along_sublane` (HAS) and `seq_along_lane` (SAL).
  2. `region`: Per-layer slope of the compiled `append + StreamIdxTC` region
     (`(t(layers) - t(1)) / (layers - 1)`) plus HLO full-cache relayout check
     at `T in {1, 8, 1024, 4096}` for XLA HAS, Pallas HAS, XLA SAL, and Pallas
     SAL.

Usage:
  python scripts/benchmark_indexer_kv_cache_append.py --mode=all
  python scripts/benchmark_indexer_kv_cache_append.py --mode=append
  python scripts/benchmark_indexer_kv_cache_append.py --mode=region --layers=4
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_append,
    kv_cache_utils,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    streamindex_topk as st_lib,
)

_KV_PACKING = 4
_HAS = "head_along_sublane"
_SAL = "seq_along_lane"
_LAYOUTS = (_HAS, _SAL)
_SHORT, _LONG, _REPS, _NUM_CACHES = 4, 36, 5, 4


# ---------------------------------------------------------------------------
# Baseline XLA append implementations
# ---------------------------------------------------------------------------


def _page_and_slot(num_tokens, seq_lens, page_indices, cu_q_lens, page_size):
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
    q_len = cu_q_lens[seq + 1] - cu_q_lens[seq]
    pos = seq_lens[seq] - q_len + (tok - cu_q_lens[seq])
    page = jnp.where(
        tok < cu_q_lens[-1],
        page_indices.reshape(seq_lens.shape[0], -1)[seq, pos // page_size],
        jnp.int32(2**30),
    )
    return page, pos % page_size


def has_append_4d(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    """`HEAD_ALONG_SUBLANE`: `cache.at[page, slot // 4, slot % 4].set(k)`."""
    _, quads_per_page, _, lkv_dim = cache_kv.shape
    width = k_packed.shape[-1]
    k = (
        jnp.pad(k_packed, ((0, 0), (0, lkv_dim - width)))
        if width < lkv_dim
        else k_packed[:, :lkv_dim]
    )
    page, slot = _page_and_slot(
        k.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        quads_per_page * _KV_PACKING,
    )
    return cache_kv.at[page, slot // _KV_PACKING, slot % _KV_PACKING].set(k)


def sal_append_4d(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    """`SEQ_ALONG_LANE`: `cache.at[page, :, :, slot].set(k_record)`."""
    _, sublane_groups, _, page_size = cache_kv.shape
    head_dim = (sublane_groups - 1) * _KV_PACKING
    rec_dim = sublane_groups * _KV_PACKING
    width = k_packed.shape[-1]
    if width == rec_dim:
        k = k_packed
    elif width <= head_dim + 1:
        k = jnp.pad(k_packed, ((0, 0), (0, rec_dim - width)))
    else:
        k = jnp.pad(k_packed[:, : head_dim + 1], ((0, 0), (0, _KV_PACKING - 1)))
    page, slot = _page_and_slot(
        k.shape[0], seq_lens, page_indices, cu_q_lens, page_size
    )
    return cache_kv.at[page, :, :, slot].set(k.reshape(-1, sublane_groups, _KV_PACKING))


def has_append_4d_dcp(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dcp_rank: jax.Array,
    *,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Baseline XLA DCP scatter in `HEAD_ALONG_SUBLANE` layout."""
    _, quads_per_page, _, lkv_dim = cache_kv.shape
    page_size = quads_per_page * _KV_PACKING
    width = k_packed.shape[-1]
    k = (
        jnp.pad(k_packed, ((0, 0), (0, lkv_dim - width)))
        if width < lkv_dim
        else k_packed[:, :lkv_dim]
    )
    page, slot = kv_cache_utils.get_dcp_page_and_slot(
        k.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        dcp_rank,
        page_size=page_size,
        num_pages=cache_kv.shape[0],
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    return cache_kv.at[page, slot // _KV_PACKING, slot % _KV_PACKING].set(k)


def sal_append_4d_dcp(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dcp_rank: jax.Array,
    *,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Baseline XLA DCP scatter in `SEQ_ALONG_LANE` layout."""
    _, sublane_groups, _, page_size = cache_kv.shape
    head_dim = (sublane_groups - 1) * _KV_PACKING
    rec_dim = sublane_groups * _KV_PACKING
    width = k_packed.shape[-1]
    if width == rec_dim:
        k = k_packed
    elif width <= head_dim + 1:
        k = jnp.pad(k_packed, ((0, 0), (0, rec_dim - width)))
    else:
        k = jnp.pad(k_packed[:, : head_dim + 1], ((0, 0), (0, _KV_PACKING - 1)))
    page, slot = kv_cache_utils.get_dcp_page_and_slot(
        k.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        dcp_rank,
        page_size=page_size,
        num_pages=cache_kv.shape[0],
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    return cache_kv.at[page, :, :, slot].set(k.reshape(-1, sublane_groups, _KV_PACKING))


_TODAYS_APPEND = {
    _HAS: has_append_4d,
    _SAL: sal_append_4d,
}
_TODAYS_APPEND_DCP = {
    _HAS: has_append_4d_dcp,
    _SAL: sal_append_4d_dcp,
}


def _in_layout(cache_has: jax.Array, layout: str, head_dim: int = 128) -> jax.Array:
    if layout == _HAS:
        return cache_has
    return st_lib.convert_cache_to_seq_along_lane(cache_has, actual_head_dim=head_dim)


# ---------------------------------------------------------------------------
# Part 1: Append-only microbenchmark (`coalesced_kv_append_benchmark.py`)
# ---------------------------------------------------------------------------


def _per_call_us(step, cache: jax.Array, args: tuple[jax.Array, ...]) -> float:
    """Per-call device+dispatch time of `cache = step(cache, *args)`."""

    def chain(num_calls):
        def run(caches, args):
            caches = list(caches)
            for i in range(num_calls):
                j = i % _NUM_CACHES
                caches[j], args = jax.lax.optimization_barrier((caches[j], args))
                caches[j] = step(caches[j], *args)
            return caches

        return jax.jit(run, donate_argnums=0)

    def seconds(fn):
        caches = jax.block_until_ready([jnp.copy(cache) for _ in range(_NUM_CACHES)])
        start = time.perf_counter()
        jax.block_until_ready(fn(caches, args))
        return time.perf_counter() - start

    short, long = chain(_SHORT), chain(_LONG)
    seconds(short)
    seconds(long)
    t_short = np.median([seconds(short) for _ in range(_REPS)])
    t_long = np.median([seconds(long) for _ in range(_REPS)])
    return float((t_long - t_short) / (_LONG - _SHORT) * 1e6)


def _batch(q_lens, seq_lens, *, page_size, num_pages, seed):
    """Random cache, K rows and metadata; pages written this step are distinct."""
    rng = np.random.default_rng(seed)
    q_lens = np.asarray(q_lens, np.int32)
    seq_lens = np.asarray(seq_lens, np.int32)
    pages_per_seq = int(seq_lens.max()) // page_size + 1
    table = rng.integers(0, num_pages, size=(len(q_lens), pages_per_seq))
    free = iter(rng.permutation(num_pages))
    for s, (q_len, seq_len) in enumerate(zip(q_lens, seq_lens)):
        if q_len:
            first, last = (seq_len - q_len) // page_size, (seq_len - 1) // page_size
            for ordinal in range(first, last + 1):
                table[s, ordinal] = next(free)
    cache = rng.integers(
        0, 256, size=(num_pages, page_size // 4, 4, 256), dtype=np.uint8
    )
    k = rng.integers(0, 256, size=(max(int(q_lens.sum()), 1), 129), dtype=np.uint8)
    return jnp.asarray(cache), (
        jnp.asarray(k),
        jnp.asarray(seq_lens),
        jnp.asarray(table, jnp.int32),
        jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32),
    )


def _decode(num_seqs, seq_len):
    stagger = (np.arange(num_seqs) * 16) % 64
    return [1] * num_seqs, seq_len - stagger


def _mixed(num_decodes, seq_len, chunk_offset):
    q_lens, seq_lens = _decode(num_decodes, seq_len)
    return q_lens + [1024 - num_decodes], list(seq_lens) + [seq_len + chunk_offset]


_SCENARIOS = (
    ("Decode B=16, 8K", _decode(16, 8192)),
    ("Decode B=32, 8K", _decode(32, 8192)),
    ("Decode B=64, 8K", _decode(64, 8192)),
    ("Decode B=128, 8K", _decode(128, 8192)),
    ("Decode B=256, 8K", _decode(256, 8192)),
    ("Decode B=16, 128K", _decode(16, 131072)),
    ("Prefill T=1024 aligned, 8K", ([1024], [8192])),
    ("Prefill T=1024 unaligned, 8K", ([1024], [8199])),
    ("Prefill T=1024 aligned, 128K", ([1024], [131072])),
    ("Prefill T=1024 unaligned, 128K", ([1024], [131079])),
    ("Prefill T=4096 aligned, 64K", ([4096], [65536])),
    ("Prefill T=4096 unaligned, 64K", ([4096], [65543])),
    ("Mixed 32x1 + 1x992, 8K", _mixed(32, 8192, 7)),
    ("Mixed 63x1 + 1x961, 8K serving", _mixed(63, 8192, 37)),
    ("Mixed 8x1 + 1x1016, 128K serving", _mixed(8, 131072, 91)),
    ("Mixed 15x1 + 1x1009, 128K serving", _mixed(15, 131072, 5)),
)

# AgentX DCP16: page_size=256, 16 ranks, interleave_size=4, 8 sequence slots.
# (name, q_lens, num_tokens, rank, local slot offset of rank 0's first write)
_AGENTX_DCP_SIZE, _AGENTX_INTERLEAVE, _AGENTX_PAGE_SIZE = 16, 4, 256
_AGENTX_SCENARIOS = (
    ("AgentX DCP16 decode T=1, active rank", [1], 1, 0, 1),
    ("AgentX DCP16 decode T=8, active rank", [1] * 8, 8, 0, 1),
    ("AgentX DCP16 decode T=16, idle rank", [1, 1, 1, 1], 16, 15, 1),
    ("AgentX DCP16 decode T=16, active rank", [1, 1, 1, 1], 16, 0, 1),
    ("AgentX DCP16 chunk T=1024, off=0", [1, 1, 1, 1021], 1024, 0, 0),
    ("AgentX DCP16 chunk T=1024, off=253", [1, 1, 1, 1021], 1024, 0, 253),
    ("AgentX DCP16 chunk T=4096, off=0", [1, 1, 1, 4093], 4096, 0, 0),
    ("AgentX DCP16 chunk T=4096, off=253", [1, 1, 1, 4093], 4096, 0, 253),
)


def _agentx_batch(q_lens, num_tokens, offset):
    max_seqs, pages_per_seq = 8, 64
    local = _AGENTX_PAGE_SIZE * 8 + offset
    base = (local // _AGENTX_INTERLEAVE) * _AGENTX_DCP_SIZE * _AGENTX_INTERLEAVE
    base += local % _AGENTX_INTERLEAVE
    q_full = np.zeros(max_seqs, np.int32)
    q_full[: len(q_lens)] = q_lens
    seq_lens = np.zeros(max_seqs, np.int32)
    for i, q_len in enumerate(q_lens):
        seq_lens[i] = base + i * _AGENTX_INTERLEAVE + q_len
    rng = np.random.default_rng(2026 + num_tokens + offset)
    num_pages = max_seqs * pages_per_seq
    cache = rng.integers(
        0, 256, size=(num_pages, _AGENTX_PAGE_SIZE // 4, 4, 256), dtype=np.uint8
    )
    return jnp.asarray(cache), (
        jnp.asarray(rng.integers(0, 256, (num_tokens, 129), dtype=np.uint8)),
        jnp.asarray(seq_lens),
        jnp.arange(num_pages, dtype=jnp.int32).reshape(max_seqs, pages_per_seq),
        jnp.asarray(np.concatenate([[0], np.cumsum(q_full)]), jnp.int32),
    )


def _print_row(name, cells):
    print(f"| {name} | " + " | ".join(cells) + " |", flush=True)


def run_append_benchmark() -> None:
    print("=== Part 1: Append-Only Per-Call Slope (us/call) ===")
    header = []
    for layout in _LAYOUTS:
        header += [f"today XLA ({layout})", f"coalesced_kv_append ({layout})"]
    _print_row("Scenario (us/call)", header)
    _print_row("---", ["--:"] * len(header))

    for seed, (name, (q_lens, seq_lens)) in enumerate(_SCENARIOS):
        cache_has, args = _batch(
            q_lens, seq_lens, page_size=1024, num_pages=640, seed=seed
        )
        cells = []
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            kernel = functools.partial(
                kv_cache_append.coalesced_kv_append, kv_layout=layout
            )
            for step in (_TODAYS_APPEND[layout], kernel):
                cells.append(f"{_per_call_us(step, cache, args):.2f}")
        _print_row(name, cells)

    for name, q_lens, num_tokens, rank, offset in _AGENTX_SCENARIOS:
        cache_has, args = _agentx_batch(q_lens, num_tokens, offset)
        cells = []
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            xla_dcp = functools.partial(
                _TODAYS_APPEND_DCP[layout],
                dcp_size=_AGENTX_DCP_SIZE,
                interleave_size=_AGENTX_INTERLEAVE,
            )
            kernel = functools.partial(
                kv_cache_append.coalesced_kv_append_dcp,
                dcp_size=_AGENTX_DCP_SIZE,
                interleave_size=_AGENTX_INTERLEAVE,
                kv_layout=layout,
            )
            dcp_args = args + (jnp.int32(rank),)
            for step in (xla_dcp, kernel):
                cells.append(f"{_per_call_us(step, cache, dcp_args):.2f}")
        _print_row(name, cells)
    print()


# ---------------------------------------------------------------------------
# Part 2: Compiled `append + StreamIdxTC` region benchmark (`reshape_verify_bench.py`)
# ---------------------------------------------------------------------------


def _hal_update(cache, k_packed, page, slot):
    return cache.at[page, slot // _KV_PACKING, slot % _KV_PACKING].set(k_packed)


def _hal_quad_jax_update(cache, k_packed, quad_row):
    orig_shape = cache.shape
    c3 = cache.reshape(-1, _KV_PACKING, orig_shape[-1])
    k3 = k_packed.reshape(-1, _KV_PACKING, orig_shape[-1])
    return c3.at[quad_row].set(k3).reshape(orig_shape)


def _sal_update(cache_sal, k_packed, page, slot, head_dim):
    kv_sublane_groups = head_dim // _KV_PACKING + 1
    k_rec = jnp.pad(
        k_packed[:, : head_dim + 1], ((0, 0), (0, _KV_PACKING - 1))
    ).reshape(-1, kv_sublane_groups, _KV_PACKING)
    return cache_sal.at[page, :, :, slot].set(k_rec)


def _time_best(fn, make_cache, args, iters: int) -> float:
    jax.block_until_ready(fn(make_cache(), *args))
    best = float("inf")
    for _ in range(iters):
        buf = make_cache()
        buf.block_until_ready()
        t0 = time.perf_counter()
        jax.block_until_ready(fn(buf, *args))
        best = min(best, time.perf_counter() - t0)
    return best


def _count_hlo_cache_relayouts(fn, cache, args) -> int:
    """Counts HLO instructions that copy/reshape/transpose the full cache buffer."""
    hlo = fn.lower(cache, *args).compile().as_text()
    cache_bytes = int(np.prod(cache.shape))
    count = 0
    for line in hlo.splitlines():
        if "= u8[" in line and any(
            op in line for op in (" copy(", " reshape(", " transpose(", " bitcast(")
        ):
            # Parse leading u8[...] shape and check if it matches full cache bytes.
            lhs = line.split("= u8[", 1)[1].split("]", 1)[0]
            try:
                dims = [int(d) for d in lhs.split(",") if d]
                if dims and int(np.prod(dims)) == cache_bytes:
                    count += 1
            except ValueError:
                pass
    return count


def run_region_benchmark_for_tokens(
    *,
    num_tokens: int,
    pages: int,
    page_size: int,
    head_dim: int,
    num_q_heads: int,
    seq_len: int,
    topk: int,
    layers: int,
    iters: int,
    dcp_size: int,
    interleave_size: int,
) -> None:
    width = -(-(head_dim + 1) // 128) * 128
    cache_bytes = pages * page_size * width
    print(
        f"=== Region Benchmark: T={num_tokens},"
        f" cache u8[{pages},{page_size // _KV_PACKING},{_KV_PACKING},{width}]"
        f" ({cache_bytes / 2**20:.1f} MiB), seq_len={seq_len},"
        f" layers={layers}, dcp_size={dcp_size}, interleave={interleave_size} ==="
    )

    rng = np.random.default_rng(42 + num_tokens)
    max_num_seqs = 16
    pages_per_seq = max(128, (seq_len // max(1, dcp_size)) // page_size + 2)
    assert pages >= pages_per_seq

    page_table_np = np.zeros((max_num_seqs, pages_per_seq), dtype=np.int32)
    seq_pages = rng.choice(pages, size=pages_per_seq, replace=False)
    page_table_np[0] = seq_pages.astype(np.int32)
    page_indices_d = jnp.asarray(page_table_np.reshape(-1), dtype=jnp.int32)

    seq_lens_np = np.zeros((max_num_seqs,), dtype=np.int32)
    seq_lens_np[0] = seq_len
    seq_lens_d = jnp.asarray(seq_lens_np, dtype=jnp.int32)

    cu_q_lens_np = np.zeros((max_num_seqs + 1,), dtype=np.int32)
    cu_q_lens_np[1:] = num_tokens
    cu_q_lens_d = jnp.asarray(cu_q_lens_np, dtype=jnp.int32)
    if num_tokens == 1:
        distribution_d = jnp.asarray([1, 0, 0], dtype=jnp.int32)
    else:
        distribution_d = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    k_all = jnp.asarray(
        rng.integers(1, 255, size=(layers, num_tokens, width)).astype(np.uint8)
    )
    q_d = jnp.asarray(
        rng.standard_normal((num_tokens, num_q_heads, head_dim)), jnp.bfloat16
    )
    w_d = jnp.asarray(rng.standard_normal((num_tokens, num_q_heads)), jnp.bfloat16)

    base_cache_np = rng.integers(
        1,
        127,
        size=(pages, page_size // _KV_PACKING, _KV_PACKING, width),
        dtype=np.uint8,
    )

    def mk_cache():
        return jnp.asarray(base_cache_np, dtype=jnp.uint8)

    base_cache_sal_np = np.asarray(
        st_lib.convert_cache_to_seq_along_lane(mk_cache(), actual_head_dim=head_dim)
    )

    def mk_cache_sal():
        return jnp.asarray(base_cache_sal_np, dtype=jnp.uint8)

    dcp_rank_d = jnp.int32(0)

    def _score(cache, kv_layout: str):
        return st_lib.streamindex_topk(
            q_d,
            w_d,
            cache,
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            distribution_d,
            k=topk,
            compression_ratio=1,
            num_kv_pages_per_block=(2, 2, 2),
            num_queries_per_block=(1, 64, 64) if num_tokens == 1 else (16, 64, 64),
            kv_layout=kv_layout,
            cp_size=dcp_size,
            cp_rank=dcp_rank_d if dcp_size > 1 else 0,
            interleave_size=interleave_size if dcp_size > 1 else 1,
        )

    if dcp_size == 1:
        pos_np = np.arange(seq_len - num_tokens, seq_len, dtype=np.int32)
        phys_page_np = seq_pages[pos_np // page_size].astype(np.int32)
        slot_np = (pos_np % page_size).astype(np.int32)
        page_d = jnp.asarray(phys_page_np)
        slot_d = jnp.asarray(slot_np)

        def v0_xla_has(cache, k_p):
            cache = _hal_update(cache, k_p, page_d, slot_d)
            return cache, _score(cache, _HAS)

        def v2_pallas_has(cache, k_p):
            cache = kv_cache_append.coalesced_kv_append(
                cache, k_p, seq_lens_d, page_indices_d, cu_q_lens_d, kv_layout=_HAS
            )
            return cache, _score(cache, _HAS)

        def v3a_xla_sal(cache_sal, k_p):
            cache_sal = _sal_update(cache_sal, k_p, page_d, slot_d, head_dim)
            return cache_sal, _score(cache_sal, _SAL)

        def v3b_pallas_sal(cache_sal, k_p):
            cache_sal = kv_cache_append.coalesced_kv_append(
                cache_sal,
                k_p,
                seq_lens_d,
                page_indices_d,
                cu_q_lens_d,
                kv_layout=_SAL,
            )
            return cache_sal, _score(cache_sal, _SAL)

        c_ref = _hal_update(mk_cache(), k_all[0], page_d, slot_d)
        c_has = kv_cache_append.coalesced_kv_append(
            mk_cache(), k_all[0], seq_lens_d, page_indices_d, cu_q_lens_d
        )
        c_sal_ref = _sal_update(mk_cache_sal(), k_all[0], page_d, slot_d, head_dim)
        c_sal = kv_cache_append.coalesced_kv_append(
            mk_cache_sal(),
            k_all[0],
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            kv_layout=_SAL,
        )
    else:

        def v0_xla_has(cache, k_p):
            cache = has_append_4d_dcp(
                cache,
                k_p,
                seq_lens_d,
                page_indices_d,
                cu_q_lens_d,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )
            return cache, _score(cache, _HAS)

        def v2_pallas_has(cache, k_p):
            cache = kv_cache_append.coalesced_kv_append_dcp(
                cache,
                k_p,
                seq_lens_d,
                page_indices_d,
                cu_q_lens_d,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                kv_layout=_HAS,
            )
            return cache, _score(cache, _HAS)

        def v3a_xla_sal(cache_sal, k_p):
            cache_sal = sal_append_4d_dcp(
                cache_sal,
                k_p,
                seq_lens_d,
                page_indices_d,
                cu_q_lens_d,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )
            return cache_sal, _score(cache_sal, _SAL)

        def v3b_pallas_sal(cache_sal, k_p):
            cache_sal = kv_cache_append.coalesced_kv_append_dcp(
                cache_sal,
                k_p,
                seq_lens_d,
                page_indices_d,
                cu_q_lens_d,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                kv_layout=_SAL,
            )
            return cache_sal, _score(cache_sal, _SAL)

        c_ref = has_append_4d_dcp(
            mk_cache(),
            k_all[0],
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            dcp_rank_d,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )
        c_has = kv_cache_append.coalesced_kv_append_dcp(
            mk_cache(),
            k_all[0],
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            dcp_rank_d,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
            kv_layout=_HAS,
        )
        c_sal_ref = sal_append_4d_dcp(
            mk_cache_sal(),
            k_all[0],
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            dcp_rank_d,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )
        c_sal = kv_cache_append.coalesced_kv_append_dcp(
            mk_cache_sal(),
            k_all[0],
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            dcp_rank_d,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
            kv_layout=_SAL,
        )

    assert bool(jnp.array_equal(c_ref, c_has)), "coalesced_kv_append HAS mismatch"
    assert bool(
        jnp.array_equal(
            c_sal_ref,
            st_lib.convert_cache_to_seq_along_lane(c_ref, actual_head_dim=head_dim),
        )
    ), "SAL ref vs converted HAS mismatch"
    assert bool(jnp.array_equal(c_sal_ref, c_sal)), "coalesced_kv_append SAL mismatch"

    def chain(step_fn, reps):
        def body(cache, ks):
            acc = jnp.zeros((num_tokens, topk), dtype=jnp.int32)
            for r in range(reps):
                cache, idx = step_fn(cache, ks[r])
                cache, idx = jax.lax.optimization_barrier((cache, idx))
                acc = acc ^ idx
            return cache, acc

        return jax.jit(body, donate_argnums=(0,))

    variants = [
        ("0. Today XLA (HAS) + StreamIdxTC (HAS)", v0_xla_has, mk_cache),
        ("2B. coalesced_kv_append (HAS) + StreamIdxTC (HAS)", v2_pallas_has, mk_cache),
        ("3A. Naive XLA (SAL) + StreamIdxTC (SAL)", v3a_xla_sal, mk_cache_sal),
        (
            "3B. coalesced_kv_append (SAL) + StreamIdxTC (SAL)",
            v3b_pallas_sal,
            mk_cache_sal,
        ),
    ]

    _print_row(
        f"Variant (T={num_tokens})",
        ["1-layer (us)", f"{layers}-layer (us)", "slope (us/layer)", "HLO relayouts"],
    )
    _print_row("---", ["--:", "--:", "--:", "--:"])
    for label, step_fn, cache_maker in variants:
        fn1 = chain(step_fn, 1)
        fnl = chain(step_fn, layers)
        t1 = _time_best(fn1, cache_maker, (k_all[:1],), iters)
        tl = _time_best(fnl, cache_maker, (k_all[:layers],), iters)
        slope = (tl - t1) / max(1, layers - 1) * 1e6
        relayouts = _count_hlo_cache_relayouts(fn1, cache_maker(), (k_all[:1],))
        _print_row(
            label,
            [f"{t1 * 1e6:.1f}", f"{tl * 1e6:.1f}", f"{slope:.1f}", f"{relayouts}"],
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "append", "region"),
        default="all",
        help="Benchmark mode to run.",
    )
    parser.add_argument("--pages", type=int, default=424, help="Pages in cache.")
    parser.add_argument("--page-size", type=int, default=256, help="Tokens per page.")
    parser.add_argument("--head-dim", type=int, default=128, help="Indexer head dim.")
    parser.add_argument("--num-q-heads", type=int, default=64, help="Query heads.")
    parser.add_argument("--seq-len", type=int, default=32768, help="Active seq len.")
    parser.add_argument("--topk", type=int, default=2048, help="Top-K retrieved.")
    parser.add_argument("--layers", type=int, default=4, help="Chained layers.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations.")
    parser.add_argument("--dcp-size", type=int, default=16, help="DCP size for region.")
    parser.add_argument(
        "--interleave-size", type=int, default=4, help="DCP interleave size."
    )
    parser.add_argument(
        "--region-tokens",
        type=str,
        default="1,8,1024,4096",
        help="Comma-separated token counts for region benchmark.",
    )
    args = parser.parse_args()

    print(f"device: {jax.devices()[0].device_kind}  n={len(jax.devices())}\n")
    if args.mode in ("all", "append"):
        run_append_benchmark()
    if args.mode in ("all", "region"):
        for tok_str in args.region_tokens.split(","):
            tok = int(tok_str.strip())
            run_region_benchmark_for_tokens(
                num_tokens=tok,
                pages=args.pages,
                page_size=args.page_size,
                head_dim=args.head_dim,
                num_q_heads=args.num_q_heads,
                seq_len=args.seq_len,
                topk=args.topk,
                layers=args.layers,
                iters=args.iters,
                dcp_size=args.dcp_size,
                interleave_size=args.interleave_size,
            )


if __name__ == "__main__":
    main()

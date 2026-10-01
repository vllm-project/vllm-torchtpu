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

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_append,
    kv_cache_utils,
)

_KV_PACKING = 4
_HAS = "head_along_sublane"
_SAL = "seq_along_lane"
_LAYOUTS = (_HAS, _SAL)
_SHORT, _LONG, _REPS, _NUM_CACHES = 2, 10, 3, 2


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

    caches = jax.block_until_ready([jnp.copy(cache) for _ in range(_NUM_CACHES)])

    def seconds(fn):
        nonlocal caches
        start = time.perf_counter()
        caches = jax.block_until_ready(fn(caches, args))
        return time.perf_counter() - start

    short, long = chain(_SHORT), chain(_LONG)
    seconds(short)
    seconds(long)
    t_short = np.median([seconds(short) for _ in range(_REPS)])
    t_long = np.median([seconds(long) for _ in range(_REPS)])
    return float((t_long - t_short) / (_LONG - _SHORT) * 1e6)


def _batch(q_lens, seq_lens, *, page_size, num_pages, seed):
    """Cache, K rows and metadata; pages written this step are distinct."""
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
    cache = jnp.zeros((num_pages, page_size // 4, 4, 256), dtype=jnp.uint8)
    k = rng.integers(0, 256, size=(max(int(q_lens.sum()), 1), 129), dtype=np.uint8)
    return cache, (
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
    cache = jnp.zeros((num_pages, _AGENTX_PAGE_SIZE // 4, 4, 256), dtype=jnp.uint8)
    return cache, (
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


def _time_best(fn, make_cache, args, iters: int) -> float:
    buf = make_cache()
    buf, _ = jax.block_until_ready(fn(buf, *args))
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        buf, _ = jax.block_until_ready(fn(buf, *args))
        best = min(best, time.perf_counter() - t0)
    return best


def _count_hlo_cache_relayouts_from_text(hlo: str, cache_bytes: int) -> int:
    """Counts HLO instructions that copy/reshape/transpose the full cache buffer."""
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


def _count_hlo_cache_relayouts(fn, cache, args) -> int:
    """Counts HLO instructions that copy/reshape/transpose the full cache buffer."""
    hlo = fn.lower(cache, *args).compile().as_text()
    return _count_hlo_cache_relayouts_from_text(hlo, int(np.prod(cache.shape)))


def _compile_and_time(fn, make_cache, args, iters: int) -> tuple[np.ndarray, str]:
    """Compiles `fn` once, warms up once, and returns `(samples_us, hlo_text)`."""
    buf = make_cache()
    compiled = fn.lower(buf, *args).compile()
    hlo_text = compiled.as_text()
    buf, _ = jax.block_until_ready(compiled(buf, *args))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        buf, _ = jax.block_until_ready(compiled(buf, *args))
        samples.append((time.perf_counter() - t0) * 1e6)
    return np.asarray(samples, dtype=np.float64), hlo_text


def _fmt_stat(samples_us: np.ndarray, *, med_override: float | None = None) -> str:
    med = float(np.median(samples_us)) if med_override is None else med_override
    lo = float(np.min(samples_us))
    hi = float(np.max(samples_us))
    return f"{med:.1f} [{lo:.1f}..{hi:.1f}]"


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
    dcp_rank: int = 0,
    max_num_seqs: int = 16,
    num_active_seqs: int | None = None,
    min_pages_per_seq: int = 128,
    barrier_metadata: bool = False,
    scenario_label: str | None = None,
) -> None:
    iters = max(5, iters)
    width = -(-(head_dim + 1) // 128) * 128
    cache_bytes = pages * page_size * width
    title = scenario_label or (
        f"T={num_tokens}, seq_len={seq_len}, dcp_size={dcp_size},"
        f" dcp_rank={dcp_rank}, interleave={interleave_size}"
    )
    print(
        f"=== Region Benchmark ({title}):"
        f" cache HAS u8[{pages},{page_size // _KV_PACKING},{_KV_PACKING},{width}]"
        f" ({cache_bytes / 2**20:.1f} MiB), layers={layers}, iters={iters},"
        f" barrier_metadata={barrier_metadata} ==="
    )

    rng = np.random.default_rng(42 + num_tokens)
    pages_per_seq = max(
        min_pages_per_seq, (seq_len // max(1, dcp_size)) // page_size + 2
    )
    assert pages >= pages_per_seq

    if num_active_seqs is None:
        is_decode = num_tokens <= 8
        num_active_seqs = num_tokens if is_decode else 1
    else:
        is_decode = num_active_seqs == num_tokens

    seq_lens_np = np.zeros((max_num_seqs,), dtype=np.int32)
    cu_q_lens_np = np.zeros((max_num_seqs + 1,), dtype=np.int32)
    if is_decode:
        for s in range(num_active_seqs):
            if dcp_size > 1:
                cycle = dcp_size * interleave_size
                base_seq = seq_len - (seq_len % cycle) - cycle
                seq_lens_np[s] = base_seq + 1 + (s % interleave_size)
            else:
                seq_lens_np[s] = seq_len
            cu_q_lens_np[s + 1] = s + 1
        cu_q_lens_np[num_active_seqs + 1 :] = num_active_seqs
        distribution_d = jnp.asarray(
            [num_active_seqs, num_active_seqs, num_active_seqs], dtype=jnp.int32
        )
    else:
        seq_lens_np[0] = seq_len
        cu_q_lens_np[1:] = num_tokens
        distribution_d = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    seq_lens_d = jnp.asarray(seq_lens_np, dtype=jnp.int32)
    cu_q_lens_d = jnp.asarray(cu_q_lens_np, dtype=jnp.int32)

    page_table_np = np.zeros((max_num_seqs, pages_per_seq), dtype=np.int32)
    if pages >= num_active_seqs * pages_per_seq:
        perm = rng.permutation(pages).astype(np.int32)
        for s in range(num_active_seqs):
            page_table_np[s] = perm[s * pages_per_seq : (s + 1) * pages_per_seq]
    else:
        reserved_tail_pages = rng.choice(pages, size=num_active_seqs, replace=False)
        remaining_pool = np.setdiff1d(
            np.arange(pages, dtype=np.int32), reserved_tail_pages
        )
        virtual_page_size = page_size * max(1, dcp_size)
        for s in range(num_active_seqs):
            seq_pages = rng.choice(
                remaining_pool, size=pages_per_seq, replace=False
            ).astype(np.int32)
            tail_ord = (int(seq_lens_np[s]) - 1) // virtual_page_size
            seq_pages[tail_ord] = int(reserved_tail_pages[s])
            page_table_np[s] = seq_pages
    page_indices_d = jnp.asarray(page_table_np.reshape(-1), dtype=jnp.int32)

    k_all = jnp.asarray(
        rng.integers(1, 255, size=(layers, num_tokens, width)).astype(np.uint8)
    )
    q_d = jnp.asarray(
        rng.standard_normal((num_tokens, num_q_heads, head_dim)), jnp.bfloat16
    )
    w_d = jnp.asarray(rng.standard_normal((num_tokens, num_q_heads)), jnp.bfloat16)

    # Materialize base caches once on device so per-variant copies stay on HBM.
    tile_pages = min(pages, 64)
    tile_np = rng.integers(
        1,
        127,
        size=(tile_pages, page_size // _KV_PACKING, _KV_PACKING, width),
        dtype=np.uint8,
    )
    tile_dev = jnp.asarray(tile_np, dtype=jnp.uint8)
    reps_pages = -(-pages // tile_pages)
    base_cache_has = jax.block_until_ready(
        jnp.tile(tile_dev, (reps_pages, 1, 1, 1))[:pages]
    )
    base_cache_sal = jax.block_until_ready(
        st_lib.convert_cache_to_seq_along_lane(base_cache_has, actual_head_dim=head_dim)
    )

    def mk_cache():
        return jnp.copy(base_cache_has)

    def mk_cache_sal():
        return jnp.copy(base_cache_sal)

    dcp_rank_d = jnp.int32(dcp_rank)

    def _score(cache, kv_layout: str, sl, pi, cq):
        return st_lib.streamindex_topk(
            q_d,
            w_d,
            cache,
            sl,
            pi,
            cq,
            distribution_d,
            k=topk,
            compression_ratio=1,
            num_kv_pages_per_block=(2, 2, 2),
            num_queries_per_block=(1, 64, 64),
            kv_layout=kv_layout,
            cp_size=dcp_size,
            cp_rank=dcp_rank_d if dcp_size > 1 else 0,
            interleave_size=interleave_size if dcp_size > 1 else 1,
        )

    if dcp_size == 1:

        def v0_xla_has(cache, k_p, sl, pi, cq):
            cache = has_append_4d(cache, k_p, sl, pi, cq)
            return cache, _score(cache, _HAS, sl, pi, cq)

        def v2_pallas_has(cache, k_p, sl, pi, cq):
            cache = kv_cache_append.coalesced_kv_append(
                cache, k_p, sl, pi, cq, kv_layout=_HAS
            )
            return cache, _score(cache, _HAS, sl, pi, cq)

        def v3a_xla_sal(cache_sal, k_p, sl, pi, cq):
            cache_sal = sal_append_4d(cache_sal, k_p, sl, pi, cq)
            return cache_sal, _score(cache_sal, _SAL, sl, pi, cq)

        def v3b_pallas_sal(cache_sal, k_p, sl, pi, cq):
            cache_sal = kv_cache_append.coalesced_kv_append(
                cache_sal,
                k_p,
                sl,
                pi,
                cq,
                kv_layout=_SAL,
            )
            return cache_sal, _score(cache_sal, _SAL, sl, pi, cq)

        c_ref = has_append_4d(
            mk_cache(), k_all[0], seq_lens_d, page_indices_d, cu_q_lens_d
        )
        c_has = kv_cache_append.coalesced_kv_append(
            mk_cache(), k_all[0], seq_lens_d, page_indices_d, cu_q_lens_d
        )
        c_sal_ref = sal_append_4d(
            mk_cache_sal(), k_all[0], seq_lens_d, page_indices_d, cu_q_lens_d
        )
        c_sal = kv_cache_append.coalesced_kv_append(
            mk_cache_sal(),
            k_all[0],
            seq_lens_d,
            page_indices_d,
            cu_q_lens_d,
            kv_layout=_SAL,
        )
    else:

        def v0_xla_has(cache, k_p, sl, pi, cq):
            cache = has_append_4d_dcp(
                cache,
                k_p,
                sl,
                pi,
                cq,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )
            return cache, _score(cache, _HAS, sl, pi, cq)

        def v2_pallas_has(cache, k_p, sl, pi, cq):
            cache = kv_cache_append.coalesced_kv_append_dcp(
                cache,
                k_p,
                sl,
                pi,
                cq,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                kv_layout=_HAS,
            )
            return cache, _score(cache, _HAS, sl, pi, cq)

        def v3a_xla_sal(cache_sal, k_p, sl, pi, cq):
            cache_sal = sal_append_4d_dcp(
                cache_sal,
                k_p,
                sl,
                pi,
                cq,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )
            return cache_sal, _score(cache_sal, _SAL, sl, pi, cq)

        def v3b_pallas_sal(cache_sal, k_p, sl, pi, cq):
            cache_sal = kv_cache_append.coalesced_kv_append_dcp(
                cache_sal,
                k_p,
                sl,
                pi,
                cq,
                dcp_rank_d,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                kv_layout=_SAL,
            )
            return cache_sal, _score(cache_sal, _SAL, sl, pi, cq)

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
                if barrier_metadata:
                    cache, k_r, sl, pi, cq = jax.lax.optimization_barrier(
                        (cache, ks[r], seq_lens_d, page_indices_d, cu_q_lens_d)
                    )
                else:
                    k_r, sl, pi, cq = ks[r], seq_lens_d, page_indices_d, cu_q_lens_d
                cache, idx = step_fn(cache, k_r, sl, pi, cq)
                cache, idx = jax.lax.optimization_barrier((cache, idx))
                acc = acc ^ idx
            return cache, acc

        return jax.jit(body, donate_argnums=(0,))

    variants = [
        (
            "0. Today XLA (HAS) + StreamIdxTC (HAS)",
            v0_xla_has,
            mk_cache,
            int(np.prod(base_cache_has.shape)),
        ),
        (
            "2B. coalesced_kv_append (HAS) + StreamIdxTC (HAS)",
            v2_pallas_has,
            mk_cache,
            int(np.prod(base_cache_has.shape)),
        ),
        (
            "3A. Naive XLA (SAL) + StreamIdxTC (SAL)",
            v3a_xla_sal,
            mk_cache_sal,
            int(np.prod(base_cache_sal.shape)),
        ),
        (
            "3B. coalesced_kv_append (SAL) + StreamIdxTC (SAL)",
            v3b_pallas_sal,
            mk_cache_sal,
            int(np.prod(base_cache_sal.shape)),
        ),
    ]

    _print_row(
        f"Variant ({title})",
        [
            "1-layer (us)",
            f"{layers}-layer (us)",
            "slope (us/layer)",
            "78L step (ms)",
            "HLO relayouts",
        ],
    )
    _print_row("---", ["--:", "--:", "--:", "--:", "--:"])
    denom = max(1, layers - 1)
    for label, step_fn, cache_maker, v_cache_bytes in variants:
        fn1 = chain(step_fn, 1)
        fnl = chain(step_fn, layers)
        t1_us, hlo1 = _compile_and_time(fn1, cache_maker, (k_all[:1],), iters)
        tl_us, _ = _compile_and_time(fnl, cache_maker, (k_all[:layers],), iters)
        slope_samples = (tl_us - t1_us) / denom
        slope_med = float(np.median(tl_us) - np.median(t1_us)) / denom
        step78_ms = slope_med * 78.0 / 1000.0
        relayouts = _count_hlo_cache_relayouts_from_text(hlo1, v_cache_bytes)
        _print_row(
            label,
            [
                _fmt_stat(t1_us),
                _fmt_stat(tl_us),
                _fmt_stat(slope_samples, med_override=slope_med),
                f"{step78_ms:.2f}",
                f"{relayouts}",
            ],
        )
    print()


def run_served_region_benchmarks(*, iters: int = 5, layers: int = 4) -> None:
    """Runs Gate D served-shape Table B benchmarks (DCP16, block_size=256,
    max_num_seqs=8) plus the DCP1 T=1/T=8 diagnostic comparison."""
    iters = max(5, iters)
    common = dict(
        page_size=256,
        head_dim=128,
        num_q_heads=64,
        topk=2048,
        layers=layers,
        iters=iters,
    )

    # Served DCP16 scenarios (--block-size=256, --max-num-batched-tokens=4096,
    # --max-num-seqs=8, DCP16, interleave=4):
    # - P90 context: 655360 + 8192 = 663552 (~41.5K tok/rank/seq = 162 pages/rank/seq)
    #   With pages_per_seq=164 and max_num_seqs=8, total pool = 1312 pages/rank.
    # - P50 context: 163840 + 2048 = 165888 (~10.4K tok/rank/seq = 40.5 pages/rank/seq)
    #   With pages_per_seq=44 and max_num_seqs=8, total pool = 352 pages/rank.
    p90_decode_seq_len = 655360 + 8192
    p50_decode_seq_len = 163840 + 2048
    p90_prefill_seq_len = 655360 + 4096
    p50_prefill_seq_len = 163840 + 2048

    served_scenarios = [
        dict(
            scenario_label="Served P90 decode B=8 T=8 ctx=663552 active (rank=0)",
            num_tokens=8,
            num_active_seqs=8,
            max_num_seqs=8,
            min_pages_per_seq=164,
            pages=8 * 164,
            seq_len=p90_decode_seq_len,
            dcp_size=16,
            interleave_size=4,
            dcp_rank=0,
        ),
        dict(
            scenario_label="Served P90 decode B=8 T=8 ctx=663552 idle (rank=15)",
            num_tokens=8,
            num_active_seqs=8,
            max_num_seqs=8,
            min_pages_per_seq=164,
            pages=8 * 164,
            seq_len=p90_decode_seq_len,
            dcp_size=16,
            interleave_size=4,
            dcp_rank=15,
        ),
        dict(
            scenario_label="Served P50 decode B=4 T=4 ctx=165888 active (rank=0)",
            num_tokens=4,
            num_active_seqs=4,
            max_num_seqs=8,
            min_pages_per_seq=44,
            pages=8 * 44,
            seq_len=p50_decode_seq_len,
            dcp_size=16,
            interleave_size=4,
            dcp_rank=0,
        ),
        dict(
            scenario_label="Served P50 decode B=4 T=4 ctx=165888 idle (rank=15)",
            num_tokens=4,
            num_active_seqs=4,
            max_num_seqs=8,
            min_pages_per_seq=44,
            pages=8 * 44,
            seq_len=p50_decode_seq_len,
            dcp_size=16,
            interleave_size=4,
            dcp_rank=15,
        ),
        dict(
            scenario_label="Served P90 prefill B=1 T=4096 off=655360 (rank=0)",
            num_tokens=4096,
            num_active_seqs=1,
            max_num_seqs=8,
            min_pages_per_seq=164,
            pages=8 * 164,
            seq_len=p90_prefill_seq_len,
            dcp_size=16,
            interleave_size=4,
            dcp_rank=0,
        ),
        dict(
            scenario_label="Served P50 prefill B=1 T=2048 off=163840 (rank=0)",
            num_tokens=2048,
            num_active_seqs=1,
            max_num_seqs=8,
            min_pages_per_seq=44,
            pages=8 * 44,
            seq_len=p50_prefill_seq_len,
            dcp_size=16,
            interleave_size=4,
            dcp_rank=0,
        ),
    ]

    for sc in served_scenarios:
        run_region_benchmark_for_tokens(**common, **sc)

    # Diagnostic: DCP1 T=1 and T=8 at seq_len=32768 (pages=424) with and without
    # per-layer optimization_barrier on metadata (seq_lens, page_indices, cu_q_lens).
    for barrier_meta in (False, True):
        tag = "barrier_meta=True" if barrier_meta else "barrier_meta=False (hoisted)"
        for tok in (1, 8):
            run_region_benchmark_for_tokens(
                **common,
                scenario_label=f"DCP1 T={tok} seq=32768 {tag}",
                num_tokens=tok,
                pages=424,
                seq_len=32768,
                dcp_size=1,
                interleave_size=4,
                barrier_metadata=barrier_meta,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "append", "region", "served"),
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
    parser.add_argument(
        "--dcp-sizes",
        type=str,
        default="1,16",
        help="Comma-separated DCP sizes for region benchmark.",
    )
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
    if args.mode == "served":
        run_served_region_benchmarks(iters=args.iters, layers=args.layers)
    elif args.mode in ("all", "region"):
        for dcp_str in args.dcp_sizes.split(","):
            dcp_size = int(dcp_str.strip())
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
                    dcp_size=dcp_size,
                    interleave_size=args.interleave_size,
                )


if __name__ == "__main__":
    main()

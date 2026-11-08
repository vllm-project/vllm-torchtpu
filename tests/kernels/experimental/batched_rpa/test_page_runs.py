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
"""Decode steps that fetch contiguous runs of KV pages into a page-major window
(`kv_run_pages`).

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

Page tables come from manager blocks of `block_pages` consecutive kernel pages
in shuffled order, as vLLM builds them. Each kernel case runs the batched
kernel with one DMA per page and with page runs on the same inputs; outputs
and KV caches must be bitwise equal.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    bref_override,
    configs,
    kernel,
    wrapper,
)

_FP8 = jnp.dtype("float8_e4m3fn")
_VERIFY = configs.BlockSizes(bq_sz=4, bq_c_sz=4, bkv_sz=3072, batch_size=8, n_buffer=3)
_ONE = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=3072, batch_size=8, n_buffer=3)

# Histories around the 3072-token block and 2176-token manager block edges,
# short ones and 128K contexts.
_HIST = [
    131068,
    3068,
    3069,
    70000,
    255,
    131000,
    6140,
    1,
    9000,
    120001,
    4093,
    50000,
    132099,
    3072,
    6144,
    0,
    128,
    129,
    2175,
    2176,
    2177,
    40000,
    132099,
]


def _cdiv(a, b):
    return -(-a // b)


def block_page_table(kv_lens, page_size, block_pages, pages_per_seq, seed):
    """Page table of manager blocks (`block_pages` consecutive pages each) in
    shuffled order; block 0 is the null block that padding points to.
    Returns the table and the number of pages in the pool."""
    rng = np.random.default_rng(seed)
    blocks = [_cdiv(_cdiv(int(n), page_size), block_pages) for n in kv_lens]
    pool_blocks = sum(blocks) + max(8, sum(blocks) // 10) + 1
    order = rng.permutation(np.arange(1, pool_blocks, dtype=np.int32))
    table = np.zeros((len(kv_lens), pages_per_seq), np.int32)
    start = 0
    for s, n in enumerate(blocks):
        ids = order[start : start + n]
        start += n
        pages = (ids[:, None] * block_pages + np.arange(block_pages)).ravel()
        pages = pages[:pages_per_seq]
        table[s, : len(pages)] = pages
    return table, pool_blocks * block_pages


def _run(
    *,
    histories,
    q_lens,
    live,
    page_size=128,
    block_pages=17,
    block=_VERIFY,
    window=4,
    num_q_heads=16,
    num_kv_heads=1,
    head_dim=256,
    dtype=_FP8,
    sm_scale=1 / 16,
    k_scale=1.0,
    v_scale=1.0,
    pages_per_seq=None,
    stream_page_indices=None,
):
    kv_lens_host = np.asarray(histories) + np.asarray(q_lens)
    if pages_per_seq is None:
        used = _cdiv(int(kv_lens_host.max()), page_size)
        pages_per_seq = (_cdiv(used, block_pages) + 1) * block_pages
    table, total_pages = block_page_table(
        kv_lens_host, page_size, block_pages, pages_per_seq, seed=1
    )
    layout = configs.KVLayout.SEQ_ALONG_LANE
    shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype, kv_layout=layout
    )
    keys = jax.random.split(jax.random.key(2), 4)
    tokens = max(int(sum(q_lens)), 1)
    cache = jax.random.normal(keys[0], shape, jnp.float32).astype(dtype)
    q = (jax.random.normal(keys[1], (tokens, num_q_heads, head_dim)) * 3).astype(
        jnp.bfloat16
    )
    k = jax.random.normal(keys[2], (tokens, num_kv_heads, head_dim)).astype(dtype)
    v = jax.random.normal(keys[3], (tokens, num_kv_heads, head_dim)).astype(dtype)
    cu = jnp.asarray(np.concatenate(([0], np.cumsum(q_lens))), jnp.int32)
    kv_lens = jnp.asarray(kv_lens_host, jnp.int32)
    dist = jnp.array([live] * 3, jnp.int32)
    scales = {} if dtype != _FP8 else {"k_scale": k_scale, "v_scale": v_scale}

    def call(kv_run_pages):
        jax.clear_caches()
        out, new_cache = wrapper.ragged_paged_attention(
            q,
            k,
            v,
            cache.copy(),
            kv_lens,
            jnp.asarray(table.ravel()),
            cu,
            dist,
            sm_scale=sm_scale,
            kv_layout=layout,
            decode_query_size=window,
            decode_block_sizes=block,
            stream_page_indices=stream_page_indices,
            kv_run_pages=kv_run_pages,
            **scales,
        )
        jax.block_until_ready((out, new_cache))
        live_tokens = int(cu[live])
        return np.asarray(out[:live_tokens]), np.asarray(new_cache)

    return call


def _assert_same(call, kv_run_pages=17):
    ref_out, ref_cache = call(None)
    out, new_cache = call(kv_run_pages)
    np.testing.assert_array_equal(out.view(np.uint16), ref_out.view(np.uint16))
    np.testing.assert_array_equal(
        new_cache.view(np.uint8 if new_cache.itemsize == 1 else np.uint16),
        ref_cache.view(np.uint8 if ref_cache.itemsize == 1 else np.uint16),
    )


@pytest.fixture
def tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("batched RPA Pallas kernels require a TPU")


def test_served_verify_windows(tpu):
    # 26 requests near 132K with the 1M-token table: the compact-table path.
    hist = list(np.random.default_rng(0).integers(131072, 132096, 26))
    _assert_same(_run(histories=hist, q_lens=[4] * 26, live=26, pages_per_seq=8194))


def test_edge_histories(tpu):
    n = len(_HIST)
    _assert_same(_run(histories=_HIST, q_lens=[4] * n, live=n))


def test_partial_windows_and_batch(tpu):
    n = len(_HIST)
    q_lens = [1 + s % 4 for s in range(n)]
    _assert_same(_run(histories=_HIST, q_lens=q_lens, live=17))


def test_one_query_kernel(tpu):
    n = len(_HIST)
    _assert_same(_run(histories=_HIST, q_lens=[1] * n, live=n, block=_ONE, window=1))


@pytest.mark.parametrize(
    ("block_pages", "kv_run_pages"), [(1, 1), (2, 2), (5, 5), (24, 24), (32, 32)]
)
def test_run_lengths(tpu, block_pages, kv_run_pages):
    n = len(_HIST)
    call = _run(histories=_HIST, q_lens=[4] * n, live=n, block_pages=block_pages)
    _assert_same(call, kv_run_pages)


def test_runs_shorter_than_blocks(tpu):
    # Runs of 17 pages inside 34-page blocks are contiguous, as is every page.
    n = len(_HIST)
    call = _run(histories=_HIST, q_lens=[4] * n, live=n, block_pages=34)
    _assert_same(call, 17)
    _assert_same(call, 1)


def test_dynamic_run_sizes_on_full_blocks(tpu, monkeypatch):
    # Every step derives its runs from the cache lengths, unmasked blocks too.
    monkeypatch.setattr(bref_override, "_STATIC_FULL_BLOCK_COPIES", False)
    monkeypatch.setattr(kernel, "_UNMASKED_INTERIOR_BLOCKS", False)
    n = len(_HIST)
    _assert_same(_run(histories=_HIST, q_lens=[4] * n, live=n))


def test_256_token_pages(tpu):
    n = len(_HIST)
    call = _run(histories=_HIST, q_lens=[4] * n, live=n, page_size=256, block_pages=9)
    _assert_same(call, 9)


def test_streamed_page_table(tpu):
    hist = [132099, 2176, 70000, 1, 40961, 131071, 5000, 3072] * 3
    call = _run(
        histories=hist,
        q_lens=[4] * len(hist),
        live=len(hist),
        pages_per_seq=8194,
        stream_page_indices=True,
    )
    _assert_same(call)


@pytest.mark.parametrize("window", [4, 16])
def test_two_kv_heads_bf16(tpu, window):
    hist = [5000, 4095, 4096, 300, 9000, 1, 12000, 8191, 2175, 2176]
    q_lens = [window if s % 3 == 2 else 1 + s % window for s in range(len(hist))]
    block = configs.BlockSizes(
        bq_sz=window, bq_c_sz=window, bkv_sz=1024, batch_size=4, n_buffer=2
    )
    call = _run(
        histories=hist,
        q_lens=q_lens,
        live=len(hist),
        block=block,
        window=window,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        dtype=jnp.bfloat16,
        sm_scale=128**-0.5,
        block_pages=3,
    )
    _assert_same(call, 3)


def _cfgs(kv_run_pages=17, **serve):
    base = configs.ServingConfigs(
        num_seqs=26,
        page_size=128,
        total_q_tokens=104,
        num_page_indices=26 * 1536,
        dtype_q=jnp.bfloat16,
        dtype_kv=_FP8,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=4,
        kv_run_pages=kv_run_pages,
    )
    return configs.RpaConfigs(
        block=_VERIFY,
        model=configs.ModelConfigs(
            num_q_heads=16, num_kv_heads=1, head_dim=256, mask_value=-1.0
        ),
        serve=dataclasses.replace(base, **serve),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=1 << 26,
    )


@pytest.mark.cpu_test
def test_page_runs_only_where_valid_pages_follow_cache_lengths():
    assert _cfgs().page_runs
    assert not _cfgs(kv_run_pages=None).page_runs
    assert not _cfgs(skip_kv_update=True).page_runs
    assert not _cfgs(cp_group_size=2).page_runs
    assert not _cfgs(attention_scope=configs.AttentionScope.CACHE_ONLY).page_runs
    assert not _cfgs(kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE).page_runs
    assert not dataclasses.replace(_cfgs(), mode=configs.RpaCase.MIXED).page_runs
    windowed = dataclasses.replace(
        _cfgs(), model=dataclasses.replace(_cfgs().model, sliding_window=1024)
    )
    assert not windowed.page_runs


@pytest.mark.cpu_test
@pytest.mark.parametrize("divisor", [1, 2, 3, 7, 17, 24, 64, 127, 1536])
@pytest.mark.parametrize("bound", [1, 1536, 8194, 40000])
def test_mod_const(divisor, bound):
    x = jnp.arange(bound, dtype=jnp.int32)
    got = np.asarray(bref_override._mod_const(x, divisor, bound))
    np.testing.assert_array_equal(got, np.arange(bound) % divisor)


@pytest.mark.cpu_test
@pytest.mark.parametrize("run_pages", [1, 2, 5, 17, 23, 24, 25, 64])
def test_block_page_runs_cover_valid_pages_inside_runs(run_pages):
    cfgs = _cfgs(kv_run_pages=run_pages)
    num_pages = cfgs.bkv_p_cache
    # Every block offset within a run, every count of valid pages.
    rel = np.repeat(np.arange(64) * num_pages, num_pages + 1)
    valid = np.tile(np.arange(num_pages + 1), 64)
    runs = bref_override._block_page_runs(
        jnp.asarray(rel, jnp.int32), jnp.asarray(valid, jnp.int32), cfgs
    )
    pos = np.zeros_like(rel)
    for start, size in runs:
        start, size = np.broadcast_to(np.asarray(start), rel.shape), np.asarray(size)
        np.testing.assert_array_equal(start, pos)
        assert (size >= 0).all()
        used = size > 0
        first, last = rel + start, rel + start + size - 1
        np.testing.assert_array_equal(first[used] // run_pages, last[used] // run_pages)
        pos = pos + size
    np.testing.assert_array_equal(pos, valid)

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
"""Interior decode steps: no masks, no stitch, static page copies.

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

Each kernel case runs the batched kernel twice on the same inputs: once with
masks, stitches and per-page copy lookups on every step, once as served.
Outputs and KV caches must be bitwise equal.
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

# Histories around the 3072-token block edges, short ones and 128K contexts.
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
    9215,
    9216,
    9217,
    40000,
    132099,
]


def _run(
    *,
    histories,
    q_lens,
    live,
    page_size=128,
    block=_VERIFY,
    window=4,
    num_q_heads=16,
    num_kv_heads=1,
    head_dim=256,
    dtype=_FP8,
    layout=configs.KVLayout.SEQ_ALONG_LANE,
    sm_scale=1 / 16,
    k_scale=1.0,
    v_scale=1.0,
    pages_per_seq=None,
):
    num_seqs = len(histories)
    used = [(h + q + page_size - 1) // page_size for h, q in zip(histories, q_lens)]
    total_pages = sum(used) + 1
    if pages_per_seq is None:
        pages_per_seq = max(used) + 1
    shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype, kv_layout=layout
    )
    rng = np.random.default_rng(1)
    table = np.zeros((num_seqs, pages_per_seq), np.int32)
    perm = rng.permutation(total_pages).astype(np.int32)
    start = 0
    for s in range(num_seqs):
        table[s, : used[s]] = perm[start : start + used[s]]
        start += used[s]
    keys = jax.random.split(jax.random.key(2), 4)
    tokens = max(int(sum(q_lens)), 1)
    cache = jax.random.normal(keys[0], shape, jnp.float32).astype(dtype)
    q = (jax.random.normal(keys[1], (tokens, num_q_heads, head_dim)) * 3).astype(
        jnp.bfloat16
    )
    k = jax.random.normal(keys[2], (tokens, num_kv_heads, head_dim)).astype(dtype)
    v = jax.random.normal(keys[3], (tokens, num_kv_heads, head_dim)).astype(dtype)
    cu = jnp.asarray(np.concatenate(([0], np.cumsum(q_lens))), jnp.int32)
    kv_lens = jnp.asarray(np.asarray(histories) + np.asarray(q_lens), jnp.int32)
    dist = jnp.array([live] * 3, jnp.int32)

    def call():
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
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=layout,
            decode_query_size=window,
            decode_block_sizes=block,
        )
        jax.block_until_ready((out, new_cache))
        live_tokens = int(cu[live])
        return np.asarray(out[:live_tokens]), np.asarray(new_cache)

    return call


def _assert_same(monkeypatch, call):
    served_out, served_cache = call()
    monkeypatch.setattr(kernel, "_UNMASKED_INTERIOR_BLOCKS", False)
    monkeypatch.setattr(bref_override, "_STATIC_FULL_BLOCK_COPIES", False)
    ref_out, ref_cache = call()
    np.testing.assert_array_equal(served_out.view(np.uint16), ref_out.view(np.uint16))
    np.testing.assert_array_equal(served_cache.view(np.uint8), ref_cache.view(np.uint8))


@pytest.fixture
def tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("batched RPA Pallas kernels require a TPU")


def test_full_verify_windows(tpu, monkeypatch):
    n = len(_HIST)
    _assert_same(monkeypatch, _run(histories=_HIST, q_lens=[4] * n, live=n))


def test_mixed_windows_and_partial_batch(tpu, monkeypatch):
    n = len(_HIST)
    q_lens = [1 + s % 4 for s in range(n)]
    _assert_same(monkeypatch, _run(histories=_HIST, q_lens=q_lens, live=17))


def test_one_query_kernel(tpu, monkeypatch):
    n = len(_HIST)
    call = _run(histories=_HIST, q_lens=[1] * n, live=n, block=_ONE, window=1)
    _assert_same(monkeypatch, call)


def test_256_token_pages(tpu, monkeypatch):
    n = len(_HIST)
    call = _run(histories=_HIST, q_lens=[4] * n, live=n, page_size=256)
    _assert_same(monkeypatch, call)


def test_served_page_table_width(tpu, monkeypatch):
    # 1M-token table (8194 columns): the wrapper's compact-table path.
    hist = [132099] * 23
    call = _run(histories=hist, q_lens=[4] * 23, live=23, pages_per_seq=8194)
    _assert_same(monkeypatch, call)


def test_other_scales(tpu, monkeypatch):
    n = len(_HIST)
    call = _run(
        histories=_HIST, q_lens=[4] * n, live=n, sm_scale=0.1, k_scale=0.5, v_scale=2.0
    )
    _assert_same(monkeypatch, call)


def test_head_along_sublane(tpu, monkeypatch):
    hist = [5000, 4095, 4096, 300, 9000, 1, 12000, 8191]
    block = configs.BlockSizes(
        bq_sz=4, bq_c_sz=4, bkv_sz=1024, batch_size=4, n_buffer=2
    )
    call = _run(
        histories=hist,
        q_lens=[4] * 8,
        live=8,
        block=block,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        dtype=jnp.bfloat16,
        layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        sm_scale=128**-0.5,
        k_scale=None,
        v_scale=None,
    )
    _assert_same(monkeypatch, call)


def _cfgs(**serve):
    base = configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=32,
        num_page_indices=8 * 64,
        dtype_q=jnp.bfloat16,
        dtype_kv=_FP8,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=4,
    )
    return configs.RpaConfigs(
        block=_VERIFY,
        model=configs.ModelConfigs(
            num_q_heads=16,
            num_kv_heads=1,
            head_dim=256,
            mask_value=-1.0,
            sm_scale=1 / 16,
        ),
        serve=dataclasses.replace(base, **serve),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=1 << 26,
    )


def test_masks_kept_where_full_blocks_can_hold_queries():
    assert kernel._skips_masks_on_interior_blocks(_cfgs())
    assert not kernel._skips_masks_on_interior_blocks(_cfgs(skip_kv_update=True))
    assert not kernel._skips_masks_on_interior_blocks(
        _cfgs(attention_scope=configs.AttentionScope.CACHE_ONLY)
    )
    assert not kernel._skips_masks_on_interior_blocks(_cfgs(cp_group_size=2))
    mixed = dataclasses.replace(_cfgs(), mode=configs.RpaCase.MIXED)
    assert not kernel._skips_masks_on_interior_blocks(mixed)
    windowed = dataclasses.replace(
        _cfgs(), model=dataclasses.replace(_cfgs().model, sliding_window=1024)
    )
    assert not kernel._skips_masks_on_interior_blocks(windowed)

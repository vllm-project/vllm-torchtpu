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
"""Tests for DCP (decode context parallel) StreamIndex Top-K"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME, _select_owned_winners, cp_local_to_global, streamindex_topk,
    streamindex_topk_dcp)

P = jax.sharding.PartitionSpec


def _filter_all_ranks(merged, dcp_size, interleave_c):
    """What the `dcp_size` ranks each keep, stacked as [dcp_size, rows, k]."""
    return np.stack([
        np.asarray(
            _select_owned_winners(jnp.asarray(merged), dcp_size, interleave_c,
                                  jnp.int32(r))) for r in range(dcp_size)
    ])


@pytest.mark.parametrize("dcp_size,interleave_c", [(2, 1), (4, 4), (8, 4),
                                                   (8, 64)])
def test_owner_filters_partition_the_list(dcp_size, interleave_c):
    rng = np.random.default_rng(20260906)
    num_rows, k, span = 16, 128, 4096
    merged = np.stack([
        rng.choice(span, size=k, replace=False).astype(np.int32)
        for _ in range(num_rows)
    ])
    merged[0, -5:] = -1

    lists = _filter_all_ranks(merged, dcp_size, interleave_c)
    assert lists.shape == (dcp_size, num_rows, k)

    for t in range(num_rows):
        want = set(merged[t][merged[t] >= 0].tolist())
        got = []
        for r in range(dcp_size):
            row = lists[r, t]
            for local in row[row >= 0].tolist():
                g = int(cp_local_to_global(local, r, dcp_size, interleave_c))
                # A rank must only ever keep positions it actually owns.
                assert (g // interleave_c) % dcp_size == r
                got.append(g)
        # Multiplicity, not just set equality: the LSE merge double-counts a
        # position delivered to two ranks.
        assert len(got) == len(set(got))
        assert set(got) == want


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


def _global_records(kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX),
                             jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(vals.astype(jnp.float8_e4m3fn),
                                       jnp.uint8).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_dcp_reassembles_the_unsharded_global_topk(dcp_size):
    """Union the per-rank lists back together; it must be the global top-k"""
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")

    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(0))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) *
         0.5).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (Q_LEN, H_IDX),
                                jnp.float32).astype(jnp.bfloat16)
    seq_lens = jnp.array([KV_LEN], jnp.int32)
    cu_q_lens = jnp.array([0, Q_LEN], jnp.int32)
    distribution = jnp.array([0, 0, 1], jnp.int32)

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(q,
                         weights,
                         rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
                         seq_lens,
                         jnp.arange(pages, dtype=jnp.int32),
                         cu_q_lens,
                         distribution,
                         k=K,
                         compression_ratio=1,
                         num_kv_pages_per_block=2,
                         num_queries_per_block=128))

    # interleave_size=1: global position g lives on rank g % dcp_size at local
    # index g // dcp_size, so the shards are a strided de-interleave.
    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate([
        rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
        for r in range(dcp_size)
    ], 0)
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME, ))
    shards = jax.device_put(shards,
                            jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)))

    local_topk = jax.jit(
        functools.partial(
            streamindex_topk_dcp,
            mesh=mesh,
            k=K,
            compression_ratio=1,
            dcp_size=dcp_size,
            interleave_size=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=128))(q, weights, shards, seq_lens,
                                        jnp.arange(vpages, dtype=jnp.int32),
                                        cu_q_lens, distribution)

    # Global leading dim is dcp_size * Q_LEN, rank-major.
    local_topk = np.asarray(local_topk).reshape(dcp_size, Q_LEN, K)
    assert local_topk.max() < KV_LEN // dcp_size, "a rank got a foreign index"

    for t in range(Q_LEN):
        got = set()
        for r in range(dcp_size):
            row = local_topk[r, t]
            for local in row[row >= 0].tolist():
                got.add(int(cp_local_to_global(local, r, dcp_size, 1)))
        want = set(reference[t][reference[t] >= 0].tolist())
        assert got == want, (f"token {t}: {len(want - got)} missing, "
                             f"{len(got - want)} spurious")

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
"""Decode context parallelism inside the Qwen3.5 megakernel.

With `MKConfig.dcp = d` the d ranks of a KV-head group hold page-interleaved shards of the pool
(global kernel page p of a sequence on DCP rank p % d at page-table column p // d, vLLM's DCP
layout). These tests pin the rank-local length rule and run the 8-rank kernel in interpret mode
at dcp 1 and dcp 4 on the same logical KV: the hidden outputs agree, the new tokens land only on
their owner rank, every other page is untouched. Needs 8 CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=8`); skips otherwise.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from qwen35_megakernel_test_utils import (
    jnp_leaves,
    make_tiny_backbone,
    setup_8_cpu_devices,
)

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel.attention_block import PagedAttentionBlock
from vllm_torchtpu.kernels.qwen35_megakernel.context import MKBuildContext
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import pack_weights

R = 8
PS = 128


def _tiny(dcp: int, **over) -> K.MKConfig:
    c = dataclasses.replace(
        K.TINY_EP,
        R=R,
        B=8,
        T=4,
        L=1,
        att_every=1,
        paged_kv=True,
        num_pages=64,
        page_size=PS,
        max_pages_per_seq=8,
        ctx=1024,
        dcp=dcp,
    )
    return dataclasses.replace(c, **over)


def _tiles(c):
    return dataclasses.replace(
        K.default_tiles(c), hier=0, pack=0, mb8=0, kv_blk=256, nkv=2, kv_sub=0
    )


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_kv_len_counts_the_pages_this_rank_holds(dcp):
    c = _tiny(dcp)
    ctx = MKBuildContext(c, _tiles(c), interpret=True)
    assert ctx.DCP == dcp and ctx.TS == c.G // dcp
    blk = PagedAttentionBlock(ctx, None, kv_dma=object())
    for dr in range(dcp):
        ctx.dr = jnp.int32(dr)
        for plen in list(range(0, 1100, 37)) + [127, 128, 129, 511, 512, 513, 1024]:
            want = sum(1 for t in range(plen) if (t // PS) % dcp == dr)
            assert int(blk.kv_len(jnp.int32(plen))) == want, (dcp, dr, plen)


def _assert_in_band(got, ref):
    """The dcp 4 step merges the G partials over other KV subsets than the dcp 1 step, so the
    bf16 roundings of the attention output differ in order: the band item1 measured for
    all-reduce order changes (a few 1e-4 of the elements outside atol = rtol = 2e-2)."""
    d = np.abs(got - ref)
    over = d > 2e-2 + 2e-2 * np.abs(ref)
    assert float(over.mean()) <= 1e-3, (float(over.mean()), float(d.max()))
    assert float(d.max()) <= 0.25, float(d.max())


def _starts():
    # sequence 0: four tokens on page 2 (owner 2 at dcp 4); sequence 1: two on page 4 (owner 0)
    # and two on page 5 (owner 1), the non-same-page write path
    return [300, 638]


def _replicate(x, lead=0):
    return jnp.concatenate([x] * R, axis=lead)


def _group_pools(c, seed, pt):
    """DCP1 pools (R, num_pages, 2, 64, 4, 128) fp8: identical over a KV-head group."""
    rng = np.random.default_rng(seed)
    u8 = rng.integers(0, 0x40, (2, c.num_pages, 2, 64, 4, PS), dtype=np.uint8)
    u8 |= (rng.integers(0, 2, u8.shape, dtype=np.uint8) << 7).astype(np.uint8)
    return np.stack([u8[r // c.G] for r in range(R)])


def _shard_pools(c, pools1_u8, pt, starts):
    """DCP4 pools: rank r's column j of sequence s holds global page dcp * j + (r % dcp)."""
    d = c.dcp
    out = pools1_u8.copy()
    for r in range(R):
        dr = r % d
        for s in range(c.NS):
            for j in range(c.max_pages_per_seq):
                gp = d * j + dr
                if gp < c.max_pages_per_seq:
                    out[r, int(pt[s, j])] = pools1_u8[r, int(pt[s, gp])]
    return out


def _expected_dcp_pools(c, pools4_in, pools1_out, pt, starts, gdn_pages=()):
    """DCP4 pools after the step: the input with each new token's lanes copied from the DCP1
    result into the owner rank's local page and the GatedDeltaNet checkpoint pages (per rank,
    written before the attention layer) taken from the DCP1 result, nothing else changed."""
    d = c.dcp
    out = pools4_in.copy()
    for pg in gdn_pages:
        out[:, pg] = pools1_out[:, pg]
    for s, s0 in enumerate(starts):
        for t in range(c.T):
            p = s0 + t
            gp, ln = p // PS, p % PS
            for r in range(R):
                if r % d != gp % d:
                    continue
                src = pools1_out[r, int(pt[s, gp])]
                dst = int(pt[s, gp // d])
                out[r, dst, :, :, :, ln] = src[:, :, :, ln]
    return out


def _run_attn_only(mesh, c, tiles, w, pools_u8, pt, pos):
    fn, _, _, _ = pool_io.make_flat_attn_step(
        mesh, c, tiles, interpret=True, skip=frozenset({"experts"})
    )
    pools = jax.lax.bitcast_convert_type(
        jnp.asarray(pools_u8.reshape(-1, 2, 64, 4, PS)), K.FP8
    )
    hid = jax.random.normal(jax.random.PRNGKey(1), (c.M, c.H), jnp.float32)
    hid = jnp.where(jnp.arange(c.M)[:, None] < c.B, hid, 0.0).astype(jnp.bfloat16)
    act = jnp.ones((c.NS,), jnp.int32)
    rows = jnp.ones((c.B,), jnp.int32)
    out, pool = fn(
        _replicate(pos),
        _replicate(hid),
        *w,
        pools,
        _replicate(jnp.asarray(pt)),
        _replicate(act),
        _replicate(rows),
    )
    out = np.asarray(out.astype(jnp.float32)).reshape(R, c.M, c.H)
    pool = np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8)).reshape(
        pools_u8.shape
    )
    return out, pool


def test_dcp4_attention_step_matches_dcp1_and_writes_only_owner_pages():
    devs = setup_8_cpu_devices()
    if len(devs) < 8:
        pytest.skip("needs 8 CPU devices")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    c1, c4 = _tiny(1), _tiny(4)
    t1, t4 = _tiles(c1), _tiles(c4)
    torch.manual_seed(0)
    w = [
        _replicate(x)
        for x in jnp_leaves(
            pack_weights(make_tiny_backbone(c1, e_slots=c1.EW), c1, 0, t1)
        )
    ]
    starts = _starts()
    pos = jnp.asarray([s0 + t for s0 in starts for t in range(c1.T)], jnp.int32)
    pt = np.zeros((c1.NS, c1.max_pages_per_seq), np.int32)
    for s in range(c1.NS):
        pt[s] = 8 + 8 * s + np.arange(c1.max_pages_per_seq)
    pools1 = _group_pools(c1, 3, pt)
    pools4 = _shard_pools(c4, pools1, pt, starts)

    out1, pool1 = _run_attn_only(mesh, c1, t1, w, pools1, pt, pos)
    out4, pool4 = _run_attn_only(mesh, c4, t4, w, pools4, pt, pos)

    assert np.isfinite(out1).all() and np.isfinite(out4).all()
    for r in range(R):
        _assert_in_band(out4[r, : c1.B], out1[r, : c1.B])
    # dcp 1 writes every rank of the group alike
    for r in range(R):
        assert np.array_equal(pool1[r], pool1[(r // c1.G) * c1.G])
    assert np.array_equal(pool4, _expected_dcp_pools(c4, pools4, pool1, pt, starts))
    # and the owner pages did change
    assert not np.array_equal(pool4, pools4)


def test_dcp4_fused_pool_step_matches_dcp1():
    """The serving path (fused pool I/O: kv_rmw writes the new tokens inside the kernel) with
    GatedDeltaNet layers in front of the attention layer."""
    devs = setup_8_cpu_devices()
    if len(devs) < 8:
        pytest.skip("needs 8 CPU devices")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    base = dict(L=4, att_every=4, num_pages=160)
    c1, c4 = _tiny(1, **base), _tiny(4, **base)
    t1, t4 = _tiles(c1), _tiles(c4)
    pt, ckpt, roff, min_pages = pool_io.default_pool_layout(c1)
    assert min_pages <= c1.num_pages
    pt = np.asarray(pt)
    torch.manual_seed(0)
    w = [
        _replicate(x)
        for x in jnp_leaves(
            pack_weights(make_tiny_backbone(c1, e_slots=c1.EW), c1, 0, t1)
        )
    ]
    starts = _starts()
    pos = jnp.asarray([s0 + t for s0 in starts for t in range(c1.T)], jnp.int32)
    pools1 = _group_pools(c1, 5, pt)
    # GatedDeltaNet checkpoint pages are per rank: zero them so the conv taps are finite
    for sub in range(ckpt.shape[0]):
        for s in range(c1.NS):
            p0 = int(ckpt[sub, s, 0]) * c1.gdn_stride
            pools1[:, p0 : p0 + c1.gdn_stride * c1.T] = 0
    pools4 = _shard_pools(c4, pools1, pt, starts)
    hid = jax.random.normal(jax.random.PRNGKey(2), (c1.M, c1.H), jnp.float32)
    hid = jnp.where(jnp.arange(c1.M)[:, None] < c1.B, hid, 0.0).astype(jnp.bfloat16)

    def run(c, tiles, pools_u8):
        fn, _, _, _ = pool_io.make_flat_pooled_step(
            mesh,
            c,
            tiles,
            skip=frozenset({"experts"}),
            fused_pool_io=True,
            interpret=True,
        )
        raw = [
            jax.lax.bitcast_convert_type(
                jnp.asarray(pools_u8.reshape(-1, 2, 64, 4, PS)), K.FP8
            )
            for _ in range(c.NATT)
        ]
        out = fn(
            _replicate(pos),
            _replicate(hid),
            *w,
            *raw,
            _replicate(jnp.asarray(pt)),
            _replicate(ckpt),
            _replicate(roff),
        )
        hidden = np.asarray(out[0].astype(jnp.float32)).reshape(R, -1, c.H)
        pools = [
            np.asarray(jax.lax.bitcast_convert_type(p, jnp.uint8)).reshape(
                pools_u8.shape
            )
            for p in out[1:]
        ]
        return hidden, pools

    out1, pools1_out = run(c1, t1, pools1)
    out4, pools4_out = run(c4, t4, pools4)
    assert np.isfinite(out1).all() and np.isfinite(out4).all()
    for r in range(R):
        _assert_in_band(out4[r, : c1.B], out1[r, : c1.B])
    gdn_pages = [
        int(ckpt[sub, s, t]) * c1.gdn_stride + k
        for sub in range(ckpt.shape[0])
        for s in range(c1.NS)
        for t in range(c1.T)
        for k in range(c1.gdn_pages)
    ]
    for p1, p4 in zip(pools1_out, pools4_out):
        assert np.array_equal(
            p4, _expected_dcp_pools(c4, pools4, p1, pt, starts, gdn_pages)
        )

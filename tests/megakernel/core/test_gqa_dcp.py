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
"""Decode context parallelism (DCP) in the megakernel library.

With `CacheExec.dcp = d` the `d` ranks of a DCP group share a KV head and
hold page-interleaved shards of the pool: global kernel page `p` of a
sequence on DCP rank `p % d` at page-table column `p // d` (vLLM's DCP
layout). These tests pin the ownership rule and the rank-local length, and
run an 8-rank Qwen3.5-shaped step (two KV heads, so G = 4 ranks per KV head,
as at TP8) in Pallas interpret mode at dcp 1 and at dcp `d` on the same
logical KV, through the split pool writer and the fused pool I/O: the hidden
outputs agree within the band the merge order allows, each new token lands
only on its owner rank's local page, and every other page is untouched.
Needs 8 CPU devices (the megakernel conftest sets them up); skips otherwise.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.megakernel.core.pool import dcp as dcp_rule
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import TINY_TILES

R = 8
PS = 128
NUM_PAGES = 256
CTX = 1024
# Sequence 0: four tokens on page 2 (owner 2 at dcp 4, owner 0 at dcp 2).
# Sequence 1: 638, 639 on page 4 and 640, 641 on page 5: two owners, and at
# dcp > 1 one page-table column, so one physical page id on both owners.
STARTS = (300, 638)


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


# --- the ownership rule ------------------------------------------------------


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_local_kv_len_counts_the_pages_a_rank_holds(dcp: int) -> None:
    plens = list(range(0, 1100, 37)) + [1, 127, 128, 129, 255, 256, 511, 512, 513]
    for rank in range(dcp):
        for plen in plens:
            want = sum(1 for t in range(plen) if (t // PS) % dcp == rank)
            got = dcp_rule.local_kv_len(jnp.int32(plen), PS, dcp, jnp.int32(rank))
            assert int(got) == want, (dcp, rank, plen)
            assert dcp_rule.host_local_kv_len(plen, PS, dcp, rank) == want
        for plen in (-1, -129):  # a padded row
            got = int(dcp_rule.local_kv_len(jnp.int32(plen), PS, dcp, jnp.int32(rank)))
            assert got == (plen if dcp == 1 else 0)
            assert dcp_rule.host_local_kv_len(plen, PS, dcp, rank) == got


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_dcp_shards_partition_the_dcp1_pages(dcp: int) -> None:
    """DCP 1 against DCP d: every page has exactly one (owner, column), the
    owners' columns tile the DCP 1 page list, and the rank-local lengths sum
    to the sequence length."""
    owners = [dcp_rule.host_page_owner_col(lp, dcp) for lp in range(64)]
    assert len(set(owners)) == 64
    for lp, (owner, col) in enumerate(owners):
        assert col * dcp + owner == lp
        for rank in range(dcp):
            own, c = dcp_rule.page_owner_col(jnp.int32(lp), dcp, jnp.int32(rank))
            assert bool(own) == (rank == owner) and int(c) == col
    for plen in range(0, 3000, 61):
        assert (
            sum(dcp_rule.host_local_kv_len(plen, PS, dcp, r) for r in range(dcp))
            == plen
        )
    if dcp > 1:
        assert dcp_rule.host_page_owner_col(5, dcp) == (5 % dcp, 5 // dcp)


@pytest.mark.parametrize(
    "me, dcp, want", [(0, 1, 0), (7, 1, 0), (5, 2, 1), (6, 4, 2), (3, 4, 3)]
)
def test_dcp_rank_is_tp_rank_mod_dcp(me: int, dcp: int, want: int) -> None:
    assert int(dcp_rule.dcp_rank(jnp.int32(me), dcp)) == want


def test_cache_exec_dcp_validation() -> None:
    assert mk.CacheExec().dcp == 1
    assert mk.CacheExec(paged_kv=True, dcp=4).dcp == 4
    with pytest.raises(ValueError, match="dcp must be an int >= 1"):
        mk.CacheExec(paged_kv=True, dcp=0)
    with pytest.raises(ValueError, match="needs paged_kv"):
        mk.CacheExec(dcp=2)
    # One equal to its defaults is still stored as `None` on the ExecSpec.
    assert mk.ExecSpec(cache=mk.CacheExec(dcp=1)).cache is None


def test_warmup_sentinel_checks_the_kv_page_on_its_owner_only() -> None:
    """Only the DCP rank holding the sentinel token's page writes its KV: on the
    others that page must stay unchanged, the recurrent slot must change."""
    from vllm_torchtpu.megakernel.core.runtime import verify_sentinel_step

    kv0, kv1 = [np.zeros(8, np.uint8)], [np.ones(8, np.uint8)]
    g0, g1 = [np.zeros(4, np.uint8)], [np.ones(4, np.uint8)]
    out = np.ones((2, 4), np.float32)
    kw: dict[str, Any] = dict(
        tokens_per_seq=1, bucket_t=2, variant_key=(1, False, False)
    )
    verify_sentinel_step(out, kv0, kv1, g0, g1, **kw)
    verify_sentinel_step(out, kv0, kv1, g0, g1, kv_owner=True, **kw)
    verify_sentinel_step(out, kv0, kv0, g0, g1, kv_owner=False, **kw)
    with pytest.raises(RuntimeError, match="kv_ok=False, kv_owner=False"):
        verify_sentinel_step(out, kv0, kv1, g0, g1, kv_owner=False, **kw)
    with pytest.raises(RuntimeError, match="kv_ok=False, kv_owner=True"):
        verify_sentinel_step(out, kv0, kv0, g0, g1, **kw)
    with pytest.raises(RuntimeError, match="gdn_ok=False"):
        verify_sentinel_step(out, kv0, kv0, g0, g0, kv_owner=False, **kw)


# --- GQA parameters ------------------------------------------------------------


def _geom() -> Any:
    return qwen35.Qwen35Geometry(
        L=4,
        H=256,
        I=256,
        IS=1024,  # 128 shared-expert columns per rank at R = 8
        E=128,
        TOPK=2,
        NH=8,
        NKV=2,
        HD=256,
        ROT=64,
        KH=8,
        VH=16,  # two value heads per key head per rank (the fused GDN needs it)
        KD=128,
        VD=128,
        att_every=4,
        moe_tp=False,
        e_slots=0,
    )


def _exec(*, T: int, B: int, dcp: int = 1, split: bool = True) -> mk.ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10).with_modules(
        {"gqa": {"kv_blk": 256, "nkv": 2, "kv_sub": 0, "split": split}}
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=CTX,
        stack_e=False,
        interpret=True,
        tiles=tiles,
        debug=mk.DebugOptions(skip=frozenset({"experts"})),
        cache=mk.CacheExec(
            paged_kv=True, num_pages=NUM_PAGES, dcp=dcp, pool_gather=False
        ),
    )


def _gqa_params(ex: mk.ExecSpec) -> Any:
    spec = qwen35.qwen35_spec(_geom())
    gqa = next(m for m in spec.all_modules() if isinstance(m, mk.GQAModule))
    return mk.planner.plan(spec, ex).module_params[gqa.name]


@pytest.mark.parametrize("dcp, ts", [(1, 4), (2, 2), (4, 1)])
def test_gqa_split_runs_over_one_dcp_shard(dcp: int, ts: int) -> None:
    ap = _gqa_params(_exec(T=4, B=8, dcp=dcp))
    got = {"G": ap.G, "SPLIT": ap.SPLIT, "DCP": ap.DCP, "TS": ap.TS}
    assert got == {"G": 4, "SPLIT": True, "DCP": dcp, "TS": ts}


def test_gqa_declines_dcp_it_cannot_shard() -> None:
    with pytest.raises(mk.MegakernelConfigError, match="divide"):
        _gqa_params(_exec(T=1, B=2, dcp=8))  # G = 4 ranks per KV head
    with pytest.raises(mk.MegakernelConfigError, match="sequence-split"):
        _gqa_params(_exec(T=1, B=2, dcp=2, split=False))


# --- the 8-rank step at dcp 1 and dcp d ------------------------------------------


def _mesh() -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"needs {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _to_u8(pools: Any) -> np.ndarray:
    """`(NATT, R, num_pages, *page)` uint8 view of rank-folded pools."""
    return np.stack(
        [
            np.asarray(jax.lax.bitcast_convert_type(p, jnp.uint8)).reshape(
                R, NUM_PAGES, *p.shape[1:]
            )
            for p in pools
        ]
    )


@functools.cache
def _inputs(T: int) -> tuple[Any, ...]:
    """Weights, hidden rows, layout and DCP 1 pools whose KV pages hold the
    same random fp8 bytes on every rank of a KV-head group."""
    mesh = _mesh()
    g = _geom()
    ex = _exec(T=T, B=len(STARTS) * T)
    _, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(611 + T), g, ex)
    pos = jnp.asarray([s0 + t for s0 in STARTS for t in range(T)], jnp.int32)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, min_pages = mk.default_pool_layout(g, ex)
    assert min_pages <= NUM_PAGES
    pools1 = _to_u8(mk.states_to_pool(g, ex, st_l, pt, ckpt, jnp.maximum(roff, 0)))
    pt_np = np.asarray(pt)
    kv_pages = sorted({int(x) for x in pt_np.ravel() if int(x) > 0})
    G = R // g.NKV
    rng = np.random.default_rng(5)
    page_shape = pools1.shape[3:]
    for a in range(pools1.shape[0]):
        for kp in kv_pages:
            for grp in range(R // G):
                u8 = rng.integers(0, 0x40, page_shape, dtype=np.uint8)
                u8 |= (rng.integers(0, 2, page_shape, dtype=np.uint8) << 7).astype(
                    np.uint8
                )
                pools1[a, grp * G : (grp + 1) * G, kp] = u8
    return mesh, g, pos, hid_mk, w_mk, pt_np, ckpt, roff, pools1, kv_pages


def _shard(pools1: np.ndarray, pt: np.ndarray, dcp: int) -> np.ndarray:
    """DCP d pools: rank r's column j of sequence s holds global page
    `d * j + r % d` of the DCP 1 pools."""
    out = pools1.copy()
    ns, width = pt.shape
    for r in range(R):
        for s in range(ns):
            for j in range(width):
                gp = dcp * j + r % dcp
                if gp < width:
                    out[:, r, pt[s, j]] = pools1[:, r, pt[s, gp]]
    return out


@functools.cache
def _run(T: int, fused: bool, dcp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh, g, pos, hid_mk, w_mk, pt, ckpt, roff, pools1, _ = _inputs(T)
    ex = _exec(T=T, B=len(STARTS) * T, dcp=dcp)
    pools_in = pools1 if dcp == 1 else _shard(pools1, pt, dcp)
    # The fused pool I/O writes the pools through outer buffer donation.
    step, _ = mk.PallasWriter(fused_pool_io=fused).make_step(
        mesh, qwen35.qwen35_spec(g), ex, donate=fused
    )
    tp = NamedSharding(mesh, P("tp"))
    pools = tuple(
        jax.device_put(
            jax.lax.bitcast_convert_type(
                jnp.asarray(p.reshape(R * NUM_PAGES, *p.shape[2:])), jnp.float8_e4m3fn
            ),
            tp,
        )
        for p in pools_in
    )
    out, raw = step(pos, hid_mk, w_mk, pools, jnp.asarray(pt), ckpt, roff)
    hid = np.asarray(mk.unfold(out, R).astype(jnp.float32))
    return hid, pools_in, _to_u8(raw)


def _assert_in_band(got: np.ndarray, ref: np.ndarray) -> None:
    """The dcp `d` step merges the G partials over other KV subsets than the
    dcp 1 step, so the attention output rounds in another order: allow a few
    1e-4 of the elements outside atol = rtol = 2e-2 (the all-reduce-order
    band), and nothing far off."""
    d = np.abs(got - ref)
    over = d > 2e-2 + 2e-2 * np.abs(ref)
    assert float(over.mean()) <= 1e-3, (float(over.mean()), float(d.max()))
    assert float(d.max()) <= 0.25, float(d.max())


def _expected_pools(
    pools_in: np.ndarray,
    out1: np.ndarray,
    pt: np.ndarray,
    kv_pages: list[int],
    T: int,
    dcp: int,
) -> np.ndarray:
    """DCP d pools after the step: the DCP 1 result on every non-KV page (the
    recurrent checkpoints are written alike, before the attention layer), the
    DCP d input on every KV page, plus each new token's lane copied from the
    DCP 1 result into its owner rank's local page."""
    exp = out1.copy()
    exp[:, :, kv_pages] = pools_in[:, :, kv_pages]
    for s, s0 in enumerate(STARTS):
        for t in range(T):
            gp, ln = divmod(s0 + t, PS)
            owner, col = dcp_rule.host_page_owner_col(gp, dcp)
            for r in range(R):
                if r % dcp == owner:
                    exp[:, r, pt[s, col], ..., ln] = out1[:, r, pt[s, gp], ..., ln]
    return exp


@pytest.mark.parametrize(
    "T, fused, dcp",
    [
        (4, False, 4),  # split pool writer (scatter kernel), MTP verify rows
        (4, True, 4),  # fused pool I/O (in-kernel KV RMW), the serving path
        (4, True, 2),  # two ranks per shard: the split rotates inside a shard
        (1, False, 4),  # decode rows through the scatter kernel
    ],
)
def test_dcp_step_matches_dcp1_and_writes_only_owner_pages(
    T: int, fused: bool, dcp: int
) -> None:
    _, g, _, _, _, pt, _, _, _, kv_pages = _inputs(T)
    G = R // g.NKV
    hid1, _, out1 = _run(T, fused, 1)
    hid_d, pools_in, out_d = _run(T, fused, dcp)
    B = len(STARTS) * T
    assert np.isfinite(hid1[:, :B]).all() and np.isfinite(hid_d[:, :B]).all()
    for r in range(R):
        _assert_in_band(hid_d[r, :B], hid1[r, :B])
    # dcp 1 writes every rank of a KV-head group alike
    for r in range(R):
        np.testing.assert_array_equal(
            out1[:, r, kv_pages], out1[:, (r // G) * G, kv_pages]
        )
    np.testing.assert_array_equal(
        out_d, _expected_pools(pools_in, out1, pt, kv_pages, T, dcp)
    )
    # and the owner pages did change
    assert not np.array_equal(out_d[:, :, kv_pages], pools_in[:, :, kv_pages])

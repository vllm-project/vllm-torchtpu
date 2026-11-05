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
"""Standalone unit tests for slice L-8b (`modules/moe` routing and ring helpers)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel.dma import stream as dma_stream
from vllm_torchtpu.megakernel.dma import vmem
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules.moe import dedup, ring, topk
from vllm_torchtpu.megakernel.modules.moe.context import MoEEmit
from vllm_torchtpu.megakernel.modules.moe.params import (
    MoEDebug,
    MoEParams,
    MoETunables,
    derive_moe_params,
)
from vllm_torchtpu.megakernel.modules.moe.routing import route
from vllm_torchtpu.megakernel.ops import F32, I32, round_up
from vllm_torchtpu.megakernel.planner import plan, weight_shapes
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec


def _simulate_expert_ring_no_collision(mp: MoEParams, n_hit: int = 8) -> None:
    """Simulate `ring.py` / `experts.py` `e_ring` (`NBE`) and `es` (`NSS`) lifetimes."""
    nbe, nss, te, t13, t2 = mp.NBE, mp.NSS, mp.TE, mp.T13, mp.T2
    assert nbe >= 1 and nss >= 1 and te == t13 + t2
    e_inflight: dict[int, int] = {}
    es_inflight: dict[int, int] = {}

    def fetch_unit(u: int, t: int | None = None) -> None:
        if u >= n_hit * te:
            return
        if t is not None:
            nt = (t + nbe) % te
            assert (u - nt) % te == 0, f"misaligned unit {u} for tile {nt}"
            if mp.PIPE_EP and nbe >= te:
                assert nt == t, f"PIPE_EP requires (t + NBE) % TE == t, got {t}/{nbe}"
        slot = u % nbe
        assert slot not in e_inflight, f"e_ring slot {slot} collision"
        e_inflight[slot] = u
        if u % te == 0:
            exp_i = u // te
            es_slot = exp_i % nss
            assert es_slot not in es_inflight, f"es slot {es_slot} collision"
            es_inflight[es_slot] = exp_i

    def fetch_tile(
        u: int, idx: int, count: int, t_base: int, grouped: bool, tiled: bool
    ) -> None:
        if grouped:
            if idx % 2 == 1:
                fetch_unit(u - 1 + nbe, t=t_base + idx - 1)
                fetch_unit(u + nbe, t=t_base + idx)
            elif idx == count - 1:
                fetch_unit(u + nbe, t=t_base + idx)
        else:
            fetch_unit(u + nbe, t=(t_base + idx) if tiled else None)

    for u0 in range(min(nbe, n_hit * te)):
        fetch_unit(u0)

    if mp.PIPE:
        assert t13 == 1 and t2 == 1
        assert e_inflight.pop(0 % nbe, None) == 0 and es_inflight.get(0 % nss) == 0
        fetch_unit(0 + nbe, t=0)
        for i in range(n_hit - 1):
            u13, u2 = (i + 1) * te, i * te + t13
            assert e_inflight.pop(u13 % nbe, None) == u13
            assert es_inflight.get((i + 1) % nss) == i + 1
            assert e_inflight.pop(u2 % nbe, None) == u2
            assert es_inflight.pop(i % nss, None) == i
            fetch_unit(u13 + nbe, t=0)
            fetch_unit(u2 + nbe, t=t13)
        last = n_hit - 1
        u2_last = last * te + t13
        assert e_inflight.pop(u2_last % nbe, None) == u2_last
        assert es_inflight.pop(last % nss, None) == last
        fetch_unit(u2_last + nbe, t=t13)
    elif mp.PIPE_EP:
        assert t13 == 2 * t2
        for t in range(t13):
            assert e_inflight.pop(t % nbe, None) == t
            fetch_tile(t, t, t13, 0, mp.group_rel, True)
        assert es_inflight.get(0 % nss) == 0
        for i in range(n_hit - 1):
            for s_step in range(t2):
                for u_req in (
                    i * te + t13 + s_step,
                    (i + 1) * te + 2 * s_step,
                    (i + 1) * te + 2 * s_step + 1,
                ):
                    assert e_inflight.get(u_req % nbe) == u_req, (
                        f"unit {u_req} not prefetched before body_ep({i})"
                    )
            for s_step in range(t2):
                u2 = i * te + t13 + s_step
                t0, t1 = 2 * s_step, 2 * s_step + 1
                u13_0, u13_1 = (i + 1) * te + t0, (i + 1) * te + t1
                assert e_inflight.pop(u2 % nbe, None) == u2
                assert e_inflight.pop(u13_0 % nbe, None) == u13_0
                fetch_unit(u2 + nbe, t=t13 + s_step)
                fetch_unit(u13_0 + nbe, t=t0)
                assert e_inflight.pop(u13_1 % nbe, None) == u13_1
                fetch_unit(u13_1 + nbe, t=t1)
            assert es_inflight.pop(i % nss, None) == i
            assert es_inflight.get((i + 1) % nss) == i + 1
        i_last = n_hit - 1
        for s_step in range(t2):
            u2 = i_last * te + t13 + s_step
            assert e_inflight.pop(u2 % nbe, None) == u2
            fetch_tile(u2, s_step, t2, t13, mp.group_rel, True)
        assert es_inflight.pop(i_last % nss, None) == i_last
    else:
        for i in range(n_hit):
            base = i * te
            for t in range(t13):
                u = base + t
                assert e_inflight.pop(u % nbe, None) == u
                fetch_tile(
                    u,
                    t,
                    t13,
                    0,
                    mp.unroll and mp.group_rel and t13 > 1,
                    mp.unroll and t13 > 1,
                )
            assert es_inflight.get(i % nss) == i
            u2_base = base + t13
            if t2 == 1:
                assert e_inflight.pop(u2_base % nbe, None) == u2_base
                assert es_inflight.pop(i % nss, None) == i
                fetch_unit(u2_base + nbe)
            else:
                for t in range(t2):
                    u = u2_base + t
                    assert e_inflight.pop(u % nbe, None) == u
                    fetch_tile(u, t, t2, t13, mp.unroll and mp.group_rel, mp.unroll)
                assert es_inflight.pop(i % nss, None) == i
    assert not e_inflight and not es_inflight


def test_batcher_net_and_topk_rows_on_lanes():
    """Verify `_batcher_net` sorting network and `topk_rows_on_lanes` exact top-k."""
    net = topk._batcher_net(16)
    rng = np.random.default_rng(12)
    arr = rng.normal(size=(16, 32)).astype(np.float32)
    for i, j in net:
        hi, lo = np.maximum(arr[i], arr[j]), np.minimum(arr[i], arr[j])
        arr[i], arr[j] = hi, lo
    np.testing.assert_allclose(arr, np.sort(arr, axis=0)[::-1], rtol=0, atol=0)

    mr, e, k_top, em = 8, 128, 4, 127
    probs = jax.random.uniform(
        jax.random.PRNGKey(7), (mr, e), dtype=F32, minval=0.01, maxval=1.0
    )
    lane_e8 = lax.broadcasted_iota(I32, (mr, e), 1)
    kb = lax.bitcast_convert_type(probs, I32)
    key = lax.bitcast_convert_type((kb & ~em) | (em - lane_e8), F32)

    ids, sel = topk.topk_rows_on_lanes(key, k_top, em, need_ids=True)
    ids_none, sel2 = topk.topk_rows_on_lanes(key, k_top, em, need_ids=False)
    assert ids_none is None
    np.testing.assert_array_equal(sel, sel2)

    _, ref_idx = lax.top_k(key, k_top)
    np.testing.assert_array_equal(ids[:, :k_top], ref_idx)
    assert jnp.all(jnp.sum(sel.astype(I32), axis=-1) == k_top)


def test_moe_ring_schedule_invariants_on_fake_geometry():
    """Verify expert ring schedule invariants on `MoEParams` with `FakeGeometry`."""
    g = FakeGeometry(
        L=1, att_every=1, H=256, NH=16, NKV=8, I=256, IS=1024, E=128, TOPK=4
    )
    ms = ModelSpec(
        "moe_ring_sim",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    tiles_1t = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=256 << 10, nbe=6)
    ex_1t = ExecSpec(R=2, B=4, T=1, stack_e=True, tiles=tiles_1t)
    p_1t = plan(ms, ex_1t)
    for tt, is_pipe in (
        (MoETunables(pipe=True), True),
        (MoETunables(pipe=False, moe_unroll=True), False),
    ):
        mp = derive_moe_params(
            g,
            ex_1t,
            p_1t,
            tt,
            MoEDebug(),
            moe_tp=False,
            e_slots=0,
            module_name="fake_moe",
        )
        assert mp.PIPE is is_pipe
        _simulate_expert_ring_no_collision(mp, n_hit=8)

    tiles_ep = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=64 << 10, nbe=6)
    ex_ep = ExecSpec(R=2, B=4, T=1, stack_e=True, tiles=tiles_ep)
    p_ep = plan(ms, ex_ep)
    mp_pipe_ep = derive_moe_params(
        g,
        ex_ep,
        p_ep,
        MoETunables(pipe=False, pipe_ep=True),
        MoEDebug(),
        moe_tp=False,
        e_slots=0,
        module_name="fake_moe",
    )
    assert mp_pipe_ep.PIPE_EP is True and mp_pipe_ep.T13 == 2 and mp_pipe_ep.T2 == 1
    _simulate_expert_ring_no_collision(mp_pipe_ep, n_hit=8)

    with pytest.raises(AssertionError, match="es slot .* collision"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(mp_pipe_ep, NSS=mp_pipe_ep.NSS - 1), n_hit=8
        )
    with pytest.raises(AssertionError, match="not prefetched before body_ep"):
        _simulate_expert_ring_no_collision(
            dataclasses.replace(mp_pipe_ep, NBE=mp_pipe_ep.TE - 1), n_hit=8
        )


def _sync_copy(src, dst, sem) -> None:
    pltpu.make_async_copy(src, dst, sem).start()
    pltpu.make_async_copy(src, dst, sem).wait()


@pytest.mark.parametrize(
    ("ecopy", "route_vec", "ecopy_opt"),
    [
        (False, False, 0),
        (True, False, 0),
        (True, False, 1),
        (True, True, 0),
    ],
    ids=["local_dedup", "ecopy_opt0", "ecopy_opt1", "ecopy_route_vec"],
)
def test_moe_routing_and_ring_helpers_in_pallas_interpret(
    ecopy: bool, route_vec: bool, ecopy_opt: int
):
    """Execute `MoEEmit.build`, `route_topk`, `dedup_*`, `route`, and `ring.py`."""
    R, B, H, E, TOPK, IR, ISR = 2, 4, 256, 128, 2, 128, 128
    e_slots = 70 if ecopy else 0
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=256 << 10, nbe=4)
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, stack_e=True, tiles=tiles, interpret=True)
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=H,
        NH=4,
        NKV=2,
        E=E,
        TOPK=TOPK,
        I=IR,
        IS=ISR * R,
        e_slots=e_slots,
    )
    ms = ModelSpec(
        "moe_r_ring",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    p = cp.plan
    tt = MoETunables(
        topk=True, topk_lanes=True, route_vec=route_vec, ecopy_opt=ecopy_opt
    )
    mp = derive_moe_params(
        g, ex, p, tt, MoEDebug(), moe_tp=False, e_slots=e_slots, module_name="fake_moe"
    )
    assert mp.ECOPY is ecopy and mp.route_vec is route_vec
    nb_picks = round_up(B * TOPK, 128)
    scratch["hitv"] = pltpu.VMEM((ex.MR, 128), I32)
    scratch["hits"] = pltpu.SMEM((ex.MR, 128), I32)
    scratch["elist"] = pltpu.SMEM((nb_picks,), I32)
    scratch["seen"] = pltpu.SMEM((8, 128), I32)
    scratch["misc_sem"] = pltpu.SemaphoreType.DMA(())
    scratch["st"] = pltpu.SMEM((1,), I32)
    scratch["es"] = pltpu.VMEM((mp.NSS, 1, 2 * IR + H), F32)
    scratch["es_sem"] = pltpu.SemaphoreType.DMA((mp.NSS,))
    if ecopy:
        scratch["ecopy"] = pltpu.SMEM((2, 1, E), F32)
        scratch["ec_sem"] = pltpu.SemaphoreType.DMA((2,))
        for k_ec in ("wlist", "cplist", "cpchoice"):
            scratch[k_ec] = pltpu.SMEM((nb_picks,), I32)
        scratch["rload"] = pltpu.SMEM((R,), I32)
        if ecopy_opt == 1:
            scratch["ulist"] = pltpu.SMEM((nb_picks,), I32)

    wsh = weight_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    def pallas_body(pexp_ref, *rest):
        n_w = len(p.w_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        wn_out_ref, nhit_out_ref, e13_out_ref = rest[n_w : n_w + 3]
        scr = dict(zip(s_names, rest[n_w + 3 :]))
        me = jnp.int32(0)
        two_ring = dma_stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ctx = MiniCtx(g, ex, cp, None, W, {}, {}, scr, me, stream=two_ring, comm=None)
        scr["hitv"][...] = jnp.zeros((ex.MR, 128), I32)
        _sync_copy(scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"])
        _sync_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0])
        if ecopy:
            o_ec = cp.stream.lay["ecopy"][0]
            _sync_copy(
                W["smalls"].at[0, :, pl.ds(o_ec, E)],
                scr["ecopy"].at[0],
                scr["ec_sem"].at[0],
            )

        m = MoEEmit.build(ctx, mp, MoEDebug())
        with two_ring.expert_stream():
            n_hit = route(m, jnp.int32(0), pexp_ref[...])
            ring.es_wait(m, jnp.int32(0))
            t0 = ring.etake(m, "e13", 0)
            e13_out_ref[...] = t0[...]
            ring.egive(m, 0, jnp.int32(0), t=0)
            t1 = ring.etake(m, "e2", 1)
            _ = t1[...]
            ring.egive_group(m, [(1, 1)], jnp.int32(0))
        wn_out_ref[...] = scr["wn"][...]
        nhit_out_ref[0] = n_hit

    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MR, E), F32),
            jax.ShapeDtypeStruct((1,), I32),
            jax.ShapeDtypeStruct((H, 2 * IR), wsh["e13"][1]),
        ],
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        scratch_shapes=[scratch[k] for k in s_names],
        interpret=pltpu.InterpretParams(),
    )

    pexp = jnp.ones((ex.MR, E), dtype=F32) * 0.01
    pexp = pexp.at[0, 2].set(5.0).at[0, 5].set(4.0)
    pexp = pexp.at[1, 3].set(5.0).at[1, 66].set(4.0)
    pexp = pexp.at[2, 2].set(5.0).at[2, 3].set(4.0)
    pexp = pexp.at[3, 5].set(5.0).at[3, 66].set(4.0)

    w_in = {k: jnp.zeros(wsh[k][0], dtype=wsh[k][1]) for k in p.w_names}
    if ecopy:
        o_ec, w_ec = cp.stream.lay["ecopy"]
        ec_tbl = jnp.zeros((w_ec,), dtype=F32).at[66].set(1.0)
        w_in["smalls"] = w_in["smalls"].at[0, 0, o_ec : o_ec + w_ec].set(ec_tbl)

    wn_out, nhit_out, e13_out = jax.jit(call)(pexp, *[w_in[k] for k in p.w_names])
    assert int(nhit_out[0]) >= 3
    np.testing.assert_allclose(jnp.sum(wn_out[:B], axis=-1), jnp.ones((B,)), atol=1e-5)
    assert e13_out.shape == (H, 2 * IR)
    assert callable(dedup.po2c)

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
"""Unit tests for `modules/moe/{context,topk,dedup,ring,routing}` helpers."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel.core.megakernel_fakes import (
    TINY_TILES,
    FakeFFNModule,
    FakeGeometry,
    FakeMixerModule,
    MiniCtx,
)
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.moe import ring, topk
from vllm_torchtpu.megakernel.core.modules.moe.context import MoEEmit
from vllm_torchtpu.megakernel.core.modules.moe.params import (
    MoEDebug,
    MoEParams,
    MoETunables,
    derive_moe_params,
)
from vllm_torchtpu.megakernel.core.modules.moe.routing import route
from vllm_torchtpu.megakernel.core.ops import BF16, F32, I32, round_up
from vllm_torchtpu.megakernel.core.planner import plan
from vllm_torchtpu.megakernel.core.spec import LayerType, ModelSpec


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
    assert ids is not None
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


def test_moe_emit_build_and_ring_unit_meta():
    """Verify `MoEEmit.build` iotas/masks and `ring._unit_meta` without DMA deps."""
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=256 << 10, nbe=6)
    ex = ExecSpec(R=2, B=4, T=1, stack_e=True, tiles=tiles)
    g = FakeGeometry(
        L=1, att_every=1, H=256, NH=16, NKV=8, I=256, IS=1024, E=128, TOPK=4
    )
    ms = ModelSpec(
        "moe_emit",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    p = plan(ms, ex)
    mp = derive_moe_params(
        g, ex, p, MoETunables(), MoEDebug(), moe_tp=False, e_slots=0, module_name="m"
    )
    scr = {"elist": jnp.arange(16, dtype=I32), "wlist": jnp.arange(16, dtype=I32)}
    fake_stream = SimpleNamespace(sp=SimpleNamespace(tk={"e13": 1, "e2": 1}))
    fake_ctx = SimpleNamespace(
        ex=ex,
        W={},
        scr=scr,
        stream=fake_stream,
        me=jnp.int32(0),
        skip=frozenset(),
        sc=lambda *_: None,
    )
    m = MoEEmit.build(fake_ctx, mp, MoEDebug())
    assert m.EM == 127 and m.lane_e8.shape == (ex.MR, 128)
    i, w, bank, t_out, td = ring._unit_meta(m, 3, None)
    assert i == 3 // mp.TE and bank == 3 % mp.NBE and t_out == 3 % mp.TE and td is None


class _FakeMoEStream:
    """Minimal stream stub for exercising `route` and `ring` helpers in Pallas."""

    def __init__(self, ebuf: dict[str, Any], esem: dict[str, Any], E: int) -> None:
        self._ebuf = ebuf
        self._esem = esem
        self.sp = SimpleNamespace(
            L=1,
            tk={"e13": 1, "e2": 1},
            lay={"ecopy": (0, E)},
        )

    def mm(self, a: Any, b: Any, dims: Any) -> Any:
        return lax.dot_general(a, b, dims, preferred_element_type=F32)

    def ecopy(self, kind: str, src: Any, bank: Any) -> Any:
        return pltpu.make_async_copy(
            src, self._ebuf[kind].at[bank], self._esem[kind].at[bank]
        )

    def eview(self, kind: str, bank: Any) -> Any:
        return self._ebuf[kind].at[bank]


def _numpy_po2c_rank0(
    uniq_picks: list[int],
    ecopy_row: np.ndarray,
    R: int,
    ER: int,
    EXTRA: int,
    route_vec: bool,
    *,
    mode: str = "exact",
) -> tuple[list[int], list[int]]:
    """NumPy reference for rank-0 Po2C `elist`/`wlist` placement."""
    n_tot = len(uniq_picks)
    rload = [sum(1 for e in uniq_picks if (e // ER) == r) for r in range(R)]
    if route_vec:
        uniq_set = set(uniq_picks)
        elist_out = [s for s in range(ER) if s in uniq_set and int(ecopy_row[s]) == 0]
        wlist_out = list(elist_out)
        code_to_e = {int(ecopy_row[e]): e for e in uniq_picks if int(ecopy_row[e]) > 0}
        cplist = [code_to_e[j + 1] for j in range(R * EXTRA) if (j + 1) in code_to_e]
    else:
        elist_out = [e for e in uniq_picks if (e // ER) == 0 and int(ecopy_row[e]) == 0]
        wlist_out = [e % ER for e in elist_out]
        cplist = [e for e in uniq_picks if int(ecopy_row[e]) > 0]

    if mode == "no_move":
        for e in cplist:
            if (e // ER) == 0:
                elist_out.append(e)
                wlist_out.append(e % ER)
        return elist_out, wlist_out

    cpchoice: list[bool] = []
    for e in cplist:
        r0 = e // ER
        c0 = int(ecopy_row[e]) - 1
        r1 = c0 // EXTRA
        l0, l1 = rload[r0], rload[r1]
        use_c = (l1 + 1 < l0) or (l1 < l0 and (l0 * R > n_tot or l1 * R < n_tot))
        if use_c:
            rload[r0] -= 1
            rload[r1] += 1
        cpchoice.append(use_c)

    for e, use_c in zip(cplist, cpchoice):
        r0 = e // ER
        s0 = e - r0 * ER
        c0 = int(ecopy_row[e]) - 1
        r1 = c0 // EXTRA
        s1 = ER + (c0 - r1 * EXTRA)
        cur, alt = (r1, r0) if use_c else (r0, r1)
        l_cur, l_alt = rload[cur], rload[alt]
        flip = (l_alt + 1 < l_cur) or (
            l_alt < l_cur and l_cur * R > n_tot and l_alt * R < n_tot
        )
        if mode == "invert_flip":
            flip = not flip
        if flip:
            rload[cur] -= 1
            rload[alt] += 1
        final_c = use_c ^ flip
        pick_r = r1 if final_c else r0
        pick_w = s1 if final_c else s0
        if pick_r == 0:
            elist_out.append(e)
            wlist_out.append(pick_w)
    return elist_out, wlist_out


def _run_moe_route_and_ring_pallas(
    *,
    moe_tp: bool,
    e_slots: int,
    ecopy_opt: int,
    route_vec: bool,
    interpret: bool = True,
) -> None:
    R = 2
    B = 8
    E = 128
    TOPK = 4
    H = 256
    I = 128 * R if moe_tp else 128  # noqa: E741
    IS = 256
    ER = E // R
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10, ebank=256 << 10, nbe=4)
    g = FakeGeometry(
        L=1,
        att_every=1,
        H=H,
        NH=4,
        NKV=2,
        I=I,
        IS=IS,
        E=E,
        TOPK=TOPK,
        moe_tp=moe_tp,
        e_slots=e_slots,
    )
    ex = ExecSpec(
        R=R, B=B, T=1, ctx=128, stack_e=True, tiles=tiles, interpret=interpret
    )
    ms = ModelSpec(
        "moe_route_pallas",
        config=g,
        layer_type=LayerType("lt", slots=(FakeMixerModule(), FakeFFNModule())),
    )
    p = plan(ms, ex)
    tt = MoETunables(
        topk=True,
        ecopy_opt=ecopy_opt,
        route_vec=route_vec,
    )
    mp = derive_moe_params(
        g,
        ex,
        p,
        tt,
        MoEDebug(),
        moe_tp=moe_tp,
        e_slots=e_slots,
        module_name="fake_moe",
    )
    assert mp.route_vec is route_vec
    assert mp.ECOPY is (not moe_tp and e_slots > ER)

    nb_picks = round_up(ex.B * mp.TOPK, 128)
    scr_shapes: dict[str, Any] = {
        "wn": pltpu.VMEM((ex.MR, mp.E), F32),
        "hitv": pltpu.VMEM((ex.MR, 128), I32),
        "hits": pltpu.SMEM((ex.MR, 128), I32),
        "elist": pltpu.SMEM((nb_picks,), I32),
        "seen": pltpu.SMEM((8, 128), I32),
        "misc_sem": pltpu.SemaphoreType.DMA(()),
        "st": pltpu.SMEM((1,), I32),
        "es": pltpu.VMEM((mp.NSS, 1, 2 * mp.IR + mp.H), F32),
        "es_sem": pltpu.SemaphoreType.DMA((mp.NSS,)),
        "sm": pltpu.VMEM((2, 1, mp.E), F32),
        "ebuf_e13": pltpu.VMEM((mp.NBE, mp.H, 2 * mp.IR), BF16),
        "ebuf_e2": pltpu.VMEM((mp.NBE, mp.IR, mp.H), BF16),
        "esem_e13": pltpu.SemaphoreType.DMA((mp.NBE,)),
        "esem_e2": pltpu.SemaphoreType.DMA((mp.NBE,)),
    }
    if mp.ECOPY:
        scr_shapes.update(
            {
                "ecopy": pltpu.SMEM((2, 1, mp.E), F32),
                "wlist": pltpu.SMEM((nb_picks,), I32),
                "cplist": pltpu.SMEM((nb_picks,), I32),
                "cpchoice": pltpu.SMEM((nb_picks,), I32),
                "rload": pltpu.SMEM((ex.R,), I32),
            }
        )
        if mp.ecopy_opt == 1:
            scr_shapes["ulist"] = pltpu.SMEM((nb_picks,), I32)
    s_names = list(scr_shapes.keys())

    def kernel_body(
        p8_ref,
        ecopy_in_ref,
        w_e13_ref,
        w_e2_ref,
        w_es_ref,
        wn_out_ref,
        meta_out_ref,
        e13_0_out_ref,
        es_0_out_ref,
        *scr_refs,
    ):
        scr = dict(zip(s_names, scr_refs))
        for hi in range(8):
            for lo in range(128):
                scr["seen"][hi, lo] = jnp.int32(0)
        for idx in range(nb_picks):
            scr["elist"][idx] = jnp.int32(-1)
            if mp.ECOPY:
                scr["wlist"][idx] = jnp.int32(-1)
        scr["sm"][0] = ecopy_in_ref[...]
        scr["sm"][1] = ecopy_in_ref[...]
        if mp.ECOPY:
            for e_i in range(mp.E):
                val = ecopy_in_ref[0, e_i]
                scr["ecopy"][0, 0, e_i] = val
                scr["ecopy"][1, 0, e_i] = val
        ebuf = {"e13": scr["ebuf_e13"], "e2": scr["ebuf_e2"]}
        esem = {"e13": scr["esem_e13"], "e2": scr["esem_e2"]}
        fake_stream = _FakeMoEStream(ebuf, esem, mp.E)
        W = {"e13": w_e13_ref, "e2": w_e2_ref, "es": w_es_ref}
        ctx = MiniCtx(
            g=g,
            ex=ex,
            core_plan=SimpleNamespace(plan=p, stream=fake_stream.sp, comm=None),
            pos_ref=None,
            W=W,
            ST={},
            O={},
            scr=scr,
            me=jnp.int32(0),
            stream=fake_stream,
            comm=None,
        )
        m = MoEEmit.build(ctx, mp, MoEDebug())
        n_hit = route(m, jnp.int32(0), p8_ref[...])

        # Wait on the first prefetched expert's e13, e2, and scale via `ring`
        # helpers, then release bank 0/1 via `egive` and drain remaining units.
        e13_0_out_ref[...] = jnp.zeros_like(e13_0_out_ref[...])
        es_0_out_ref[...] = jnp.zeros_like(es_0_out_ref[...])

        @pl.when(n_hit > 0)
        def _():
            v13 = ring.etake(m, "e13", 0)
            v2 = ring.etake(m, "e2", 1)
            ring.es_wait(m, 0)
            e13_0_out_ref[...] = v13[...]
            es_0_out_ref[...] = scr["es"][0, ...]
            _ = v2[...]
            ring.egive(m, 0, jnp.int32(0), 0)
            ring.egive(m, 1, jnp.int32(0), 1)

            def drain_rest(u_idx, _):
                k_bank = lax.rem(u_idx, jnp.int32(mp.NBE))
                t_tile = lax.rem(u_idx, jnp.int32(mp.TE))

                @pl.when(t_tile < mp.T13)
                def _():
                    fake_stream.ecopy("e13", W["e13"].at[0, 0, 0], k_bank).wait()

                @pl.when(t_tile >= mp.T13)
                def _():
                    fake_stream.ecopy("e2", W["e2"].at[0, 0, 0], k_bank).wait()

                @pl.when(t_tile == 0)
                def _():
                    s_slot = lax.rem(
                        lax.div(u_idx, jnp.int32(mp.TE)), jnp.int32(mp.NSS)
                    )
                    pltpu.make_async_copy(
                        W["es"].at[0, 0],
                        scr["es"].at[s_slot],
                        scr["es_sem"].at[s_slot],
                    ).wait()

                return 0

            nf_total = jnp.minimum(jnp.int32(mp.NBE + 2), n_hit * jnp.int32(mp.TE))
            lax.fori_loop(2, nf_total, drain_rest, 0)

        wn_out_ref[...] = scr["wn"][...]
        meta_out_ref[0] = n_hit
        meta_out_ref[1] = scr["st"][0]
        for idx in range(32):
            meta_out_ref[2 + idx] = scr["elist"][idx]
            meta_out_ref[34 + idx] = (
                scr["wlist"][idx] if mp.ECOPY else scr["elist"][idx]
            )

    call = pl.pallas_call(
        kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MR, mp.E), F32),
            jax.ShapeDtypeStruct((66,), I32),
            jax.ShapeDtypeStruct((mp.H, 2 * mp.IR), BF16),
            jax.ShapeDtypeStruct((1, 2 * mp.IR + mp.H), F32),
        ],
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        scratch_shapes=[scr_shapes[k] for k in s_names],
        interpret=pltpu.InterpretParams() if interpret else False,
    )

    rng = np.random.default_rng(42 + int(moe_tp) + 2 * ecopy_opt + 4 * int(route_vec))
    p8_np = rng.uniform(0.05, 1.0, size=(ex.MR, mp.E)).astype(np.float32)
    # Ensure distinct probabilities per row and force hits on both rank 0 and rank 1
    for b in range(B):
        top_e = [b, (b + 1) % ER, ER + (b % ER), ER + ((b + 2) % ER)]
        for k_i, e_id in enumerate(top_e):
            p8_np[b, e_id] = 10.0 - k_i * 1.5

    ecopy_np = np.zeros((1, mp.E), dtype=np.float32)
    if mp.ECOPY:
        # Replicate expert 0 (on rank 0) onto rank 1's extra slot 0 (`code = EXTRA + 1`)
        # and expert ER (on rank 1) onto rank 0's extra slot 0 (`code = 1`).
        ecopy_np[0, 0] = float(mp.EXTRA + 1)
        ecopy_np[0, ER] = 1.0

    w_e13_np = (
        np.arange(mp.EW * mp.H * (2 * mp.IR), dtype=np.float32).reshape(
            1, mp.EW, 1, mp.H, 2 * mp.IR
        )
        * 1e-3
    )
    w_e2_np = np.ones((1, mp.EW, 1, mp.IR, mp.H), dtype=np.float32) * 0.1
    w_es_np = (
        np.arange(mp.EW * (2 * mp.IR + mp.H), dtype=np.float32).reshape(
            1, mp.EW, 1, 2 * mp.IR + mp.H
        )
        * 1e-2
    )

    wn_out, meta_out, e13_0_out, es_0_out = jax.jit(call)(
        jnp.asarray(p8_np, dtype=F32),
        jnp.asarray(ecopy_np, dtype=F32),
        jnp.asarray(w_e13_np, dtype=BF16),
        jnp.asarray(w_e2_np, dtype=BF16),
        jnp.asarray(w_es_np, dtype=F32),
    )

    # 1. Verify normalized top-k weights `wn[:B]` against NumPy
    wn_ref = np.zeros((B, mp.E), dtype=np.float32)
    uniq_picks: list[int] = []
    for b in range(B):
        picks_b = np.argsort(-p8_np[b], kind="stable")[:TOPK]
        w_b = p8_np[b, picks_b]
        wn_ref[b, picks_b] = w_b / np.sum(w_b)
        for e_id in picks_b:
            if int(e_id) not in uniq_picks:
                uniq_picks.append(int(e_id))
    np.testing.assert_allclose(np.asarray(wn_out[:B]), wn_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(wn_out[B:]), 0.0)

    # 2. Verify `n_hit`, `st[0] == n_hit * TE`, `elist[:n_hit]`, and `wlist[:n_hit]`
    meta = np.asarray(meta_out)
    n_hit = int(meta[0])
    assert int(meta[1]) == n_hit * mp.TE
    elist = [int(x) for x in meta[2 : 2 + n_hit]]
    wlist = [int(x) for x in meta[34 : 34 + n_hit]]
    assert len(elist) == len(set(elist)), f"duplicate experts in elist: {elist}"
    if moe_tp:
        assert elist == uniq_picks
    elif not mp.ECOPY:
        expected_r0 = [e for e in uniq_picks if (e // ER) == 0]
        assert elist == expected_r0
    else:
        exp_elist, exp_wlist = _numpy_po2c_rank0(
            uniq_picks, ecopy_np[0], R, ER, mp.EXTRA, route_vec, mode="exact"
        )
        assert elist == exp_elist
        assert wlist == exp_wlist
        # Must-fail controls: a no-move Po2C and an inverted-flip Po2C both differ
        no_move_elist, no_move_wlist = _numpy_po2c_rank0(
            uniq_picks, ecopy_np[0], R, ER, mp.EXTRA, route_vec, mode="no_move"
        )
        assert (elist, wlist) != (no_move_elist, no_move_wlist)
        inv_elist, inv_wlist = _numpy_po2c_rank0(
            uniq_picks, ecopy_np[0], R, ER, mp.EXTRA, route_vec, mode="invert_flip"
        )
        assert (elist, wlist) != (inv_elist, inv_wlist)

    # 3. Verify `ring.etake` / `ring.es_wait` fetched slot `w0` for expert 0
    assert n_hit > 0
    w0 = wlist[0] if mp.ECOPY else elist[0]
    np.testing.assert_array_equal(
        np.asarray(e13_0_out), np.asarray(jnp.asarray(w_e13_np[0, w0, 0], dtype=BF16))
    )
    np.testing.assert_allclose(
        np.asarray(es_0_out), w_es_np[0, w0], rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    ("moe_tp", "e_slots", "ecopy_opt", "route_vec"),
    [
        (True, 0, 0, False),
        (False, 0, 0, False),
        (False, 72, 0, False),
        (False, 72, 1, False),
        (False, 72, 0, True),
    ],
    ids=[
        "tp_local",
        "ep_local",
        "ep_ecopy_opt0_po2c",
        "ep_ecopy_opt1_po2c",
        "ep_route_vec_po2c",
    ],
)
def test_moe_route_topk_dedup_po2c_and_ring_in_pallas(
    moe_tp: bool, e_slots: int, ecopy_opt: int, route_vec: bool
):
    """Execute `route` (`route_topk`, `dedup_*`, `po2c`) and `ring` in Pallas."""
    _run_moe_route_and_ring_pallas(
        moe_tp=moe_tp,
        e_slots=e_slots,
        ecopy_opt=ecopy_opt,
        route_vec=route_vec,
        interpret=True,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("moe_tp", "e_slots", "ecopy_opt", "route_vec"),
    [
        (False, 72, 0, False),
        (False, 72, 1, False),
        (False, 72, 0, True),
    ],
    ids=["tpu_ecopy_opt0", "tpu_ecopy_opt1", "tpu_route_vec"],
)
def test_moe_route_and_ring_on_tpu(
    moe_tp: bool, e_slots: int, ecopy_opt: int, route_vec: bool
):
    """Execute `route` and `ring` helpers on TPU (`interpret=False`)."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if not tpu_devs:
        pytest.skip("TPU device required for interpret=False Pallas execution")
    with jax.default_device(tpu_devs[0]):
        _run_moe_route_and_ring_pallas(
            moe_tp=moe_tp,
            e_slots=e_slots,
            ecopy_opt=ecopy_opt,
            route_vec=route_vec,
            interpret=False,
        )

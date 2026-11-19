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
"""q38 X-K-LIVE, MoE half (L3-LIVE): pad rows of the D14 whole step issue no routed-expert work (CPU interpret).

With `ExecSpec.live_rows` (VLLM_TPU_MK_LIVE_ROWS; L1-LIVE's plumbing) the MoE hit list is built only from live rows
(`dedup._row_on` reads the planner's `live_rows` input), so an expert picked only by pad rows issues no expert unit
and no expert-weight DMA. The whole step here is test_wholestep's stage-C D14 harness (`_run_c`'s inputs: long mode,
union attention, request-sharded selection, request loop; main-KV DCP 1 / 2 / 4 at R = 2 / 2 / 4) at the 16-row
bucket (4 requests x T = 4), with random (not identical) experts so the expert set matters numerically.

Bars (stated before any run, 2026-11-13; class B = bitwise):
  B1  with the flag on, every live row's residual and per-QSA-layer step rows (KV_NEW / KR_NEW / IR_NEW) are
      bit-identical to the flag off, at DCP 1 / 2 / 4 and 4 / 8 / 12 / 16 live rows (pads a suffix, the runner's
      layout); at 16 live rows every row and the routing trace are bit-identical;
  E1  engagement from the kernel's own routing trace (`MoEDebug.hit_trace`): with the flag on each rank's hit list is
      EXACTLY the first-appearance list of its local experts over the live rows (so pad-only expert units = 0 and no
      unit belongs to nobody); with the flag off it is the list over all rows, and pad-only units > 0 at every live
      level below 16 (the inputs exercise pad work, or the comparison is vacuous);
  E2  a non-suffix mask (live requests 1 and 3): hit list exact over the live rows, and at least one expert is first
      picked by a pad row ahead of a live row that also picks it (the case a pad row's `seen` write would hide);
      live rows within test_wholestep's budget of the flag off (the expert sets are equal, only the f32 summation
      order of a live row's experts may differ: class R for non-suffix masks, which the runner does not produce);
  T1  the trace is passive: flag-on and flag-off outputs bit-identical with and without `hit_trace`;
  R1  refusals: live_rows + ECOPY (e_slots > E / R) and live_rows + `MoEDebug.row_mask` raise MegakernelConfigError,
      controls plan; the MoE block never declares `live_rows`.
Pad rows' outputs are don't-care: they carry the shared expert plus only those of their routed experts that some live
row also picked; they stay finite (asserted) and the share of pad-row elements that differ from the flag off is printed.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel import test_wholestep as tw
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.blocks import MoEBlock, MoEDebug
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.blocks.moe import trace as tr
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.live_rows import LIVE_ROWS
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

T = 4
L = tw.L
DCP_ENV = "VLLM_TPU_QSA_MAIN_KV_DCP"


def _mask(live_reqs, nreq: int = 4) -> tuple[int, ...]:
    """Row mask (T rows per request) with the requests in `live_reqs` live."""
    return tuple(int(q in set(live_reqs)) for q in range(nreq) for _ in range(T))


def _bits(a) -> np.ndarray:
    a = np.asarray(a)
    return a.view({2: np.uint16, 4: np.uint32}[a.dtype.itemsize])


@contextlib.contextmanager
def _dcp_env(dcp: int):
    old = os.environ.get(DCP_ENV)
    if dcp > 1:
        os.environ[DCP_ENV] = str(dcp)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(DCP_ENV, None)
        else:
            os.environ[DCP_ENV] = old


@functools.lru_cache(maxsize=None)
def _run(dcp: int, mask: tuple[int, ...] | None, nreq: int = 4, trace: bool = True, key: int = 17) -> dict:
    """One whole step (test_wholestep `_run_c`'s D14 inputs, no reference) with the live-row mask `mask` (None = flag
    off). Returns bit views of the outputs and the routing trace."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import IR_NEW, KR_NEW

    R = 4 if dcp == 4 else 2
    Gx = tw.G if R == 2 else dataclasses.replace(tw.G, NH=3 * R)
    ctx = tw.CTX_D4 if dcp == 4 else tw.CTX_C
    B = nreq * T
    dbg = mk.DebugOptions(blocks={MoEBlock.name: MoEDebug(hit_trace=True)}) if trace else mk.DebugOptions()
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tw._TILES, debug=dbg, live_rows=mask is not None)
    with _dcp_env(dcp):
        d = wi.random_wholestep_inputs(key, Gx, ex, L, budget=tw.BUDGET, n_pages=tw.PAGES)
        d = wi.add_indexer_inputs(key, Gx, ex, d, L, max_groups=tw.MAX_GROUPS, ctx=ctx)
        d = wi.add_step_inputs(key, Gx, ex, d, L, ctx=ctx)
        d = wi.add_carrier(key, d)
        d["hbm_tables"] = True
        if dcp > 1:  # the ranks of a DCP group share one KV head (test_wholestep `shared_kv`)
            kv = 2 * Gx.HD
            d["qkvg"] = d["qkvg"].at[1:, ..., -kv:].set(
                jnp.broadcast_to(d["qkvg"][:1, ..., -kv:], d["qkvg"][1:, ..., -kv:].shape)
            )
            d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
        pps = d["mtab"].shape[1]
        if dcp > 1:
            tw._dcp_split(d, dcp)
        long_kw = dict(carrier="bf16", chunk_pages=8, hbm_tables=True, per_request_scores=True, bounded_chunks=True,
                       shard_index=True, union_attn=True, union_stats=False, union_phys=False, loop_requests=True,
                       dcp=dcp, compressed_dcp=False)
        spec = wholestep_spec(Gx, R, L=L, qsa=dict(budget=tw.BUDGET), step=True,
                              indexed=dict(max_groups=tw.MAX_GROUPS, pages_per_seq=pps, debug=True, **long_kw))
        w_mk, states = wi.pack_wholestep(spec, Gx, ex, d, L)
        mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
        names = [x.name for x in kernel.extra_inputs(spec, ex)]
        extras = list(wi.extras_c(d, mesh, R))
        if mask is not None:
            assert names[-1] == LIVE_ROWS and names.count(LIVE_ROWS) == 1, names
            extras.append(jax.device_put(jnp.asarray(mask, jnp.int32), NamedSharding(mesh, P())))
        else:
            assert LIVE_ROWS not in names, names
        assert len(extras) == len(names), (len(extras), names)
        step, info = kernel.make_step(mesh, spec, ex)
        _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *extras)
    M, n_att = ex.M, L // Gx.att_every
    mp = info["plan"].block_params["moe"]
    out = dict(R=R, B=B, TE=mp.TE, ER=Gx.E // R, TOPK=Gx.TOPK)
    out["res"] = _bits(outs[RESIDUAL_OUT]).reshape(R, M, -1)[:, :B]
    out["res_f32"] = np.asarray(outs[RESIDUAL_OUT], np.float32).reshape(R, M, -1)[:, :B]
    for k, name in (("kv", KV_NEW), ("kr", KR_NEW), ("ir", IR_NEW)):
        out[k] = _bits(outs[name]).reshape(R, n_att, M, -1)[:, :, :B]
    if trace:
        out["hits"] = np.asarray(outs[tr.TRACE_HITS]).reshape(R, L, ex.MR, 128)[:, :, :B, : Gx.TOPK]
        out["elist"] = np.asarray(outs[tr.TRACE_ELIST]).reshape(R, L, -1)
        out["units"] = np.asarray(outs[tr.TRACE_UNITS]).reshape(R, L)
    return out


def _engagement(o: dict, cls_mask, adds_mask) -> dict:
    """Classify every issued expert unit by the rows that picked its expert (from the trace's own picks: the mask
    under test is never trusted). `cls_mask` marks live rows; `adds_mask` the rows expected to add experts. Returns
    expert counts per class (units = count x TE), whether every list is exactly the expected first-appearance list,
    and the number of experts first picked by a pad row ahead of a live row that also picks them."""
    R, B, TE, ER = o["R"], o["B"], o["TE"], o["ER"]
    live = [bool(x) for x in cls_mask]
    adds = [bool(x) for x in adds_mask]
    c = dict(live=0, pad=0, foreign=0, exact=True, mismatch=[], pad_first=0, bad_units=0)
    for r in range(R):
        for lyr in range(L):
            u = int(o["units"][r, lyr])
            if u % TE:
                c["bad_units"] += 1
            got = [int(e) for e in o["elist"][r, lyr, : u // TE]]
            picks = [[int(e) for e in o["hits"][r, lyr, b]] for b in range(B)]
            want: list[int] = []
            for b in range(B):
                for e in picks[b]:
                    if adds[b] and e // ER == r and e not in want:
                        want.append(e)
            if got != want:
                c["exact"] = False
                c["mismatch"].append((r, lyr, got[:8], want[:8]))
            live_set = {e for b in range(B) if live[b] for e in picks[b]}
            pad_set = {e for b in range(B) if not live[b] for e in picks[b]}
            for e in got:
                c["live" if e in live_set else ("pad" if e in pad_set else "foreign")] += 1
            first: dict[int, int] = {}
            for b in range(B):
                for e in picks[b]:
                    first.setdefault(e, b)
            c["pad_first"] += sum(
                1 for e, b0 in first.items()
                if e // ER == r and not live[b0] and any(live[b] and e in picks[b] for b in range(b0 + 1, B))
            )
    return c


def _live_rows(mask) -> np.ndarray:
    return np.asarray(mask, bool)


def _assert_live_bitwise(on: dict, off: dict, mask) -> None:
    lr = _live_rows(mask)
    for k in ("res",):
        np.testing.assert_array_equal(on[k][:, lr], off[k][:, lr], err_msg=f"{k} live rows")
    for k in ("kv", "kr", "ir"):
        np.testing.assert_array_equal(on[k][:, :, lr], off[k][:, :, lr], err_msg=f"{k} live rows")


def _pad_note(on: dict, off: dict, mask) -> str:
    pr = ~_live_rows(mask)
    if not pr.any():
        return "no pad rows"
    a, b = on["res_f32"][:, pr], off["res_f32"][:, pr]
    assert np.isfinite(a).all(), "pad rows turned non-finite with the flag on"
    return "pad rows finite; %.3f of pad-row residual elements differ from flag off" % float((on["res"][:, pr] != off["res"][:, pr]).mean())


LEVELS = [(1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 2, 3))]
CASES = [(dcp, n, reqs) for dcp in (1, 2, 4) for n, reqs in LEVELS]


@pytest.mark.parametrize(("dcp", "nreq_live", "reqs"), CASES, ids=[f"dcp{d}-live{4 * n}" for d, n, _ in CASES])
def test_live_rows_moe_bitwise_and_no_pad_experts(dcp, nreq_live, reqs):
    """B1 + E1 at one DCP / live level of the 16-row bucket."""
    mask = _mask(reqs)
    off = _run(dcp, None)
    on = _run(dcp, mask)
    ones = (1,) * len(mask)
    e_on = _engagement(on, mask, mask)
    e_off = _engagement(off, mask, ones)
    TE = on["TE"]
    print(f"L3LIVE-CASE dcp={dcp} live_rows={sum(mask)}/{len(mask)}: expert units ON live {e_on['live'] * TE} pad "
          f"{e_on['pad'] * TE} foreign {e_on['foreign'] * TE} | OFF live {e_off['live'] * TE} pad {e_off['pad'] * TE} "
          f"foreign {e_off['foreign'] * TE} (experts x TE {TE}; one expert-weight DMA per unit; summed over {on['R']} "
          f"ranks x {L} layers); lists exact on/off {e_on['exact']}/{e_off['exact']}; {_pad_note(on, off, mask)}",
          flush=True)
    assert e_on["bad_units"] == 0 and e_off["bad_units"] == 0, "unit count not a multiple of TE"
    assert e_off["exact"], ("flag-off hit list is not the all-rows first-appearance list", e_off["mismatch"][:3])
    assert e_on["exact"], ("flag-on hit list is not the live-rows first-appearance list", e_on["mismatch"][:3])
    assert e_on["pad"] == 0 and e_on["foreign"] == 0 and e_on["live"] > 0, e_on
    assert e_off["foreign"] == 0, e_off
    if sum(mask) < len(mask):
        assert e_off["pad"] > 0, "vacuous: the pad rows picked no pad-only expert with the flag off"
    _assert_live_bitwise(on, off, mask)
    if sum(mask) == len(mask):
        for k in ("res", "kv", "kr", "ir", "hits", "elist", "units"):
            np.testing.assert_array_equal(on[k], off[k], err_msg=f"all-live {k}")
    print(f"L3LIVE-B1 dcp={dcp} live_rows={sum(mask)}: live rows bitwise (residual + KV_NEW / KR_NEW / IR_NEW) OK",
          flush=True)


def test_live_rows_moe_wide_bucket():
    """B1 + E1 on the WIDE dedup path (B = 32 > wide_rows 16; fori_loop over traced rows): 20 of 32 rows live."""
    reqs = (0, 1, 2, 3, 4)
    mask = _mask(reqs, nreq=8)
    off = _run(1, None, nreq=8)
    on = _run(1, mask, nreq=8)
    e_on, e_off = _engagement(on, mask, mask), _engagement(off, mask, (1,) * len(mask))
    TE = on["TE"]
    print(f"L3LIVE-WIDE dcp=1 live_rows=20/32: units ON live {e_on['live'] * TE} pad {e_on['pad'] * TE} | OFF live "
          f"{e_off['live'] * TE} pad {e_off['pad'] * TE}; exact {e_on['exact']}/{e_off['exact']}; "
          f"{_pad_note(on, off, mask)}", flush=True)
    assert e_on["exact"] and e_off["exact"], (e_on["mismatch"][:2], e_off["mismatch"][:2])
    assert e_on["pad"] == 0 and e_on["foreign"] == 0 and e_off["pad"] > 0
    _assert_live_bitwise(on, off, mask)


def test_live_rows_moe_interleaved_mask():
    """E2: live requests 1 and 3 (pad rows ahead of live rows)."""
    mask = _mask((1, 3))
    off = _run(1, None)
    on = _run(1, mask)
    e_on = _engagement(on, mask, mask)
    TE = on["TE"]
    lr = _live_rows(mask)
    a, b = on["res_f32"][:, lr], off["res_f32"][:, lr]
    scale = float(np.abs(b).mean())
    d = np.abs(a - b)
    print(f"L3LIVE-INTERLEAVED live requests 1, 3: units ON live {e_on['live'] * TE} pad {e_on['pad'] * TE} foreign "
          f"{e_on['foreign'] * TE}; exact {e_on['exact']}; experts first picked by a pad row ahead of a live row "
          f"{e_on['pad_first']}; live rows vs flag off: bitwise share {float((on['res'][:, lr] == off['res'][:, lr]).mean()):.4f}, "
          f"mean {d.mean() / scale:.2e} max {d.max() / scale:.2e} of mean |off|", flush=True)
    assert e_on["pad_first"] > 0, "vacuous: no pad row picks a live row's expert first"
    assert e_on["exact"], e_on["mismatch"][:3]
    assert e_on["pad"] == 0 and e_on["foreign"] == 0
    assert np.isfinite(a).all()
    assert d.mean() < 1e-2 * scale and np.quantile(d, 0.999) < 6e-2 * scale and d.max() < 1e-1 * scale


def test_hit_trace_is_passive():
    """T1: the trace changes no output bit, flag on (8 live rows) and off."""
    mask = _mask((0, 1))
    for m in (mask, None):
        a, b = _run(1, m, trace=True), _run(1, m, trace=False)
        for k in ("res", "kv", "kr", "ir"):
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} mask={m}")
    print("L3LIVE-T1 hit_trace passive (flag on and off) OK", flush=True)


def test_live_rows_refusals_and_declarations():
    """R1."""
    from vllm_torchtpu.megakernel.models import qwen35
    from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES

    # ECOPY (redundant expert slots) needs an EP geometry with e_slots > E / R: Qwen3.5's tiny EP geometry (Qwen3.8's
    # geometry on this base has no e_slots field).
    g35 = qwen35.Qwen35Geometry(L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64, KH=2, VH=4,
                                KD=128, VD=128, att_every=4, moe_tp=False, e_slots=68)
    tiles35 = dataclasses.replace(TINY_TILES, bank=512 << 10).with_blocks({"gqa": {"kv_blk": 256}})
    ex35 = ExecSpec(R=2, B=2, T=1, ctx=256, stack_e=True, interpret=True, tiles=tiles35)
    spec35 = qwen35.qwen35_spec(g35)
    assert planner.plan(spec35, ex35).block_params["moe"].ECOPY  # control: ECOPY alone plans, with ECOPY on
    with pytest.raises(mk.MegakernelConfigError, match="ECOPY"):
        planner.plan(spec35, dataclasses.replace(ex35, live_rows=True))
    spec = qwen38.segment_spec(qwen38.TINY, 2, L=4)
    base = ExecSpec(R=2, B=8, T=4, ctx=128, interpret=True, tiles=tw._TILES)
    rm = dataclasses.replace(base, debug=mk.DebugOptions(blocks={MoEBlock.name: MoEDebug(row_mask=True)}))
    planner.plan(spec, rm)  # control: the debug row mask alone plans
    on = dataclasses.replace(base, live_rows=True)
    planner.plan(spec, on)  # control: live_rows alone plans
    with pytest.raises(mk.MegakernelConfigError, match="row_mask"):
        planner.plan(spec, dataclasses.replace(rm, live_rows=True))
    assert all(x.name != LIVE_ROWS for b in spec.all_blocks() for x in b.extra_inputs(spec.config, on))
    names_on = [x.name for x in planner.extra_inputs(spec, on)]
    assert names_on[-1] == LIVE_ROWS and names_on.count(LIVE_ROWS) == 1, names_on
    assert LIVE_ROWS not in [x.name for x in planner.extra_inputs(spec, base)]
    assert tr.trace_outputs(MoEDebug(), qwen38.TINY, base) == []
    print("L3LIVE-R1 refusals (ECOPY, row_mask) raise, controls plan; MoE declares no live_rows input OK", flush=True)


def test_live_engagement_line(monkeypatch, caplog):
    """The car's MoE engagement marker: one INFO line per process and bucket, at trace time, only on the live path."""
    import logging

    from vllm_torchtpu.megakernel.blocks.moe import dedup

    monkeypatch.setattr(dedup, "_LIVE_LOGGED", set())
    with caplog.at_level(logging.INFO, logger="vllm." + dedup.__name__):
        for _ in range(2):
            dedup._log_live_once(ExecSpec(B=16, R=8, live_rows=True))
        dedup._log_live_once(ExecSpec(B=32, R=8, live_rows=True))
    lines = [r.getMessage() for r in caplog.records if "q38 X-K-LIVE MoE traced" in r.getMessage()]
    assert lines == [
        "q38 X-K-LIVE MoE traced: bucket B=16 R=8, pad rows add no routed experts (dedup_local live_rows)",
        "q38 X-K-LIVE MoE traced: bucket B=32 R=8, pad rows add no routed experts (dedup_local live_rows)",
    ], lines
    src = open(dedup.__file__).read()
    body = src[src.index("def _row_on("):src.index("_LIVE_LOGGED: set")]
    assert body.index("if m.ex.live_rows:") < body.index("_log_live_once(m.ex)") < body.index("return row_live(m.ctx, b)")

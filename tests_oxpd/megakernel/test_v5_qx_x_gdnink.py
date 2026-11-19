# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): X-DCP-QX-EARLY (VLLM_TPU_MK_DCP_QX_EARLY=1) x GDN-REL-FETCH (VLLM_TPU_MK_GDN_INK=2)
together, CPU interpret: the L1 successor's ruling arms this pair in the v5 car, and each is bitwise alone; this is the
composition check.

Runner: test_l1idx `_idx_run` (the served indexed options `_d08s1_options()` + `_idx_options()` over the test geometry, test
tiles `test_wholestep._TILES`, union probe on), as L1-QX's identity test uses it. GDN-REL-FETCH is applied as L1-GDNINK's own
test applies it: VLLM_TPU_MK_GDN_INK=2 and `rel_fetch=True` on the test tiles (the test kernels take explicit tiles, not
`wholestep_tiles`; L1-GDNINK's test_gdnink_wholestep_tiles_rel_fetch proves the env -> tiles plumbing).
Rules (stated before data), per scenario at main-KV DCP 2 / 4 and 16 / 48 rows with D08 S1 (+ the IDX package on two legs, extra coverage):
  * engagement: the both-on run builds the Qx block with `qx_early` in its options and calls `TwoRing.fetch_rel` during its
    build; the both-off run builds neither (no Qx block, no fetch_rel call) and does issue `TwoRing.fetch` refills;
  * bitwise: both on == both off (test_wholestep `_s1_same`: residual, ck_new, ring_new, index queries, kr_new / ir_new and
    the union state on every rank).
"""
from __future__ import annotations

import dataclasses

import pytest

from tests.megakernel import test_l1idx as tl
from tests.megakernel import test_wholestep as tw

QX_ENV = "VLLM_TPU_MK_DCP_QX_EARLY"
INK_ENV = "VLLM_TPU_MK_GDN_INK"
EXTRA_SCEN = {
    # L1-QX's 48-row DCP-4 scenario and its DCP-2 twin.
    "v5_d4_r4_48": dict(R=4, nreq=12, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, ties=True, page512=True),
    "v5_d2_r4_48": dict(R=4, nreq=12, dcp=2, loop=True, budget=64, ctx=tw.CTX_D4, ties=True, page512=True),
}
CASES = [
    ("d2_r2_16", "BASE"),  # DCP 2, 16 rows
    ("d4_r4_16", "BASE"),  # DCP 4, 16 rows, per-layer tables
    ("d4_r4_16", "ALLNP"),  # + the IDX package (TSLIM 1)
    ("d4_r4_16", "ALL"),  # + the IDX package with TSLIM 2 prefetch: extra coverage only (IDX is OFF in v5's car, T98k NO VERDICT)
    ("v5_d4_r4_48", "BASE"),  # DCP 4, 48 rows
    ("v5_d2_r4_48", "BASE"),  # DCP 2, 48 rows
]


def _run(monkeypatch, scen: str, cfg: str, on: bool, calls: dict) -> tuple:
    from vllm_torchtpu.megakernel import stream

    base_tiles = tw._TILES
    if on:
        monkeypatch.setenv(QX_ENV, "1")
        monkeypatch.setenv(INK_ENV, "2")
        monkeypatch.setattr(tw, "_TILES", dataclasses.replace(base_tiles, rel_fetch=True))
    else:
        monkeypatch.delenv(QX_ENV, raising=False)
        monkeypatch.delenv(INK_ENV, raising=False)
    n0 = dict(calls)
    try:
        r = tl._idx_run(monkeypatch, scen, cfg)
    finally:
        monkeypatch.setattr(tw, "_TILES", base_tiles)
        monkeypatch.delenv(QX_ENV, raising=False)
        monkeypatch.delenv(INK_ENV, raising=False)
    assert stream.TwoRing.fetch_rel is not None
    return r, {k: calls[k] - n0[k] for k in calls}


@pytest.mark.parametrize(("scen", "cfg"), CASES, ids=[f"{s}-{c}" for s, c in CASES])
def test_qx_and_gdn_rel_fetch_together_are_bitwise(monkeypatch, scen, cfg):
    from vllm_torchtpu.megakernel import stream

    if scen in EXTRA_SCEN:
        monkeypatch.setitem(tl.SCEN, scen, EXTRA_SCEN[scen])
    kw = tl.SCEN[scen]
    D, KU = kw["dcp"], tl._ku(kw["budget"])
    calls = {"fetch_rel": 0, "fetch": 0}
    real_rel, real_fetch = stream.TwoRing.fetch_rel, stream.TwoRing.fetch

    def fetch_rel(self, *a, **k):
        calls["fetch_rel"] += 1
        return real_rel(self, *a, **k)

    def fetch(self, *a, **k):
        calls["fetch"] += 1
        return real_fetch(self, *a, **k)

    monkeypatch.setattr(stream.TwoRing, "fetch_rel", fetch_rel)
    monkeypatch.setattr(stream.TwoRing, "fetch", fetch)
    b, d_off = _run(monkeypatch, scen, cfg, False, calls)
    r, d_on = _run(monkeypatch, scen, cfg, True, calls)
    assert "Qx" not in b[8]["_block"] and "qx_early" not in b[8]["_opts"], (b[8]["_block"], b[8]["_opts"])
    assert r[8]["_block"].endswith("QxBlock") and r[8]["_opts"].get("qx_early") is True, (r[8]["_block"], r[8]["_opts"])
    assert d_off["fetch_rel"] == 0 and d_off["fetch"] > 0, d_off
    assert d_on["fetch_rel"] > 0, d_on
    tw._s1_same(r, b, D, KU, f"{scen} {cfg} QX + INK=2 vs both off")
    print(f"V5 QXxGDNINK {scen} {cfg}: DCP {D}, {kw['nreq'] * 4} rows: bitwise identical (block {r[8]['_block']} vs "
          f"{b[8]['_block']}); fetch_rel on {d_on['fetch_rel']} / off {d_off['fetch_rel']}; fetch off {d_off['fetch']}",
          flush=True)

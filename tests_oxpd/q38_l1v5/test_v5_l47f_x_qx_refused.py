# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): L47-FOLD-S1 (VLLM_TPU_MK_L47_FOLD) and X-DCP-QX-EARLY (VLLM_TPU_MK_DCP_QX_EARLY) are not
composed: with both on, building the whole step's QSA block must raise (ValueError naming L1-L47F) when the spec is built, never
fall back silently to one of the two. The L1 successor's ruling: the v5 car arms L47, not QX; the render check forbids both.

Path (C): build_wholestep_call merges `_d08s1_options()`, `_idx_options()` (which adds `qx_early` under main-KV DCP 2 / 4) and
`_l47f_options(L, att_every)` (`l47_sink` at L 45) into the indexed options; wholestep_spec -> qsa_full_l47f.lean_step_block ->
qsa_full_idx.lean_step_block routes `qx_early` to its Qx block class, which has no L47F counterpart, so the L47F router raises.
In serving, that build runs inside the driver's whole-step setup (qwen38_driver.py `_attach_whole`), whose handler logs the
traceback ('Qwen3.8 whole-step: setup failed; segment program only'), which the v5 prereg's D14-engagement rule reads as VOID.
Controls: each flag alone builds its own block (L47F alone -> the L47F lean block; QX alone -> the Qx block)."""

import pytest

from tests.megakernel import test_l1l47f as L

QX_ENV = "VLLM_TPU_MK_DCP_QX_EARLY"
K = 64


def _indexed(monkeypatch, *, fold: bool, qx: bool, dcp: int = 4) -> dict:
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    for e in tuple(L.S1) + tuple(L.IDX_ENVS) + (L.FOLD_ENV, QX_ENV, "VLLM_TPU_QSA_COMPRESSED_DCP"):
        monkeypatch.delenv(e, raising=False)
    for e, v in L.S1.items():
        monkeypatch.setenv(e, v)
    monkeypatch.setenv(L.DCP_ENV, str(dcp))
    if fold:
        monkeypatch.setenv(L.FOLD_ENV, "1")
    if qx:
        monkeypatch.setenv(QX_ENV, "1")
    wsc._QX_LOGGED = False
    indexed = dict(max_groups=1024, pages_per_seq=8, debug=False, carrier="fp8", chunk_pages=8, hbm_tables=True,
                   per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, union_stats=False,
                   union_phys=False, loop_requests=True, dcp=dcp, compressed_dcp=False, n_layers=12)
    indexed.update(wsc._d08s1_options())
    indexed.update(wsc._idx_options())
    indexed.update(wsc._l47f_options(45, 4))
    return indexed


@pytest.mark.parametrize("dcp", [2, 4])
def test_l47f_and_qx_together_refuse_at_build(monkeypatch, dcp):
    from vllm_torchtpu.megakernel.wholestep.qsa_full_l47f import lean_step_block

    indexed = _indexed(monkeypatch, fold=True, qx=True, dcp=dcp)
    assert indexed.get("qx_early") is True and indexed.get("l47_sink") is True, indexed
    with pytest.raises(ValueError, match="L1-L47F") as e:
        lean_step_block(dict(indexed), dict(budget=K))
    assert "Qx" in str(e.value), str(e.value)
    print(f"V5 L47FxQX dcp={dcp}: refused at build: {str(e.value)[:200]}", flush=True)


@pytest.mark.parametrize("dcp", [2, 4])
def test_each_flag_alone_builds_its_block(monkeypatch, dcp):
    from vllm_torchtpu.megakernel.wholestep.qsa_full_l47f import lean_step_block

    a = lean_step_block(dict(_indexed(monkeypatch, fold=True, qx=False, dcp=dcp)), dict(budget=K))
    assert type(a).__name__.endswith("L47FLeanBlock") and a.l47_sink, type(a).__name__
    ix = _indexed(monkeypatch, fold=False, qx=True, dcp=dcp)
    assert "l47_sink" not in ix and ix.get("qx_early") is True
    b = lean_step_block(dict(ix), dict(budget=K))
    assert "Qx" in type(b).__name__ and not getattr(b, "l47_sink", False), type(b).__name__
    print(f"V5 L47FxQX dcp={dcp}: alone L47F -> {type(a).__name__}, alone QX -> {type(b).__name__}", flush=True)


def test_l47f_and_gdn_rel_fetch_together_refuse_at_plan(monkeypatch):
    """STACK-D v5 (q38 L1-V5): L47-FOLD-S1's partial last cycle (L 45) and GDN-REL-FETCH (VLLM_TPU_MK_GDN_INK bit 2:
    the whole step's tiles with rel_fetch=True) are not composed either: L47's planner refuses a partial cycle with rel_fetch
    on (planner.py, 'needs rel_fetch off'), at plan time, i.e. at the whole step's build. Controls: INK=2 at L 44 plans; L 45
    with INK=0 plans."""
    from vllm_torchtpu.megakernel import planner
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import wholestep_tiles

    g, R = qwen38.FULL, 8

    def plan(L, ink):
        monkeypatch.setenv("VLLM_TPU_MK_GDN_INK", ink)
        nq = -(-L // 4)
        spec = wholestep_spec(g, R, L=L, first="qsa", pool_gdn=True, qsa=dict(budget=512), step=True,
                              indexed=dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=nq))
        t = wholestep_tiles(16, 65536)
        assert bool(t.rel_fetch) == (ink == "2"), (ink, t.rel_fetch)
        return planner.plan(spec, ExecSpec(R=R, B=16, T=4, ctx=128, tiles=t))

    with pytest.raises(AssertionError, match="rel_fetch off"):
        plan(45, "2")
    assert plan(44, "2").T_TOTAL > 0
    assert plan(45, "0").T_TOTAL > 0
    print("V5 L47FxGDNINK: L 45 + rel_fetch refused at plan; L 44 + INK=2 and L 45 + INK=0 plan", flush=True)

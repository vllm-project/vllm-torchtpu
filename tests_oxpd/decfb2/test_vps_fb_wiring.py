# SPDX-License-Identifier: Apache-2.0
"""PD DEC-FB-2: runner wiring of the VPS per-row fallback and per-candidate draws (torch CPU).

The runner's own methods (`_sample_spec_verify_vps`, `_vps_verify_fb`, `_vps_fallback_rows`, `_check_vps`) and module
functions (`_vps_verify_fb_impl`) are extracted from tpu_runner.py by ast into the harness of
tests/q38mm_dstock/test_vps_wiring.py (FakeRunner, TP1 head, stock fused verify as the reference).
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/decfb2/test_vps_fb_wiring.py
Mutation-checked by tests/decfb2/q38_decfb2_vpsfb_wiring_mutrun.py.
"""

import ast
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38mm_dstock"))
import test_vps_wiring as w  # noqa: E402

VP, VW, NS, B, K, V = w.VP, w.VW, w.NS, w.B, w.K, w.V
w._fns(w.RUNNER, ["_vps_verify_fb_impl"], NS)
NS["_vps_verify_fb_compiled"] = NS["_vps_verify_fb_impl"]
NS["vps_wiring"] = VW
NS["logger"].warning_once = NS["logger"].warning     # vllm's logger has warning_once; the harness logger does not
# q38 L4 VPSHOST: the runner's l6_ab lever reads (sampsnap) are off in this harness (tests/q38_l4_vpshost runs them on).
NS["_l6_ab"] = types.SimpleNamespace(on=lambda lever: False, note=lambda *a, **k: None)
FBV, FBROWS, FBLOG = w._fns(w.RUNNER, ["_vps_verify_fb", "_vps_fallback_rows", "_log_vps_fallback"], NS,
                           cls="TPUModelRunner")


class _NoStatsInFlight:
    """q38 L4 T-1(i): the runner's AsyncCounts with nothing in flight (statsasync is off in this harness)."""

    pending = ()

    @staticmethod
    def ready(wait=False):
        return []


@pytest.fixture(autouse=True, scope="module")
def _harness_modules_installed():
    """test_vps_wiring's stubs (its vllm.distributed all-gather is the TP1 identity) only while these tests run."""
    with w.HARNESS.active():
        yield


class FbRunner(w.FakeRunner):
    def __init__(self, fb_max=8, tiegap=0, cand=False, **kw):
        super().__init__(**kw)
        self._vps_fb_path = fb_max > 0 or tiegap > 0 or cand
        self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = fb_max, tiegap, cand
        self._h2d = []
        self._vps_fb_async = _NoStatsInFlight()
        self.top_k = torch.full((B,), 20, dtype=torch.int32)
        self.temps = torch.full((B,), 0.6)

    def _padded_sampling_params(self, s, e, n, dtype, device):
        k = torch.where(self.top_k >= V, 0, self.top_k)
        return (self.temps.to(dtype).view(-1, 1)[:n], k.view(-1, 1)[:n].clone(),
                torch.full((n, 1), 0.95, dtype=torch.float32))

    def _sampler_topk_candidates(self, s, e, grammar):
        k = self.top_k[s:e]
        return 64 if bool(((k >= 1) & (k <= 32)).all()) else 0

    def _upload_unless_unchanged(self, slot, host):
        self._h2d.append((slot, host.tolist()))
        return host

    _vps_verify_fb = FBV
    _log_vps_fallback = FBLOG
    _vps_fallback_rows = FBROWS


def test_flags_off_keeps_mm_path():
    """No PD flag: `_sample_spec_verify_vps` calls today's `_vps_verify_compiled` (vp_verify_fb is never built)."""
    logits, md = w.problem(1)
    r = w.FakeRunner()
    called = []
    real = NS["_vps_verify_fb_compiled"]
    NS["_vps_verify_fb_compiled"] = lambda *a, **k: called.append(1) or real(*a, **k)
    try:
        r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(1))
    finally:
        NS["_vps_verify_fb_compiled"] = real
    assert not called


@pytest.mark.parametrize("seed", range(3))
def test_fallback_vpbx_equals_fused_and_keeps_the_rng_stream(seed):
    """VPBX + fallback, two requests with top_k unset: equals the runner's own stock fused verify (which takes the full
    path for this chunk), with the same generator state after."""
    logits, md = w.problem(seed)
    r = FbRunner(fb_max=8)
    r.top_k[2] = 0
    r.top_k[5] = 10 ** 6                       # >= vocab: unset
    g1, g2 = torch.Generator().manual_seed(7 + seed), torch.Generator().manual_seed(7 + seed)
    a = w.FUSED(r, logits, md, 0, B, g1)
    b = r._sample_spec_verify_vps(logits, md, 0, B, g2, 8)
    assert torch.equal(a, b)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert r._vps_fb_stats.tolist() == [8, 8, 0] and r._vps_fb_chunks == 1
    assert ("vps_num_active", [B]) in r._h2d


def test_vpb_draw_shapes_and_order():
    """VPB: [B, TP*KC], [T], [T, TP*KC], then the fallback rows' [F, V]; output = vp_verify_fb on those draws."""
    logits, md = w.problem(4)
    r = FbRunner(fb_max=8, cand=True)
    r.top_k[3] = 0
    W = 1 * VP.VP_KC
    seen, real = [], torch.rand

    def spy(*a, **kw):
        seen.append(tuple(a[0]))
        return real(*a, **kw)
    torch.rand = spy
    try:
        out = r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(3), 4)
    finally:
        torch.rand = real
    assert seen == [(B, W), tuple(md.draft_token_ids.shape), (B * K, W), (4, V)]
    g = torch.Generator().manual_seed(3)
    bu, au, ru = real((B, W), generator=g), real(md.draft_token_ids.shape, generator=g), real((B * K, W), generator=g)
    fbu = real((4, V), generator=g)
    temps, top_k, top_p = r._padded_sampling_params(0, B, B, logits.dtype, None)
    ref, _ = VP.vp_verify_fb(logits, 0, lambda x: x, md.bonus_logits_indices, md.target_logits_indices, temps, top_k,
                             top_p, bu, au, ru, md.draft_token_ids, md.draft_lengths, md.segment_ids,
                             md.group_indices, r.rejection_sampler.synthetic_conditional_rates, K, True,
                             torch.tensor([B], dtype=torch.int32), fallback_rows=4, cand_draws=True, fb_u=fbu)
    assert torch.equal(out, ref)


def test_check_skipped_under_candidate_draws():
    logits, md = w.problem(5)
    r = FbRunner(fb_max=0, cand=True, checks=2)
    r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(3), 0)
    assert r._vps_checks_left == 2 and r._vps_mismatches is None


def test_check_runs_with_fallback_vpbx():
    logits, md = w.problem(6)
    r = FbRunner(fb_max=8, checks=2, faults=1)
    r.top_k[1] = 0
    for _ in range(2):
        r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(3), 4)
    assert r._vps_mismatches.tolist() == [0, 1]


def test_stats_logged(caplog, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_VPS_STATS_EVERY", "2")
    logits, md = w.problem(7)
    r = FbRunner(fb_max=8)
    r.top_k[0] = 0
    NS["logger"].setLevel("INFO")
    with caplog.at_level("INFO"):
        for _ in range(2):
            r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(3), 4)
    msgs = [m for m in caplog.messages if m.startswith("VPS_FALLBACK")]
    assert msgs == ["VPS_FALLBACK chunks=2 flagged_rows=8 covered_rows=8 uncovered_rows=0 max_rows=8 tiegap_slots=0 "
                    "draws=full"]


# ---- host side: fallback row counts ----
def test_fallback_rows_host():
    r = FbRunner(fb_max=8)
    r.input_batch = types.SimpleNamespace(top_k_cpu_tensor=r.top_k, temperature_cpu_tensor=r.temps)
    r.vocab_size = V
    assert r._vps_fallback_rows(0, B) == 0
    r.top_k[1] = 0
    assert r._vps_fallback_rows(0, B) == 4
    r.temps[2] = 0.0
    r.top_k[2] = 0                              # greedy: needs no row
    assert r._vps_fallback_rows(0, B) == 4
    r.top_k[3] = V + 5                          # unset (>= vocab)
    assert r._vps_fallback_rows(0, B) == 8
    r.top_k[4] = 33                             # above KC
    assert r._vps_fallback_rows(0, B) is None   # 12 rows > 8: today's path
    assert r._vps_fallback_rows(0, 3) == 4      # only requests 0-2 in this chunk
    r2 = FbRunner(fb_max=0, tiegap=2)
    r2.input_batch, r2.vocab_size = r.input_batch, V
    assert r2._vps_fallback_rows(5, B) == 2     # reserved tie rows only
    assert r2._vps_fallback_rows(0, 2) is None  # a range request cannot run without fallback rows


def test_wiring_helpers():
    assert VW.fallback_slots(0, 0, 8) == 0
    assert VW.fallback_slots(3, 0, 8) == 4 and VW.fallback_slots(5, 0, 8) == 8 and VW.fallback_slots(9, 0, 8) is None
    assert VW.fallback_slots(5, 0, 6) == 6 and VW.fallback_slots(1, 2, 8) == 4
    assert VW.fallback_variants(8, 0) == [0, 4, 8] and VW.fallback_variants(6, 0) == [0, 4, 6]
    assert VW.fallback_variants(0, 2) == [2]
    k = torch.tensor([0, 5, 33, 32, 1, 10 ** 6], dtype=torch.int32)
    t = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.0, 1.0])
    assert VW.range_rows(k, t, 248320, 32, 4) == 12


def test_draws_and_flags_parse(monkeypatch):
    for v, want in (("", False), ("full", False), ("candidate", True), ("vpb", True)):
        monkeypatch.setenv("VLLM_TPU_VPS_DRAWS", v)
        assert VW.cand_draws() is want
    monkeypatch.setenv("VLLM_TPU_VPS_DRAWS", "bogus")
    with pytest.raises(ValueError):
        VW.cand_draws()
    monkeypatch.delenv("VLLM_TPU_VPS_DRAWS")
    monkeypatch.setenv("VLLM_TPU_VPS_FALLBACK_ROWS", "8")
    assert VW.fallback_path() and VW.fallback_max() == 8
    monkeypatch.delenv("VLLM_TPU_VPS_FALLBACK_ROWS")
    assert not VW.fallback_path()
    monkeypatch.setenv("VLLM_TPU_VPS_DRAWS", "candidate")      # per-candidate draws alone take the fb path
    assert VW.fallback_path()
    monkeypatch.delenv("VLLM_TPU_VPS_DRAWS")
    monkeypatch.setenv("VLLM_TPU_VPS_TIEGAP_SLOTS", "1")       # reserved tie rows alone too
    assert VW.fallback_path()
    monkeypatch.delenv("VLLM_TPU_VPS_TIEGAP_SLOTS")
    monkeypatch.delenv("VLLM_TPU_VPS_STATS_EVERY", raising=False)
    assert VW.stats_every() == 1000
    for name in ("VLLM_TPU_VPS_FALLBACK_ROWS", "VLLM_TPU_VPS_TIEGAP_SLOTS", "VLLM_TPU_VPS_DRAWS"):
        assert name in VW.HASHED_ENVS
    assert "VLLM_TPU_VPS_STATS_EVERY" in VW.IGNORED_ENVS


def test_head_refused_under_dp_lockstep_and_needs_kc_plus_one_blocks():
    model = w._model(w._Proc(), w._head())
    ok, why = VW.resolve_vps_head(model, 160, 32)
    assert ok is not None, why
    none, why = VW.resolve_vps_head(model, 160, 32, dp_lockstep=True)
    assert none is None and "lockstep" in why
    assert VW.resolve_vps_head(w._model(w._Proc(), w._head(width=160 * 32)), 160, 32)[0] is None   # kc blocks
    assert VW.resolve_vps_head(w._model(w._Proc(), w._head(width=160 * 33)), 160, 32)[0] is not None


# ---- execute_model: the fallback branch of the chunk dispatch ----
def _fb_block():
    src = open(w.RUNNER).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                and ast.get_source_segment(src, n.test) == "vps_step and spec_decode_metadata is not None and "
                                                         "self._vps_fb_path")
    return compile(ast.Module(body=[node], type_ignores=[]), "<fb>", "exec")


FB_BLOCK = _fb_block()


@pytest.mark.parametrize("fb_path,step,rows,want", [
    (True, True, 4, (True, 4)), (True, True, 0, (True, 0)), (True, True, None, (False, 0)),
    (False, True, 4, ("mm", 0)), (True, False, 4, ("mm", 0)),
])
def test_execute_model_fallback_branch(fb_path, step, rows, want):
    me = types.SimpleNamespace(_vps_fb_path=fb_path, _vps_fallback_rows=lambda s, e: rows)
    loc = {"self": me, "vps_step": step, "spec_decode_metadata": object(), "start_index": 0, "end_index": 8,
           "vps_chunk": "mm", "vps_fb": 0}
    exec(FB_BLOCK, {}, loc)
    assert (loc["vps_chunk"], loc["vps_fb"]) == want


# ---- the chunk forwards F; warm-up compiles every F; cache keys; state plumbing ----
def test_chunk_forwards_fallback_rows():
    logits, md = w.problem(6)
    r = FbRunner()
    got = []
    r._sample_spec_verify_vps = lambda *a, **k: got.append(a[-1] if len(a) > 5 else k) or "VPS"
    assert w.CHUNK(r, logits, md, None, None, 0, B, False, torch.Generator().manual_seed(1), False, vps=True,
                   vps_fallback=4) == "VPS"
    assert got == [4]


def test_precompile_covers_every_fallback_variant():
    (pre,) = w._fns(w.RUNNER, ["_precompile_vps_verify"], NS, cls="TPUModelRunner")
    NS["synchronize_tensors"] = lambda *a: None
    seen = []
    real = NS["_vps_verify_fb_compiled"]
    NS["_vps_verify_fb_compiled"] = lambda *a: seen.append((a[-3], a[-2], None if a[-1] is None else tuple(a[-1].shape))) or real(*a)
    try:
        logits, md = w.problem(2)
        for cand, want in ((False, [(0, False, None), (4, False, None), (8, False, None)]),
                           (True, [(0, True, None), (4, True, (4, V)), (8, True, (8, V))])):
            seen.clear()
            r = FbRunner(fb_max=8, cand=cand)
            r.device = None
            pre(r, logits, md.bonus_logits_indices, md.target_logits_indices, md.draft_token_ids, md.draft_lengths,
                md.segment_ids, md.group_indices, torch.zeros(md.draft_token_ids.shape))
            assert seen == want
    finally:
        NS["_vps_verify_fb_compiled"] = real


def test_compile_factors_include_pd_flags(monkeypatch):
    f = w._real_factors()
    for n in VW.HASHED_ENVS + VW.IGNORED_ENVS:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("VLLM_TPU_VPS_FALLBACK_ROWS", "8")
    monkeypatch.setenv("VLLM_TPU_VPS_TIEGAP_SLOTS", "1")
    monkeypatch.setenv("VLLM_TPU_VPS_DRAWS", "candidate")
    monkeypatch.setenv("VLLM_TPU_VPS_STATS_EVERY", "5")
    got = f()
    assert (got["VLLM_TPU_VPS_FALLBACK_ROWS"], got["VLLM_TPU_VPS_TIEGAP_SLOTS"], got["VLLM_TPU_VPS_DRAWS"]) == (
        8, 1, "candidate")
    assert "VLLM_TPU_VPS_STATS_EVERY" not in got


def _src_fn(name):
    src = open(w.RUNNER).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return src, fn


def test_state_plumbing_structure():
    """Structural (source) checks of the plumbing the CPU harness cannot run: execute_model records F per chunk and
    puts it in the state; sample_tokens zips it and passes it to the chunk; load_model passes DP lockstep."""
    src, fn = _src_fn("execute_model")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ExecuteModelState"]
    kw = {k.arg: ast.get_source_segment(src, k.value) for k in calls[0].keywords}
    assert kw["vps_fallback"] == "vps_fb_list if any(vps_fb_list) else []"
    apps = [ast.get_source_segment(src, n) for n in ast.walk(fn) if isinstance(n, ast.Call)
            and ast.get_source_segment(src, n.func) == "vps_fb_list.append"]
    assert apps == ["vps_fb_list.append(int(vps_fb) if vps_chunk else 0)"]
    src, fn = _src_fn("sample_tokens")
    chunk_calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and ast.get_source_segment(src, n.func) == "self._sample_spec_verify_chunk"]
    assert any(ast.get_source_segment(src, k.value) == "vps_fb" for c in chunk_calls for k in c.keywords
               if k.arg == "vps_fallback")
    assert "vps_fbs = state.vps_fallback or [0] * len(state.logits_list)" in src
    src, fn = _src_fn("load_model")
    res = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
           and ast.get_source_segment(src, n.func) == "vps_wiring.resolve_vps_head"]
    assert [ast.get_source_segment(src, k.value) for k in res[0].keywords if k.arg == "dp_lockstep"] == [
        "self._dp_lockstep_enabled()"]


def test_fallback_rows_host_rows_per_request():
    r = FbRunner(fb_max=6)
    r.input_batch = types.SimpleNamespace(top_k_cpu_tensor=r.top_k, temperature_cpu_tensor=r.temps)
    r.vocab_size = V
    r.top_k[1] = 0
    assert r._vps_fallback_rows(0, B) == 4      # one range request = 1 + K = 4 rows
    r.top_k[3] = 0
    assert r._vps_fallback_rows(0, B) is None   # two = 8 rows > 6

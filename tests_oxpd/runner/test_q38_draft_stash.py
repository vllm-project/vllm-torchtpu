# SPDX-License-Identifier: Apache-2.0
"""q38 L1-DSTASH (runner/draft_stash.py, VLLM_TPU_DECODE_TRIM_DRAFT_STASH) on CPU.

D1  bitwise draft carry: under DECODE_TRIM, a held-back request re-enters with its stashed [bonus, draft_1..K] row
    intact and, request by request, every served step (substituted input ids, corrected positions, GDN read offset,
    accepted tokens, next drafts) equals the untrimmed reference run; without the stash the same re-entries verify -1
    drafts (positive control) and yield one token.
D2  the rejection correction and the GDN read offset at 0 / 1 / many holds per step: positions and read offsets equal
    the reference for held-back and never-held requests, with and without the stash; re-entering requests are never in
    the rejection map (carried rejected count 0).
D3  env off: no stash state is created and the run is identical; the stash flag without DECODE_TRIM is off.
D4  no new device program: every stash call of _substitute_placeholder_token has a signature the real precompile warms
    (VLLM_TPU_WARMED_SUBSTITUTE_SOURCE, as served), or, without it, the regular call's signature.
D5  engagement and counters: install line, window line, per-step applied set == re-entering set, entries only for
    held-back requests, every entry resolved.
D6  DraftStash units (capture / plan edges, padding identical to _apply_async_token_substitution's).
D7  the env gate and the refusals (PCP, multi-stage pipeline, no async spec).
D8  runner wiring (the four hooks sit where the proof assumes).
D9  compile-key registration (host-only flag).
The harness is tests/q38_dstash/dstash_harness.py (real AsyncScheduler + real DECODE_TRIM + the runner's real async
bookkeeping on an object.__new__ runner + a deterministic fake model)."""

from __future__ import annotations

import inspect
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_dstash"))
import dstash_harness as H  # noqa: E402

_CACHE: dict = {}


def run(name, **kw):
    key = (name, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = H.run(name, **kw)
    return _CACHE[key]


def mt(name):
    return H.SCENARIOS[name]["max_tokens"]


def by_step(res, rid, step):
    (e,) = [x for x in res["records"][rid] if x["step"] == step]
    return e


def prev_record(res, rid, step):
    return [x for x in res["records"][rid] if x["step"] < step][-1]


# ----------------------------------------------------------------------------------------------------- D1
@pytest.mark.parametrize("name,min_reentries", [("S1", 8), ("S2", 20)])
def test_d1_bitwise_draft_carry(name, min_reentries):
    ref = run(name, stash=False, table=None)
    on = run(name, stash=True)
    off = run(name, stash=False)
    re_on, re_off = H.reentries(on), H.reentries(off)
    assert len(re_on) >= min_reentries, re_on
    # every re-entry verifies, bit for bit, the drafts its request proposed at its last served step
    for step, rid in re_on:
        e, p = by_step(on, rid, step), prev_record(on, rid, step)
        assert e["x"] == [p["out"][-1]] + p["drafts"][: len(e["x"]) - 1], (name, step, rid, e, p)
        assert e["host"][1:] == [-1] * (len(e["host"]) - 1)  # the scheduler shipped placeholders
        assert not e["in_pre"]
    # positive control: without the stash the same kind of re-entry verifies placeholders and yields one token
    assert re_off
    for step, rid in re_off:
        e = by_step(off, rid, step)
        assert e["x"][1:] == [-1] * (len(e["x"]) - 1) and len(e["out"]) == 1, (step, rid, e)
    # request by request, the trimmed + stashed run IS the untrimmed reference (inputs, positions, offsets, outputs)
    rec_ref, rec_on, rec_off = (H.trimmed_records(x, mt(name)) for x in (ref, on, off))
    assert rec_on == rec_ref
    assert rec_off != rec_ref
    assert on["finals"] == ref["finals"] == off["finals"]
    assert all(len(v) == m for v, m in zip(on["finals"].values(), mt(name)))
    acc_on = [len(by_step(on, rid, s)["out"]) for s, rid in re_on]
    acc_off = [len(by_step(off, rid, s)["out"]) for s, rid in re_off]
    print(f"L1-DSTASH D1 {name}: re-entries stash {len(re_on)} / off {len(re_off)}; tokens per re-entry stash "
          f"{sum(acc_on) / len(acc_on):.3f} off {sum(acc_off) / len(acc_off):.3f}; steps on {len(on['log'])} off "
          f"{len(off['log'])} ref {len(ref['log'])}; records match the reference: stash {rec_on == rec_ref} "
          f"off {rec_off == rec_ref}; totals {on['totals']}")
    assert sum(acc_on) > sum(acc_off) * len(acc_on) / len(acc_off)  # the stash recovers accepted drafts


# ----------------------------------------------------------------------------------------------------- D2
@pytest.mark.parametrize("name,holds", [("S0i", {0}), ("S1", {1}), ("S2", {2, 4})])
def test_d2_rejection_correction_and_gdn_offset(name, holds):
    ref = run(name, stash=False, table=None)
    seen = set()
    for stash in (True, False):
        res = run(name, stash=stash)
        seen |= {len(e["held"]) for e in res["log"] if len(e["sched"]) > 0}
        idx_ref = {rid: {e["before"]: e for e in recs} for rid, recs in ref["records"].items()}
        held_ever = {rid for e in res["log"] for rid in e["held"]}
        reent = set(H.reentries(res))
        mtd = dict(zip([f"r{i}" for i in range(len(mt(name)))], mt(name)))
        diverged: set = set()
        for rid, recs in res["records"].items():
            for e in recs:
                if e["before"] >= mtd[rid] or rid in diverged:
                    continue
                r = idx_ref[rid][e["before"]]
                # the correction (positions) and the GDN read offset match the reference at every served step up
                # to and including the first re-entry without the stash (where only the drafts differ)
                assert e["pos"] == r["pos"] and e["off"] == r["off"], (name, stash, rid, e, r)
                if (e["step"], rid) in reent:
                    assert not e["in_pre"], (name, stash, rid, e)  # no rejected count applied at re-entry
                    if not stash:
                        diverged.add(rid)
                        continue
                assert e["x"] == r["x"] and e["out"] == r["out"], (name, stash, rid, e, r)
            if rid not in held_ever:
                assert rid not in diverged
    assert holds <= seen and max(seen) == max(holds), (name, seen)
    print(f"L1-DSTASH D2 {name}: holds per step seen {sorted(seen)}")


# ----------------------------------------------------------------------------------------------------- D3
def _bytes(res):
    keep = {k: res[k] for k in ("records", "finals", "calls", "trim_totals")}
    keep["log"] = [{k: v for k, v in e.items() if k not in ("applied", "entries", "dstash_attrs")} for e in res["log"]]
    return json.dumps(keep, sort_keys=True, default=str).encode()


def test_d3_env_off_identity_and_gate():
    a = run("S1", stash=False)  # flag off, trim on
    b = run("S1", stash=True, trim_env=False)  # flag on, the runner's env has no DECODE_TRIM
    for res in (a, b):
        assert res["dstash_attrs"] == [] and res["totals"] is None and res["refused"] is None
        assert all(e["dstash_attrs"] == [] for e in res["log"])
        assert not [c for c in res["calls"] if c[0] == "_dstash_apply"]
        assert not res["lines"]
    assert _bytes(a) == _bytes(b)
    # and it is the plain trimmed run: re-entries verify placeholders
    assert H.reentries(a) and all(by_step(a, rid, s)["x"][1:] == [-1] * H.K for s, rid in H.reentries(a))
    c = run("S0", stash=True, table=None)  # flag on, no trim at all
    ref = run("S0", stash=False, table=None)
    assert c["dstash_attrs"] == [] and _bytes(c) == _bytes(ref)
    d = run("S0i", stash=True)  # flag on, trim installed but never holds: stash created, never applied
    t = d["totals"]
    # vLLM itself skips a request whose previous step already reached max_tokens; the runner sees it as held back and
    # stashes it, and the entry is dropped when the request finishes: it is never applied
    assert t is not None and t["hits"] == 0 and t["conflict"] == 0 and not d["live_entries"]
    assert t["captured"] == t["dropped_invalidated"] + t["dropped_dead"]
    assert all(set(e["entries"]) == set(e["unsched"]) for e in d["log"])
    assert not [x for x in d["calls"] if x[0] == "_dstash_apply"]
    assert H.trimmed_records(d, mt("S0i")) == H.trimmed_records(run("S0i", stash=False, table=None), mt("S0i"))
    print(f"L1-DSTASH D3: env-off runs identical ({len(_bytes(a))} bytes); no-trim run identical; never-holding "
          f"trim: stash totals {d['totals']}")


# ----------------------------------------------------------------------------------------------------- D4
def test_d4_no_new_device_program():
    warm = H.warmed_signatures()
    assert warm, "the precompile warmed nothing"
    on = run("S2", stash=True, warmed=True)
    st = {c[1] for c in on["calls"] if c[0] == "_dstash_apply"}
    reg = {c[1] for c in on["calls"] if c[0] == "_apply_async_token_substitution"}
    other = {c[0] for c in on["calls"]} - {"_dstash_apply", "_apply_async_token_substitution"}
    assert st and reg and not other, (st, reg, other)
    assert st <= warm and reg <= warm, (st - warm, reg - warm)
    cold = run("S2", stash=True, warmed=False)
    st_c = {c[1] for c in cold["calls"] if c[0] == "_dstash_apply"}
    reg_c = {c[1] for c in cold["calls"] if c[0] == "_apply_async_token_substitution"}
    assert st_c and st_c <= reg_c, (st_c, reg_c)  # the regular call's own program
    assert not st_c <= warm  # control: the warmed set discriminates (int32 sources are not warmed)
    assert H.trimmed_records(cold, mt("S2")) == H.trimmed_records(run("S2", stash=False, table=None), mt("S2"))
    print(f"L1-DSTASH D4: stash call signatures {sorted(st)} all in the {len(warm)} warmed; cold {sorted(st_c)}")


# ----------------------------------------------------------------------------------------------------- D5
def test_d5_engagement_and_counters():
    on = run("S1", stash=True, log_every=5)
    t = on["totals"]
    re_on = H.reentries(on)
    installs = [x for x in on["lines"] if x.startswith("q38 DSTASH on:")]
    windows = [x for x in on["lines"] if x.startswith("q38 DSTASH steps=")]
    assert installs == ["q38 DSTASH on: k 3, source stride 4, warmed source True, window 5 steps"], installs
    assert len(windows) >= 2 and all("hits=" in w and "accepted_per_hit=" in w for w in windows), windows
    unsched_total = sum(len(e["unsched"]) for e in on["log"])
    prev_held: set = set()
    for e in on["log"]:
        reent = {rid for rid in e["sched"] if rid in prev_held}
        assert set(e["applied"]) == reent, (e, reent)  # applied exactly to the re-entering requests
        # an entry exactly for each request held back this step (the trim's holds, plus vLLM skipping a request whose
        # previous step already reached max_tokens, whose entry is dropped when it finishes); none left after re-entry
        assert set(e["entries"]) == set(e["unsched"]) and set(e["held"]) <= set(e["unsched"]), e
        prev_held = set(e["held"])
    assert t["hits"] == len(re_on) and t["reentry_nostash"] == 0 and t["conflict"] == 0
    assert t["captured"] == unsched_total and t["captured"] >= sum(len(e["held"]) for e in on["log"])
    assert t["captured"] == t["hits"] + t["dropped_invalidated"] + t["dropped_dead"] + len(on["live_entries"])
    assert t["skip_shape"] == t["skip_rows"] == t["skip_structured"] == t["held_nonspec"] == 0
    assert sum(t["accept_hist"].values()) == t["accept_n"] and t["accept_n"] + len(on["pending_accept"]) == t["hits"]
    assert t["accept_tokens"] == sum(k * v for k, v in t["accept_hist"].items())
    # the last re-entry of a request still pending at the end has no committed result yet; all others do
    last = {rid: s for s, rid in re_on}
    committed = [len(by_step(on, rid, s)["out"]) for s, rid in re_on
                 if not (rid in on["pending_accept"] and last[rid] == s)]
    assert len(committed) == t["accept_n"] and sum(committed) == t["accept_tokens"], (committed, t)
    print(f"L1-DSTASH D5: {installs[0]} | {windows[-1][:400]}")


# ----------------------------------------------------------------------------------------------------- D6
def _pre(mapping, src=None, rej=True):
    return SimpleNamespace(req_id_to_index_copy=dict(mapping),
                           next_tokens_tpu=src if src is not None else torch.arange(32, dtype=torch.int32),
                           spec_decode_num_rejected_tokens=torch.zeros(8, dtype=torch.int32) if rej else None)


def _so(scheduled, cached=(), resumed=()):
    return SimpleNamespace(num_scheduled_tokens={r: 4 for r in scheduled},
                           scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached), resumed_req_ids=set(resumed)))


def test_d6_draft_stash_units(monkeypatch):
    from vllm_torchtpu.runner import draft_stash as ds
    from vllm_torchtpu.runner import tpu_runner as r
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    st = ds.DraftStash(3, 0)
    live = {"a": 1, "b": 1, "c": 1}
    st.capture(_pre({"a": 0, "b": 1, "c": 2}), _so(["a"], ["a"]), set(), live, {"a": 0, "b": 1, "c": 2})
    assert sorted(st.entries) == ["b", "c"] and st.entries["b"][1] == 1 and st.totals["captured"] == 2
    # b invalidated (preempted / finished), c dead (gone from the runner): both dropped
    st.capture(_pre({"a": 0}), _so(["a"], ["a"]), {"b"}, {"a": 1}, {"a": 0})
    assert st.entries == {} and st.totals["dropped_invalidated"] == 1 and st.totals["dropped_dead"] == 1
    # stride-1 (non-spec) results: nothing to carry; a 2-D or too-short source: skipped
    st.capture(_pre({"a": 0, "b": 1}, rej=False), _so(["a"]), set(), live, {})
    st.capture(_pre({"a": 0, "b": 1}, src=torch.zeros(2, 4, dtype=torch.int32)), _so(["a"]), set(), live, {})
    st.capture(_pre({"a": 0, "b": 9}, src=torch.zeros(8, dtype=torch.int32)), _so(["a"]), set(), live, {})
    assert st.entries == {} and st.totals["held_nonspec"] == 1 and st.totals["skip_shape"] == 2
    # a re-entry without an entry is counted
    st.capture(None, _so(["b"], ["b"]), set(), live, {"a": 0})
    assert st.totals["reentry_nostash"] == 1
    # plan: slots of the re-entering request, its row, consumed; others untouched
    src = torch.arange(100, 132, dtype=torch.int32)
    st.entries = {"b": (src, 2), "c": (src, 5), "d": (src, 1), "e": (src, 3)}
    groups = st.plan(["a", "b", "c", "d", "e"], 0, 5, np.array([4, 4, 2, 4, 5]), {"d": 0}, None, False)
    ((g_src, cur, sidx),) = groups
    assert g_src is src and cur == [4, 5, 6, 7, 8, 9] and sidx == [8, 9, 10, 11, 20, 21]
    assert st.entries == {} and st.last_applied == ("b", "c") and st.totals["conflict"] == 1
    assert st.totals["skip_rows"] == 1 and st.pending_accept == {"b", "c"}
    # a step with structured-output requests: consumed, not applied
    st.entries = {"b": (src, 2)}
    assert st.plan(["b"], 0, 1, np.array([4]), None, None, True) == [] and st.totals["skip_structured"] == 1
    # the committed-token counter reads exactly the stash-hit requests of the committed results
    st.note_committed([(0, None, 0, "b"), (1, None, 0, "x")], [[1, 2, 3], [4]])
    assert st.totals["accept_n"] == 1 and st.totals["accept_tokens"] == 3 and st.pending_accept == {"c"}
    # padding = _apply_async_token_substitution's: run both through the eager program, same result
    comp = r._substitute_placeholder_token
    eager = getattr(comp, "_torchdynamo_orig_callable", None) or comp.__wrapped__
    ids = torch.arange(16, dtype=torch.int32)
    cur_np, src_np = ds.padded_indices([4, 5, 6, 7], [8, 9, 10, 11], 16)
    a = eager(ids.clone(), torch.from_numpy(cur_np), torch.from_numpy(src_np), src.to(torch.int64))
    rn = object.__new__(TPUModelRunner)
    rn.device = torch.device("cpu")
    rn._fast_token_substitution = False
    rn._pre_async_results = SimpleNamespace(next_tokens_tpu=src.to(torch.int64))
    monkeypatch.setattr(r, "_substitute_placeholder_token", eager)
    monkeypatch.delenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", raising=False)
    b = rn._apply_async_token_substitution(ids.clone(), np.array([4, 5, 6, 7], np.int32),
                                           np.array([8, 9, 10, 11], np.int32))
    assert a.tolist() == b.tolist() == list(range(4)) + [108, 109, 110, 111] + list(range(8, 16))


# ----------------------------------------------------------------------------------------------------- D7
def test_d7_gate_and_refusals(monkeypatch):
    from vllm_torchtpu.runner import draft_stash as ds

    for stash, trim, want in [(None, None, False), ("1", None, False), ("1", "0", False), ("1", "off", False),
                              (None, "64:1", False), ("0", "64:1", False), ("1", "64:1", True)]:
        for k, v in (("VLLM_TPU_DECODE_TRIM_DRAFT_STASH", stash), ("VLLM_TPU_DECODE_TRIM", trim)):
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        assert ds.enabled() is want, (stash, trim)
    monkeypatch.setenv("VLLM_TPU_DECODE_TRIM_LOG", "0")
    for attr, val, why in [("_pcp_enabled", True, "prefill context parallelism"), ("_pp_is_last", False, "pipeline"),
                           ("speculative_config", None, "async scheduling with speculative decoding")]:
        rn = H.make_runner()
        setattr(rn, attr, val)
        assert rn._dstash_create() is None and why in rn._dstash_refused and "_dstash" not in rn.__dict__
    rn = H.make_runner()
    st = rn._dstash_create()
    assert st is rn._dstash and st.stride == 1 + H.K and "_dstash_refused" not in rn.__dict__


# ----------------------------------------------------------------------------------------------------- D8
def test_d8_runner_wiring():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    src = inspect.getsource(TPUModelRunner._prepare_inputs)
    up = src.index("self.input_ids = self.input_ids_cpu[local_token_slice].to(")
    ap = src.index("self._dstash_apply(")
    assert up < ap < src.index("self.position_ids = self.positions_cpu[local_token_slice]")
    assert ap < src.index("self._postfwd_fused_prepare(") and ap < src.index(
        "self._prepare_async_token_substitution_indices(")
    assert src.count("_dstash") == 2  # the guard and the call
    ex = inspect.getsource(TPUModelRunner.execute_model)
    pop = ex.index('dstash_chunk = self.__dict__.pop("_dstash_chunk", None) == start_index')
    cond = ex.index("and (len(cur_input_indices) > 0 or dstash_chunk)")
    assert ex.index("input_ids = self._apply_async_token_substitution(") < pop < cond < ex.index(
        "extract_draft_token_ids(")
    us = inspect.getsource(TPUModelRunner._update_states)
    assert us.index("record[3] not in invalidated_req_ids") < us.index("dstash.capture(") < us.index(
        "super()._update_states(")
    mp = inspect.getsource(TPUModelRunner._modify_prev_results)
    assert mp.index("valid_sampled_token_ids[i].clear()") < mp.index("dstash.note_committed(") < mp.index(
        "for pre_req_idx, req_state, seq_len, req_id in pre_request_seq_lens:")
    ap_src = inspect.getsource(TPUModelRunner._dstash_apply)
    assert "self._warmed_next_tokens(src) if warmed else src" in ap_src


# ----------------------------------------------------------------------------------------------------- D9
def test_d9_compile_env_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_DECODE_TRIM_DRAFT_STASH" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv("VLLM_TPU_DECODE_TRIM_DRAFT_STASH", raising=False)
    assert envs.VLLM_TPU_DECODE_TRIM_DRAFT_STASH is False
    monkeypatch.setenv("VLLM_TPU_DECODE_TRIM_DRAFT_STASH", "1")
    assert envs.VLLM_TPU_DECODE_TRIM_DRAFT_STASH is True

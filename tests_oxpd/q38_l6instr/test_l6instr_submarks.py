# SPDX-License-Identifier: Apache-2.0
"""q38 L6-INSTR (H1 rung 1): the log-only sub-marks of PR-0, DH-E and T-5.

Sub-marks (``host_phases.sub``) split a HOSTPHASES phase; with
VLLM_TPU_HOST_PHASES_SUBS=1 (and VLLM_TPU_HOST_PHASES=N > 0) they are summed
per step and logged on the HOSTSUBS line; with it unset they record nothing.

* Real paths, on a virtual clock where a bracket is claimed exact:
  ``~pi_seed_sync`` is exactly the blocking ``previous.wait_for_copy()`` of the
  mamba seed-copy crossing path; ``~df0_call`` is exactly the draft model call
  of step 0; the adapter's ``~ap_*`` split brackets the window build, the
  window upload, the scalars upload, the compiled call and the table views;
  ``_prepare_inputs`` (driven through L6-CARRY's CPU harness, lever on and
  off) emits the ``~pi_*`` / ``~pp_*`` split.
* Structure, where the runner path needs a device (execute_model,
  sample_tokens, the drafter's propose and seed calls): each sub-mark sits
  between the marks of its parent phase and right next to the call it
  brackets.
* Replay: every host_phases call of every instrumented function, in source
  order, reaches the HOSTSUBS line with the census on and nothing with it off.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import host_phases

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
ROOT = os.path.dirname(TESTS)
SRC = os.path.join(ROOT, "src", "vllm_torchtpu")
RUNNER = os.path.join(SRC, "runner", "tpu_runner.py")
CARRY = os.path.join(SRC, "runner", "decode_carry.py")
EAGLE = os.path.join(SRC, "spec_decode", "eagle3.py")
INPUTS = os.path.join(SRC, "models", "vllm", "qwen4_exp", "inputs.py")
DRIVER = os.path.join(SRC, "megakernel_runtime", "qwen38_driver.py")

# The new sub-marks and the phase each one splits (its closing mark).
NEW_SUBS = {
    "pi_lhost": "prep_host_and_uploads", "pi_carry": "prep_host_and_uploads",
    "pi_h2d": "prep_host_and_uploads", "pp_mamba": "prep_host_and_uploads",
    "pp_subst_idx": "prep_host_and_uploads", "pp_key": "prep_host_and_uploads",
    "pp_fused": "prep_host_and_uploads", "pp_record": "prep_host_and_uploads",
    "pp_upload": "prep_host_and_uploads", "pp_rest": "prep_host_and_uploads",
    "pi_subst2": "prepare_inputs", "pi_collect": "prepare_inputs",
    "pi_seed_sync": "prepare_inputs", "pi_cross": "prepare_inputs",
    "ap_drafts": "adapter_prepare_inputs", "ap_fence": "adapter_prepare_inputs",
    "ap_inputs": "adapter_prepare_inputs", "ap_scalars": "adapter_prepare_inputs",
    "ap_wbuild": "adapter_prepare_inputs", "ap_wh2d": "adapter_prepare_inputs",
    "ap_sup": "adapter_prepare_inputs", "ap_fused": "adapter_prepare_inputs",
    "ap_tables": "adapter_prepare_inputs", "ap_rest": "adapter_prepare_inputs",
    "dpi_rest": "dp_prepare_inputs", "dpi_seed": "dp_prepare_inputs",
    "df0_call": "dp_forward0", "df0_rest": "dp_forward0",
    "dc_rest": "dp_carries", "dc_call": "dp_carries",
    "ss_attrs": "sample_setup", "ss_draft_chunks": "sample_setup", "ss_rest": "sample_setup",
}
# The sub-marks that existed before (DEC-HOST-B / D1k / L4-C HOSTSUBS).
OLD_SUBS = {
    "kc_pre", "kc_bind", "kc_start_load", "am_entry", "am_steady", "am_stage_bt",
    "us_inval", "us_base", "us_copies", "pi_entry", "pi_fence", "pi_tokens", "pi_spec_qsl",
    "pf_counters", "pf_logits", "st_kvconn_out", "st_async", "st_assembly",
    "fd_pre_whole", "fd_whole_call",
}
# q38 L6 P-METER (follow-up commit): the copy-wait split inside _modify_prev_results
# (tests/q38_l6/test_l6_pmeter.py owns their placement and values).
PMETER_SUBS = {"mpr_pre", "mpr_wait", "mpr_post", "vr_post"}  # ~vr_draw / ~vr_call live in runner/l6_nodefer.py


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


class _Clock:
    """Virtual time: reads do not advance it; the test advances it."""

    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def tick(self, us):
        self.ns += us * 1000


def _reset(monkeypatch, every, subs, virtual=True):
    monkeypatch.setattr(host_phases, "_EVERY", every)
    monkeypatch.setattr(host_phases, "_SLOW_NS", 0)
    monkeypatch.setattr(host_phases, "_SUBS", subs)
    monkeypatch.setattr(host_phases, "_GC", False)
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    monkeypatch.setattr(host_phases, "_cur", {})
    monkeypatch.setattr(host_phases, "_ctx", [])
    monkeypatch.setattr(host_phases, "_sub_last", None)
    monkeypatch.setattr(host_phases, "_sub_acc", {})
    monkeypatch.setattr(host_phases, "_sub_samples", {})
    monkeypatch.setattr(host_phases, "_rank", 3)
    log = _Log()
    monkeypatch.setattr(host_phases, "logger", log)
    clock = _Clock()
    if virtual:
        monkeypatch.setattr(host_phases.time, "perf_counter_ns", clock)
    return log, clock


def _subs_line(log, needle):
    got = [ln for ln in log.lines if ln.startswith("HOSTSUBS") and needle in ln]
    assert len(got) == 1, log.lines
    return got[0].split("(p50/p90/mean-per-step/n-steps) ", 1)[1]


def _subs_dict(line):
    out = {}
    for kv in line.split():
        k, v = kv[1:].split("=", 1)
        out[k] = v
    return out


def _assert_off(log):
    assert host_phases._sub_acc == {} and host_phases._sub_samples == {}
    assert not any(ln.startswith("HOSTSUBS") for ln in log.lines), log.lines
    assert not any("~" in ln for ln in log.lines if ln.startswith("HOSTPHASES"))


# ------------------------------------------------------------------ real paths


def _seed_copies_module():
    path = os.path.join(TESTS, "runner", "test_mamba_state_seed_copies.py")
    spec = importlib.util.spec_from_file_location("q38_l6instr_seed_copies", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("subs", [True, False])
def test_seed_sync_is_exactly_the_blocking_copy_wait(monkeypatch, subs):
    """The crossing path waits on the previous step's sampled-token copy;
    ~pi_seed_sync is that wait alone (777 us), the host work before it goes to
    ~pi_collect (50 us), and the phase is their sum."""
    import vllm_torchtpu.runner.tpu_runner as runner_mod
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    m = _seed_copies_module()
    log, clock = _reset(monkeypatch, every=1, subs=subs)
    monkeypatch.setattr(runner_mod, "get_pcp_group", lambda: NS(world_size=1))
    offsets = torch.zeros(m.NUM_BLOCKS, dtype=torch.int32)
    offsets[5] = 3
    fake, _ = m._make_self(["a"], [383], [[[5, 6, 7, 8, 9]]], state_pos={"a": 0},
                           state_block_size=384, ckpt_window=4, read_offsets=offsets)
    fake.scheduler_config = NS(async_scheduling=True)
    fake.speculative_config = NS(num_speculative_tokens=3)

    class _TickMap(dict):
        def get(self, k, d=None):
            clock.tick(50)
            return dict.get(self, k, d)

    waits = []

    def wait():
        waits.append(1)
        clock.tick(777)
        return torch.tensor([[77, 88, 99, 11]])

    fake._pre_async_results = NS(req_id_to_index_copy=_TickMap({"a": 0}), num_draft_per_req={0: 3},
                                 wait_for_copy=wait)
    host_phases.begin_step()
    host_phases.mark("prep_attention_metadata")
    TPUModelRunner._collect_mamba_state_seed_copies(fake, NS(num_scheduled_tokens={"a": 4}), 0, 1)
    host_phases.mark("prepare_inputs")
    host_phases.begin_step()
    assert waits == [1] and fake._mamba_state_pos["a"] == 1  # the crossing path ran
    ph = [ln for ln in log.lines if ln.startswith("HOSTPHASES") and "prepare_inputs=" in ln]
    assert len(ph) == 1 and "prepare_inputs=827/827/827" in ph[0], log.lines
    if not subs:
        _assert_off(log)
        return
    assert _subs_line(log, "~pi_seed_sync") == "~pi_collect=50/50/50/1 ~pi_seed_sync=777/777/777/1"


def test_seed_sync_absent_when_no_crossing_waits(monkeypatch):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    m = _seed_copies_module()
    log, clock = _reset(monkeypatch, every=1, subs=True)
    fake, _ = m._make_self(["a"], [100], [[[5, 6, 7, 8]]], state_pos={"a": 0}, state_block_size=384)
    fake.scheduler_config = NS(async_scheduling=True)
    fake.speculative_config = NS(num_speculative_tokens=3)

    def wait():
        raise AssertionError("no wait away from a boundary")

    fake._pre_async_results = NS(req_id_to_index_copy={"a": 0}, num_draft_per_req={0: 3},
                                 wait_for_copy=wait)
    host_phases.begin_step()
    host_phases.mark("prep_attention_metadata")
    TPUModelRunner._collect_mamba_state_seed_copies(fake, NS(num_scheduled_tokens={"a": 4}), 0, 1)
    clock.tick(4)
    host_phases.mark("prepare_inputs")
    host_phases.begin_step()
    d = _subs_dict(_subs_line(log, "~pi_collect"))
    assert "pi_seed_sync" not in d and d["pi_collect"] == "0/0/0/1", d


@pytest.mark.parametrize("subs", [True, False])
@pytest.mark.parametrize("step_idx", [0, 1])
def test_df0_call_is_exactly_the_draft_model_call(monkeypatch, subs, step_idx):
    import vllm_torchtpu.spec_decode.eagle3 as e3
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    log, clock = _reset(monkeypatch, every=1, subs=subs)
    monkeypatch.setattr(e3, "set_model_tag", lambda _tag: contextlib.nullcontext())
    monkeypatch.setattr(e3, "set_forward_context", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(e3, "set_vllm_model_wrapper_context", lambda **k: contextlib.nullcontext())
    monkeypatch.setattr(e3._ffc, "enabled", lambda: False)
    hidden = torch.ones(3, 2)
    calls = []

    def model(**kwargs):
        calls.append(kwargs["input_ids"].shape)
        clock.tick(500)
        return hidden

    fake = NS(
        _build_draft_attn_metadata=lambda **kw: (clock.tick(16), {"draft": "md"})[1],
        speculative_config=NS(method="eagle3"),
        _draft_model_for_step=lambda step: model,
        draft_vllm_config=object(),
        runner=NS(_dp_num_tokens_across_dp=lambda n: n, mesh=object()),
        _draft_moe_ctx=lambda: contextlib.nullcontext(),
        _mtp_loop_model=None,
        _mtp_loop_model_warm=True,
        _unwrap_model_out=Eagle3Proposer._unwrap_model_out,
    )
    host_phases.begin_step()
    host_phases.mark("dp_prepare_inputs")
    last, carry = Eagle3Proposer._forward_draft(
        fake, chunk=NS(), input_ids=torch.zeros(3, dtype=torch.int32),
        positions=torch.arange(3, dtype=torch.int32), target_hidden_states=hidden,
        step_idx=step_idx, seq_lens_delta=0, num_rejected_np=None,
    )
    clock.tick(20)
    host_phases.sub("df0_rest")  # propose's own closing sub (structure test below)
    host_phases.mark("dp_forward0")
    host_phases.begin_step()
    assert len(calls) == 1 and torch.equal(last, hidden) and torch.equal(carry, hidden)
    if not subs:
        _assert_off(log)
        return
    line = _subs_line(log, "~df0_rest")
    if step_idx == 0:
        assert line == "~df0_call=500/500/500/1 ~df0_rest=20/20/20/1", line
        ph = [ln for ln in log.lines if "dp_forward0=" in ln][0]
        assert "dp_fwd0_metadata=16/16/16" in ph and "dp_forward0=520/520/520" in ph, ph
    else:
        # Later draft steps run no step-0 marks and no ~df0_call.
        assert line == "~df0_rest=536/536/536/1", line


@pytest.mark.parametrize("subs", [True, False])
def test_adapter_sub_marks_bracket_their_calls(monkeypatch, subs):
    """Qwen4ExpAdapter._prepare_inputs_fused (VLLM_TPU_QWEN4_FUSED_PREP, n-gram
    window on): window build 40 us, each H2D 70 us (window, then the scalars),
    the compiled call 300 us, the table views 9 us."""
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import inputs as inputs_mod
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
    from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot

    log, clock = _reset(monkeypatch, every=1, subs=subs)
    rows, max_reqs, max_len, num_tokens = 4, 6, 64, 16
    rng = np.random.default_rng(38)
    runner = NS(device="cpu", max_num_reqs=max_reqs)
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    runner.vllm_config = NS(model_config=runner.model_config, parallel_config=NS(pipeline_parallel_size=1))
    runner.input_batch = NS(
        token_ids_cpu_tensor=torch.from_numpy(rng.integers(100, 999, (max_reqs, max_len)).astype(np.int32)),
        num_computed_tokens_cpu=np.zeros(max_reqs, dtype=np.int64),
        req_ids=["a", "b", "c"],
    )
    adapter = Qwen4ExpAdapter(runner)
    adapter.pooled = False
    adapter.spec_window = 4
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_NGRAM_WINDOW", "1", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP_CHECK", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", False, raising=False)
    real_window = inputs_mod.ngram_window_from_host
    real_upload = adapter._host_upload
    real_tables = adapter._block_tables
    out4 = tuple(torch.zeros(rows, dtype=torch.int32) for _ in range(4))
    fused_calls = []

    def fused(*args, **kwargs):
        fused_calls.append(kwargs["window"])
        clock.tick(300)
        return out4

    monkeypatch.setattr(inputs_mod, "ngram_window_from_host",
                        lambda *a, **k: (clock.tick(40), real_window(*a, **k))[1])
    monkeypatch.setattr(adapter, "_host_upload", lambda arr: (clock.tick(70), real_upload(arr))[1])
    monkeypatch.setattr(adapter, "_fused_fn", lambda: fused)
    monkeypatch.setattr(adapter, "_block_tables", lambda snap, r: (clock.tick(9), real_tables(snap, r))[1])
    attention = f"{adapter.attention_prefix}.self_attn"
    tables = {
        adapter._main_owner(attention): NS(block_tables=torch.arange(rows * 3).reshape(rows, 3)),
        adapter._compressed_owner(attention): NS(block_tables=torch.arange(rows * 2).reshape(rows, 2)),
    }
    starts = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)
    runner.input_batch.num_computed_tokens_cpu[:3] = [20, 21, 22]
    runner._attn_metadata_builder_ctx = NS(
        query_start_loc=starts,
        seq_lens=torch.tensor([24, 25, 26, 0], dtype=torch.int32),
        mamba_state_indices=torch.tensor([5, 3, 1, 0], dtype=torch.int32),
    )
    ids = torch.from_numpy(rng.integers(100, 999, num_tokens).astype(np.int32))
    store = adapter.begin_step()
    host_phases.begin_step()
    host_phases.mark("seed_reset_subst")
    out = adapter.prepare_inputs(ModelInputSnapshot(store, 0, 3, num_tokens, tables, ids))
    clock.tick(3)
    host_phases.sub("ap_rest")  # execute_model's own closing sub (structure test below)
    host_phases.mark("adapter_prepare_inputs")
    host_phases.begin_step()
    assert fused_calls == [True] and out["state_slots"] is out4[2]
    if not subs:
        _assert_off(log)
        return
    assert _subs_line(log, "~ap_fused") == (
        "~ap_scalars=0/0/0/1 ~ap_wbuild=40/40/40/1 ~ap_wh2d=70/70/70/1 ~ap_sup=70/70/70/1 "
        "~ap_fused=300/300/300/1 ~ap_tables=9/9/9/1 ~ap_rest=3/3/3/1"
    )


CARRY_ENV = (
    "VLLM_TPU_L6_DECODE_CARRY", "VLLM_TPU_L6_DECODE_CARRY_CHECK", "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_CLASSES", "VLLM_TPU_L6_ENVS_CACHE", "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
)
PREP_SUBS = ["pi_entry", "pi_fence", "pi_tokens", "pi_lhost", "pi_carry", "pi_h2d", "pi_spec_qsl",
             "pp_mamba", "pp_subst_idx", "pp_key", "pp_fused", "pp_rest", "pi_subst2", "pi_cross"]


@pytest.mark.parametrize("mode", ["carry", "upload", "off"])
def test_prepare_inputs_sub_marks_through_the_carry_harness(monkeypatch, mode):
    """The real _prepare_inputs (L6-CARRY's CPU harness, async spec decode,
    POSTFWD_FUSED): every ~pi_* / ~pp_* split appears once per step, the carry
    path adds ~pp_record, and the census off records nothing."""
    sys.path.insert(0, os.path.join(TESTS, "q38_l6"))
    try:
        import l6_carry_harness as H
    finally:
        sys.path.pop(0)
    from vllm_torchtpu.core import l6_ab

    for k in CARRY_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    if mode == "carry":
        monkeypatch.setenv("VLLM_TPU_L6_DECODE_CARRY", "1")
    l6_ab._resolve()
    undo = H.install_eager_programs()
    try:
        log, _ = _reset(monkeypatch, every=3, subs=mode != "off", virtual=False)
        r = H.make_runner(seed=0, nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True)
        sim = H.Sim(r, nreq=3, seed=0)
        for _ in range(7):
            host_phases.begin_step()
            host_phases.mark("kv_connector")
            sim.step(())
            host_phases.mark("prepare_inputs")
        host_phases.begin_step()
    finally:
        undo()
        for k in CARRY_ENV:
            monkeypatch.delenv(k, raising=False)
        l6_ab._resolve()
    ph = [ln for ln in log.lines if ln.startswith("HOSTPHASES")]
    assert len(ph) == 2 and all("prep_host_and_uploads=" in ln for ln in ph), log.lines
    if mode == "off":
        _assert_off(log)
        return
    subs = [ln for ln in log.lines if ln.startswith("HOSTSUBS")]
    assert len(subs) == 2, log.lines
    d = _subs_dict(subs[1].split("(p50/p90/mean-per-step/n-steps) ", 1)[1])
    # The second window holds three steps (harness steps 2..4, all carried with the lever on).
    want = PREP_SUBS + (["pp_record"] if mode == "carry" else [])
    assert {k: d.get(k, "ABSENT").rsplit("/", 1)[-1] for k in want} == {k: "3" for k in want}, d
    assert "pi_layout" not in d and "pp_upload" not in d, d
    if mode == "upload":
        assert "pp_record" not in d, d


# ------------------------------------------------------------------ structure


def _fn(path, name):
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not in {path}")


def _hp_seq(fn):
    """'M:<name>' / 'S:<name>' for every host_phases.mark / sub with a literal
    name, in source order."""
    calls = []
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "host_phases"
                and n.func.attr in ("mark", "sub") and n.args and isinstance(n.args[0], ast.Constant)):
            calls.append((n.lineno, n.col_offset, ("M:" if n.func.attr == "mark" else "S:") + n.args[0].value))
    return [c[2] for c in sorted(calls)]


def _run(seq, want):
    """`want` occurs in `seq` as one contiguous run."""
    i = seq.index(want[0])
    assert seq[i:i + len(want)] == want, seq[i:i + len(want) + 3]


def _stmt_lists(fn):
    for n in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            lst = getattr(n, field, None)
            if isinstance(lst, list) and lst and isinstance(lst[0], ast.stmt):
                yield lst


def _is_sub(stmt, name):
    return isinstance(stmt, ast.Expr) and ast.unparse(stmt) == f"host_phases.sub('{name}')"


def _calls(stmt):
    return {ast.unparse(c.func).split(".")[-1] for c in ast.walk(stmt) if isinstance(c, ast.Call)}


def _bracketed(fn, before, after):
    """Statements with sub(before) right before them and sub(after) right after."""
    out = []
    for lst in _stmt_lists(fn):
        for i in range(1, len(lst) - 1):
            if _is_sub(lst[i - 1], before) and _is_sub(lst[i + 1], after):
                out.append(lst[i])
    return out


def _span(fn, before, after):
    """q38 L6-V5: the statements between sub(before) and the next sub(after) in one statement list (one entry per
    sub(before) that has a later sub(after) in its list)."""
    out = []
    for lst in _stmt_lists(fn):
        for i, s in enumerate(lst):
            if _is_sub(s, before):
                for j in range(i + 1, len(lst)):
                    if _is_sub(lst[j], after):
                        out.append(lst[i + 1:j])
                        break
    return out


def _is_call_stmt(stmt, func):
    """An expression statement that is a direct call to `func` (not a block
    that merely contains one)."""
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and ast.unparse(stmt.value.func) == func)


def _next_after(fn, pred):
    for lst in _stmt_lists(fn):
        for i, s in enumerate(lst[:-1]):
            if pred(s):
                return lst[i + 1]
    raise AssertionError("statement not found")


def test_seed_sync_bracket_in_source():
    fn = _fn(RUNNER, "_collect_mamba_state_seed_copies")
    hits = _bracketed(fn, "pi_collect", "pi_seed_sync")
    assert [ast.unparse(s) for s in hits] == ["sampled_cpu = previous.wait_for_copy()"]
    assert _hp_seq(fn) == ["S:pi_collect", "S:pi_seed_sync", "S:pi_collect"]


def test_prepare_inputs_split_order():
    _run(_hp_seq(_fn(RUNNER, "_prepare_inputs")), [
        "S:pi_entry", "S:pi_fence", "S:pi_tokens", "S:pi_lhost", "S:pi_carry", "S:pi_h2d",
        "S:pi_spec_qsl", "S:pp_mamba", "S:pp_rest", "M:prep_host_and_uploads",
        "M:prep_attention_metadata", "S:pi_subst2", "S:pi_cross"])
    fn = _fn(RUNNER, "_prepare_inputs")
    nxt = _next_after(fn, lambda s: "_collect_mamba_state_seed_copies" in _calls(s))
    assert _is_sub(nxt, "pi_cross")
    fn = _fn(RUNNER, "_postfwd_fused_prepare")
    seq = _hp_seq(fn)
    assert seq == ["S:pp_subst_idx", "S:pp_key", "S:pp_fused"], seq
    lines = {s: n.lineno for n in ast.walk(fn) if isinstance(n, ast.Expr)
             for s in [ast.unparse(n)] if s.startswith("host_phases.sub")}
    call = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "_postfwd_subst_fused"]
    assert len(call) == 1
    assert lines["host_phases.sub('pp_key')"] < call[0] < lines["host_phases.sub('pp_fused')"]


def test_carry_sub_marks_bracket_the_carry_program():
    fn = _fn(CARRY, "_run_carry")
    # q38 L6-V5: with L6-ADC composed, ~pp_key / ~pp_fused bracket the carry program choice (the merged carry program
    # when adcarry is set, else carry_fused), several statements instead of one; both calls must sit inside.
    spans = _span(fn, "pp_key", "pp_fused")
    assert len(spans) == 1 and spans[0], [[ast.unparse(s) for s in sp] for sp in spans]
    calls = set().union(*(_calls(s) for s in spans[0]))
    assert "carry_fused" in calls, calls
    merged = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("run_merged")]
    assert not merged or "run_merged" in calls, calls
    assert _is_sub(_next_after(fn, lambda s: _is_call_stmt(s, "_record")), "pp_record")
    assert _hp_seq(fn) == ["S:pp_subst_idx", "S:pp_key", "S:pp_fused", "S:pp_record"]
    fn = _fn(CARRY, "fused_prepare")
    assert _is_sub(_next_after(fn, lambda s: ast.unparse(s) == "seq_lens = upload(runner, ctx)"), "pp_upload")
    assert _is_sub(_next_after(fn, lambda s: _is_call_stmt(s, "_record")), "pp_record")


def test_adapter_split_in_execute_model_and_the_adapter():
    fn = _fn(RUNNER, "execute_model")
    _run(_hp_seq(fn), ["M:seed_reset_subst", "S:ap_drafts", "S:ap_fence", "S:ap_inputs", "S:ap_rest",
                       "M:adapter_prepare_inputs", "M:forward_dispatch"])
    assert _is_sub(_next_after(fn, lambda s: ast.unparse(s) == "self._record_input_staging_fence()"),
                   "ap_fence")
    for lst in _stmt_lists(fn):
        idx = [i for i, s in enumerate(lst) if _is_sub(s, "ap_inputs") or _is_sub(s, "ap_rest")]
        if len(idx) == 2:
            between = lst[idx[0] + 1:idx[1]]
            assert [ast.unparse(s).split("\n")[0] for s in between] == [
                "model_kwargs = None", "if self.model_adapter is not None:"]
            assert "prepare_inputs" in _calls(between[1])
            break
    else:
        raise AssertionError("ap_inputs / ap_rest not in one block")
    fn = _fn(INPUTS, "_prepare_inputs_fused")
    assert _hp_seq(fn) == ["S:ap_scalars", "S:ap_wbuild", "S:ap_wh2d", "S:ap_sup", "S:ap_fused", "S:ap_tables"]
    (call,) = _bracketed(fn, "ap_sup", "ap_fused")
    assert ast.unparse(call).startswith("out = fused(")
    (up,) = _bracketed(fn, "ap_wbuild", "ap_wh2d")
    assert ast.unparse(up) == "store = self._host_upload(window_np)"


def test_sample_setup_census_in_source():
    fn = _fn(RUNNER, "sample_tokens")
    _run(_hp_seq(fn), ["M:between_execute_and_sample", "S:ss_attrs", "S:ss_draft_chunks", "S:ss_rest",
                       "M:sample_setup"])
    (hand,) = _bracketed(fn, "ss_attrs", "ss_draft_chunks")
    assert ast.unparse(hand) == ("if self._is_async_drafter:\n"
                                 "    self._async_drafter.draft_chunks = state.draft_chunks")


def test_draft_sub_marks_in_source():
    fn = _fn(EAGLE, "propose")
    _run(_hp_seq(fn), ["M:dp_target_hidden", "S:dpi_rest", "M:dp_prepare_inputs", "S:df0_rest",
                       "M:dp_forward0", "S:dc_rest", "S:dc_call", "S:dc_rest", "S:dc_call", "S:dc_rest",
                       "M:dp_carries", "M:dp_token0"])
    hits = _bracketed(fn, "dc_rest", "dc_call")
    assert sorted(c for h in hits for c in _calls(h) if c.startswith("_draft_rows_carry")) == [
        "_draft_rows_carry_and_token", "_draft_rows_carry_token_fresh"]
    prev = [lst[i - 1] for lst in _stmt_lists(fn) for i, s in enumerate(lst) if _is_sub(s, "df0_rest")]
    assert len(prev) == 1 and "_forward_draft" in _calls(prev[0])
    prev = [lst[i - 1] for lst in _stmt_lists(fn) for i, s in enumerate(lst)
            if _is_sub(s, "dpi_rest") and i > 0]
    assert len(prev) == 1 and "_prepare_draft_inputs" in _calls(prev[0])
    fn = _fn(EAGLE, "_prepare_draft_inputs")
    hits = _bracketed(fn, "dpi_rest", "dpi_seed")
    seeds = sorted(c for h in hits for c in _calls(h)
                   if c in ("_async_seed_inputs_compiled", "_async_draft_seed", "_draft_seed_input_ids"))
    assert seeds == ["_async_draft_seed", "_async_seed_inputs_compiled", "_draft_seed_input_ids",
                     "_draft_seed_input_ids"], seeds
    assert _hp_seq(fn).count("S:dpi_seed") == 4
    fn = _fn(EAGLE, "_forward_draft")
    nxt = _next_after(fn, lambda s: ast.unparse(s) == "out = model(**kwargs)")
    assert ast.unparse(nxt) == "if step_idx == 0:\n    host_phases.sub('df0_call')"


def _literal_subs(path):
    names = []
    for n in ast.walk(ast.parse(open(path).read())):
        if (isinstance(n, ast.Call) and ast.unparse(n.func) == "host_phases.sub"
                and n.args and isinstance(n.args[0], ast.Constant)):
            names.append(n.args[0].value)
    return set(names)


def test_source_carries_exactly_the_listed_sub_marks():
    got = set().union(*(_literal_subs(p) for p in (RUNNER, CARRY, EAGLE, INPUTS, DRIVER)))
    want = OLD_SUBS | set(NEW_SUBS) | PMETER_SUBS
    assert got == want, (sorted(got - want), sorted(want - got))
    assert "pi_layout" not in got  # split into pi_lhost / pi_carry / pi_h2d


def test_reader_maps_every_sub_mark_to_its_phase():
    path = os.path.join(TESTS, "q38_hostsub", "q38_hostsub_read.py")
    spec = importlib.util.spec_from_file_location("q38_l6instr_reader", path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    assert {k: reader.phase_of(k) for k in NEW_SUBS} == NEW_SUBS
    assert reader.phase_of("pi_tokens") == "prep_host_and_uploads"
    assert reader.phase_of("pf_logits") == "post_forward"


REPLAY = [(RUNNER, "_prepare_inputs"), (RUNNER, "_collect_mamba_state_seed_copies"),
          (RUNNER, "_postfwd_fused_prepare"), (RUNNER, "execute_model"), (RUNNER, "sample_tokens"),
          (CARRY, "_run_carry"), (CARRY, "fused_prepare"), (INPUTS, "_prepare_inputs_fused"),
          (EAGLE, "propose"), (EAGLE, "_prepare_draft_inputs"), (EAGLE, "_forward_draft")]


@pytest.mark.parametrize("subs", [True, False])
def test_replay_every_instrumented_function_reaches_the_census(monkeypatch, subs):
    """Each function's host_phases calls, in source order, replayed one step
    per function: every new sub-mark is on the HOSTSUBS line with the census
    on; with it off there is no HOSTSUBS line and HOSTPHASES is unchanged."""
    lines = {}
    for on in (subs, False):
        log, clock = _reset(monkeypatch, every=1, subs=on)
        seen = set()
        host_phases.begin_step()
        for path, name in REPLAY:
            for item in _hp_seq(_fn(path, name)):
                clock.tick(3)
                kind, label = item.split(":", 1)
                (host_phases.mark if kind == "M" else host_phases.sub)(label)
                seen.add(label)
            host_phases.mark("replay_" + name)
        host_phases.begin_step()
        lines[on] = log
    log = lines[subs]
    assert set(NEW_SUBS) <= seen
    ph_on = [ln for ln in log.lines if ln.startswith("HOSTPHASES")]
    ph_off = [ln for ln in lines[False].lines if ln.startswith("HOSTPHASES")]
    assert ph_on == ph_off
    if not subs:
        _assert_off(log)
        return
    d = _subs_dict(_subs_line(log, "~ss_draft_chunks"))
    assert set(NEW_SUBS) <= set(d), sorted(set(NEW_SUBS) - set(d))


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6INSTR_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6INSTR_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)

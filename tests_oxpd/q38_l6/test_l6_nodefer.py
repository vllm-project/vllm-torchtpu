# SPDX-License-Identifier: Apache-2.0
"""q38 L6 NODEFER census (host_levers_h2.md section 4.1 row 3): runner/l6_nodefer.py
and the ~vr_draw / ~vr_call / ~vr_post split of vfy_rejection.

Real TPUModelRunner._sample_spec_verify_vps (stock VPS call and the DEC-FB-2
fallback call) on a virtual clock with the compiled calls stubbed: the sub-marks
bracket exactly the pre-call work, the call and the post-call work; the census
counts the call's arguments a torch_tpu probe stand-in reports as not yet
materializing (the fresh uniform draws), on the sampled calls only (1st,
(E+1)-th, ...), and records nothing with the lever off.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import host_phases, l6_nodefer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src", "vllm_torchtpu", "runner", "tpu_runner.py")
P, K, V = 4, 3, 64


class _Clk:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def tick(self, us):
        self.ns += us * 1000


@pytest.fixture
def nd(monkeypatch):
    import vllm_torchtpu.runner.tpu_runner as rm

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)
    log = []
    for name, v in (("_EVERY", 1), ("_SLOW_NS", 0), ("_SUBS", True), ("_GC", False), ("_last", None),
                    ("_steps", 0), ("_samples", {}), ("_cur", {}), ("_ctx", []), ("_sub_last", None),
                    ("_sub_acc", {}), ("_sub_samples", {}), ("_rank", 0)):
        monkeypatch.setattr(host_phases, name, v)
    monkeypatch.setattr(host_phases, "logger", NS(info=lambda fmt, *a: log.append(fmt % a)))
    clk = _Clk()
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", clk)
    # probe stand-in: a tensor counts as materializing once marked (no id() bookkeeping: freed tensors' ids recur)
    def mark(t):
        t._q38_started = True
        return t

    def is_materializing(t):
        return bool(getattr(t, "_q38_started", False))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.sync", NS(is_materializing=is_materializing))

    # the compiled programs inside the real wrappers; a call starts materializing its arguments
    def stock(local_logits, shard_offset, *args):
        clk.tick(200)
        for a in (local_logits, *args):
            if isinstance(a, torch.Tensor):
                mark(a)
        return torch.zeros(P, K + 1, dtype=torch.int64)

    def fb(local_logits, shard_offset, *args):
        clk.tick(250)
        for a in (local_logits, *args):
            if isinstance(a, torch.Tensor):
                mark(a)
        return torch.zeros(P, K + 1, dtype=torch.int64), torch.zeros(3, dtype=torch.int32)
    monkeypatch.setattr(rm._vps_verify_compiled, "fn", stock)
    monkeypatch.setattr(rm._vps_verify_fb_compiled, "fn", fb)

    def stats_every():
        clk.tick(15)
        return 0
    monkeypatch.setattr(rm.vps_wiring, "stats_every", stats_every)

    def setup(env=None, fb_path=False):
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        l6_nodefer._resolve()
        if l6_ab.ACTIVE:
            l6_ab.begin_step()
        logits = torch.randn(P * (K + 1), V)
        md = NS(bonus_logits_indices=torch.arange(P), target_logits_indices=torch.arange(P * K),
                draft_token_ids=torch.zeros(P, K, dtype=torch.int64), draft_lengths=torch.full((P,), K),
                segment_ids=torch.zeros(P * K, dtype=torch.int64), group_indices=torch.zeros(P, dtype=torch.int64))
        temps, top_k, top_p = torch.ones(P), torch.zeros(P, dtype=torch.int32), torch.ones(P)

        def upload(name, t):
            clk.tick(30)
            return mark(t)
        fake = NS(_vps_head=NS(full_vocab=V, tp_size=8, shard_offset=0),
                  rejection_sampler=NS(synthetic_conditional_rates=None, synthetic_mode=0),
                  speculative_config=NS(num_speculative_tokens=K), _vps_cand_draws=fb_path,
                  _padded_sampling_params=lambda *a: (temps, top_k, top_p), _vps_fb_path=fb_path,
                  _vps_checks_left=0, _upload_unless_unchanged=upload, _vps_fb_stats=None, _vps_fb_chunks=0,
                  _vps_fb_max=8, _vps_tiegap=0)
        fake._vps_verify_fb = lambda *a, **k: rm.TPUModelRunner._vps_verify_fb(fake, *a, **k)
        for x in (logits, md.bonus_logits_indices, md.target_logits_indices, md.draft_token_ids, md.draft_lengths,
                  md.segment_ids, md.group_indices, temps, top_k, top_p):
            mark(x)
        return fake, logits, md
    yield setup, log, clk
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    l6_nodefer._resolve()


def _call(fake, logits, md, clk, fb_rows=0):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    host_phases.begin_step()
    host_phases.mark("sample_setup")
    out = TPUModelRunner._sample_spec_verify_vps(fake, logits, md, 0, P, None, fallback_rows=fb_rows)
    clk.tick(5)
    host_phases.sub("vr_post")  # sample_tokens' tail (test_census_wrapper_and_vr_post_in_source pins it)
    host_phases.mark("vfy_rejection")
    host_phases.begin_step()
    return out


def _subs(log):
    got = [ln for ln in log if ln.startswith("HOSTSUBS") and "~vr_" in ln]
    assert len(got) == 1, log
    return got[0].split("(p50/p90/mean-per-step/n-steps) ", 1)[1]


def _phase(log, name):
    got = [ln for ln in log if ln.startswith("HOSTPHASES") and f" {name}=" in ln]
    assert len(got) == 1, log
    return got[0].split(f" {name}=", 1)[1].split()[0]


def _counts():
    return {k: v for k, v in l6_ab._state["counts"].items() if k.startswith("nodefer.")}


def test_fallback_call_split_and_census(nd):
    setup, log, clk = nd
    fake, logits, md = setup({"VLLM_TPU_L6_NODEFER_CENSUS": "1", "VLLM_TPU_L6_NODEFER_EVERY": "1"}, fb_path=True)
    _call(fake, logits, md, clk, fb_rows=2)
    # upload 30 before the call; the call 250; the fallback stats 15 after it + 5 more before the mark
    assert _subs(log) == "~vr_draw=30/30/30/1 ~vr_call=250/250/250/1 ~vr_post=20/20/20/1"
    assert _phase(log, "vfy_rejection") == "300/300/300"
    # positions: 7 bonus_u, 8 accept_u, 9 recover_u, 20 fb_u are fresh draws (deferred); the rest is known
    assert _counts() == {"nodefer.calls": 1, "nodefer.sampled": 1, "nodefer.deferred": 4,
                         "nodefer.a7": 1, "nodefer.a8": 1, "nodefer.a9": 1, "nodefer.a20": 1}, _counts()


def test_stock_call_split_and_sampling_every_e(nd):
    setup, log, clk = nd
    fake, logits, md = setup({"VLLM_TPU_L6_NODEFER_CENSUS": "1", "VLLM_TPU_L6_NODEFER_EVERY": "2"})
    _call(fake, logits, md, clk)
    assert _subs(log) == "~vr_draw=0/0/0/1 ~vr_call=200/200/200/1 ~vr_post=5/5/5/1"
    for _ in range(2):
        _call(fake, logits, md, clk)
    assert _counts() == {"nodefer.calls": 3, "nodefer.sampled": 2, "nodefer.deferred": 6,
                         "nodefer.a7": 2, "nodefer.a8": 2, "nodefer.a9": 2}, _counts()


def test_off_records_nothing(nd):
    setup, log, clk = nd
    fake, logits, md = setup({"VLLM_TPU_L6_PMETER": "1"})  # another L6 lever on, nodefer off
    _call(fake, logits, md, clk)
    assert _counts() == {}


def test_probe_unavailable_is_counted_not_raised(nd, monkeypatch):
    setup, log, clk = nd
    fake, logits, md = setup({"VLLM_TPU_L6_NODEFER_CENSUS": "1"})

    def no_sync():
        raise ImportError("no torch_tpu here")
    monkeypatch.setattr(l6_nodefer, "_sync", no_sync)
    _call(fake, logits, md, clk)
    assert _counts() == {"nodefer.calls": 1, "nodefer.nosync": 1}


def test_census_wrapper_and_vr_post_in_source():
    from vllm_torchtpu.runner import tpu_runner as rm

    stmts = [ln.strip() for ln in open(RUNNER).read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert "_vps_verify_compiled = _l6_nodefer.wrap(_static_shape_args(_tpu_compile(_vps_verify_impl)))" in stmts
    assert "_vps_verify_fb_compiled = _l6_nodefer.wrap(_static_shape_args(_tpu_compile(_vps_verify_fb_impl)))" in stmts
    assert isinstance(rm._vps_verify_compiled, l6_nodefer.VfyCall)
    assert isinstance(rm._vps_verify_fb_compiled, l6_nodefer.VfyCall)
    k = stmts.index('host_phases.mark("vfy_rejection")')
    assert stmts[k - 1] == 'host_phases.sub("vr_post")' and stmts.count('host_phases.sub("vr_post")') == 1
    # the runner's VPS method bodies are the base text (the MM / DEC-FB-2 harnesses exec them in their own namespace)
    assert not any("_l6_nodefer" in s or "vr_draw" in s or "vr_call" in s for s in stmts
                   if not s.startswith("_vps_verify") and not s.startswith("from vllm_torchtpu.runner import l6_nodefer"))


def test_registry_rows():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert l6_ab.LEVERS["nodefer"] == "VLLM_TPU_L6_NODEFER_CENSUS"
    for name, default in (("VLLM_TPU_L6_NODEFER_CENSUS", ""), ("VLLM_TPU_L6_NODEFER_EVERY", 64)):
        assert name in _TPU_COMPILE_ENV_IGNORED
        if name not in os.environ:
            assert getattr(envs, name) == default

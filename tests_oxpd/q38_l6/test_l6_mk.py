# SPDX-License-Identifier: Apache-2.0
"""q38 L6-MK levers mkctx (D14 driver whole-step call trims) and mkthin (torch_tpu single-partition
thin dispatch): bitwise equality with the stock path over random served-shaped streams, A/B
windows, CHECK modes, refusals, registration (runner/l6_mk.py). CDK CPU pod only."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l6_mk_harness as H  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.runner import l6_mk  # noqa: E402

COUNTS: dict = {}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    def wipe():
        for k in list(os.environ):
            if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_SERVED_AB"):
                monkeypatch.delenv(k, raising=False)
        l6_ab._resolve()
        l6_mk._reset_for_tests()
        COUNTS.clear()

    wipe()
    orig_log = l6_ab._log_window

    def log_window():  # keep every window's counts for the assertions
        for k, v in l6_ab._state["counts"].items():
            COUNTS[k] = COUNTS.get(k, 0) + v
        orig_log()

    monkeypatch.setattr(l6_ab, "_log_window", log_window)
    yield monkeypatch
    wipe()
    try:
        l6_mk.uninstall_mkthin_for_tests()
    except ImportError:
        pass


def _set(mp, **env):
    for k, v in env.items():
        mp.setenv(k, v)
    l6_ab._resolve()
    l6_mk._reset_for_tests()


def counts() -> dict:
    out = dict(COUNTS)
    for k, v in l6_ab._state["counts"].items():
        out[k] = out.get(k, 0) + v
    return out


# --------------------------------------------------------------------------------------- mkctx
SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_mkctx_matches_stock_over_random_streams(clean, seed):
    c_off, c_on = {}, {}
    off = H.mkctx_stream(seed, counts=c_off)
    _set(clean, VLLM_TPU_L6_MK_CTX="1")
    on = H.mkctx_stream(seed, counts=c_on)
    assert on[0] == off[0], "outputs / routes differ"
    assert on[1] == off[1], "program calls (tag, config, inputs) or shadow calls differ"
    assert on[2] == off[2], "driver stats / counters / budgets differ"
    whole = off[2][2]
    assert whole > 50, whole
    got = counts()
    # Engagement: every whole-step call took the lever's path; fewer numpy reads than stock.
    assert got.get("mkctx.whole", 0) == whole, (got, whole)
    assert c_on["whole_eligible"] < c_off["whole_eligible"], (c_on, c_off)
    assert c_on["real_traffic"] < c_off["real_traffic"], (c_on, c_off)
    print(f"ENGAGEMENT mkctx seed={seed} whole={whole} whole_eligible {c_off['whole_eligible']}->"
          f"{c_on['whole_eligible']} real_traffic {c_off['real_traffic']}->{c_on['real_traffic']}")


def test_mkctx_whole_calls_run_under_the_stock_tag_and_config(clean):
    _set(clean, VLLM_TPU_L6_MK_CTX="1")
    _, rec, _ = H.mkctx_stream(5, steps=200)
    whole = [r for r in rec if r[0] == "whole"]
    seg = [r for r in rec if r[0] == "seg"]  # the segment program runs only off the whole-step route
    assert whole
    assert {(r[1], r[2]) for r in whole} == {("model_mk_whole", "whole_cfg")}
    assert {(r[1], r[2]) for r in seg} <= {("model_mk_segments", "seg_cfg")}


def test_mkctx_ab_windows_match_stock(clean):
    off = H.mkctx_stream(7)
    _set(clean, VLLM_TPU_L6_MK_CTX="ab:1", VLLM_TPU_L6_AB_STEPS="7", VLLM_TPU_L6_AB_CONFIRM="2")
    on = H.mkctx_stream(7)
    assert on == off
    n = counts().get("mkctx.whole", 0)
    assert 0 < n < off[2][2], (n, off[2][2])


def test_mkctx_noaot_falls_back_to_the_stock_contexts(clean):
    off = H.mkctx_stream(3, noaot=True)
    _set(clean, VLLM_TPU_L6_MK_CTX="1")
    on = H.mkctx_stream(3, noaot=True)
    assert on == off


def test_mkctx_lever_off_never_builds_the_cache(clean, monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

    built = []
    orig = l6_mk.whole_contexts
    monkeypatch.setattr(l6_mk, "whole_contexts", lambda *a: built.append(1) or orig(*a))
    H.mkctx_stream(2, steps=120)
    assert built == []
    _set(clean, VLLM_TPU_L6_MK_CTX="1")
    H.mkctx_stream(2, steps=120)
    assert built, "the lever never entered the cached contexts"
    assert drv is not None


def test_mkctx_check_mode_finds_no_mismatch(clean):
    off = H.mkctx_stream(4)
    _set(clean, VLLM_TPU_L6_MK_CTX="1", VLLM_TPU_L6_MK_CTX_CHECK="1")
    on = H.mkctx_stream(4)
    assert on == off
    got = counts()
    assert got.get("mkctx.check_ok", 0) == off[2][2] and got.get("mkctx.mismatch", 0) == 0, got


def test_mkctx_check_catches_a_broken_eligibility_memo(clean):
    """An eligibility that changes between its sites (what the memo assumes cannot happen) is caught
    by the CHECK on the first whole-step call; the lever then stays off for the process."""
    import numpy as np
    from types import SimpleNamespace as NS

    from vllm_torchtpu.megakernel_runtime import driver as base_drv
    from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

    _set(clean, VLLM_TPU_L6_MK_CTX="1", VLLM_TPU_L6_MK_CTX_CHECK="1")
    rec = []
    d = H.make_driver(drv, rec)
    flip = [0]

    def we(runner, n_req, rows):
        flip[0] += 1
        return flip[0] % 2 == 1  # True, False, True, ...

    d._whole_eligible = we
    clean.setattr(base_drv.MegakernelDriver, "_inspect_step_detail", staticmethod(lambda *a: (True, 4, drv.T_VERIFY, "")))
    runner = NS(input_batch=NS(num_computed_tokens_cpu=np.full(8, 100, np.int32)), warmup_done=True,
                vllm_config=NS(label="runner_cfg"))
    for _ in range(3):
        l6_ab.begin_step()
        d.try_forward(runner, torch.zeros(16, dtype=torch.int32), torch.zeros(16, dtype=torch.int32), None, None)
    assert d.__dict__.get("_l6_mkctx_off") is True
    assert counts().get("mkctx.mismatch", 0) == 1


def test_mkctx_off_keeps_the_stock_eligibility_reads(clean, monkeypatch):
    """Lever off: _whole_eligible is read at each site as before (an eligibility that changes between
    the sites routes exactly as stock does)."""
    import numpy as np
    from types import SimpleNamespace as NS

    from vllm_torchtpu.megakernel_runtime import driver as base_drv
    from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

    rec = []
    d = H.make_driver(drv, rec)
    flip = [0]

    def we(runner, n_req, rows):
        flip[0] += 1
        return flip[0] % 2 == 1

    d._whole_eligible = we
    monkeypatch.setattr(base_drv.MegakernelDriver, "_inspect_step_detail", staticmethod(lambda *a: (True, 4, drv.T_VERIFY, "")))
    runner = NS(input_batch=NS(num_computed_tokens_cpu=np.full(8, 100, np.int32)), warmup_done=True,
                vllm_config=NS(label="runner_cfg"))
    d.try_forward(runner, torch.zeros(16, dtype=torch.int32), torch.zeros(16, dtype=torch.int32), None, None)
    assert flip[0] == 2 and rec[-1][0] == "seg", (flip, rec[-1])


def test_mkctx_check_steps_bound_the_check(clean):
    _set(clean, VLLM_TPU_L6_MK_CTX="1", VLLM_TPU_L6_MK_CTX_CHECK="1", VLLM_TPU_L6_CHECK_STEPS="50")
    H.mkctx_stream(4, steps=300)
    got = counts()
    assert 0 < got.get("mkctx.check_ok", 0) <= 50, got


# --------------------------------------------------------------------------------------- mkthin
def _thin_on(mp, **extra):
    _set(mp, **{"VLLM_TPU_L6_MK_THIN_TT": "1", **extra})
    assert l6_mk.install_mkthin(), "mkthin did not install (torch_tpu sources differ?)"


def test_mkthin_verified_sources_are_the_pod_torch_tpu():
    """The thin path mirrors these torch_tpu bodies; the pod runs the served image's torch_tpu."""
    got = l6_mk._source_hashes()
    print(f"MKTHIN_SOURCES {got}")
    assert got == l6_mk.THIN_SOURCES


@pytest.mark.parametrize("seed", SEEDS)
def test_mkthin_matches_stock_over_random_streams(clean, seed):
    off = H.mkthin_stream(seed)
    _thin_on(clean)
    on = H.mkthin_stream(seed)
    assert on[0] == off[0], "outputs differ"
    assert on[1] == off[1], "fast-runner arguments differ"
    got = counts()
    n_two = sum(1 for t in off[0] if t[1] == "two_part")
    assert got.get("mkthin.thin", 0) == len(off[0]) - n_two, (got, n_two)
    assert got.get("mkthin.stock", 0) == n_two and got.get("mkthin.plan_refused", 0) == 1, got
    print(f"ENGAGEMENT mkthin seed={seed} calls={len(off[0])} thin={got.get('mkthin.thin', 0)} "
          f"stock={got.get('mkthin.stock', 0)} refused={got.get('mkthin.plan_refused', 0)}")


def test_mkthin_plans(clean):
    log = H.RunnerLog()
    plans = {}
    for name, fn, shapes, parts, as_list in H.thin_graphs():
        exe = H.build_split_exe(fn, log, name, parts, as_list)
        plans[name] = l6_mk.thin_plan(exe)
    assert isinstance(plans["two_part"], str) and "partitions" in plans["two_part"], plans["two_part"]
    for name in ("two_out", "single", "passthrough", "unused"):
        assert not isinstance(plans[name], str), (name, plans[name])
    assert plans["single"].single and plans["single"].unwrap
    assert ("a", 0) in plans["passthrough"].outs
    assert len(plans["unused"].arg_idx) == 3 and plans["unused"].nph == 4


def test_mkthin_ab_windows_match_stock(clean):
    off = H.mkthin_stream(9)
    _thin_on(clean, VLLM_TPU_L6_MK_THIN_TT="ab:0", VLLM_TPU_L6_AB_STEPS="5", VLLM_TPU_L6_AB_CONFIRM="2")
    on = H.mkthin_stream(9)
    assert on == off
    got = counts()
    eligible = sum(1 for t in off[0] if t[1] != "two_part")
    assert 0 < got.get("mkthin.thin", 0) < eligible, (got, eligible)


def test_mkthin_module_hooks_take_the_stock_path(clean):
    _thin_on(clean)
    log = H.RunnerLog()
    name, fn, shapes, parts, as_list = H.thin_graphs()[0]
    exe = H.build_split_exe(fn, log, name, parts, as_list)
    fired = []
    h = exe._split_gm.register_forward_hook(lambda m, a, o: fired.append(1))
    try:
        for _ in range(4):
            l6_ab.begin_step()
            exe([torch.randn(s) for s in shapes])
    finally:
        h.remove()
    assert len(fired) == 4
    assert counts().get("mkthin.thin", 0) == 0


def test_mkthin_refuses_recompile_fn_and_kwargs(clean):
    log = H.RunnerLog()
    name, fn, shapes, parts, as_list = H.thin_graphs()[0]
    exe = H.build_split_exe(fn, log, name, parts, as_list)
    exe.recompile_fn = lambda: exe
    assert "recompile_fn" in l6_mk.thin_plan(exe)


def test_mkthin_check_mode_finds_no_mismatch(clean):
    off = H.mkthin_stream(11)
    _thin_on(clean, VLLM_TPU_L6_MK_THIN_TT_CHECK="1")
    on = H.mkthin_stream(11)
    assert on == off
    got = counts()
    n_two = sum(1 for t in off[0] if t[1] == "two_part")
    assert got.get("mkthin.check_ok", 0) == len(off[0]) - n_two and got.get("mkthin.mismatch", 0) == 0, got


def test_mkthin_check_catches_a_broken_plan(clean):
    """A plan whose output mapping is wrong is caught by the CHECK (which returns torch_tpu's own
    result), and the executable keeps torch_tpu's path from then on."""
    _thin_on(clean, VLLM_TPU_L6_MK_THIN_TT_CHECK="1")
    log = H.RunnerLog()
    name, fn, shapes, parts, as_list = H.thin_graphs()[0]
    exe = H.build_split_exe(fn, log, name, parts, as_list)
    l6_ab.begin_step()
    args = [torch.randn(s) for s in shapes]
    good = exe(list(args))
    plan = exe.__dict__[l6_mk._PLAN_ATTR]
    plan.outs = tuple(reversed(plan.outs))
    l6_ab.begin_step()
    out = exe(list(args))
    assert all(torch.equal(a, b) for a, b in zip(out, good))
    assert plan.off and counts().get("mkthin.mismatch", 0) == 1
    l6_ab.begin_step()
    out = exe(list(args))
    assert all(torch.equal(a, b) for a, b in zip(out, good))


def test_mkthin_check_catches_wrong_arguments(clean):
    """A plan whose argument order is wrong is caught by the CHECK before any thin call."""
    _thin_on(clean, VLLM_TPU_L6_MK_THIN_TT_CHECK="1")
    log = H.RunnerLog()
    name, fn, shapes, parts, as_list = H.thin_graphs()[0]
    exe = H.build_split_exe(fn, log, name, parts, as_list)
    plan = l6_mk.thin_plan(exe)
    assert not isinstance(plan, str)
    plan.arg_idx = tuple(reversed(plan.arg_idx))
    plan.identity = False
    exe.__dict__[l6_mk._PLAN_ATTR] = plan
    args = [torch.randn(s) for s in shapes]
    l6_ab.begin_step()
    exe(list(args))
    assert plan.off and counts().get("mkthin.mismatch", 0) == 1


def test_mkthin_refuses_unverified_sources(clean, monkeypatch):
    _set(clean, VLLM_TPU_L6_MK_THIN_TT="1")
    from torch_tpu._internal.compile import split_compiler as sc

    before = sc._SplitCompiledExecutable.__dict__["__call__"]
    monkeypatch.setitem(l6_mk.THIN_SOURCES, "_WrapperModule.forward", "0000000000000000")
    assert l6_mk.install_mkthin() is False
    assert sc._SplitCompiledExecutable.__dict__["__call__"] is before


def test_mkthin_off_leaves_torch_tpu_unpatched(clean):
    from torch_tpu._internal.compile import split_compiler as sc

    l6_mk.install_worker_patches()  # flag unset: no-op
    assert not getattr(sc._SplitCompiledExecutable.__dict__["__call__"], "_l6_thin", False)


# ---------------------------------------------------------------------------------- registration
def test_levers_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    want = {"mkctx": "VLLM_TPU_L6_MK_CTX", "mkthin": "VLLM_TPU_L6_MK_THIN_TT", "mkdirect": "VLLM_TPU_L6_MK_DIRECT",
            "fuserpc": "VLLM_TPU_L6_FUSE_RPC", "e2kvonly": "VLLM_TPU_L6_OUT_KVONLY"}
    for lever, env in want.items():
        assert l6_ab.LEVERS[lever] == env
        assert env in envs.environment_variables and env in _TPU_COMPILE_ENV_IGNORED
    for env in ("VLLM_TPU_L6_MK_CTX_CHECK", "VLLM_TPU_L6_MK_THIN_TT_CHECK", "VLLM_TPU_L6_MK_DIRECT_CHECK",
                "VLLM_TPU_L6_OUT_KVONLY_CHECK"):
        assert env in envs.environment_variables and env in _TPU_COMPILE_ENV_IGNORED


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)

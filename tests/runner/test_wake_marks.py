# SPDX-License-Identifier: Apache-2.0
"""runner/wake_marks.py: the thread-wake marks and the prewarm arm."""

import types

import pytest
import torch
from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
    TorchTpuCompiledExecutable,
)

from vllm_torchtpu.runner import host_phases
from vllm_torchtpu.runner import wake_marks as wm

_ORIG_TO = torch.Tensor.to
_ORIG_CALL = TorchTpuCompiledExecutable.__call__


@pytest.fixture
def hp(monkeypatch):
    """HOST_PHASES on with a fresh sample buffer; every patch undone after."""

    def setup(on=True, prewarm=0):
        monkeypatch.setattr(host_phases, "_EVERY", 10**9 if on else 0)
        monkeypatch.setattr(host_phases, "_samples", {})
        monkeypatch.setattr(host_phases, "_cur", [])
        monkeypatch.setattr(host_phases, "_kind", "dec")
        monkeypatch.setattr(host_phases, "_last", None)
        monkeypatch.setattr(wm, "ON", on)
        monkeypatch.setattr(wm, "PREWARM", prewarm)
        monkeypatch.setattr(wm, "FC_FLUSH", False)
        monkeypatch.setattr(wm, "_orig_exec_call", None)
        monkeypatch.setattr(TorchTpuCompiledExecutable, "__call__", _ORIG_CALL)
        monkeypatch.setattr(wm._S, "site", None)
        monkeypatch.setattr(wm._S, "armed", False)
        monkeypatch.setattr(wm._S, "runner", None)
        monkeypatch.setattr(wm._S, "noop_ok", False)

    yield setup
    torch.Tensor.to = _ORIG_TO


def _executable(fn):
    """A TorchTpuCompiledExecutable whose C++ runner is ``fn(args)``."""
    ex = TorchTpuCompiledExecutable.__new__(TorchTpuCompiledExecutable)
    ex._fast_runner = fn
    return ex


def _fold_module():
    ex = _executable(lambda args: tuple(a * 2 + 1 for a in args))

    def fold(a, b, *, k):
        return ex(a + k, b)

    def verify(a):
        return ex(a)

    return types.SimpleNamespace(
        _prep_fold_compiled=fold, _verify_one_draw_ro_compiled=verify
    )


class _Runner:
    def __init__(self):
        self.device = torch.device("meta")
        self.warmup_done = True
        self.fences = 0

    def _wait_input_staging_fence(self):
        self.fences += 1

    def capture_model(self):
        pass


def _names():
    return [n for n, _ in host_phases._cur]


def test_off_patches_nothing(hp):
    hp(on=False, prewarm=0)
    mod, runner = _fold_module(), _Runner()
    fold, call = mod._prep_fold_compiled, TorchTpuCompiledExecutable.__call__
    fence = runner._wait_input_staging_fence
    wm.install(runner, mod)
    assert mod._prep_fold_compiled is fold
    assert TorchTpuCompiledExecutable.__call__ is call
    assert runner._wait_input_staging_fence == fence
    assert "capture_model" not in vars(runner)
    assert host_phases._samples == {}


def test_fold_split_front_runner_post_outputs_unchanged(hp):
    hp(on=True)
    ref_mod = _fold_module()
    a, b = torch.arange(4), torch.ones(3)
    ref = ref_mod._prep_fold_compiled(a, b, k=3)
    mod = _fold_module()
    wm.install(_Runner(), mod)
    out = mod._prep_fold_compiled(a, b, k=3)
    assert all(torch.equal(x, y) for x, y in zip(out, ref))
    assert _names() == ["x_pu_fc_front", "x_pu_fc_runner", "x_pu_fc_post"]
    assert all(ns >= 0 for _, ns in host_phases._cur)
    # Verify is split under its own prefix; a bare executable call (no site)
    # records nothing.
    host_phases._cur.clear()
    mod._verify_one_draw_ro_compiled(a)
    assert _names() == ["x_vf_front", "x_vf_runner", "x_vf_post"]
    host_phases._cur.clear()
    _executable(lambda args: args)(a)
    assert _names() == []


def test_fold_without_executable_notes_norun(hp):
    hp(on=True)
    mod = types.SimpleNamespace(_prep_fold_compiled=lambda x: x + 1)
    wm.install(_Runner(), mod)
    assert torch.equal(mod._prep_fold_compiled(torch.zeros(2)), torch.ones(2))
    assert _names() == ["x_pu_fc_norun"]


def _prepare_inputs(runner, host):
    runner._wait_input_staging_fence()
    first = host.to(runner.device, non_blocking=True)
    second = host.to(runner.device, non_blocking=True)
    return first, second


def _dummy_run(runner, host):
    runner._wait_input_staging_fence()
    return host.to(runner.device, non_blocking=True)


def test_first_h2d_split_only_on_the_serving_prep_path(hp):
    hp(on=True)
    runner = _Runner()
    wm.install(runner, types.SimpleNamespace())
    host = torch.arange(8, dtype=torch.int32)
    first, second = _prepare_inputs(runner, host)
    assert runner.fences == 1
    assert first.shape == second.shape == host.shape and first.device.type == "meta"
    # Only the first CPU->device .to after the fence, then .to is restored.
    assert _names() == ["x_pu_h2d1_pre", "x_pu_h2d1_to"]
    assert torch.Tensor.to is _ORIG_TO
    host_phases._cur.clear()
    _dummy_run(runner, host)
    assert _names() == [] and torch.Tensor.to is _ORIG_TO
    runner.warmup_done = False
    _prepare_inputs(runner, host)
    assert _names() == [] and torch.Tensor.to is _ORIG_TO


def test_cpu_to_cpu_and_dtype_casts_pass_through_while_armed(hp):
    hp(on=True)
    runner = _Runner()
    wm.install(runner, types.SimpleNamespace())

    def _prepare_inputs():
        runner._wait_input_staging_fence()
        x = torch.arange(4).to(torch.float32)
        y = x.to("cpu")
        assert torch.Tensor.to is not _ORIG_TO  # still armed
        return x, y

    x, y = _prepare_inputs()
    assert torch.equal(x, torch.arange(4, dtype=torch.float32)) and torch.equal(x, y)
    assert _names() == []


@pytest.mark.parametrize("on", [True, False])
def test_prewarm_keeps_outputs_and_is_timed(hp, monkeypatch, on):
    a, b = torch.arange(4), torch.ones(3)
    hp(on=on, prewarm=0)
    ref_mod, ref_runner = _fold_module(), _Runner()
    wm.install(ref_runner, ref_mod)
    host = torch.arange(8, dtype=torch.int32)
    ref_h2d = _prepare_inputs(ref_runner, host)
    ref = ref_mod._prep_fold_compiled(a, b, k=3)

    hp(on=on, prewarm=7)
    # Exec prewarm program: an eager stand-in for the TPU-compiled 1-op program
    # (whose executable call is split like the fold's).
    noop_ex = _executable(lambda args: (args[0] + 1,))
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: (lambda x: noop_ex(x)[0]))
    mod, runner = _fold_module(), _Runner()
    runner.device = torch.device("cpu")  # the noop warm-up reads its output
    wm.install(runner, mod)
    runner.capture_model()
    assert wm._S.noop_ok
    runner.device = torch.device("meta")
    h2d = _prepare_inputs(runner, host)
    out = mod._prep_fold_compiled(a, b, k=3)
    assert all(torch.equal(x, y) for x, y in zip(out, ref))
    assert [x.shape for x in h2d] == [x.shape for x in ref_h2d]
    if on:
        assert _names() == [
            "x_pu_h2d1_pre",
            "x_pw_h2d_first",
            "x_pu_h2d1_to",
            "x_pw_h2d",
            "x_pw_ex_front",
            "x_pw_ex_runner",
            "x_pw_ex_post",
            "x_pw_ex_total",
            "x_pu_fc_front",
            "x_pu_fc_runner",
            "x_pu_fc_post",
        ]
    else:
        assert host_phases._samples == {}
    assert torch.Tensor.to is _ORIG_TO

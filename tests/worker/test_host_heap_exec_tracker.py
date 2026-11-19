# SPDX-License-Identifier: Apache-2.0
"""TPU_HOST_HEAP_REPORT>=4 executable tracker and the synthetic-verify-only
precompile flag (both default off, both outside the compile-cache key)."""

import sys
import types

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.worker import host_heap_report as hhr


class _FakeExec:
    def __init__(self, x):
        self.x = x

    def __call__(self, *a):
        return self.x


@pytest.fixture
def fake_torch_tpu(monkeypatch):
    mod = types.ModuleType("torch_tpu._internal.compile.torch_tpu_compiled_executable")

    class TorchTpuCompiledExecutable(_FakeExec):
        pass

    mod.TorchTpuCompiledExecutable = TorchTpuCompiledExecutable
    for name in ("torch_tpu", "torch_tpu._internal", "torch_tpu._internal.compile"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch_tpu._internal.compile"].torch_tpu_compiled_executable = mod
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(hhr, "_records", [])
    monkeypatch.setattr(hhr, "_phase", "startup")
    return TorchTpuCompiledExecutable


def test_flags_default_off_and_ignored_by_cache_key(monkeypatch):
    monkeypatch.delenv("TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY", raising=False)
    assert envs.TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY is False
    from vllm_torchtpu.compilation import tpu_compiler

    for name in ("TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY", "TPU_HOST_HEAP_USAGE_AFTER_S"):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_tracker_counts_per_phase_and_served_calls(fake_torch_tpu, caplog):
    cls = fake_torch_tpu
    assert hhr.install_exec_tracker(0)
    hhr.set_phase("rejection_sampler")
    a, b = cls(1), cls(2)
    hhr.set_phase("model backbone")
    c = cls(3)
    assert a() == 1 and c() == 3  # warm-up calls
    # An async proxy that copies the real executable's __dict__ shares its record.
    proxy = object.__new__(cls)
    proxy.__dict__.update(b.__dict__)
    hhr.mark_warm()
    c()
    proxy()
    assert [r[:2] for r in hhr._records] == [
        ["rejection_sampler", 1],
        ["rejection_sampler", 1],
        ["model backbone", 2],
    ]
    with caplog.at_level("INFO"):
        hhr.usage_report("t", 0)
    line = [r.getMessage() for r in caplog.records if "HOSTHEAP-USAGE" in r.getMessage()]
    assert line, caplog.text
    assert "rejection_sampler: n=2 warm_called=1 served_called=1 served_calls=1" in line[-1]
    assert "model backbone: n=1 warm_called=1 served_called=1 served_calls=1" in line[-1]
    assert hhr._phase == "serving"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

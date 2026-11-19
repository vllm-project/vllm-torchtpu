# SPDX-License-Identifier: Apache-2.0
"""DEC-HC engagement counters (VLLM_TPU_HC_DECODE_COUNT): built / disabled fused decode HC modules, and the
host count of fused-eligible target forwards."""

import logging

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement as E
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual, HyperConnectionConfig


@pytest.fixture(autouse=True)
def _reset():
    E.reset()
    yield
    E.reset()


def _module(inject=True):
    m = GatedResidual(
        HyperConnectionConfig(
            hc_count=4, hidden_size=128, params_dtype=torch.bfloat16, hc_lowrank=32, rms_norm_eps=1e-6,
            hc_per_branch_norm=True,
        )
    )
    return m


@pytest.fixture
def fused_flags(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_FUSED", True)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_UP_T", True)
    monkeypatch.setattr(vllm_adapter, "build_decode_mix_op", lambda *a: object())
    monkeypatch.setattr(vllm_adapter, "build_decode_combine_mix_op", lambda *a: object())
    return monkeypatch


def test_built_and_disabled_modules(fused_flags):
    mods = [_module() for _ in range(3)]
    assert E.snapshot()["modules"] == 3 and E.snapshot()["up_t"] == 3
    mods[0].disable_decode_fused()
    mods[0].disable_decode_fused()  # the MTP draft may call it more than once: counted once
    assert E.snapshot()["modules"] == 2 and E.snapshot()["up_t"] == 2
    fused_flags.setattr(envs, "TPU_QWEN4_HC_DECODE_UP_T", False)
    _module()
    assert E.snapshot()["modules"] == 3 and E.snapshot()["up_t"] == 2
    fused_flags.setattr(envs, "TPU_QWEN4_HC_DECODE_FUSED", False)
    _module()
    assert E.snapshot()["modules"] == 3


def test_flag_off_records_nothing(monkeypatch):
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_FUSED", False)
    _module()
    assert E.snapshot() == {"modules": 0, "up_t": 0, "steps": 0, "eligible": 0}


def test_count_step_eligibility_and_log(caplog):
    assert E.count_step(256, 0) is False  # no fused module on this rank
    E.record_built(True)
    with caplog.at_level(logging.INFO):
        hits = [E.count_step(t, 6) for t in (256, 112, 1024, 136, 64)]
    assert hits == [True, True, False, False, True]
    assert E.snapshot() == {"modules": 1, "up_t": 1, "steps": 6, "eligible": 3}
    lines = [r.getMessage() for r in caplog.records if "DEC-HC engagement" in r.getMessage()]
    assert lines == ["DEC-HC engagement: fused decode HC on 1 HC modules (UP_T 1); fused-eligible forwards 3 "
                     "of 6 on this rank"]


def test_flags_and_compile_key(monkeypatch):
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    for name in ("TPU_QWEN4_HC_DECODE_UP_T", "VLLM_TPU_HC_DECODE_COUNT", "TPU_QWEN4_HC_DECODE_FUSED"):
        monkeypatch.delenv(name, raising=False)
        assert name in envs.environment_variables
    assert envs.environment_variables["TPU_QWEN4_HC_DECODE_UP_T"]() is False
    assert envs.environment_variables["VLLM_TPU_HC_DECODE_COUNT"]() == 0
    # UP_T changes the compiled program (hashed); the counter is host-side logging only (ignored).
    assert "TPU_QWEN4_HC_DECODE_UP_T" not in _TPU_COMPILE_ENV_IGNORED
    assert "TPU_QWEN4_HC_DECODE_FUSED" not in _TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_HC_DECODE_COUNT" in _TPU_COMPILE_ENV_IGNORED

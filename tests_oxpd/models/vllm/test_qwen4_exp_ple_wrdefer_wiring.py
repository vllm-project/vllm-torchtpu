# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_WRDEFER bind-time engagement in Qwen4ExpPooledPLE (q38 L1 PLE-WRDEFER; CPU).

The writer reads the flag at trace time (tests/kernels/test_qwen4_exp_ple_wrdefer.py
proves its bytes). A compiled program that comes from a cache is not traced again, so
the layer also logs one bind-time line per process when the flag is set, and nothing
when it is unset. The ops it builds do not change with the flag (the writer is chosen
inside them). Harness: the l1_ple fold wiring test's adapter (D07-b2 stand-ins).
"""

import logging

import pytest

from tests.models.vllm import test_qwen4_exp_ple_d14_fold_wiring as fw
from tests.models.vllm import test_qwen4_exp_ple_decode_bounded_wiring as w

FLAG = "VLLM_TPU_PLE_WRDEFER"
LOGGER = "vllm.vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple"


def _bind(monkeypatch, env, binds=1):
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_ple

    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(pooled_ple, "_WRDEFER_NOTED", [])
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger(LOGGER)
    h = _Keep(level=logging.INFO)
    log.addHandler(h)
    old = log.level
    log.setLevel(logging.INFO)
    try:
        adapted, pool, built = fw._adapter(monkeypatch, env)
        for _ in range(binds - 1):
            adapted.bind_pool(pool)
    finally:
        log.removeHandler(h)
        log.setLevel(old)
    return [m for m in records if "PLE-WRDEFER" in m], built


@pytest.mark.parametrize("depth", ["2", "4"])
def test_wrdefer_logs_its_bind_line_once(monkeypatch, depth):
    lines, built = _bind(monkeypatch, {w.FLAG: "1", FLAG: depth}, binds=2)
    print("Q38 L1-PLEWD bind engagement:", lines)
    assert lines == [
        f"q38 PLE-WRDEFER on: ple PLE checkpoint writer keeps {depth} DMAs in flight "
        "(VLLM_TPU_PLE_WRDEFER; D14 fold buckets write inside the fold program)"
    ], lines
    # Two binds, and each builds exactly the ops it builds without the flag (the writer is chosen
    # inside them); proof1 j-9c9ff591 failed here on [(0,), (1,)] for two binds (test bug).
    assert built == [(0,), (1,)] * 2, built


@pytest.mark.parametrize("env", [{w.FLAG: "1"}, {w.FLAG: "1", FLAG: "0"}], ids=["unset", "zero"])
def test_wrdefer_unset_logs_nothing(monkeypatch, env):
    lines, built = _bind(monkeypatch, env)
    assert lines == [], lines
    assert built == [(0,), (1,)], built

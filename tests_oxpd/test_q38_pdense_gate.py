# SPDX-License-Identifier: Apache-2.0
"""The P-DENSE trace-time record calls are gated at import on VLLM_TPU_PDENSE_TRACE_RECORDS (q38 p75 j-8eb772a8).

With it off (the default, whatever VLLM_TPU_PDENSE_COUNT is) the call sites never reach record_*, so a forward traced with the token dimension
marked dynamic (vLLM's profile run) does not specialize it. With it on the counters are traced,
and the old _rows still specializes: the gate, not _rows, is what protects the default runs.
"""

import importlib

import pytest
import torch
import torch._dynamo

from vllm_torchtpu.models.vllm.qwen4_exp import pdense_engagement as E


def _call_site(x):
    # The shape of pooled_ple.py's and model.py's guarded call sites.
    rows = x.shape[0]
    if E.ENABLED:
        E.record_ple("ple_onehot", rows)
        E.record_hc_defer(rows, 2)
    return x * 2


def _trace_dynamic():
    torch._dynamo.reset()
    f = torch.compile(_call_site, backend="eager", fullgraph=True)
    x = torch.randn(64, 4)
    torch._dynamo.mark_dynamic(x, 0)
    return torch.equal(f(x), x * 2)


@pytest.mark.parametrize("records,count,enabled", [(None, None, False), (None, "200", False), ("0", "200", False),
                                                   ("1", None, True)])
def test_enabled_follows_trace_records_not_count(monkeypatch, records, count, enabled):
    for name, value in (("VLLM_TPU_PDENSE_TRACE_RECORDS", records), ("VLLM_TPU_PDENSE_COUNT", count)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    try:
        assert importlib.reload(E).ENABLED is enabled
    finally:
        for name in ("VLLM_TPU_PDENSE_TRACE_RECORDS", "VLLM_TPU_PDENSE_COUNT"):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(E)


def test_trace_records_is_registered_and_compile_neutral():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_PDENSE_TRACE_RECORDS" in envs.environment_variables
    assert "VLLM_TPU_PDENSE_TRACE_RECORDS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_gate_off_dynamic_trace_compiles(monkeypatch):
    monkeypatch.setattr(E, "ENABLED", False)
    E.reset()
    assert _trace_dynamic()
    assert E._TRACED["ple_onehot"] == set() and E._TRACED["hc_defer"] == {}


def test_gate_on_still_specializes_with_the_old_rows(monkeypatch):
    """Documents the boundary: with counting on, the old _rows specializes the dynamic size."""
    monkeypatch.setattr(E, "ENABLED", True)
    E.reset()
    with pytest.raises(Exception, match="Constraints violated"):
        _trace_dynamic()
    E.reset()

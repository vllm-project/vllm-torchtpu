# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PDENSE_COUNT: trace-time records of the P-DENSE paths and the host-side per-step count
(models/vllm/qwen4_exp/pdense_engagement.py), for the accuracy bundle's engagement rule."""

import logging

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.qwen4_exp import pdense_engagement as E


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # These tests are about recording with counting on (VLLM_TPU_PDENSE_COUNT > 0): the call sites gate on
    # E.ENABLED, which is decided at import.
    monkeypatch.setattr(E, "ENABLED", True)
    E.reset()
    yield
    E.reset()


def test_count_step_counts_from_registered_builds_not_trace_records(caplog):
    """Host-side counting (P-MOE-6): A / B / C come from what the stage process registered at build and the shared
    row rules; trace-time records alone count nothing (serving's traces are all dynamic, so they never fire there)."""
    E.record_ple("ple_onehot", 8)          # a trace-time record with no registration: must not count
    E.register_ple(object(), onehot=True, fused_tile=0)
    E.register_hc_defer(4, 3)              # whole tile 4: rows >= 8 and rows % 4 == 0
    with caplog.at_level(logging.INFO):
        hits = [E.count_step(t, 5) for t in (8, 10, 4, 16, 12)]
    assert [h["ple_onehot"] for h in hits] == [True] * 5
    assert [h["hc_defer"] for h in hits] == [True, False, False, True, True]
    assert [h["ple_fused"] for h in hits] == [False] * 5
    assert E._COUNTS == {"steps": 5, "ple_onehot": 5, "ple_fused": 0, "hc_defer": 3, "hc_handoffs": 9}
    lines = [r.getMessage() for r in caplog.records if "P-DENSE engagement" in r.getMessage()]
    assert lines == ["P-DENSE engagement: PLE one-hot conv steps 5, PLE fused steps 0, HC deferred steps 3 "
                     "(hand-offs 9) of 5 prefill steps on this stage process"]


def test_fused_steps_replace_the_onehot_conv():
    E.register_ple(object(), onehot=True, fused_tile=16)
    hits = [E.count_step(t, 0) for t in (32, 24, 16, 8)]
    assert [h["ple_fused"] for h in hits] == [True, False, True, False]
    assert [h["ple_onehot"] for h in hits] == [False, True, False, True]


def test_base_arm_counts_zero():
    """Flags off: the builds register nothing that counts, so A = B = C = 0 (the base arm's clause)."""
    E.register_ple(object(), onehot=False, fused_tile=0)
    E.register_hc_defer(0, 3)
    for t in (8, 6400, 8192):
        E.count_step(t, 0)
    assert E._COUNTS == {"steps": 3, "ple_onehot": 0, "ple_fused": 0, "hc_defer": 0, "hc_handoffs": 0}


def _stage(monkeypatch, whole, ple_at=None):
    from types import SimpleNamespace

    from tests.models.vllm.test_qwen4_exp_hc_defer import _TiledLayer
    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpBackbone

    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DEFER_COMBINE", False)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DEFER_WHOLE", whole)
    log = []
    stage = [6, 7, 8, 9, 10]
    fake = SimpleNamespace(
        host_ple_index=None, stage=range(stage[0], stage[-1] + 1), first_stage=False, last_stage=False,
        layers={i: _TiledLayer(log, i, ple=(i == ple_at)) for i in stage}, hc_count=4, sp_pad_rows=0,
        embed_tokens=None, hyper_connection_mixer=None, emit_mtp_hidden=False,
    )
    Qwen4ExpBackbone._register_pdense_engagement(fake)

    def run(rows):
        del log[:]
        hidden = torch.arange(rows * 4, dtype=torch.float64).reshape(rows, 4)
        Qwen4ExpBackbone._forward_impl(
            fake, torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, dtype=torch.int64), None,
            sequence_parallel=False, intermediate_tensors={"hidden_states": hidden},
            query_start_loc=None, ngram_context=None, state_slots=None, context_lens=None,
            valid_tokens=None, main_block_table=None, compressed_block_table=None,
        )
        return sum(1 for e in log if e[0] == "combine_mix")

    return run


@pytest.mark.parametrize("ple_at", [None, 8])
def test_stage_loop_and_host_count_agree(monkeypatch, ple_at):
    """The traced decision (the real stage loop) and the host count agree on every row count, hand-offs included
    (tile 4; with a PLE layer at position 2 the hand-off into it is skipped)."""
    run = _stage(monkeypatch, whole=True, ple_at=ple_at)
    for rows in (8, 10, 12, 6, 16, 4, 20):
        E.reset_counts()
        handed = run(rows)
        hit = E.count_step(rows, 0)
        assert hit["hc_defer"] == (handed > 0), rows
        assert E._COUNTS["hc_handoffs"] == handed, rows


def test_stage_loop_registers_nothing_without_defer_whole(monkeypatch):
    run = _stage(monkeypatch, whole=False)
    assert run(16) == 0 and E._BUILT["hc"] is None and not E.count_step(16, 0)["hc_defer"]


@pytest.mark.parametrize("env,path", [({"VLLM_TPU_PLE_CONV_ONEHOT": "1"}, "ple_onehot"),
                                      ({"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "16"}, "ple_fused"),
                                      ({}, None)])
def test_pooled_ple_registration_matches_the_path_taken(monkeypatch, env, path):
    """The real pooled PLE layer registers at bind_pool; for each step size the host count names exactly the path
    its forward took (the trace-time record, concrete sizes on CPU, is the oracle here)."""
    from tests.models.vllm.test_qwen4_exp_ple_fused_wiring import _adapter, _step

    E.reset()
    adapted, _, _ = _adapter(monkeypatch, env)
    for rows in (32, 24):
        E.reset_traced()
        _step(adapted, rows)
        taken = {k for k in ("ple_onehot", "ple_fused") if rows in E._TRACED[k]}
        hit = E.count_step(rows, 0)
        assert {k for k in ("ple_onehot", "ple_fused") if hit[k]} == taken, (rows, taken, hit)
    if path is None:
        assert E._COUNTS["ple_onehot"] == E._COUNTS["ple_fused"] == 0
    else:
        assert E._COUNTS[path] > 0


def test_counters_are_not_called_inside_the_forward(monkeypatch):
    """Registration happens at bind_pool / build and counting in the runner: the forward never calls them."""
    from tests.models.vllm.test_qwen4_exp_ple_fused_wiring import _adapter, _step

    adapted, _, _ = _adapter(monkeypatch, {"VLLM_TPU_PLE_CONV_ONEHOT": "1"})

    def boom(*a, **k):
        raise AssertionError("host-side counter called inside the forward")

    for name in ("register_ple", "register_hc_defer", "count_step"):
        monkeypatch.setattr(E, name, boom)
    _step(adapted, 32)


def test_no_records_no_counts():
    E.count_step(6400, 0)
    assert E._COUNTS["steps"] == 1 and E._COUNTS["ple_onehot"] == E._COUNTS["hc_defer"] == 0


@pytest.mark.parametrize("rows,handoffs", [(8, 3), (10, None)])
def test_stage_loop_records_hc_handoffs(monkeypatch, rows, handoffs):
    """The real stage loop (Qwen4ExpBackbone._forward_impl) records its hand-offs per row count: 3 for a
    4-layer stage at a whole-tile step under DEFER_WHOLE, none at a partial-tile step."""
    from tests.models.vllm.test_qwen4_exp_hc_defer import _run_whole

    _run_whole(monkeypatch, rows, whole=True)
    assert E._TRACED["hc_defer"].get(rows) == handoffs


@pytest.mark.parametrize("flag", [False, True])
def test_pooled_ple_records_the_onehot_conv(monkeypatch, flag):
    """forward_with_history records its row count for the one-hot conv exactly when the flag is on."""
    from tests.models.vllm.test_qwen4_exp_ple_onehot_wiring import _adapter, _step

    env = {"VLLM_TPU_PLE_CONV_ONEHOT": "1"} if flag else {}
    adapted, _, _ = _adapter(monkeypatch, env)
    _step(adapted, 32)
    assert (32 in E._TRACED["ple_onehot"]) == flag


@pytest.mark.parametrize("fused", [False, True])
def test_pooled_ple_records_the_fused_path(monkeypatch, fused):
    """forward_with_history records its row count for the fused PLE exactly when the fused path applies."""
    from tests.models.vllm.test_qwen4_exp_ple_fused_wiring import _adapter, _step

    env = {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "16"} if fused else {}
    adapted, _, _ = _adapter(monkeypatch, env)
    _step(adapted, 32)
    assert (32 in E._TRACED["ple_fused"]) == fused

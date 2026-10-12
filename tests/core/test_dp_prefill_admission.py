# SPDX-License-Identifier: Apache-2.0
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_torchtpu.core import dp_prefill_admission as admission


def engine():
    scheduler = SimpleNamespace(
        running=[],
        waiting=[object()],
        kv_holding_waiting=[],
        pause_state=admission.PauseState.UNPAUSED,
    )
    return SimpleNamespace(
        scheduler=scheduler,
        batch_queue=[],
        pending_pause=False,
        input_queue=queue.Queue(),
        dp_group=object(),
        dp_size=8,
        dp_rank=0,
        current_wave=3,
        is_running=lambda: True,
        _handle_client_request=Mock(),
    )


def test_zero_window_repolls_after_rendezvous(monkeypatch):
    core = engine()

    def reduce(state, **kwargs):
        assert kwargs["group"] is core.dp_group
        # Request becomes visible while another rank enters the rendezvous.
        core.input_queue.put(("add", "cached-owner-request"))
        state.copy_(state.new_tensor([1, 0, 8]))

    monkeypatch.setattr(admission.dist, "all_reduce", reduce)
    admission.coalesce_prefill_wave(core, 0)
    core._handle_client_request.assert_called_once_with("add", "cached-owner-request")


@pytest.mark.parametrize("global_state", [[8, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 0]])
def test_common_exit_decision_does_not_require_full_batch(monkeypatch, global_state):
    core = engine()
    reduce = Mock(
        side_effect=lambda state, **kw: state.copy_(state.new_tensor(global_state))
    )
    monkeypatch.setattr(admission.dist, "all_reduce", reduce)
    monkeypatch.setattr(
        admission.time, "sleep", Mock(side_effect=AssertionError("must not wait"))
    )
    admission.coalesce_prefill_wave(core, 5)
    assert reduce.call_count == 1


@pytest.mark.parametrize("active", ["running", "batch_queue", "pending_pause"])
def test_resume_decode_and_pause_vote_to_bypass(monkeypatch, active):
    core = engine()
    if active == "running":
        core.scheduler.running.append(object())
    elif active == "batch_queue":
        core.batch_queue.append(object())
    else:
        core.pending_pause = True

    def reduce(state, **kwargs):
        assert state[1].item() == 1

    monkeypatch.setattr(admission.dist, "all_reduce", reduce)
    admission.coalesce_prefill_wave(core, 5)


def test_burst_is_drained_before_next_ready_vote(monkeypatch):
    core = engine()
    votes = []

    def reduce(state, **kwargs):
        votes.append(state.tolist())
        if len(votes) == 1:
            core.input_queue.put(("add", "second"))
            state.copy_(state.new_tensor([1, 0, 0]))
        else:
            core._handle_client_request.assert_called_once_with("add", "second")
            state.copy_(state.new_tensor([8, 0, 0]))

    monkeypatch.setattr(admission.dist, "all_reduce", reduce)
    monkeypatch.setattr(admission.time, "sleep", lambda _: None)
    admission.coalesce_prefill_wave(core, 100)
    assert len(votes) == 2


@pytest.mark.parametrize("window", [-1, float("inf"), float("nan")])
def test_reject_invalid_budget(window):
    with pytest.raises(ValueError):
        admission.coalesce_prefill_wave(engine(), window)


@pytest.mark.parametrize(
    "window,step,prefix,ep,work,expected",
    [
        (None, 0, True, True, True, 0),
        (0, 0, True, True, True, 1),
        (2, 1, True, True, True, 0),
        (2, 0, False, True, True, 0),
        (2, 0, True, False, True, 0),
        (2, 0, True, True, False, 0),
    ],
)
def test_hook_only_coalesces_cache_affine_fresh_wave(
    monkeypatch, window, step, prefix, ep, work, expected
):
    from vllm.v1.engine.core import DPEngineCoreProc

    original = Mock(return_value=True)
    coalesce = Mock()
    monkeypatch.setattr(
        DPEngineCoreProc, "_tpu_prefill_admission_patch", False, raising=False
    )
    monkeypatch.setattr(DPEngineCoreProc, "_process_engine_step", original)
    monkeypatch.setattr(admission, "coalesce_prefill_wave", coalesce)
    admission.patch_dp_prefill_admission()
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dp_prefill_admission_ms": window},
            parallel_config=SimpleNamespace(enable_expert_parallel=ep),
            cache_config=SimpleNamespace(enable_prefix_caching=prefix),
        ),
        step_counter=step,
        has_work=lambda: work,
    )
    assert DPEngineCoreProc._process_engine_step(core) is True
    assert coalesce.call_count == expected
    original.assert_called_once_with(core)

# SPDX-License-Identifier: Apache-2.0
"""Where the runner submits Stage-3 loads under
TPU_RAIDEN_STAGE3_SUBMIT_AFTER_DISPATCH: held in maybe_setup_kv_connector,
submitted at the start of get_finished_kv_transfers, on every path a step
takes."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.runner.test_tpu_runner_kv_connector_lifecycle import (
    _fake_runner,
    _host_copy,
    _recording_connector,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

_RUNNER = "vllm_torchtpu.runner.tpu_runner"
_HELD = object()


def _switch(on: bool):
    return patch(
        f"{_RUNNER}.dist_utils.get_stage3_submit_after_dispatch", return_value=on
    )


@contextlib.contextmanager
def _group(connector):
    with (
        patch(f"{_RUNNER}.has_kv_transfer_group", return_value=True),
        patch(f"{_RUNNER}.get_kv_transfer_group", return_value=connector),
    ):
        yield


def _holding_connector(events: list[str], held=_HELD) -> MagicMock:
    """A recording connector that also logs the hold and the held submit."""
    connector = _recording_connector(events)
    connector.handle_preemptions.side_effect = lambda _m: events.append("preempt")
    connector.bind_connector_metadata.side_effect = lambda _m: events.append("bind")
    connector.start_load_kv.side_effect = lambda *_a, **_k: events.append("load")

    def hold():
        events.append("hold")
        return held

    def submit(entries):
        assert entries is held
        events.append("submit_held")

    connector.start_load_kv_holding_loads.side_effect = hold
    connector.submit_held_loads.side_effect = submit
    return connector


def _bind(runner, *names):
    for name in names:
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))


def _setup_runner(events):
    runner = SimpleNamespace(_held_kv_loads=None, _stage_timing=MagicMock())
    runner._stage_timing.take_samples.return_value = ([], [])
    _bind(
        runner,
        "maybe_setup_kv_connector",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
        "_submit_held_kv_loads",
        "kv_connector_no_forward",
        "_sample_tokens_pp_intermediate_stage",
    )
    return runner


def _scheduler_output():
    return SimpleNamespace(finished_req_ids=set(), kv_connector_metadata=object())


def test_on_holds_at_setup_and_submits_before_the_poll():
    events: list[str] = []
    connector = _holding_connector(events)
    runner = _setup_runner(events)
    so = _scheduler_output()
    with _switch(True), _group(connector):
        runner.maybe_setup_kv_connector(so)
        assert runner._held_kv_loads is _HELD
        events.append("forward")
        runner.get_finished_kv_transfers(so)
    assert events == [
        "preempt",
        "bind",
        "hold",
        "forward",
        "submit_held",
        "poll",
        "clear",
    ]
    assert runner._held_kv_loads is None
    connector.start_load_kv.assert_not_called()


def test_off_is_todays_calls_one_for_one():
    connector = MagicMock()
    connector.get_finished.return_value = (set(), set())
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.build_connector_worker_meta.return_value = None
    connector.get_kv_connector_stats.return_value = None
    runner = _setup_runner([])
    so = _scheduler_output()
    with _switch(False), _group(connector):
        runner.maybe_setup_kv_connector(so)
        runner.get_finished_kv_transfers(so)
    assert [c[0] for c in connector.mock_calls] == [
        "handle_preemptions",
        "bind_connector_metadata",
        "start_load_kv",
        "get_finished",
        "get_block_ids_with_load_errors",
        "build_connector_worker_meta",
        "get_kv_connector_stats",
        "clear_connector_metadata",
    ]
    connector.start_load_kv.assert_called_once_with(None)
    assert runner._held_kv_loads is None


def test_inline_load_and_connectors_without_hold_load_as_today():
    """Inline mode needs the loads before the forward; a connector without
    the holding form (MultiConnector, offloading) loads in full."""
    runner = _setup_runner([])
    so = _scheduler_output()
    inline = MagicMock()
    plain = MagicMock(
        spec=["handle_preemptions", "bind_connector_metadata", "start_load_kv"]
    )
    with _switch(True):
        with _group(inline):
            runner.maybe_setup_kv_connector(
                so, wait_for_completion=True, report_completion=False
            )
        with _group(plain):
            runner.maybe_setup_kv_connector(so)
    inline.start_load_kv.assert_called_once_with(
        None, wait_for_completion=True, report_completion=False
    )
    inline.start_load_kv_holding_loads.assert_not_called()
    plain.start_load_kv.assert_called_once_with(None)
    assert runner._held_kv_loads is None


@pytest.mark.parametrize(
    "method,is_async,verify",
    [(None, False, False), ("mtp", True, False), ("mtp", True, True)],
)
def test_sample_tokens_submits_after_the_drafter_before_the_poll(
    method, is_async, verify
):
    events: list[str] = []

    def propose(**kwargs):
        events.append("draft")
        return torch.tensor([[8]]) if is_async else None

    runner = _fake_runner(method, is_async, verify, propose)
    runner._held_kv_loads = None
    _bind(runner, "maybe_setup_kv_connector", "_submit_held_kv_loads")
    so = runner.execute_model_state.scheduler_output
    so.kv_connector_metadata = object()
    connector = _holding_connector(events)
    with (
        _switch(True),
        _group(connector),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=_host_copy),
    ):
        runner.maybe_setup_kv_connector(so)
        events.append("forward")
        TPUModelRunner.sample_tokens(runner, None)
    drafted = ["draft"] if method == "mtp" else []
    assert events == (
        ["preempt", "bind", "hold", "forward"]
        + drafted
        + ["save", "submit_held", "poll", "clear"]
    )
    assert runner._held_kv_loads is None


def test_no_forward_step_submits_in_the_same_call():
    events: list[str] = []
    connector = _holding_connector(events)
    runner = _setup_runner(events)
    with (
        _switch(True),
        _group(connector),
        patch(f"{_RUNNER}.dist_utils.get_raiden_inline_load", return_value=False),
    ):
        TPUModelRunner.kv_connector_no_forward(runner, _scheduler_output(), None)
    assert events == ["preempt", "bind", "hold", "submit_held", "poll", "clear"]
    assert runner._held_kv_loads is None


def test_pipeline_intermediate_stage_submits_in_its_sample_tokens():
    """sample_tokens returns early on a non-last stage; that path polls too."""
    events: list[str] = []
    connector = _holding_connector(events)
    runner = _setup_runner(events)
    so = _scheduler_output()
    with _switch(True), _group(connector):
        runner.maybe_setup_kv_connector(so)
        runner._pp_pending_scheduler_output = so
        runner.execute_model_state = None
        runner._pp_is_last = False
        out = TPUModelRunner.sample_tokens(runner, None)
    assert events == ["preempt", "bind", "hold", "save", "submit_held", "poll", "clear"]
    assert out is not None
    assert runner._held_kv_loads is None


def test_held_loads_left_by_a_step_go_out_before_the_next_steps_setup():
    """A step that never reached the poll still submits its loads, ahead of
    every connector call of the next step."""
    events: list[str] = []
    second = object()
    connector = _holding_connector(events, held=second)
    runner = _setup_runner(events)
    stale = object()
    runner._held_kv_loads = stale
    submitted = []
    connector.submit_held_loads.side_effect = lambda e: (
        submitted.append(e),
        events.append("submit_held"),
    )
    with _switch(True), _group(connector):
        runner.maybe_setup_kv_connector(_scheduler_output())
    assert events == ["submit_held", "preempt", "bind", "hold"]
    assert submitted == [stale]
    assert runner._held_kv_loads is second


def test_held_set_none_means_nothing_to_submit():
    events: list[str] = []
    connector = _holding_connector(events, held=None)
    connector.submit_held_loads.side_effect = AssertionError("nothing was held")
    runner = _setup_runner(events)
    so = _scheduler_output()
    with _switch(True), _group(connector):
        runner.maybe_setup_kv_connector(so)
        runner.get_finished_kv_transfers(so)
    assert events == ["preempt", "bind", "hold", "poll", "clear"]

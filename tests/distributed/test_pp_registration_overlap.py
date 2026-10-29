# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for registering PP sources after device submission."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer import tpu_connector as mod
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.runner import tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

from .test_tpu_connector import _make_raiden_worker


def _connector(monkeypatch, **worker_args):
    worker = _make_raiden_worker(pp_size=4, **worker_args)
    monkeypatch.setattr(worker, "_raiden_stage3_enabled", lambda: True)
    monkeypatch.setattr(mod, "_use_raiden_dsv4_admission", lambda: False)
    monkeypatch.setattr(mod, "_use_raiden_glm_admission", lambda: False)
    monkeypatch.setattr(mod, "_uses_tp_stage3_source", lambda _: False)
    worker.process_send_load = MagicMock()
    connector = object.__new__(mod.TPURaidenConnector)
    connector.connector_worker = worker
    metadata = mod.TPUConnectorMetadata()
    metadata.reqs_to_send["r0"] = object()
    connector._connector_metadata = metadata
    return connector, worker, metadata


def test_registration_captures_metadata_and_preserves_completion_flags(monkeypatch):
    connector, worker, metadata = _connector(monkeypatch)
    callback = connector.start_load_kv_for_forward(
        None, wait_for_completion=True, report_completion=False
    )
    worker.process_send_load.assert_not_called()
    assert callback is not None
    # A callback owns its original batch, not a lookup of mutable bound metadata.
    connector._connector_metadata = mod.TPUConnectorMetadata()
    callback()
    worker.process_send_load.assert_called_once_with(
        metadata, wait_for_completion=True, report_completion=False
    )


@pytest.mark.parametrize(
    "kind",
    ["load", "release", "expire", "consumer", "no_send", "non_pp", "other_transport"],
)
def test_control_and_consumer_operations_remain_eager(monkeypatch, kind):
    connector, worker, metadata = _connector(monkeypatch)
    if kind in ("load", "release", "expire"):
        value = getattr(metadata, "reqs_to_" + kind)
        if isinstance(value, dict):
            value["control"] = object()
        else:
            value.add("control")
    elif kind == "consumer":
        worker.is_producer = False
    elif kind == "no_send":
        metadata.reqs_to_send.clear()
    elif kind == "non_pp":
        monkeypatch.setattr(mod, "_engine_is_pipeline", lambda _: False)
    else:
        monkeypatch.setattr(worker, "_raiden_stage3_enabled", lambda: False)
    assert connector.start_load_kv_for_forward(None) is None
    worker.process_send_load.assert_called_once_with(
        metadata, wait_for_completion=False, report_completion=True
    )


def test_multi_connector_loads_v6d_before_deferred_registration(monkeypatch):
    connector, worker, _ = _connector(monkeypatch)
    events = []
    worker.process_send_load.side_effect = lambda *a, **kw: events.append("register")
    v6d = SimpleNamespace(start_load_kv=lambda *a, **kw: events.append("load"))
    multi = object.__new__(TPUMultiConnector)
    multi._connectors = [connector, v6d]
    callback = multi.start_load_kv_for_forward(None)
    assert events == ["load"]
    assert callback is not None
    callback()
    assert events == ["load", "register"]


@pytest.mark.parametrize("chunks", [0, 1, 2])
@pytest.mark.parametrize("failure", [None, "forward", "register"])
def test_execute_model_registration_precedes_handoff_and_output(
    monkeypatch, chunks, failure
):
    events = []

    def record(name):
        def call(*a, **kw):
            events.append(name)
            if failure == name:
                raise RuntimeError(name)

        return call

    callback = record("register")
    runner = MagicMock(spec=TPUModelRunner)
    runner.execute_model_state = None
    runner.vllm_config = SimpleNamespace()
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner._pp_wave = MagicMock()
    runner._pp_wave.before_step.side_effect = record("before_step")
    runner._pp_wave.end_forward.side_effect = record("handoff")
    runner._update_states.side_effect = record("update")
    runner.maybe_setup_kv_connector.return_value = callback
    runner._pcp_enabled = False
    runner._pp_is_first = False
    runner._pp_is_last = False
    runner.supports_mm_inputs = False
    runner.phase_based_profiler = None
    runner._is_async_drafter = False
    runner._hidden_nan_detector = None
    runner._kv_cache_nan_detector = None
    runner._reorder_batch_for_rpa.return_value = (0, 0)
    runner._count_input_chunks.return_value = (chunks, 1)
    runner._dp_coordinated_step.return_value = (8, chunks)
    runner.input_batch = SimpleNamespace(
        num_reqs=chunks, req_ids=[str(i) for i in range(chunks)]
    )
    runner.input_ids = torch.zeros(8, dtype=torch.int32)
    runner.position_ids = torch.zeros(8, dtype=torch.int32)
    runner.mesh = None
    runner._kv_cache_bundle = None
    runner._stage_timing = MagicMock()
    runner._apply_async_token_substitution.return_value = runner.input_ids
    runner._prepare_inputs.side_effect = [
        (None, None, 1, 1, i + 1, [], [], None, None, None) for i in range(chunks)
    ]

    def forward(**kw):
        record("forward")()
        return SimpleNamespace(tensors={}), None

    runner.forward_model.side_effect = forward
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r0": 8},
    )
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "0")
    with (
        patch.object(runner_mod, "set_forward_context", return_value=nullcontext()),
        patch.object(
            runner_mod, "set_vllm_model_wrapper_context", return_value=nullcontext()
        ),
        patch.object(runner_mod, "has_kv_transfer_group", return_value=False),
        patch.object(runner_mod.TraceAnnotation, "is_enabled", return_value=False),
        patch.object(
            runner_mod.dist_utils, "get_raiden_inline_load", return_value=False
        ),
    ):
        if failure == "register" or (failure == "forward" and chunks):
            with pytest.raises(RuntimeError, match=failure):
                TPUModelRunner.execute_model(runner, scheduler_output)
            assert "handoff" not in events
            assert runner.execute_model_state is None
        else:
            assert TPUModelRunner.execute_model(runner, scheduler_output) is None
            assert runner._pp_pending_scheduler_output is scheduler_output
            assert events == ["before_step", "update"] + (
                ["forward", "register", "handoff"]
                + ["forward", "handoff"] * (chunks - 1)
                if chunks
                else ["register"]
            )
    runner.maybe_setup_kv_connector.assert_called_once_with(
        scheduler_output,
        after_forward=True,
        wait_for_completion=False,
        report_completion=True,
    )

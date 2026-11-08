# SPDX-License-Identifier: Apache-2.0
"""Final-output checks distinguish NaN/Inf from padding."""

import gc
import logging
import threading
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("model_nan_cpu")]


def checker(monkeypatch):
    from vllm_torchtpu.runner import model_nan_check

    monkeypatch.setattr(model_nan_check, "logger", logging.getLogger(__name__))
    monkeypatch.setattr(model_nan_check.socket, "gethostname", lambda: "nan-test-host")
    return model_nan_check.ModelNaNChecker(rank=3, engine_idx=2)


def plan(valid=2, padded=4, token_slice=None, mapping=None):
    return SimpleNamespace(
        global_num_tokens=valid,
        global_padded_num_tokens=padded,
        token_slice=token_slice or slice(0, padded),
        _packed_to_request_major_token_indices=mapping,
    )


def consume(check):
    check.enqueue()
    check.poll()


def test_forward_and_blocked_poll_do_not_wait_or_read_results(monkeypatch):
    from vllm_torchtpu.runner import model_nan_check

    check = checker(monkeypatch)
    event = SimpleNamespace(record=lambda: None, query=lambda: False)
    monkeypatch.setattr(model_nan_check, "TpuEvent", lambda: event)
    error = AssertionError("NaN flags must only be read after query succeeds")
    with (
        patch.object(torch.Tensor, "item", side_effect=error),
        patch.object(torch.Tensor, "tolist", side_effect=error),
    ):
        check.check_hidden(torch.ones(2, 2), plan(valid=2, padded=2))
        check.check_logits(torch.ones(2, 2), num_valid_rows=2)
        check.enqueue()
        check.poll()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("real_bad", [False, True])
def test_jax_reduction_returns_one_bool_and_excludes_padding(bad_value, real_bad):
    from vllm_torchtpu.runner import model_nan_check

    values = jnp.array([[bad_value, 1.0], [0.0, 2.0]])
    result = model_nan_check._has_nan_jax(values, jnp.array([real_bad, True]))
    assert result.shape == () and result.dtype == jnp.bool_
    assert bool(result) is real_bad


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stage", ["last_hidden", "logits"])
def test_nan_in_a_real_row_reports_the_tensor_and_rank(
    monkeypatch, caplog, stage, bad_value
):
    check = checker(monkeypatch)
    values = torch.tensor([[0.0, bad_value], [1.0, 2.0]])
    with caplog.at_level(logging.ERROR):
        if stage == "last_hidden":
            check.check_hidden(values, plan(valid=2, padded=2))
        else:
            check.check_logits(values, num_valid_rows=2)
        assert not caplog.records
        consume(check)
    assert stage in caplog.text and "rank=3" in caplog.text
    assert "host=nan-test-host" in caplog.text and "engine=2" in caplog.text
    assert "shape=(2, 2)" in caplog.text


def test_padding_nonfinite_and_finite_extremes_do_not_report(monkeypatch, caplog):
    check = checker(monkeypatch)
    values = torch.tensor(
        [
            [torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max],
            [0.0, -0.0],
            [float("nan"), float("inf")],
        ]
    )
    check.check_hidden(values, plan(valid=2, padded=3))
    check.check_logits(values, num_valid_rows=2)
    consume(check)
    assert not caplog.records


@pytest.mark.parametrize("bad_row,expected", [(0, False), (1, True), (2, True)])
def test_pcp_checks_noncontiguous_real_rows(monkeypatch, caplog, bad_row, expected):
    check = checker(monkeypatch)
    # This rank owns packed rows [-1, 1, 4]: padding, token 1, token 4.
    layout = plan(
        valid=5,
        padded=8,
        token_slice=slice(2, 5),
        mapping=np.array([0, 2, -1, 1, 4, 3, -1, -1]),
    )
    values = torch.ones(3, 2)
    values[bad_row, 0] = float("nan")
    check.check_hidden(values, layout)
    with caplog.at_level(logging.ERROR):
        consume(check)
    assert bool(caplog.records) is expected


def test_each_chunk_is_checked_and_empty_real_rows_are_clean(monkeypatch, caplog):
    check = checker(monkeypatch)
    values = torch.full((2, 3), float("nan"))
    check.check_logits(values, num_valid_rows=0)
    check.check_logits(values, num_valid_rows=1)
    check.check_logits(torch.ones(2, 3), num_valid_rows=1)
    with caplog.at_level(logging.ERROR):
        consume(check)
    assert len(caplog.records) == 1


def test_empty_round_submits_no_readback(monkeypatch):
    from vllm_torchtpu.runner import model_nan_check

    monkeypatch.setattr(
        model_nan_check, "TpuEvent", lambda: pytest.fail("Empty round issued D2H")
    )
    check = checker(monkeypatch)
    check.enqueue()
    check.poll()


def test_poll_drains_ready_prefix_on_caller_thread_once(monkeypatch, caplog):
    from vllm_torchtpu.runner import model_nan_check

    events = []
    queries = []

    class Event:
        def __init__(self):
            self.ready = False
            self.index = len(events)
            events.append(self)

        def record(self):
            pass

        def query(self):
            queries.append(self.index)
            return self.ready

    monkeypatch.setattr(model_nan_check, "TpuEvent", Event)
    check = checker(monkeypatch)
    # The later event being ready does not permit reading past a blocked head.
    for rows in (1, 2, 3):
        check.check_logits(torch.full((rows, 2), float("nan")), rows)
        check.enqueue()
    events[1].ready = True
    with caplog.at_level(logging.ERROR):
        check.poll()
        assert queries == [0]
        assert not caplog.records
        events[0].ready = True
        check.poll()
        assert queries == [0, 0, 1, 2]
        assert len(caplog.records) == 2
        check.poll()
        assert len(caplog.records) == 2
        events[2].ready = True
        check.poll()
        check.poll()
    assert len(caplog.records) == 3
    assert all(record.thread == threading.get_ident() for record in caplog.records)
    for rows, record in zip((1, 2, 3), caplog.records):
        assert f"shape=({rows}, 2)" in record.message


def test_pending_round_keeps_d2h_source_alive_until_consumed(monkeypatch):
    from vllm_torchtpu.runner import model_nan_check

    event = SimpleNamespace(record=lambda: None, query=lambda: False)
    monkeypatch.setattr(model_nan_check, "TpuEvent", lambda: event)
    check = checker(monkeypatch)
    check.check_logits(torch.ones(1, 2), 1)
    stack = torch.stack
    sources = []

    def track_source(tensors):
        result = stack(tensors)
        sources.append(weakref.ref(result))
        return result

    # A distinct destination ensures its reference cannot keep the source alive.
    with (
        patch.object(torch, "stack", side_effect=track_source),
        patch.object(torch.Tensor, "to", lambda tensor, *_, **__: tensor.clone()),
    ):
        check.enqueue()
    gc.collect()
    check.poll()
    assert sources[0]() is not None
    event.query = lambda: True
    check.poll()
    gc.collect()
    assert sources[0]() is None


@pytest.mark.parametrize("pcp", [False, True])
@pytest.mark.parametrize("warmup_raises", [False, True])
def test_capture_suspends_checks_and_restores_them_after_warmup(
    monkeypatch, caplog, pcp, warmup_raises
):
    from vllm_torchtpu.runner import tpu_runner

    check = checker(monkeypatch)

    def warmup():
        assert runner._nan_checker is None, "Synthetic warmup must not check NaNs"
        if warmup_raises:
            raise RuntimeError("warmup failed")

    runner = SimpleNamespace(
        _nan_checker=check,
        _hidden_nan_detector=None,
        _kv_cache_nan_detector=None,
        _add_pipeline_chunk_buckets=lambda: None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        enforce_eager=False,
        lora_config=None,
        maybe_setup_dummy_loras=lambda _: nullcontext(),
        _precompile_timed=lambda _: nullcontext(),
        _precompile_backbone=lambda: None,
        _precompile_sampling_subgraphs=lambda: None,
        _precompile_mamba_rollback_helpers=lambda: None,
        _precompile_spec_boundary_pairs=lambda: None,
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        model=None,
        _is_async_drafter=True,
        _async_drafter=SimpleNamespace(precompile=lambda: None),
        _pcp_enabled=pcp,
        _mtp_enabled=True,
        _precompile_rejection_sampler=lambda: None,
        _precompile_spec_logprobs=lambda: None,
        _warmup_spec_decode=warmup,
        _warmup_pcp_mtp_prefill=warmup,
    )
    monkeypatch.setattr(tpu_runner, "maybe_create_mm_encoder_manager", lambda *_: None)
    if warmup_raises:
        with pytest.raises(RuntimeError, match="warmup failed"):
            tpu_runner.TPUModelRunner.capture_model(runner)
    else:
        tpu_runner.TPUModelRunner.capture_model(runner)
    assert runner._nan_checker is check
    with caplog.at_level(logging.ERROR):
        runner._nan_checker.check_logits(
            torch.full((1, 2), float("nan")), num_valid_rows=1
        )
        consume(check)
    assert "logits" in caplog.text

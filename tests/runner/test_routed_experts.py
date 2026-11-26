# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests the runner's routed-experts steps against vLLM's real worker
connector."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from runner_output_test_utils import _host_copy_state, step_runner
from vllm.config.aux_output import AuxOutputConfig
from vllm.distributed.aux_output_connector import worker as aux_output_worker
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
)
from vllm.platforms import current_platform

from tests.layers.adapter import routed_experts_test_utils as capture_utils
from vllm_torchtpu.runner import routed_experts
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

PADDING = capture_utils.PADDING


@pytest.fixture
def on_host(monkeypatch):
    """Returns the events the capture records."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    return capture_utils.use_host_events(monkeypatch)


def _capture(monkeypatch, model, is_first_rank=True):
    tp_group = SimpleNamespace(is_first_rank=is_first_rank)
    monkeypatch.setattr(aux_output_worker, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(routed_experts, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(routed_experts, "MoERunner", capture_utils.MoELayer)
    monkeypatch.setattr(routed_experts, "VllmMoERunner", capture_utils.MoELayer)
    return routed_experts.RoutedExpertsCapture(
        capture_utils.make_vllm_config(),
        model,
        SimpleNamespace(num_blocks=1024, kv_cache_groups=[]),
    )


@pytest.fixture
def capture(monkeypatch, on_host):
    model = capture_utils.make_moe_model()
    capture = _capture(monkeypatch, model)
    yield capture, model
    capture.close()


@pytest.fixture
def build_capture(monkeypatch, on_host):
    """Builds captures on models of the given MoE layer ids, and closes them
    after the test."""
    captures = []

    def build(layer_ids=None, is_first_rank=True):
        model = capture_utils.make_moe_model(layer_ids)
        captures.append(_capture(monkeypatch, model, is_first_rank))
        return captures[-1], model

    yield build
    for capture in captures:
        capture.close()


def _forward(capture, model, num_tokens, bucket, seed):
    """Runs one forward padded to `bucket` and returns its real rows."""
    ids = capture_utils.expert_ids(bucket, num_experts=PADDING, seed=seed)
    ids[num_tokens:] = PADDING
    capture_utils.model_forward(model, ids)
    capture.take_forward(bucket, num_tokens)
    return ids[:num_tokens]


def _begin(capture, requests, block_hashes=None):
    capture.begin_step(
        AuxOutputConnectorMetadata(
            generation=0,
            requests=dict.fromkeys(requests, 0),
            block_hashes=block_hashes or {},
            finished_requests=(),
        )
    )


def _hashes(*blocks):
    return [bytes([block]) * 32 for block in blocks]


def test_every_forward_of_a_step_reaches_its_requests(capture, on_host):
    """A step runs two forwards, and each writes the table from row 0. Each
    request still gets back its own rows, including `a`, whose prefill spans
    two steps."""
    capture, model = capture
    # The connector keeps a request's full blocks only in the store, under
    # these hashes, so `a` gets its first step's rows back from there.
    _begin(capture, ["a", "b", "c"], {"a": _hashes(0, 1, 2)})
    a_rows = _forward(capture, model, 12, 16, seed=0)
    bc_rows = _forward(capture, model, 9, 16, seed=1)
    token_starts = np.zeros(3, dtype=np.int32)
    pending = capture.prepare_output(
        ["a", "b", "c"], {"a": 12, "b": 6, "c": 3}, token_starts
    )
    # The runner's next step can overwrite this array before the output is
    # processed.
    token_starts[:] = 100

    # `a` is a partial prefill, so it samples nothing and gets no rows yet.
    outputs = pending.process_output([[], [7], [8]])

    assert set(outputs) == {"b", "c"}
    assert np.array_equal(outputs["b"].rows, bc_rows[:6].numpy())
    assert np.array_equal(outputs["c"].rows, bc_rows[6:].numpy())

    _begin(capture, ["a"], {"a": _hashes(3)})
    a_rest = _forward(capture, model, 4, 16, seed=2)
    pending = capture.prepare_output(["a"], {"a": 4}, np.array([12]))
    outputs = pending.process_output([[9]])

    expected = torch.cat([a_rows, a_rest]).numpy()
    assert np.array_equal(outputs["a"].rows, expected)
    assert all(event.synchronized for event in on_host)


def test_a_rank_that_does_not_publish_copies_nothing(build_capture, on_host):
    capture, model = build_capture(is_first_rank=False)
    _begin(capture, ["a"])
    _forward(capture, model, 5, 16, seed=0)

    token_starts = np.zeros(1, dtype=np.int32)
    assert capture.prepare_output(["a"], {"a": 5}, token_starts) is None
    assert not on_host
    assert not capture._capturer._step_copies


def test_layers_capture_cannot_serve_are_refused(monkeypatch, on_host):
    """Fused kernels are armed per layer, so the check runs on the loaded
    model and names each layer it refuses."""
    model = capture_utils.make_moe_model()
    model[1].picks_experts_in_route = False

    with pytest.raises(NotImplementedError, match=r"MoE layers \[1\]"):
        _capture(monkeypatch, model)


def test_moe_layers_sharing_a_layer_id_are_refused(build_capture):
    """Two layers with one id would write the same table column, and the
    record of the layers a forward wrote could not tell them apart."""
    with pytest.raises(ValueError, match=r"MoE layers have ids \[1\]"):
        build_capture(layer_ids=(0, 1, 1))


@pytest.mark.parametrize("is_first_rank", [True, False])
def test_the_warmup_check_names_a_layer_without_its_hook(build_capture, is_first_rank):
    """Every rank checks its own forward."""
    capture, model = build_capture(is_first_rank=is_first_rank)
    model[1].capture_fn = None
    capture_utils.model_forward(model, capture_utils.expert_ids(32))

    with pytest.raises(RuntimeError, match=r"MoE layers \[1\] .* 32-token forward"):
        capture.check_forward(32)


@pytest.mark.parametrize("most_model_len", [None, 64])
@pytest.mark.parametrize("enabled", [True, False])
def test_warmup_checks_every_backbone_forward(most_model_len, enabled):
    """Each warmup forward of every context bucket is checked right after it
    runs, before the next forward replaces its table."""
    events = []

    def dummy_run(num_tokens, num_reqs, num_blocks, use_max_model_len):
        events.append(("forward", num_tokens, num_reqs, num_blocks, use_max_model_len))

    capture = SimpleNamespace(
        check_forward=lambda num_tokens: events.append(("check", num_tokens))
    )
    runner = SimpleNamespace(
        num_reqs_max_model_len=8,
        max_num_blocks_per_req=4,
        most_model_len=most_model_len,
        num_reqs_most_model_len=16,
        num_blocks_per_most_len_req=2,
        _dummy_run=dummy_run,
        _routed_experts=capture if enabled else None,
    )

    TPUModelRunner._backbone_dummy_runs(runner, 32)

    contexts = [(8, 4, True)] + ([(16, 2, False)] if most_model_len else [])
    checks = [("check", 32)] if enabled else []
    assert events == [
        event for context in contexts for event in [("forward", 32, *context), *checks]
    ]


def _reserve_config(
    monkeypatch,
    *,
    max_num_batched_tokens=None,
    max_num_seqs=None,
    max_concurrent_batches=None,
):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    vllm_config = capture_utils.make_vllm_config()
    vllm_config.aux_output_config = AuxOutputConfig(enable_return_routed_experts=True)
    scheduler_config = vllm_config.scheduler_config
    if max_num_batched_tokens is not None:
        scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    if max_num_seqs is not None:
        scheduler_config.max_num_seqs = max_num_seqs
    if max_concurrent_batches is not None:
        vllm_config.max_concurrent_batches = max_concurrent_batches
    return vllm_config


# Bytes of one table row: 12 uint16 ids.
ROW_BYTES = capture_utils.NUM_LAYERS * capture_utils.TOPK * 2


def test_hbm_reserve_covers_the_forward_tables_in_flight(monkeypatch):
    """Two steps in flight, each with at most 8 forwards (one request each at
    least), whose tables pad at most to twice their tokens or 16 rows: 2 * (2
    * 64 + 8 * 16) rows. Plus the latest table of 64 rows, which the capturer
    keeps until the next forward returns. That is more than vLLM's int32
    table of 64 rows."""
    vllm_config = _reserve_config(monkeypatch)
    paddings = capture_utils.TOKEN_PADDINGS

    reserve = routed_experts.hbm_reserve_bytes(vllm_config, paddings)

    assert reserve == (2 * (2 * 64 + 8 * 16) + 64) * ROW_BYTES
    assert reserve > 64 * 12 * 4
    vllm_config.aux_output_config = AuxOutputConfig()
    assert routed_experts.hbm_reserve_bytes(vllm_config, paddings) == 0


def test_hbm_reserve_covers_buckets_that_jump_more_than_double(monkeypatch):
    """A user's `compile_sizes` can jump more than 2x, here 8x. Seven
    forwards of 9 tokens each pad to 64 rows, 448 in all, which a bound of
    twice the step's tokens (2 * 64 + 8 * 8 = 192) would miss. The reserve
    must cover every step of equal-sized forwards."""
    vllm_config = _reserve_config(monkeypatch)
    paddings = [8, 64]
    scheduler_config = vllm_config.scheduler_config
    num_tokens = scheduler_config.max_num_batched_tokens

    reserve = routed_experts.hbm_reserve_bytes(vllm_config, paddings)

    rows = reserve // ROW_BYTES
    step_rows = (rows - paddings[-1]) // vllm_config.max_concurrent_batches
    for forward_tokens in range(1, num_tokens + 1):
        num_forwards = min(num_tokens // forward_tokens, scheduler_config.max_num_seqs)
        bucket = min(size for size in paddings if size >= forward_tokens)
        assert num_forwards * bucket <= step_rows, forward_tokens


def test_hbm_reserve_of_far_apart_buckets_counts_whole_forwards(monkeypatch):
    """With the buckets of `docs/deployment/v7x_recipes.md`, 8 and 8192 are
    1024x apart, so 1024 times the step's tokens would reserve 8,388,616 rows
    a step. A step runs at most 8 forwards, one request each at least, and
    none pads past 8192 rows, so the reserve counts 8 * 8192 rows a step."""
    vllm_config = _reserve_config(
        monkeypatch, max_num_batched_tokens=8192, max_num_seqs=8
    )

    reserve = routed_experts.hbm_reserve_bytes(vllm_config, [1, 8, 8192])

    assert reserve == (2 * 8 * 8192 + 8192) * ROW_BYTES


def test_hbm_reserve_bounds_a_step_by_its_scheduled_tokens(monkeypatch):
    """Startup accepts `--max-num-batched-tokens 16384` with
    `--max-num-scheduled-tokens 256` and buckets up to 256, because the
    scheduler stops a step at 256 tokens. The reserve must size a step by
    those 256 tokens too: 8 forwards of at most 256 rows, as 16x the step's
    tokens is larger. vLLM's int32 table still holds 16384 rows."""
    vllm_config = _reserve_config(
        monkeypatch, max_num_batched_tokens=16384, max_num_seqs=8
    )
    vllm_config.scheduler_config.max_num_scheduled_tokens = 256

    reserve = routed_experts.hbm_reserve_bytes(vllm_config, [16, 256])

    assert reserve == max(
        (2 * 8 * 256 + 256) * ROW_BYTES, 16384 * capture_utils.NUM_LAYERS
        * capture_utils.TOPK * 4,
    )  # fmt: skip


def test_hbm_reserve_covers_two_tables_of_the_largest_bucket(monkeypatch):
    """A synchronous step of one request fills one table of at most 64 rows.
    Warmup also runs the 128-row bucket, which is above the step's tokens,
    and the capturer keeps each forward's table until the next forward
    returns, so two 128-row tables are held at once."""
    vllm_config = _reserve_config(
        monkeypatch, max_num_batched_tokens=64, max_num_seqs=1,
        max_concurrent_batches=1,
    )  # fmt: skip

    reserve = routed_experts.hbm_reserve_bytes(vllm_config, [16, 32, 64, 128])

    assert reserve == 2 * 128 * ROW_BYTES


def test_a_row_format_wider_than_the_reserve_fails(monkeypatch, on_host):
    """The reserve is computed before the capturer exists. If vLLM's capturer
    stored wider ids than the reserve counts, building the capture fails."""
    monkeypatch.setattr(routed_experts, "_RESERVED_ID_DTYPE", torch.uint8)

    with pytest.raises(AssertionError, match="HBM reserve"):
        _capture(monkeypatch, capture_utils.make_moe_model())


def _step_runner(capture, model, chunks, bucket):
    """A runner stand-in that runs one step as `chunks` forwards, each of
    which writes fresh expert ids. Returns the runner and the ids each
    forward wrote."""
    written = []

    def forward_model(**_kwargs):
        ids = capture_utils.expert_ids(bucket, num_experts=PADDING, seed=len(written))
        capture_utils.model_forward(model, ids)
        written.append(ids)
        return torch.zeros(bucket, 2), None

    runner = step_runner(chunks, bucket, forward_model, _routed_experts=capture)
    return runner, written


def _scheduler_output(num_scheduled_tokens, finished_requests=()):
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_encoder_inputs={},
        num_scheduled_tokens=num_scheduled_tokens,
        finished_req_ids=set(finished_requests),
        has_structured_output_requests=False,
        aux_output_connector_metadata=AuxOutputConnectorMetadata(
            generation=0,
            requests=dict.fromkeys(num_scheduled_tokens, 0),
            block_hashes={},
            finished_requests=tuple(finished_requests),
        ),
    )


@contextlib.contextmanager
def _runner_patches():
    no_context = lambda *_args, **_kwargs: contextlib.nullcontext()  # noqa: E731
    with (
        patch("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", lambda: False),
        patch("vllm_torchtpu.runner.tpu_runner.set_forward_context", no_context),
        patch(
            "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
            no_context,
        ),
        # The sampled tokens are already on the host.
        patch.object(
            AsyncTPUCopyState,
            "from_device_chunks",
            _host_copy_state,
        ),
    ):
        yield


def test_runner_returns_each_requests_rows_from_a_two_forward_step(capture):
    """The runner takes each forward with its real token count and hands
    `prepare_output` the requests in the order the forwards ran them, so each
    request gets back the rows its forward wrote."""
    capture, model = capture
    chunks = [[("a", 12)], [("b", 6), ("c", 3)]]
    runner, written = _step_runner(capture, model, chunks, bucket=16)

    with _runner_patches():
        scheduler_output = _scheduler_output({"a": 12, "b": 6, "c": 3})
        assert runner.execute_model(scheduler_output) is None
        output = runner.sample_tokens(None)

    rows = {
        req_id: result.rows
        for req_id, result in output.aux_output_connector_output.items()
    }
    assert set(rows) == {"a", "b", "c"}
    assert np.array_equal(rows["a"], written[0][:12].numpy())
    assert np.array_equal(rows["b"], written[1][:6].numpy())
    assert np.array_equal(rows["c"], written[1][6:9].numpy())


def test_a_zero_token_step_releases_the_requests_it_finishes(capture):
    """The scheduler reports a finished request in one step's metadata only,
    and that step may schedule no tokens. The runner still hands the report
    to the connector, which then releases the request."""
    capture, model = capture
    chunks = [[("a", 12)], [("b", 6), ("c", 3)]]
    runner, _ = _step_runner(capture, model, chunks, bucket=16)

    with _runner_patches():
        runner.execute_model(_scheduler_output({"a": 12, "b": 6, "c": 3}))
        runner.sample_tokens(None)
        runner.execute_model(_scheduler_output({}, finished_requests=["a"]))

    assert set(capture._connector._requests) == {"b", "c"}


@pytest.mark.parametrize("enabled", [True, False])
def test_initialize_kv_cache_builds_capture_only_when_enabled(monkeypatch, enabled):
    built = MagicMock()
    monkeypatch.setattr(routed_experts, "RoutedExpertsCapture", built)
    aux_output_config = AuxOutputConfig(enable_return_routed_experts=enabled)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(aux_output_config=aux_output_config),
        model=object(),
        kv_cache_manager=MagicMock(),
        pcp_manager=None,
        _init_prefill_checkpoint_units=lambda _config: None,
        _routed_experts=None,
    )
    kv_cache_config = object()

    with patch("vllm_torchtpu.runner.tpu_runner.set_current_vllm_config"):
        TPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    if enabled:
        built.assert_called_once_with(runner.vllm_config, runner.model, kv_cache_config)
        assert runner._routed_experts is built.return_value
    else:
        built.assert_not_called()
        assert runner._routed_experts is None

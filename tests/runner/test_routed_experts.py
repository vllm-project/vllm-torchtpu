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
from runner_output_test_utils import _host_copy_state
from vllm.config.aux_output import AuxOutputConfig
from vllm.distributed.aux_output_connector import worker as aux_output_worker
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
)
from vllm.platforms import current_platform

from tests.layers.adapter import routed_experts_test_utils as capture_utils
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    vllm_adapter as adaptive_moe,
)
from vllm_torchtpu.layers.adapter import (
    expert_placement,
    fused_moe_ep,
    moe_tp_decode,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner import routed_experts
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

PADDING = capture_utils.PADDING


@pytest.fixture
def on_host(monkeypatch):
    """Returns the events the capture records."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    return capture_utils.use_host_events(monkeypatch)


def _capture(monkeypatch, model, is_first_rank=True, data_parallel_rank=0):
    tp_group = SimpleNamespace(is_first_rank=is_first_rank)
    monkeypatch.setattr(aux_output_worker, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(routed_experts, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(routed_experts, "MoERunner", capture_utils.MoELayer)
    return routed_experts.RoutedExpertsCapture(
        capture_utils.make_vllm_config(data_parallel_rank=data_parallel_rank),
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

    def build(layer_ids=None, is_first_rank=True, data_parallel_rank=0, **layer_kwargs):
        model = capture_utils.make_moe_model(layer_ids, **layer_kwargs)
        captures.append(_capture(monkeypatch, model, is_first_rank, data_parallel_rank))
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
    """Begins a step of `requests`: their ids, or a dict of each id's first
    position that vLLM's connector has not returned yet."""
    capture.begin_step(
        AuxOutputConnectorMetadata(
            generation=0,
            requests=requests
            if isinstance(requests, dict)
            else dict.fromkeys(requests, 0),
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
        ["a", "b", "c"], {"a": 12, "b": 6, "c": 3}, {}, token_starts
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
    pending = capture.prepare_output(["a"], {"a": 4}, {}, np.array([12]))
    outputs = pending.process_output([[9]])

    expected = torch.cat([a_rows, a_rest]).numpy()
    assert np.array_equal(outputs["a"].rows, expected)
    assert all(event.synchronized for event in on_host)


@pytest.mark.parametrize("data_parallel_rank", [1, 0])
@pytest.mark.parametrize("vllm_gathers_dp_rows", [True, False])
def test_a_dp_rank_returns_its_own_rows(
    build_capture, vllm_gathers_dp_rows, data_parallel_rank
):
    """With DP and EP, vLLM gathers the rows of every DP rank before a MoE
    layer's hook, in the order of the DP group and each rank padded to the
    same bucket, unless the MoE method gathers them itself after picking the
    experts. Either way a request on the rank at position 1 of the DP group
    gets back that rank's rows, also when the engine's DP index is 0, as when
    RayDistributedExecutorV2 places the engine with DP index 1 on the first
    chips."""
    capture, model = build_capture(
        data_parallel_rank=data_parallel_rank,
        dp_size=2,
        dp_rank=1,
        vllm_gathers_dp_rows=vllm_gathers_dp_rows,
    )
    _begin(capture, ["a"])
    rows = capture_utils.expert_ids(16, num_experts=PADDING, seed=1)
    rows[5:] = PADDING
    if vllm_gathers_dp_rows:
        other_rank_rows = capture_utils.expert_ids(16, num_experts=PADDING, seed=0)
        hook_rows = torch.cat([other_rank_rows, rows])
    else:
        hook_rows = rows
    capture_utils.model_forward(model, hook_rows)
    capture.take_forward(16, 5)

    pending = capture.prepare_output(["a"], {"a": 5}, {}, np.zeros(1, dtype=np.int32))
    outputs = pending.process_output([[7]])

    assert np.array_equal(outputs["a"].rows, rows[:5].numpy())


def test_layers_that_differ_in_who_gathers_are_rejected(monkeypatch, on_host):
    """The capturer keeps one count of gathered DP ranks for every layer, so a
    model where vLLM gathers the rows for some MoE layers and the MoE method
    for the others cannot be captured."""
    model = capture_utils.make_moe_model(dp_size=2, vllm_gathers_dp_rows=True)
    model[1].do_naive_dispatch_combine = False

    with pytest.raises(
        NotImplementedError, match=r"vLLM gathers them before MoE layers \[0, 2\]"
    ):
        _capture(monkeypatch, model)


def test_a_rank_that_does_not_publish_copies_nothing(build_capture, on_host):
    capture, model = build_capture(is_first_rank=False)
    _begin(capture, ["a"])
    _forward(capture, model, 5, 16, seed=0)

    token_starts = np.zeros(1, dtype=np.int32)
    assert capture.prepare_output(["a"], {"a": 5}, {}, token_starts) is None
    assert not on_host
    assert not capture._capturer._step_copies


def test_a_step_without_connector_metadata_leaves_no_rows_behind(capture):
    """The speculative decoding warmup runs steps that the scheduler did not
    build. They carry no connector metadata, so vLLM's connector takes no
    snapshot of them. Their forwards are not copied to the host, so none is
    left over for the next step's snapshot."""
    capture, model = capture
    capture.begin_step(None)
    _forward(capture, model, 7, 16, seed=0)
    token_starts = np.zeros(1, dtype=np.int32)
    assert capture.prepare_output(["w"], {"w": 7}, {}, token_starts) is None

    _begin(capture, ["b"])
    rows = _forward(capture, model, 3, 16, seed=1)
    pending = capture.prepare_output(["b"], {"b": 3}, {}, token_starts)
    outputs = pending.process_output([[7]])

    assert np.array_equal(outputs["b"].rows, rows.numpy())


def test_a_step_without_connector_metadata_still_checks_its_forwards(capture):
    capture, model = capture
    model[1].capture_fn = None
    capture.begin_step(None)

    with pytest.raises(RuntimeError, match=r"MoE layers \[1\]"):
        _forward(capture, model, 7, 16, seed=0)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_rows_of_rejected_drafts_are_dropped(capture, async_scheduling):
    """`b`, first in the step, runs its last token and 2 drafts and keeps all
    3 rows. `a` runs its last token and 3 drafts but keeps 1, so its last 2
    rows are dropped and its next step's row follows the kept ones. The
    scheduler lists no drafts for `c`, whose 1 sampled token keeps its row.
    Under async scheduling `a`'s next step is scheduled as if every draft
    were accepted, 2 positions late, and vLLM's connector puts its row after
    the kept ones. Each request starts at a block boundary, as after a
    prefix-cache hit."""
    capture, model = capture
    _begin(capture, {"a": 8, "b": 16, "c": 24})
    rows = _forward(capture, model, 8, 16, seed=0)
    pending = capture.prepare_output(
        ["b", "a", "c"],
        {"a": 4, "b": 3, "c": 1},
        {"a": [0, 0, 0], "b": [0, 0]},
        np.array([16, 8, 24], dtype=np.int32),
    )
    outputs = pending.process_output([[1, 2, 3], [1, 2], [1]])

    assert outputs["b"].token_start == 16
    assert np.array_equal(outputs["b"].rows, rows[:3].numpy())
    assert outputs["a"].token_start == 8
    assert np.array_equal(outputs["a"].rows, rows[3:5].numpy())
    assert np.array_equal(outputs["c"].rows, rows[7:].numpy())

    _begin(capture, {"a": 10})
    next_rows = _forward(capture, model, 1, 16, seed=1)
    scheduled_start = 12 if async_scheduling else 10
    pending = capture.prepare_output(
        ["a"], {"a": 1}, {}, np.array([scheduled_start], dtype=np.int32)
    )
    outputs = pending.process_output([[5]])

    assert outputs["a"].token_start == 10
    assert np.array_equal(outputs["a"].rows, next_rows.numpy())


@pytest.mark.parametrize(
    "owner, attr",
    [
        ("quant_method", fused_moe_ep.FUSED_MOE_EP_OP_ATTR),
        ("quant_method", adaptive_moe.FUSED_MOE_EP_OP_ATTR),
        ("quant_method", moe_tp_decode._OP_ATTR),
        ("routed_experts", expert_placement.EXPERT_ORDER_ATTR),
    ],
)
def test_layers_capture_cannot_serve_are_refused(monkeypatch, on_host, owner, attr):
    """Fused kernels and expert placement are set per layer, so the check
    runs on the loaded model and names each layer it refuses."""
    model = capture_utils.make_moe_model()
    target = model[1].routed_experts
    setattr(target.quant_method if owner == "quant_method" else target, attr, (0,))

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


def test_hbm_reserve_covers_the_forward_tables_in_flight(monkeypatch):
    """Two steps in flight, each with at most 8 forwards (one request each at
    least), whose tables pad at most to twice their tokens or 16 rows: the
    tables of 2 * (2 * 64 + 8 * 16) rows of 12 uint16 ids. That is more than
    vLLM's int32 table of 64 rows."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    vllm_config = capture_utils.make_vllm_config()
    paddings = capture_utils.TOKEN_PADDINGS

    vllm_config.aux_output_config = AuxOutputConfig(enable_return_routed_experts=True)
    reserve = routed_experts.hbm_reserve_bytes(vllm_config, paddings, 3)

    assert reserve == 2 * (2 * 64 + 8 * 16) * 12 * 2
    assert reserve > 64 * 12 * 4
    vllm_config.aux_output_config = AuxOutputConfig()
    assert routed_experts.hbm_reserve_bytes(vllm_config, paddings, 3) == 0


@pytest.mark.parametrize("enable_expert_parallel", [True, False])
def test_hbm_reserve_under_dp(monkeypatch, enable_expert_parallel):
    """With EP, every forward of a step is padded to the bucket of the DP
    engine with the most tokens, so each of a step's 3 forwards can hold the
    largest bucket's 64 rows. Without EP the engines run on their own, and
    the single-engine bound holds."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    vllm_config = capture_utils.make_vllm_config()
    vllm_config.aux_output_config = AuxOutputConfig(enable_return_routed_experts=True)
    vllm_config.parallel_config.data_parallel_size = 2
    vllm_config.parallel_config.enable_expert_parallel = enable_expert_parallel

    reserve = routed_experts.hbm_reserve_bytes(
        vllm_config, capture_utils.TOKEN_PADDINGS, 3
    )

    step_rows = 3 * 64 if enable_expert_parallel else 2 * 64 + 8 * 16
    assert reserve == 2 * step_rows * 12 * 2


def test_hbm_reserve_covers_buckets_that_jump_more_than_double(monkeypatch):
    """A user's `compile_sizes` can jump more than 2x, here 8x. Seven
    forwards of 9 tokens each pad to 64 rows, 448 in all, which a bound of
    twice the step's tokens (2 * 64 + 8 * 8 = 192) would miss. The reserve
    must cover every step of equal-sized forwards."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    vllm_config = capture_utils.make_vllm_config()
    vllm_config.aux_output_config = AuxOutputConfig(enable_return_routed_experts=True)
    paddings = [8, 64]
    scheduler_config = vllm_config.scheduler_config
    num_tokens = scheduler_config.max_num_batched_tokens

    reserve = routed_experts.hbm_reserve_bytes(vllm_config, paddings, 3)

    step_rows = reserve // (12 * 2 * vllm_config.max_concurrent_batches)
    assert step_rows == 8 * 64 + 8 * 8
    for forward_tokens in range(1, num_tokens + 1):
        num_forwards = min(num_tokens // forward_tokens, scheduler_config.max_num_seqs)
        bucket = min(size for size in paddings if size >= forward_tokens)
        assert num_forwards * bucket <= step_rows, forward_tokens


def _step_runner(capture, model, chunks, bucket):
    """A runner stand-in that runs one step as `chunks` forwards through the
    real `execute_model` and `sample_tokens`. Each chunk is a list of
    (request id, scheduled tokens), and its forward writes fresh expert ids.
    Returns the runner and the ids each forward wrote."""
    req_ids = [req_id for chunk in chunks for req_id, _ in chunk]
    chunk_starts = np.cumsum([0, *map(len, chunks)])
    written = []

    def prepare_inputs(_so, start_index, _num_decode, _num_windowed):
        chunk = chunks[list(chunk_starts).index(start_index)]
        counts = [n for _, n in chunk]
        runner.query_start_loc_np[: len(chunk) + 1] = np.cumsum([0, *counts])
        runner._last_sequence_layout_plan = SequenceLayoutPlan(
            descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(0, bucket),
            global_num_tokens=sum(counts),
            global_padded_num_tokens=bucket,
            local_num_tokens=sum(counts),
            local_padded_num_tokens=bucket,
            _packed_to_request_major_token_indices=None,
        )
        empty = torch.empty(0, dtype=torch.int32)
        logits_indices = torch.tensor(np.cumsum(counts) - 1)
        end_index = start_index + len(chunk)
        return (None, logits_indices, len(chunk), len(chunk), end_index, empty,
                empty, None, None, None)  # fmt: skip

    def forward_model(**_kwargs):
        ids = capture_utils.expert_ids(bucket, num_experts=PADDING, seed=len(written))
        capture_utils.model_forward(model, ids)
        written.append(ids)
        return torch.zeros(bucket, 2), None

    num_reqs = len(req_ids)
    runner = SimpleNamespace(
        _routed_experts=capture,
        _pp_handoff=None,
        _dp_step_other_decode=False,
        _pp_is_first=True,
        _pp_is_last=True,
        _pp_pending_scheduler_output=None,
        _pp_pending_kv_save_nan_check=None,
        execute_model_state=None,
        _update_states=lambda _so: None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _mtp_prefill_only=False,
        _reorder_batch_for_rpa=lambda _so: (0, 0),
        _count_input_chunks=lambda _so: (len(chunks), max(map(len, chunks))),
        _dp_coordinated_step=lambda *_args, **_kwargs: (bucket, len(chunks)),
        _dp_local_prompt_logprob_rows=lambda _so: 0,
        _dp_step_prompt_logprob_rows=0,
        _thinking_budget=None,
        _dp_num_tokens_across_dp=lambda _n: None,
        supports_mm_inputs=False,
        phase_based_profiler=None,
        _is_async_drafter=False,
        _greedy_top_tokens_this_step=lambda: False,
        _min_tokens_batch_state=lambda: {},
        _repetition_penalty=SimpleNamespace(begin_step=lambda _batch: False),
        vllm_config=SimpleNamespace(),
        maybe_setup_kv_connector=lambda *_args, **_kwargs: None,
        input_batch=SimpleNamespace(
            num_reqs=num_reqs,
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            max_num_logprobs=None,
            all_greedy=True,
            num_tokens_no_spec=[n for chunk in chunks for _, n in chunk],
            num_computed_tokens_cpu=np.zeros(num_reqs, dtype=np.int32),
            token_ids_cpu=np.zeros((num_reqs, 2 * bucket), dtype=np.int32),
        ),
        requests={
            req_id: SimpleNamespace(
                num_computed_tokens=0, num_tokens=n, output_token_ids=[]
            )
            for chunk in chunks
            for req_id, n in chunk
        },
        num_prompt_logprobs={},
        query_start_loc_np=np.zeros(num_reqs + 1, dtype=np.int32),
        _last_sequence_layout_plan=None,
        position_ids=torch.zeros(bucket, dtype=torch.int32),
        _flush_mamba_state_seed_copies=lambda: None,
        _reset_read_offsets_for_new_requests=lambda *_args: None,
        input_ids=torch.zeros(bucket, dtype=torch.int32),
        _apply_async_token_substitution=lambda ids, *_args: ids,
        _record_input_staging_fence=lambda: None,
        _get_model_inputs=lambda ids, _mm: (ids, None),
        _token_padding_update=lambda _n: None,
        mesh=None,
        _kv_cache_bundle=None,
        _stage_timing=MagicMock(),
        is_pooling_model=False,
        pcp_manager=None,
        sequence_layout_planner=SimpleNamespace(
            maybe_select_logits_hidden_states=lambda hs, _plan, idx: hs[idx]
        ),
        compute_logits_from_hidden_states=lambda hs: torch.zeros(hs.shape[0], 8),
        _attn_metadata_builder_ctx=SimpleNamespace(
            mamba_state_indices=None, unified_mamba_state_indices=None, seq_lens=None
        ),
        mamba_slot_read_offsets=None,
        speculative_config=None,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        _adaptive_verification=None,
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda logits: torch.full((logits.shape[0],), 7),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _prepare_inputs=prepare_inputs,
        forward_model=forward_model,
    )
    runner._stage_timing.stage.return_value = contextlib.nullcontext()
    runner._stage_timing.take_samples.return_value = ([], [])
    for name in (
        "_any_decode",
        "_readmit_trimmed_drafts",
        "_sample_chunk_logits",
        "_sample_plain_chunk",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
        "_build_kv_connector_output",
        "sample_tokens",
        "execute_model",
        "_maybe_select_logits_hidden_states",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner, written


def test_runner_returns_each_requests_rows_from_a_two_forward_step(capture):
    """The runner takes each forward with its real token count and hands
    `prepare_output` the requests in the order the forwards ran them, so each
    request gets back the rows its forward wrote."""
    capture, model = capture
    chunks = [[("a", 12)], [("b", 6), ("c", 3)]]
    runner, written = _step_runner(capture, model, chunks, bucket=16)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=21,
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"a": 12, "b": 6, "c": 3},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        has_structured_output_requests=False,
        aux_output_connector_metadata=AuxOutputConnectorMetadata(
            generation=0,
            requests={"a": 0, "b": 0, "c": 0},
            block_hashes={},
            finished_requests=(),
        ),
    )

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

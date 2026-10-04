# SPDX-License-Identifier: Apache-2.0
"""Async MTP contracts across real scheduler, runner and TPU operations.

No model weights are needed. Accelerator initialization is deferred to fixtures.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_tpu  # noqa: F401
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults,
    AsyncTPUCopyState,
)

pytestmark = pytest.mark.tpu_test


@pytest.fixture
def device():
    # Do not initialize the accelerator during test collection.
    result = torch.device("tpu")
    probe = torch.tensor([17], dtype=torch.int32, device=result)
    assert probe.cpu().tolist() == [17]
    return result


def request_state(req_id, block, output=()):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(10)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=32),
        generator=None,
        block_ids=([block],),
        num_computed_tokens=10 + max(len(output) - 1, 0),
        output_token_ids=list(output),
    )


def make_runner(device, monkeypatch, k):
    runner = object.__new__(TPUModelRunner)
    runner.device = device
    runner.speculative_config = (
        SimpleNamespace(num_speculative_tokens=k, use_ngram_gpu=lambda: False)
        if k
        else None
    )
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    # One engine, so no DP lockstep (_dp_lockstep_enabled reads this).
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=1, enable_expert_parallel=False
    )
    runner.use_async_scheduling = True
    runner.use_async_spec_decode = k > 0
    runner.prev_num_draft_tokens = SimpleNamespace(np=np.zeros(4, dtype=np.int32))
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner._pre_async_results = None
    runner._mamba_state_pos = {}
    runner._mamba_offset_seeded = set()
    runner._read_offset_scratch = {}
    runner._last_sequence_layout_plan = None
    runner._fast_token_substitution = False
    # Real upstream batch operations and CPU token buffers. The kernels being
    # checked (offset reset / source assembly / substitution / D2H) use TPU.
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=512,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=512,
        block_sizes=[256],
        kernel_block_sizes=[256],
        max_num_blocks_per_req=[8],
        num_spec_tokens=k,
    )
    # Unrelated model/encoder setup is outside these text-only unit tests.
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _ids: None,
        register_request=lambda _rid, _params: None,
    )
    runner.encoder_cache = {}
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner._may_reorder_batch = lambda _output: None
    monkeypatch.setattr(
        gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    return runner


def cached_step(runner, req_ids, *, num_scheduled=4):
    """Valid cached-request output: no finish, preemption or block replacement."""
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {rid: num_scheduled for rid in req_ids}
    output.total_num_scheduled_tokens = num_scheduled * len(req_ids)
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={
            rid: runner.requests[rid].prompt_token_ids
            + runner.requests[rid].output_token_ids
            for rid in req_ids
        },
        new_block_ids=[None] * len(req_ids),
        num_computed_tokens=[
            runner.requests[rid].num_computed_tokens for rid in req_ids
        ],
        num_output_tokens=[
            len(runner.requests[rid].output_token_ids) for rid in req_ids
        ],
    )
    k = runner.speculative_config.num_speculative_tokens
    output.scheduled_spec_decode_tokens = {rid: [-1] * k for rid in req_ids}
    return output


@pytest.mark.parametrize(
    ("layout", "transition", "offset", "expected"),
    [
        ("unified", "continue", 2, 2),
        ("unified", "pause", 0, 0),
        ("unified", "pause", 2, 2),
        ("unified", "new_owner", 2, 0),
        ("unified", "reuse_id", 2, 0),
        ("compact", "continue", 2, 2),
        ("compact", "new_owner", 2, 0),
        ("compact", "reuse_id", 2, 0),
    ],
)
def test_mamba_read_offset_follows_storage_owner(
    device, monkeypatch, transition, offset, expected, layout
):
    """Check offset ownership; compact pause needs separate state restoration."""
    runner = make_runner(device, monkeypatch, 3)
    a = request_state("A", 5, [100, 101, 102])
    b = request_state("B", 9, [200])
    a.block_ids = ([5, 6, 7, 8],)
    b.block_ids = ([9, 10, 11, 12],)
    runner.requests = {"A": a, "B": b}
    runner.mamba_slot_read_offsets = torch.zeros(16, dtype=torch.int32, device=device)
    runner._mamba_copy_plan = [(0, [torch.zeros((16, 8), device=device)])]
    runner._mamba_state_block_size = 256
    runner._mamba_ckpt_window = 4
    runner._pool_block_split = 1
    runner.cache_config = SimpleNamespace(block_size=256)
    runner._pending_mamba_state_copies = []
    runner._mamba_slot_stride = 4
    runner.mamba_state_indices_cpu = torch.zeros(2, dtype=torch.int32)
    runner._init_mamba_slot_pool(9)
    if layout == "compact":
        runner._mamba_copy_plan = []

    def enter(rid, output=None):
        if output is None:
            output = cached_step(runner, [rid])
        runner._update_states(output)  # Includes real GPUModelRunner._update_states.
        block = runner.requests[rid].block_ids[0][0]
        indices = torch.tensor([block, 0], dtype=torch.int32, device=device)
        if layout == "compact":
            indices = runner._build_mamba_state_indices(0, 1, 2)
        runner._attn_metadata_builder_ctx = SimpleNamespace(
            mamba_state_indices=indices if layout == "compact" else None,
            unified_mamba_state_indices=[indices] if layout == "unified" else None,
        )
        runner._collect_mamba_state_seed_copies(output, 0, 1)
        assert not runner._pending_mamba_state_copies  # Away from block boundaries.
        runner._reset_read_offsets_for_new_requests(0, 1)
        return indices

    indices = enter("A")
    slot = int(indices.cpu()[0])
    sampled = torch.tensor(
        [[100 + i if i <= offset else -1 for i in range(4)]],
        dtype=torch.int32,
        device=device,
    )
    runner._update_mamba_slot_read_offsets([indices], sampled, 1)
    before = runner.mamba_slot_read_offsets.cpu().tolist()[slot]
    assert before == offset

    if transition == "pause":
        enter("B")
        assert "A" not in runner.input_batch.req_id_to_index
        assert runner.requests["A"] is a and a.block_ids == ([5, 6, 7, 8],)
        if layout == "unified":
            assert runner.mamba_slot_read_offsets.cpu().tolist()[slot] == offset
        enter("A")
    elif transition == "new_owner":
        # A finishes and a distinct request receives block 5.
        runner.requests["C"] = request_state("C", 5)
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {"A"}
        runner._update_states(output)
        enter("C")
    elif transition == "reuse_id":
        # Upstream permits abort and resubmit with the same ID in one step.
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {"A"}
        output.num_scheduled_tokens = {"A": 10}
        output.total_num_scheduled_tokens = 10
        output.scheduled_new_reqs = [
            NewRequestData(
                req_id="A",
                prompt_token_ids=list(range(10)),
                mm_features=[],
                sampling_params=a.sampling_params,
                pooling_params=None,
                block_ids=a.block_ids,
                num_computed_tokens=0,
                lora_request=None,
            )
        ]
        enter("A", output)
        assert runner.requests["A"] is not a
    else:
        enter("A")

    actual = runner.mamba_slot_read_offsets.cpu().tolist()[slot]
    assert actual == expected, "same live KV owner must retain its accepted checkpoint"


def scheduler_for(request, k):
    scheduler = object.__new__(AsyncScheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.defer_block_free = False
    scheduler._inflight_prefills = set()
    scheduler.enable_return_routed_experts = False
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.use_v2_model_runner = False
    scheduler.use_pp = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=True)
    scheduler.prev_step_scheduled_req_ids = {request.request_id}
    scheduler._spec_token_placeholders = [-1] * k
    return scheduler


@pytest.mark.parametrize("k", [0, 1, 3])
def test_bootstrap_commits_once_after_upstream_update(device, monkeypatch, k):
    runner = make_runner(device, monkeypatch, k)
    state = request_state("A", 5)
    runner.requests = {"A": state}
    runner.input_batch.add_request(state)
    runner.input_batch.refresh_metadata()
    request = Request("A", list(range(10)), state.sampling_params, None)
    request.status = RequestStatus.RUNNING
    scheduler = scheduler_for(request, k)
    prefill = SchedulerOutput.make_empty()
    prefill.num_scheduled_tokens = {"A": 10}
    prefill.total_num_scheduled_tokens = 10
    prefill.num_spec_tokens_to_schedule = k
    scheduler._update_after_schedule(prefill)
    assert request.num_output_placeholders == 1  # Derived by real scheduler code.

    sampled = torch.tensor([[100]], dtype=torch.int32, device=device)
    if k:
        drafts = torch.tensor(
            [list(range(101, 101 + k))], dtype=torch.int32, device=device
        )
        chunks, next_indices, rejected, counts = (
            runner._assemble_async_prefill_bootstrap(drafts, [sampled], [1])
        )
    else:
        chunks, next_indices, rejected, counts = (
            [sampled.reshape(-1)],
            {0: 0},
            None,
            None,
        )
    records = [(0, state, 10, "A")]
    mapping = runner._update_placeholder([], records, next_indices, counts)
    runner._pre_async_results = AsyncPreResults(
        req_ids=["A"],
        next_tokens_tpu=chunks[0],
        request_seq_lens=records,
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy=mapping,
        copy_state=AsyncTPUCopyState.from_device_chunks([sampled], [1]),
        spec_decode_num_rejected_tokens=rejected,
        num_draft_per_req=counts,
    )
    decode = SchedulerOutput.make_empty()
    decode.num_scheduled_tokens = {"A": 1 + k}
    decode.total_num_scheduled_tokens = 1 + k
    decode.scheduled_spec_decode_tokens = {"A": [-1] * k} if k else {}
    decode.scheduled_cached_reqs = scheduler._make_cached_request_data(
        [request],
        [],
        decode.num_scheduled_tokens,
        decode.scheduled_spec_decode_tokens,
        {"A": KVCacheBlocks(([],))},
    )
    assert decode.scheduled_cached_reqs.num_output_tokens == [1]
    runner._update_states(decode)
    runner._modify_prev_results()
    actual = {
        "end": int(runner.input_batch.num_tokens_no_spec[0]),
        "tokens": runner.input_batch.token_ids_cpu[0, :11].tolist(),
        "history": state.output_token_ids,
    }
    expected = {"end": 11, "tokens": list(range(10)) + [100], "history": [100]}
    assert actual == expected, (
        "upstream trim and TPU rollback must commit one bootstrap output"
    )


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("mode", ["pure", "mixed"])
def test_proposed_drafts_reach_next_inputs(device, k, mode):
    runner = object.__new__(TPUModelRunner)
    runner.device = device
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=k)
    runner._last_sequence_layout_plan = None
    runner._fast_token_substitution = False
    drafts = torch.tensor(
        [list(range(101, 101 + k)), list(range(201, 201 + k))],
        dtype=torch.int32,
        device=device,
    )
    if mode == "mixed":
        verified = torch.tensor([[100] + [-1] * k], dtype=torch.int32, device=device)
        newly_sampled = torch.tensor(
            [[200] + [-1] * k], dtype=torch.int32, device=device
        )
        state = SimpleNamespace(
            num_reqs_list=[1, 1],
            spec_decode_metadata_list=[
                SimpleNamespace(
                    draft_lengths=torch.tensor([k], device=device),
                    draft_lengths_cpu=np.array([k]),
                ),
                None,
            ],
        )
        chunks, _indices, rejected, counts = runner._assemble_async_spec_substitution(
            drafts, [verified, newly_sampled], state
        )
        ids = ["verify", "new"]
    else:
        newly_sampled = torch.tensor([[200]], dtype=torch.int32, device=device)
        chunks, _indices, rejected, counts = runner._assemble_async_prefill_bootstrap(
            drafts[1:], [newly_sampled], [1]
        )
        ids = ["new"]

    runner.input_batch = SimpleNamespace(req_ids=ids)
    source = torch.cat(chunks)
    runner._pre_async_results = AsyncPreResults(
        req_ids=ids,
        next_tokens_tpu=source,
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={rid: i for i, rid in enumerate(ids)},
        spec_decode_num_rejected_tokens=rejected,
        num_draft_per_req=counts,
    )
    dst, src = runner._prepare_async_token_substitution_indices(
        0, len(ids), np.full(len(ids), 1 + k, dtype=np.int32)
    )
    placeholder_inputs = torch.full(
        (len(ids) * (1 + k),), -1, dtype=torch.int32, device=device
    )
    replaced = runner._apply_async_token_substitution(placeholder_inputs, dst, src)
    actual = replaced.cpu().reshape(len(ids), 1 + k)[-1].tolist()
    expected = [200] + list(range(201, 201 + k))
    assert actual == expected, (
        "a non-verify chunk must carry its real proposals into the next step"
    )

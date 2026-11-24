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
"""Spec tests for CircularBufferSpec KV ring lifecycle and DP+EP runner lockstep."""

import contextlib
import inspect
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from runner_input_test_utils import (
    _BlockTable,
    _fake_tpu_events,
    _make_runner,
    _scheduler_output,
)
from vllm import forward_context as fc_mod
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu import _patch_circular_buffer_block_zeroing, patch_registry
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    stage_block_table_uploads,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner import tpu_runner as tpu_runner_mod
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _spec_warmup_group_num_blocks,
)


def _make_circular_spec(block_size: int = 64) -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _make_full_spec(block_size: int = 1024) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_should_record_circular_buffer_new_block_ids_when_zeroing_enabled(
    monkeypatch,
):
    target = "vllm_torchtpu:_patch_circular_buffer_block_zeroing"
    applied = set(patch_registry._applied) - {target}  # noqa: SLF001 -- reset registry
    monkeypatch.setattr(patch_registry, "_applied", applied)
    unwrapped_init = inspect.unwrap(CircularBufferManager.__init__)
    monkeypatch.setattr(CircularBufferManager, "__init__", unwrapped_init)

    patch_registry.apply("engine_core")
    patched_once = CircularBufferManager.__init__
    assert patched_once is not unwrapped_init
    # Second application must be idempotent rather than stacking wrappers.
    _patch_circular_buffer_block_zeroing()
    assert CircularBufferManager.__init__ is patched_once

    circular_spec = _make_circular_spec(block_size=64)
    fp8_full_spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
    )
    # Mixed-precision KV cache enables KVCacheConfig.needs_kv_cache_zeroing naturally.
    zeroed_cfg = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=fp8_full_spec),
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=circular_spec),
        ],
    )
    assert zeroed_cfg.needs_kv_cache_zeroing is True
    zeroed_coord = get_kv_cache_coordinator(
        kv_cache_config=zeroed_cfg,
        max_model_len=2048,
        max_in_flight_tokens=2048,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=64,
        hash_block_size=64,
    )
    ring_mgr = zeroed_coord.single_type_managers[1]
    assert isinstance(ring_mgr, CircularBufferManager)

    claimed_r0 = zeroed_coord.allocate_new_blocks(
        "req-0", num_tokens=128, num_tokens_main_model=128
    )
    claimed_r1 = zeroed_coord.allocate_new_blocks(
        "req-1", num_tokens=128, num_tokens_main_model=128
    )
    # Re-allocating for an active request claims 0 additional ring blocks.
    claimed_r0_again = zeroed_coord.allocate_new_blocks(
        "req-0", num_tokens=64, num_tokens_main_model=64
    )
    assert claimed_r0_again[1] == []
    expected_ids = [claimed_r0[1][0].block_id, claimed_r1[1][0].block_id]
    assert len(set(expected_ids)) == 2
    assert ring_mgr.new_block_ids == expected_ids

    unzeroed_cfg = KVCacheConfig(
        num_blocks=12,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=circular_spec)
        ],
    )
    assert unzeroed_cfg.needs_kv_cache_zeroing is False
    unzeroed_coord = get_kv_cache_coordinator(
        kv_cache_config=unzeroed_cfg,
        max_model_len=2048,
        max_in_flight_tokens=2048,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=64,
        hash_block_size=64,
    )
    unzeroed_coord.allocate_new_blocks(
        "req-0", num_tokens=256, num_tokens_main_model=256
    )
    assert len(unzeroed_coord.get_blocks("req-0")[0]) == 1
    assert unzeroed_coord.single_type_managers[0].new_block_ids == []

    # Wrapper must stay compatible when upstream adds extra __init__ parameters.
    forwarded = []

    def future_init(
        self,
        kv_cache_spec,
        needs_kv_cache_zeroing: bool = False,
        future_upstream_flag: bool = False,
    ) -> None:
        del kv_cache_spec
        self._record_new_block_ids = False  # noqa: SLF001 -- mimic upstream skip
        forwarded.append((needs_kv_cache_zeroing, future_upstream_flag))

    monkeypatch.setattr(CircularBufferManager, "__init__", future_init)
    _patch_circular_buffer_block_zeroing()
    probe = CircularBufferManager(
        circular_spec,
        needs_kv_cache_zeroing=True,
        future_upstream_flag=True,
    )
    probe_pos = CircularBufferManager(circular_spec, True)
    assert forwarded == [(True, True), (True, False)]
    assert probe._record_new_block_ids is True  # noqa: SLF001 -- verify flag set
    assert probe_pos._record_new_block_ids is True  # noqa: SLF001 -- positional arg


def test_should_clamp_circular_block_table_widths_in_attention_metadata(
    monkeypatch,
):
    _fake_tpu_events(monkeypatch)
    circular_spec = _make_circular_spec(block_size=64)
    uniform_circular_spec = UniformTypeKVCacheSpecs.from_specs(
        {
            "indexer.0": _make_circular_spec(block_size=64),
            "indexer.1": _make_circular_spec(block_size=64),
        }
    )

    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[128, 256],
        scheduled_tokens=[4, 4],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=64,
    )
    runner.most_model_len = 2048
    runner.num_reqs_most_model_len = 4
    runner.position_ids = torch.zeros((8,), dtype=torch.int32)

    # Ring tables hold 1 block per request despite most_model_len=2048.
    ring_table = torch.tensor([[3], [5], [99], [88]], dtype=torch.int32)
    uniform_table = torch.tensor([[7], [9], [77], [66]], dtype=torch.int32)
    runner.input_batch.block_table = [
        _BlockTable(ring_table),
        _BlockTable(uniform_table),
    ]

    ring_builder = AttentionMetadataBuilder(
        kv_cache_spec=circular_spec,
        layer_names=["indexer.0"],
        vllm_config=runner.vllm_config,
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )
    uniform_builder = AttentionMetadataBuilder(
        kv_cache_spec=uniform_circular_spec,
        layer_names=["indexer.0", "indexer.1"],
        vllm_config=runner.vllm_config,
        device=runner.device,
        runner=runner,
        kv_cache_group_id=1,
    )
    runner.attn_groups = [
        [SimpleNamespace(metadata_builders=[ring_builder])],
        [SimpleNamespace(metadata_builders=[uniform_builder])],
    ]

    common_md = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
        seq_lens=torch.tensor([132, 260, 0, 0], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=260,
        block_table_tensor=ring_table,
        slot_mapping=torch.empty(0, dtype=torch.int32),
        causal=True,
    )

    for use_max_len in (False, True):
        ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=use_max_len,
            seq_lens=torch.tensor([132, 260, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
            sequence_layout_descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        )
        runner._attn_metadata_builder_ctx = ctx  # noqa: SLF001 -- builder context wire

        ring_md = ring_builder.build(0, common_md)
        uniform_md = uniform_builder.build(0, common_md)

        assert ring_md.block_tables.tolist() == [3, 5, 0, 0]
        assert uniform_md.block_tables.tolist() == [7, 9, 0, 0]

        stage_block_table_uploads(runner, ctx, num_reqs_padded=4)
        staged_ring_md = ring_builder.build(0, common_md)
        staged_uniform_md = uniform_builder.build(0, common_md)
        assert staged_ring_md.block_tables.tolist() == [3, 5, 0, 0]
        assert staged_uniform_md.block_tables.tolist() == [7, 9, 0, 0]


def test_should_clamp_circular_group_blocks_during_spec_and_mtp_warmup(monkeypatch):
    full_spec = _make_full_spec(block_size=1024)
    circular_spec = _make_circular_spec(block_size=64)
    uniform_circular = UniformTypeKVCacheSpecs.from_specs(
        {"indexer.0": circular_spec, "indexer.1": _make_circular_spec(block_size=64)}
    )
    full_group = KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=full_spec)
    ring_group = KVCacheGroupSpec(
        layer_names=["indexer.0"], kv_cache_spec=circular_spec
    )
    uniform_ring_group = KVCacheGroupSpec(
        layer_names=["indexer.0", "indexer.1"], kv_cache_spec=uniform_circular
    )

    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[64],
        scheduled_tokens=[8],
        token_paddings=[8, 64, 512],
        pcp_size=1,
        max_num_reqs=8,
        block_size=1024,
    )
    runner.max_model_len = 4096
    runner.max_num_tokens = 512
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner.scheduler_config = SimpleNamespace(async_scheduling=False, max_num_seqs=4)

    # Helper returns 1 for CircularBufferSpec / UniformTypeKVCacheSpecs, else cdiv.
    helper = _spec_warmup_group_num_blocks
    assert helper(runner, ring_group, 3072) == 1
    assert helper(runner, uniform_ring_group, 3072) == 1
    assert helper(runner, full_group, 2500) == 3
    assert helper(runner, full_group, 99999) == 4

    # Pool has 6 blocks: full group at P=2048 needs 3 blocks (top=6 > 3+1), whereas
    # unclamped circular group (block_size=64) would demand 33 blocks and fail.
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[full_group, ring_group],
    )
    recorded_orders = []
    runner.requests = {}

    def record_execute_model(so):
        recorded_orders.append(so)
        for req in so.scheduled_new_reqs:
            runner.requests[req.req_id] = SimpleNamespace(output_token_ids=[])
        return None

    runner.execute_model = record_execute_model
    runner.sample_tokens = lambda *_args: None
    runner.take_draft_token_ids = lambda: None
    runner._warmup_spec_decode_cleanup = MagicMock()  # noqa: SLF001 -- warmup cleanup

    warmup_mtp = TPUModelRunner._warmup_one_mtp_prefill  # noqa: SLF001 -- warmup entry
    warmup_batch = TPUModelRunner._warmup_spec_batch  # noqa: SLF001 -- warmup entry
    succeeded = warmup_mtp(runner, P=2048, quiet=True)
    assert succeeded is True
    assert len(recorded_orders) == 1
    new_req = recorded_orders[0].scheduled_new_reqs[0]
    assert len(new_req.block_ids[0]) == 3
    assert len(new_req.block_ids[1]) == 1

    # Fail-fast when pool cannot satisfy max_nblk + 1 (top == 4 <= 3 + 1).
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[full_group, ring_group],
    )
    with pytest.raises(
        RuntimeError, match="MTP prefill warmup has not enough KV blocks"
    ):
        warmup_mtp(runner, P=2048, quiet=True)

    # _warmup_spec_batch with mixed=True clamps circular mix_nblks to 1 per req.
    recorded_orders.clear()
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[full_group, ring_group],
    )
    warmup_batch(runner, R=2, P=64, greedy=True, mixed=True)
    mix_new_reqs = [
        req
        for so in recorded_orders
        for req in so.scheduled_new_reqs
        if "mix" in req.req_id
    ]
    assert mix_new_reqs
    for req in mix_new_reqs:
        assert len(req.block_ids[0]) == 1
        assert len(req.block_ids[1]) == 1


def test_should_zero_trailing_positions_under_all_sequence_layout(monkeypatch):
    _fake_tpu_events(monkeypatch)
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[3],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=16,
    )
    runner.positions_cpu.fill_(777)

    so = _scheduler_output([3])
    prepare_inputs = TPUModelRunner._prepare_inputs  # noqa: SLF001 -- runner step entry
    prepare_inputs(runner, so, start_index=0, num_decode_reqs=0)

    assert runner.position_ids.tolist() == [0, 1, 2, 0, 0, 0, 0, 0]

    # Under PARTIAL layout, positions_cpu[total:global_padded] belongs to packed PCP
    # rank slices and must not be overwritten by the ALL-layout zeroing path.
    partial_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[3],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=16,
    )
    partial_plan = SequenceLayoutPlan(
        descriptor=SequenceLayoutDescriptor(kind=SequenceLayoutKind.PARTIAL),
        token_slice=slice(4, 8),
        global_num_tokens=3,
        global_padded_num_tokens=8,
        local_num_tokens=2,
        local_padded_num_tokens=4,
    )
    partial_runner.sequence_layout_planner = SimpleNamespace(
        reserve_host_token_capacity=lambda *_args, **_kwargs: None,
        prepare_real=lambda **_kwargs: partial_plan,
    )
    partial_runner.positions_cpu[4:8] = torch.tensor([10, 11, 0, 0], dtype=torch.int32)
    prepare_inputs(partial_runner, so, start_index=0, num_decode_reqs=0)
    assert partial_runner.position_ids.tolist() == [10, 11, 0, 0]

    # Under pcp_manager (layout_plan is None, zero_pad_tail is False),
    # _prepare_inputs must guard tail zeroing via zero_pad_tail rather than
    # dereferencing layout_plan.kind.
    pcp_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[3],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=16,
    )
    pcp_plan = SimpleNamespace(
        local_num_reqs=1,
        global_num_tokens=3,
        global_padded_num_tokens=8,
        local_num_tokens=2,
        local_padded_num_tokens=4,
        token_slice=slice(4, 8),
        logits_indices_cpu=None,
        request_distribution_counts=lambda decode, total: (decode, total),
    )
    pcp_runner.pcp_manager = SimpleNamespace(
        num_reqs_in_chunk=lambda q_lens, _seq_lens, capacity: min(
            capacity, len(q_lens)
        ),
        reserve_host_token_capacity=lambda *_args, **_kwargs: None,
        prepare_real=lambda **_kwargs: None,
        sharding_plan=pcp_plan,
    )
    pcp_runner.sequence_layout_planner = None
    pcp_runner.positions_cpu[4:8] = torch.tensor([20, 21, 0, 0], dtype=torch.int32)
    prepare_inputs(pcp_runner, so, start_index=0, num_decode_reqs=0)
    assert pcp_runner.position_ids.tolist() == [20, 21, 0, 0]


def test_should_populate_dp_metadata_and_clamp_idle_logits_indices_in_dummy_run(
    monkeypatch,
):
    _fake_tpu_events(monkeypatch)
    monkeypatch.setattr(
        tpu_runner_mod,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        tpu_runner_mod, "synchronize_tensors", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        fc_mod.current_platform,
        "set_additional_forward_context",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        fc_mod,
        "coordinate_batch_across_dp",
        lambda **_kwargs: pytest.fail("unexpected coordinate_batch_across_dp call"),
    )

    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32, 64],
        pcp_size=1,
        dp_size=4,
        dp_rank=2,
        enable_ep=True,
        max_num_reqs=48,
        block_size=16,
    )
    req_paddings_fn = tpu_runner_mod._get_req_paddings  # noqa: SLF001 -- bucket builder
    runner.num_reqs_paddings = req_paddings_fn(
        tpu_runner_mod.MIN_NUM_SEQS, runner.max_num_reqs
    )
    runner._profile_inputs = {}  # noqa: SLF001 -- runner profile state
    runner.vocab_size = 128
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: 8)
    runner._hidden_states_dtype = torch.bfloat16  # noqa: SLF001 -- precompile dtype
    runner._precompile_timed = lambda *_args: contextlib.nullcontext()  # noqa: SLF001
    profile_fn = TPUModelRunner._profile_input  # noqa: SLF001 -- bind runner profiler
    runner._profile_input = types.MethodType(profile_fn, runner)  # noqa: SLF001 -- bind
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    dummy_run = TPUModelRunner._dummy_run  # noqa: SLF001 -- runner dummy step under test
    precompile_logits = TPUModelRunner._precompile_compute_selected_logits  # noqa: SLF001

    seen_dp_metadata = []
    seen_logits_shapes = []
    seen_logits_pairs = []

    def fake_forward_model(
        *, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        del positions, inputs_embeds, intermediate_tensors
        seen_dp_metadata.append(get_forward_context().dp_metadata)
        return torch.zeros((input_ids.shape[0], 8), dtype=torch.bfloat16), None

    def fake_compute_selected_logits(hidden_states, logits_indices):
        seen_logits_shapes.append(tuple(logits_indices.shape))
        seen_logits_pairs.append((hidden_states.shape[0], logits_indices.shape[0]))
        return torch.zeros((logits_indices.shape[0], 16), dtype=torch.bfloat16)

    runner.forward_model = fake_forward_model
    runner.compute_selected_logits = fake_compute_selected_logits

    precompile_logits(runner)
    precompiled_grid = set(seen_logits_pairs)
    seen_logits_shapes.clear()
    seen_logits_pairs.clear()

    # 1. Startup warmup / profiling (dp_lockstep=False) sets DPMetadata on DP+EP.
    elapsed = dummy_run(
        runner,
        num_tokens=16,
        num_reqs=48,
        num_blocks=4,
        dp_lockstep=False,
        profile_prefix=32,
    )
    assert isinstance(elapsed, float)
    assert seen_dp_metadata[-1] is not None
    assert seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [16, 16, 16, 16]
    assert seen_logits_shapes == []

    # 2. Idle lockstep (dp_lockstep=True) clamps and pads logits to the precompile grid:
    # - num_tokens=16 < num_reqs=48 -> min=16 -> padded 16
    # - num_tokens=64 > num_reqs=24 (unpadded) -> min=24 -> padded 32 (not 24 or 48)
    # - num_tokens=64, num_reqs=64 > max_num_reqs=48 -> capped at max_num_reqs 48
    for num_tokens, num_reqs, expected_logits_reqs in (
        (16, 48, 16),
        (64, 24, 32),
        (64, 64, 48),
    ):
        res = dummy_run(
            runner,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_blocks=4,
            dp_lockstep=True,
        )
        assert res is None
        assert seen_dp_metadata[-1] is not None
        assert (
            seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [num_tokens] * 4
        )
        assert seen_logits_shapes[-1] == (expected_logits_reqs,)
        assert seen_logits_pairs[-1] in precompiled_grid

    # 2b. When DP vocab logits sharding is active, idle lockstep preserves the
    # cross-engine row count from _dp_vocab_logits_rows().
    runner._dp_vocab_logits = True  # noqa: SLF001 -- exercise DP vocab parallel branch
    runner._dp_vocab_logits_rows = lambda: 32  # noqa: SLF001 -- stub DP row bucket
    dummy_run(
        runner,
        num_tokens=16,
        num_reqs=48,
        num_blocks=4,
        dp_lockstep=True,
    )
    assert seen_logits_shapes[-1] == (32,)

    # 3. Non-lockstep configuration (dp_size=1, enable_ep=False) keeps dp_metadata=None.
    non_dp_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32],
        pcp_size=1,
        dp_size=1,
        dp_rank=0,
        enable_ep=False,
        max_num_reqs=16,
        block_size=16,
    )
    non_dp_runner.maybe_select_dummy_loras = runner.maybe_select_dummy_loras
    non_dp_runner.forward_model = fake_forward_model
    non_dp_runner.compute_selected_logits = fake_compute_selected_logits

    dummy_run(
        non_dp_runner,
        num_tokens=16,
        num_reqs=16,
        num_blocks=4,
        dp_lockstep=False,
    )
    assert seen_dp_metadata[-1] is None


@pytest.mark.tpu_test
def test_dp_ep_circular_runner_tpu_metadata_and_dummy_logits(device, monkeypatch):
    circular_spec = _make_circular_spec(block_size=64)
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[128, 256],
        scheduled_tokens=[3, 2],
        token_paddings=[8, 16, 32],
        pcp_size=1,
        dp_size=2,
        dp_rank=1,
        enable_ep=True,
        max_num_reqs=32,
        block_size=64,
    )
    runner.device = device
    runner.most_model_len = 2048
    runner.num_reqs_most_model_len = 4
    runner.positions_cpu.fill_(999)

    ring_table = torch.tensor([[3], [5], [91], [92]], dtype=torch.int32)
    runner.input_batch.block_table = [_BlockTable(ring_table)]

    so = _scheduler_output([3, 2])
    prepare_inputs = TPUModelRunner._prepare_inputs  # noqa: SLF001 -- runner step entry
    prepare_inputs(runner, so, start_index=0, num_decode_reqs=0)
    assert runner.position_ids.device.type == "tpu"
    assert runner.position_ids.cpu().tolist() == [0, 1, 2, 0, 1, 0, 0, 0]

    ring_builder = AttentionMetadataBuilder(
        kv_cache_spec=circular_spec,
        layer_names=["indexer.0"],
        vllm_config=runner.vllm_config,
        device=device,
        runner=runner,
        kv_cache_group_id=0,
    )
    ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([131, 258, 0, 0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32, device=device),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32, device=device),
        sequence_layout_descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    )
    runner._attn_metadata_builder_ctx = ctx  # noqa: SLF001 -- builder context wire
    common_md = CommonAttentionMetadata(
        query_start_loc=ctx.query_start_loc,
        query_start_loc_cpu=torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32),
        seq_lens=ctx.seq_lens,
        num_reqs=4,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=258,
        block_table_tensor=ring_table,
        slot_mapping=torch.empty(0, dtype=torch.int32),
        causal=True,
    )
    ring_md = ring_builder.build(0, common_md)
    assert ring_md.block_tables.device.type == "tpu"
    assert ring_md.block_tables.cpu().tolist() == [3, 5, 0, 0]

    monkeypatch.setattr(
        tpu_runner_mod,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    dummy_run = TPUModelRunner._dummy_run  # noqa: SLF001 -- runner dummy step under test
    seen_dp_dev_types = []
    seen_logits_shapes = []

    def tpu_forward_model(
        *, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        del positions, inputs_embeds, intermediate_tensors
        dp_md = get_forward_context().dp_metadata
        assert dp_md is not None
        seen_dp_dev_types.append(dp_md.num_tokens_across_dp_cpu.device.type)
        assert dp_md.num_tokens_across_dp_cpu.cpu().tolist() == [16, 16]
        return torch.zeros(
            (input_ids.shape[0], 8), dtype=torch.bfloat16, device=device
        ), None

    def tpu_compute_selected_logits(hidden_states, logits_indices):
        del hidden_states
        seen_logits_shapes.append(
            (tuple(logits_indices.shape), logits_indices.device.type)
        )
        return torch.zeros(
            (logits_indices.shape[0], 16), dtype=torch.bfloat16, device=device
        )

    runner.forward_model = tpu_forward_model
    runner.compute_selected_logits = tpu_compute_selected_logits

    dummy_run(
        runner,
        num_tokens=16,
        num_reqs=32,
        num_blocks=4,
        dp_lockstep=True,
    )
    assert seen_dp_dev_types == ["tpu"]
    assert seen_logits_shapes == [((16,), "tpu")]

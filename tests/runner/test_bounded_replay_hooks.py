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
"""Spec tests for bounded-replay tracking and HostScheduleSnapshot metadata."""

import contextlib
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import jax
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output, init_cpu_runner
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    HostScheduleSnapshot,
)
from vllm_torchtpu.runner.bounded_replay import (
    BoundedReplayTracker,
    prefix_replay_tokens,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.vllm_v1.worker.gpu_model_runner import GPUModelRunner


def _swa_kv_config() -> KVCacheConfig:
    swa_spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=32,
        bounded_replay=True,
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=swa_spec)
        ],
    )


def _build_metadata_on_device(dev: torch.device):
    bt = SimpleNamespace(
        max_num_blocks_per_req=1,
        blocks_per_kv_block=1,
        get_cpu_tensor=lambda: torch.tensor([[3], [5], [7], [9]], dtype=torch.int32),
    )
    runner = SimpleNamespace(
        device=dev,
        block_size=8,
        max_num_reqs=4,
        most_model_len=8,
        position_ids=torch.arange(4, dtype=torch.int32, device=dev),
        _block_table_stage_cpu=None,
        _free_h2d_sources={},
        _registered_h2d_sources=set(),
        _unfenced_h2d_sources=[],
        attn_groups=[],
        pcp_manager=None,
        _unified_kv_layout=False,
        input_batch=SimpleNamespace(block_table=[bt]),
    )
    spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["attn.0"],
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
                cp_kv_cache_interleave_size=8,
            )
        ),
        device=dev,
        runner=runner,
        kv_cache_group_id=0,
    )
    runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(  # noqa: SLF001 -- test runner builder context
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([24, 16, 0, 0], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device=dev),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32, device=dev),
        host_schedule=HostScheduleSnapshot(
            query_start_loc=(0, 2, 4, 4, 4),
            seq_lens=(24, 16, 0, 0),
            prompt_lens=(20, 16, 0, 0),
            replay_starts=(8, 0, 0, 0),
        ),
    )
    return builder.build(0, SimpleNamespace(num_reqs=4))


def test_should_drop_host_schedule_snapshot_in_pytree() -> None:
    meta = _build_metadata_on_device(torch.device("cpu"))
    assert meta.host_schedule == HostScheduleSnapshot(
        query_start_loc=(0, 2, 4, 4, 4),
        seq_lens=(24, 16, 0, 0),
        prompt_lens=(20, 16, 0, 0),
        replay_starts=(8, 0, 0, 0),
    )
    leaves, treedef = jax.tree_util.tree_flatten(meta)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.host_schedule is None


def test_should_track_and_evict_replay_starts_in_update_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = init_cpu_runner(monkeypatch)
    assert isinstance(runner._bounded_replay, BoundedReplayTracker)  # noqa: SLF001 -- verify tracker initialized
    assert runner._bounded_replay.replay_starts == {}  # noqa: SLF001 -- verify initial state
    assert prefix_replay_tokens(runner.kv_cache_config) == 0

    sched_out = SimpleNamespace(
        finished_req_ids={"finished_req"},
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="new_replayed",
                replay_start=16,
                num_computed_tokens=16,
                sampling_params=None,
                prompt_token_ids=[],
            ),
            SimpleNamespace(
                req_id="new_plain",
                replay_start=0,
                num_computed_tokens=0,
                sampling_params=None,
                prompt_token_ids=[],
            ),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["resumed_req", "active_req", "resumed_zero"],
            resumed_req_ids={"resumed_req", "resumed_zero"},
            num_computed_tokens=[24, 32, 0],
        ),
    )
    runner._bounded_replay = None  # noqa: SLF001 -- verify None tracker is no-op
    with patch.object(GPUModelRunner, "_update_states", return_value=False):
        assert runner._update_states(sched_out) is False  # noqa: SLF001 -- direct unit test of state hook
    assert runner._bounded_replay is None  # noqa: SLF001 -- verify uninitialized state preserved

    runner._bounded_replay = BoundedReplayTracker()  # noqa: SLF001 -- restore active tracker before KV config
    with patch.object(GPUModelRunner, "_update_states", return_value=False):
        assert runner._update_states(sched_out) is False  # noqa: SLF001 -- no-op when kv_cache_config is None
    assert runner._bounded_replay.replay_starts == {}  # noqa: SLF001 -- no starts recorded without bounded replay

    runner.kv_cache_config = _swa_kv_config()
    assert prefix_replay_tokens(runner.kv_cache_config) == 32
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True  # noqa: SLF001 -- populate replay starts via _update_states
    assert runner._bounded_replay.replay_starts == {  # noqa: SLF001 -- verify recorded starts
        "new_replayed": 16,
        "resumed_req": 24,
    }

    runner._bounded_replay.replay_starts = {  # noqa: SLF001 -- seed pre-eviction replay state
        "finished_req": 16,
        "resumed_req": 8,
        "survivor": 12,
    }
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True  # noqa: SLF001 -- direct unit test of eviction hook
    assert runner._bounded_replay.replay_starts == {  # noqa: SLF001 -- verify finished evicted and survivor kept
        "survivor": 12,
        "new_replayed": 16,
        "resumed_req": 24,
    }


@pytest.mark.parametrize("pcp_size", [1, 2])
@pytest.mark.parametrize("dp_size_and_lockstep", [(1, False), (8, True)])
@pytest.mark.parametrize("profiling", [False, True])
def test_should_populate_host_schedule_across_layouts_and_dummy_modes(
    pcp_size: int,
    dp_size_and_lockstep: tuple[int, bool],
    profiling: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dp_size, dp_lockstep = dp_size_and_lockstep
    if pcp_size > 1:
        pcp_mod = "vllm_torchtpu.layers.core.pcp_sequence_layout"
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_rank", lambda: 0)
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_world_size", lambda: pcp_size)
    runner = _make_runner(
        num_computed_tokens=[4, 0],
        prompt_tokens=[8, 6],
        scheduled_tokens=[2, 6],
        token_paddings=[16],
        pcp_size=pcp_size,
        max_num_reqs=4,
    )
    runner.dp_size = dp_size
    runner.dp_rank = 0
    runner._dp_vocab_logits = False  # noqa: SLF001 -- stub dp vocab logits state
    runner.parallel_config.data_parallel_size = dp_size
    runner.parallel_config.enable_expert_parallel = False
    runner.vllm_config.compilation_config.fast_moe_cold_start = False
    runner._bounded_replay.replay_starts = {"req0": 4}  # noqa: SLF001 -- seed bounded-replay start
    if dp_lockstep:
        runner.kv_cache_config = _swa_kv_config()
    TPUModelRunner._prepare_inputs(runner, _scheduler_output([2, 6]), 0, 0, 0)  # noqa: SLF001 -- exercise _prepare_inputs hook
    ctx = runner._attn_metadata_builder_ctx  # noqa: SLF001 -- inspect builder context
    expected_prep = (
        HostScheduleSnapshot(
            query_start_loc=(0, 2, 8, 8, 8),
            seq_lens=(6, 6, 0, 0),
            prompt_lens=(8, 6, 0, 0),
            replay_starts=(4, 0, 0, 0) if dp_lockstep else (0, 0, 0, 0),
        )
        if pcp_size == 1
        else None
    )
    assert ctx.host_schedule == expected_prep

    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors", lambda *_a, **_kw: None
    )
    runner.maybe_select_dummy_loras = lambda *_a, **_kw: contextlib.nullcontext()
    runner._record_input_staging_fence = lambda: None  # noqa: SLF001 -- stub dummy fence
    runner._dp_lockstep_enabled = lambda: dp_lockstep  # noqa: SLF001 -- stub dp lockstep state
    runner._dp_num_tokens_across_dp = lambda _n: None  # noqa: SLF001 -- stub dp allreduce
    runner.compute_selected_logits = lambda out, _idx: out
    runner._profile_inputs = {}  # noqa: SLF001 -- initialize profile cache
    runner.vocab_size = 128
    runner._pp_topk_buffer = None  # noqa: SLF001 -- disable pp topk buffer
    runner._profile_input = MethodType(TPUModelRunner._profile_input, runner)  # noqa: SLF001 -- bind profile helper
    runner.forward_model = lambda *, positions, **_kw: (
        torch.zeros((positions.shape[-1], 4), dtype=torch.float32),
        None,
    )
    TPUModelRunner._dummy_run(  # noqa: SLF001 -- exercise _dummy_run hook
        runner,
        8,
        4,
        2,
        use_max_model_len=True,
        dp_lockstep=dp_lockstep,
        profile_prefix=0 if profiling else None,
    )
    dummy_ctx = runner._attn_metadata_builder_ctx  # noqa: SLF001 -- inspect dummy builder context
    expected = HostScheduleSnapshot.zeros(4) if dp_lockstep and not profiling else None
    assert dummy_ctx.host_schedule == expected


def test_should_slice_prompt_and_replay_snapshots_at_nonzero_start_index() -> None:
    runner = _make_runner(
        num_computed_tokens=[2, 3, 6],
        prompt_tokens=[10, 20, 30],
        scheduled_tokens=[1, 1, 1],
        token_paddings=[16],
        pcp_size=1,
        max_num_reqs=4,
    )
    runner.kv_cache_config = _swa_kv_config()
    runner._bounded_replay.replay_starts = {"req0": 55, "req2": 77}  # noqa: SLF001 -- distinct starts on rows 0 and 2
    TPUModelRunner._prepare_inputs(runner, _scheduler_output([1, 1, 1]), 1, 3, 3)  # noqa: SLF001 -- prepare chunk at start_index=1
    ctx = runner._attn_metadata_builder_ctx  # noqa: SLF001 -- inspect chunk builder context
    assert ctx.host_schedule is not None
    assert ctx.host_schedule.prompt_lens == (20, 30, 0, 0)
    assert ctx.host_schedule.replay_starts == (0, 77, 0, 0)


@pytest.mark.tpu_test
def test_should_build_bounded_replay_metadata_on_tpu() -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only runtime extension imported inside tpu_test

    meta = _build_metadata_on_device(torch.device("tpu:0"))
    assert meta.block_tables.device.type == "tpu"
    assert meta.block_tables.cpu().tolist() == [3, 5, 0, 0]
    assert meta.host_schedule is not None
    assert meta.host_schedule.replay_starts == (8, 0, 0, 0)

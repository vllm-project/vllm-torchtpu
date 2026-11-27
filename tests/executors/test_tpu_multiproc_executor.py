# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor

# Sentinel returned by the parent get_kv_cache_specs; the override plumbing must
# pass this through unchanged.
_SPECS = [{"layer": MagicMock()}]


def _make_executor(engine_override, worker_overrides, pipeline_parallel_size=1):
    """Build an executor without running __init__, with a mocked engine
    cache_config and a collective_rpc that returns the per-worker overrides."""
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.vllm_config.parallel_config.world_size = len(worker_overrides)
    executor.vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_copies_override_from_workers_when_engine_unset(mock_super):
    """Engine override is None -> copy the agreed worker value to the engine."""
    executor = _make_executor(engine_override=None, worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_called_once_with("get_num_gpu_blocks_override")
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 5428


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_does_not_overwrite_existing_engine_override(mock_super):
    """A user-supplied engine override (e.g. --num-gpu-blocks-override) wins;
    the worker value is never queried."""
    executor = _make_executor(engine_override=1234, worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_not_called()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_raises_when_workers_disagree(mock_super):
    """Workers must agree on the override; a mismatch is a sizing bug, not
    something to silently pick one of. 5428 -> 6840 is 26%, far outside the
    relative tolerance."""
    executor = _make_executor(engine_override=None, worker_overrides=[5428, 6840, 5428])

    with pytest.raises(ValueError, match="workers disagree"):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_small_blocks_tolerate_measurement_jitter(mock_super):
    """The tolerance must scale with the block count.

    Workers size the pool from their own measured free HBM, so the counts
    never match exactly. With a small per-block size the same ~0.02 GiB of
    allocator jitter is hundreds of blocks: these are the eight values
    observed serving Kimi-Linear-48B at TP=8 (~140 KiB/block), a spread of
    158. Against an absolute tolerance of 4 this fails 100% of the time, so
    the model cannot be served at all, while Kimi-K3 (~14 MiB/block) passes
    on the same code because the identical jitter is only a few blocks."""
    observed = [551506, 551425, 551415, 551403, 551426, 551358, 551348, 551356]
    executor = _make_executor(engine_override=None, worker_overrides=observed)

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 551348


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_absolute_floor_applies_to_tiny_block_counts(mock_super):
    """The relative tolerance must not shrink below the absolute floor: at
    100 blocks 0.5% rounds to 0, which would demand exact agreement."""
    executor = _make_executor(engine_override=None, worker_overrides=[100, 102, 101])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 100


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_no_override_when_workers_return_none(mock_super):
    """If no worker computed an override (uniform sizing skipped), the engine
    stays None and vLLM falls back to its own num_blocks computation."""
    executor = _make_executor(engine_override=None, worker_overrides=[None, None, None])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_pipeline_stages_may_disagree_and_the_smallest_stage_wins(mock_super):
    """Each pipeline stage holds different layers, so its workers measure a
    different count; vLLM sizes every stage's pool to the minimum."""
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[100, 100, 100, 100] + [80, 80, 80, 80],
        pipeline_parallel_size=2,
    )

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 80


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_workers_within_a_pipeline_stage_must_still_agree(mock_super):
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[100, 100, 100, 150] + [80, 80, 80, 80],
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match="workers disagree"):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_pipeline_stage_without_an_override_is_rejected(mock_super):
    """vLLM applies the override to every stage, so a stage that set none
    (no attention or no mamba layers in it) has no measured capacity for
    the value the others computed."""
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[None, None] + [80, 81],
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match=r"stage\(s\) \[0\] set no"):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_pipeline_stages_without_any_override_leave_it_unset(mock_super):
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[None, None, None, None],
        pipeline_parallel_size=2,
    )

    executor.get_kv_cache_specs()

    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


def _make_ray_executor(engine_override, worker_overrides):
    """Same shape as _make_executor but for the Ray multi-host executor."""
    from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
        RayDistributedExecutorV2,
    )

    executor = RayDistributedExecutorV2.__new__(RayDistributedExecutorV2)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.vllm_config.parallel_config.world_size = len(worker_overrides)
    executor.vllm_config.parallel_config.pipeline_parallel_size = 1
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


def test_ray_executor_tolerates_measurement_jitter():
    """Regression for dev build 155: the Ray multi-host executor demanded
    EXACT agreement (assert len(set(overrides)) == 1) while the multiproc
    executor already tolerated jitter. Kimi-K3 at TP=32 landed 6456..6459
    (~14 MiB blocks, a 3-block spread from per-worker HBM measurement) and
    the whole engine start died on the bare assert."""
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    observed = [6459, 6456] * 16
    executor = _make_ray_executor(engine_override=None, worker_overrides=observed)
    with patch.object(RayExecutorV2, "get_kv_cache_specs", return_value=_SPECS):
        specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 6456


def test_ray_executor_raises_on_real_disagreement():
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    executor = _make_ray_executor(
        engine_override=None, worker_overrides=[6456, 8000] * 16
    )
    with (
        patch.object(RayExecutorV2, "get_kv_cache_specs", return_value=_SPECS),
        pytest.raises(ValueError, match="workers disagree"),
    ):
        executor.get_kv_cache_specs()


def _compacting_executor(
    uses_mrope=False, runner_type="generate", logits_processors=None
):
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = MagicMock()
    model_config = executor.vllm_config.model_config
    model_config.uses_mrope = uses_mrope
    model_config.runner_type = runner_type
    model_config.logits_processors = logits_processors
    return executor


def _scheduler_output(*new_reqs):
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    return SchedulerOutput(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={r.req_id: 1 for r in new_reqs},
        total_num_scheduled_tokens=len(new_reqs),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _new_req(req_id, prompt, **sampling):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(**sampling),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )


def test_new_prompts_travel_as_int32_arrays_and_engine_copy_keeps_lists():
    import numpy as np

    prompt = [0, 7, 200063, 2**31 - 1]
    original = _scheduler_output(_new_req("a", prompt))

    sent = _compacting_executor()._worker_copy(original)

    ids = sent.scheduled_new_reqs[0].prompt_token_ids
    assert isinstance(ids, np.ndarray) and ids.dtype == np.int32
    assert ids.tolist() == prompt
    # The engine core keeps reading its own object: still the list.
    assert original.scheduled_new_reqs[0].prompt_token_ids is prompt


@pytest.mark.parametrize(
    "sampling", [{"prompt_logprobs": 1}, {"thinking_token_budget": 16}]
)
def test_requests_whose_ids_are_read_as_lists_keep_their_list(sampling):
    prompt = [1, 2, 3]
    out = _compacting_executor()._worker_copy(
        _scheduler_output(_new_req("a", prompt, **sampling), _new_req("b", [4, 5]))
    )
    assert out.scheduled_new_reqs[0].prompt_token_ids is prompt
    assert out.scheduled_new_reqs[1].prompt_token_ids.tolist() == [4, 5]


@pytest.mark.parametrize(
    "config",
    [
        {"uses_mrope": True},
        {"runner_type": "pooling"},
        {"logits_processors": ["my.module:Processor"]},
    ],
)
def test_configs_that_read_prompt_ids_on_the_worker_send_lists(config):
    original = _scheduler_output(_new_req("a", [1, 2, 3]))
    assert _compacting_executor(**config)._worker_copy(original) is original


def test_array_prompt_fills_the_input_batch_row_like_the_list():
    """The worker's InputBatch.add_request writes the same row from either form,
    after the round trip through the broadcast's pickling."""
    import pickle

    import torch
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    prompt = list(range(100, 1100))
    rows = []
    for compact in (False, True):
        so = _scheduler_output(_new_req("a", prompt))
        if compact:
            so = _compacting_executor()._worker_copy(so)
        req = pickle.loads(pickle.dumps(so, protocol=5)).scheduled_new_reqs[0]
        batch = InputBatch(
            max_num_reqs=2,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            device=torch.device("cpu"),
            vocab_size=200064,
            block_sizes=[128],
            max_num_blocks_per_req=[16],
        )
        state = CachedRequestState(
            req_id=req.req_id,
            prompt_token_ids=req.prompt_token_ids,
            mm_features=[],
            sampling_params=req.sampling_params,
            generator=None,
            block_ids=req.block_ids,
            num_computed_tokens=0,
            output_token_ids=[],
        )
        index = batch.add_request(state)
        assert state.num_prompt_tokens == len(prompt)
        rows.append(batch.token_ids_cpu[index, : len(prompt)].tolist())
    assert rows[0] == rows[1] == prompt


def test_cached_all_token_ids_are_cut_to_the_output_tail_the_worker_reads():
    """GPUModelRunner._update_states reads `all_token_ids[req_id][-n:]` with
    n = num_output_tokens, and only when n > 0; the tail gives the same slice."""
    from vllm.v1.core.sched.output import CachedRequestData

    prompt_then_out = list(range(1000)) + [7, 8, 9]
    held_prefill = list(range(5000))
    short = [1, 2]  # async placeholders count toward n, which can exceed the list
    full = {"dec": prompt_then_out, "pre": held_prefill, "short": short}
    num_output = {"dec": 3, "pre": 0, "short": 4}
    cached = CachedRequestData(
        req_ids=list(full),
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids=dict(full),
        new_block_ids=[None] * 3,
        num_computed_tokens=[1002, 4096, 2],
        num_output_tokens=list(num_output.values()),
    )
    original = _scheduler_output()
    original.scheduled_cached_reqs = cached

    sent = _compacting_executor()._worker_copy(original).scheduled_cached_reqs

    assert sent.all_token_ids == {"dec": [7, 8, 9], "pre": [], "short": [1, 2]}
    for req_id, n in num_output.items():
        if n > 0:
            assert sent.all_token_ids[req_id][-n:] == full[req_id][-n:]
    assert original.scheduled_cached_reqs.all_token_ids["pre"] is held_prefill


def test_output_without_token_lists_is_sent_as_is():
    original = _scheduler_output()
    assert _compacting_executor()._worker_copy(original) is original

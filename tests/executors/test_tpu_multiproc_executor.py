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

from vllm_torchtpu.executors import tpu_multiproc_executor
from vllm_torchtpu.executors.tpu_multiproc_executor import (
    TpuMultiprocExecutor,
    _may_pack_prompts,
)

_EXECUTOR_MODULE = "vllm_torchtpu.executors.tpu_multiproc_executor"

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


_PROMPT = [7, 129279, 0, 42] * 1000


def _new_request(prompt_token_ids, req_id="r0"):
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([1, 2],),
        num_computed_tokens=0,
        lora_request=None,
    )


@patch.object(MultiprocExecutor, "execute_model", return_value="out")
def test_execute_model_sends_new_prompts_as_int_arrays(mock_super):
    """New requests' prompts go to the workers as array('i') with the same
    tokens; the worker copies and reads them as before."""
    import pickle
    from array import array

    import numpy as np
    import torch

    new_req = _new_request(list(_PROMPT))
    no_ids = _new_request(None)
    scheduler_output = MagicMock(scheduled_new_reqs=[new_req, no_ids])
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor._pack_prompts = True

    assert executor.execute_model(scheduler_output, non_block=True) == "out"

    mock_super.assert_called_once_with(scheduler_output, True)
    assert no_ids.prompt_token_ids is None
    sent = new_req.prompt_token_ids
    assert isinstance(sent, array) and sent.typecode == "i"
    received = pickle.loads(pickle.dumps(new_req, protocol=pickle.HIGHEST_PROTOCOL))
    prompt = received.prompt_token_ids
    assert received.prompt_len == len(_PROMPT)
    assert type(prompt[1]) is int and list(prompt) == _PROMPT
    token_ids_cpu = np.zeros((1, len(_PROMPT) + 8), dtype=np.int32)
    token_ids_cpu[0, : len(_PROMPT)] = prompt
    np.testing.assert_array_equal(token_ids_cpu[0, : len(_PROMPT)], _PROMPT)
    assert torch.tensor(prompt).tolist() == _PROMPT


@pytest.mark.parametrize("consumer", [None, "reasoning", "configured", "plugin"])
def test_prompts_stay_lists_for_list_consumers(consumer):
    """A reasoning config (thinking-budget state) or any custom logits
    processor, configured or installed as a plugin, may use list semantics
    on the prompt: then the prompt is sent as a list."""
    vllm_config = MagicMock(reasoning_config=None)
    vllm_config.model_config.logits_processors = None
    plugins = []
    if consumer == "reasoning":
        vllm_config.reasoning_config = MagicMock()
    elif consumer == "configured":
        vllm_config.model_config.logits_processors = ["pkg.Processor"]
    elif consumer == "plugin":
        plugins = [MagicMock()]
    with patch(f"{_EXECUTOR_MODULE}.entry_points", return_value=plugins):
        assert _may_pack_prompts(vllm_config) is (consumer is None)


@patch.object(MultiprocExecutor, "execute_model", return_value="out")
def test_execute_model_keeps_lists_when_not_packing(mock_super):
    new_req = _new_request(list(_PROMPT))
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor._pack_prompts = False

    executor.execute_model(MagicMock(scheduled_new_reqs=[new_req]))

    assert type(new_req.prompt_token_ids) is list
    assert new_req.prompt_token_ids == _PROMPT


@pytest.fixture
def prepacked():
    """The input thread's packed prompts, as when the executor packs."""
    import weakref

    table = weakref.WeakValueDictionary()
    with patch.object(tpu_multiproc_executor, "_prepacked", table):
        yield table


def test_prepack_packs_in_the_input_thread_preprocess(prepacked):
    """The wrapped preprocess_add_request packs the request's prompt and
    leaves the request's list prompt untouched; the entry lives as long as
    its request."""
    from array import array
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCore

    made = {
        "core_req": SimpleNamespace(
            request_id="r7", prompt_token_ids=list(_PROMPT), resumable=False
        ),
        "chunk": SimpleNamespace(
            request_id="s1", prompt_token_ids=[5, 6], resumable=True
        ),
    }

    def original(self, core_request):
        return made.pop(core_request), 3

    with patch.object(EngineCore, "preprocess_add_request", original):
        tpu_multiproc_executor.install_prompt_prepack()
        tpu_multiproc_executor.install_prompt_prepack()  # idempotent
        assert EngineCore.preprocess_add_request._tpu_prompt_prepack
        request, wave = EngineCore.preprocess_add_request(MagicMock(), "core_req")
        EngineCore.preprocess_add_request(MagicMock(), "chunk")

    assert wave == 3 and request.request_id == "r7"
    assert type(request.prompt_token_ids) is list
    packed = prepacked["r7"]
    assert isinstance(packed, array) and packed.typecode == "i"
    assert list(packed) == _PROMPT
    # A streaming-input session's chunk is packed when it is scheduled.
    assert "s1" not in prepacked
    del request, packed
    assert "r7" not in prepacked


@patch.object(MultiprocExecutor, "execute_model", return_value="out")
def test_execute_model_sends_the_prepacked_prompt(mock_super, prepacked):
    """A prompt packed in the input thread is sent as is and leaves the
    table."""
    from array import array

    packed = array("i", _PROMPT)
    prepacked["r0"] = packed
    new_req = _new_request(list(_PROMPT))
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor._pack_prompts = True

    executor.execute_model(MagicMock(scheduled_new_reqs=[new_req]))

    assert new_req.prompt_token_ids is packed
    assert "r0" not in prepacked


@patch.object(MultiprocExecutor, "execute_model", return_value="out")
def test_execute_model_repacks_a_prompt_scheduled_with_other_tokens(
    mock_super, prepacked
):
    """A stale entry (the prompt grew since it was packed) is not sent: the
    scheduled prompt is packed instead."""
    from array import array

    shorter = array("i", _PROMPT[:-1])
    rewritten = array("i", _PROMPT[:-1] + [99])
    prepacked.update(r0=shorter, r1=rewritten)
    new_reqs = [_new_request(list(_PROMPT), req_id) for req_id in ("r0", "r1")]
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor._pack_prompts = True

    executor.execute_model(MagicMock(scheduled_new_reqs=new_reqs))

    for new_req in new_reqs:
        assert list(new_req.prompt_token_ids) == _PROMPT
    assert len(prepacked) == 0

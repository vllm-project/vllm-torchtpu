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
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor
from vllm_torchtpu.executors.tpu_uniproc_executor import TpuUniProcExecutor

# Sentinel returned by the parent determine_available_memory; the override
# plumbing must pass this through unchanged.
_AVAIL = [1024 * 1024 * 1024]


def _make_executor(
    engine_override,
    worker_overrides,
    pipeline_parallel_size=1,
    executor_cls=TpuMultiprocExecutor,
):
    """Build an executor without running __init__, with a mocked engine
    cache_config and a collective_rpc that returns the per-worker overrides."""
    executor = executor_cls.__new__(executor_cls)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.vllm_config.parallel_config.world_size = len(worker_overrides)
    executor.vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_copies_override_from_workers_when_engine_unset(mock_super):
    """Engine override is None -> copy the agreed worker value to the engine."""
    executor = _make_executor(engine_override=None, worker_overrides=[5428, 5428, 5428])

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    executor.collective_rpc.assert_called_once_with("get_sized_num_gpu_blocks")
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 5428


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_does_not_overwrite_existing_engine_override(mock_super):
    """A user-supplied engine override (e.g. --num-gpu-blocks-override) wins;
    the worker value is never queried."""
    executor = _make_executor(engine_override=1234, worker_overrides=[5428, 5428, 5428])

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    executor.collective_rpc.assert_not_called()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_raises_when_workers_disagree(mock_super):
    """Workers must agree on the override; a mismatch is a sizing bug, not
    something to silently pick one of. 5428 -> 6840 is 26%, far outside the
    relative tolerance."""
    executor = _make_executor(engine_override=None, worker_overrides=[5428, 6840, 5428])

    with pytest.raises(ValueError, match="workers disagree"):
        executor.determine_available_memory()


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
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

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 551348


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_absolute_floor_applies_to_tiny_block_counts(mock_super):
    """The relative tolerance must not shrink below the absolute floor: at
    100 blocks 0.5% rounds to 0, which would demand exact agreement."""
    executor = _make_executor(engine_override=None, worker_overrides=[100, 102, 101])

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 100


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_no_override_when_workers_return_none(mock_super):
    """If no worker computed an override (uniform sizing skipped), the engine
    stays None and vLLM falls back to its own num_blocks computation."""
    executor = _make_executor(engine_override=None, worker_overrides=[None, None, None])

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_pipeline_stages_may_disagree_and_the_smallest_stage_wins(mock_super):
    """Each pipeline stage holds different layers, so its workers measure a
    different count; vLLM sizes every stage's pool to the minimum."""
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[100, 100, 100, 100] + [80, 80, 80, 80],
        pipeline_parallel_size=2,
    )

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 80


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_workers_within_a_pipeline_stage_must_still_agree(mock_super):
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[100, 100, 100, 150] + [80, 80, 80, 80],
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match="workers disagree"):
        executor.determine_available_memory()


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
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
        executor.determine_available_memory()


@patch.object(MultiprocExecutor, "determine_available_memory", return_value=_AVAIL)
def test_pipeline_stages_without_any_override_leave_it_unset(mock_super):
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[None, None, None, None],
        pipeline_parallel_size=2,
    )

    executor.determine_available_memory()

    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


@patch.object(UniProcExecutor, "determine_available_memory", return_value=_AVAIL)
def test_uniproc_executor_pins_worker_sized_override(mock_super):
    """Single-process executor pins the worker-sized block count after
    determine_available_memory without mutating a user-supplied override."""
    executor = _make_executor(
        engine_override=None,
        worker_overrides=[5428],
        executor_cls=TpuUniProcExecutor,
    )

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    executor.collective_rpc.assert_called_once_with("get_sized_num_gpu_blocks")
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 5428


@patch.object(UniProcExecutor, "determine_available_memory", return_value=_AVAIL)
def test_uniproc_executor_preserves_user_override(mock_super):
    executor = _make_executor(
        engine_override=1234,
        worker_overrides=[5428],
        executor_cls=TpuUniProcExecutor,
    )

    avail = executor.determine_available_memory()

    assert avail is _AVAIL
    executor.collective_rpc.assert_not_called()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234


def _make_ray_executor(engine_override, worker_overrides):
    """Same shape as _make_executor but for the Ray multi-host executor."""
    from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
        RayDistributedExecutorV2,
    )

    return _make_executor(
        engine_override=engine_override,
        worker_overrides=worker_overrides,
        executor_cls=RayDistributedExecutorV2,
    )


def test_ray_executor_tolerates_measurement_jitter():
    """Regression for dev build 155: the Ray multi-host executor demanded
    EXACT agreement (assert len(set(overrides)) == 1) while the multiproc
    executor already tolerated jitter. Kimi-K3 at TP=32 landed 6456..6459
    (~14 MiB blocks, a 3-block spread from per-worker HBM measurement) and
    the whole engine start died on the bare assert."""
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    observed = [6459, 6456] * 16
    executor = _make_ray_executor(engine_override=None, worker_overrides=observed)
    with patch.object(RayExecutorV2, "determine_available_memory", return_value=_AVAIL):
        avail = executor.determine_available_memory()

    assert avail is _AVAIL
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 6456


def test_ray_executor_raises_on_real_disagreement():
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    executor = _make_ray_executor(
        engine_override=None, worker_overrides=[6456, 8000] * 16
    )
    with (
        patch.object(RayExecutorV2, "determine_available_memory", return_value=_AVAIL),
        pytest.raises(ValueError, match="workers disagree"),
    ):
        executor.determine_available_memory()

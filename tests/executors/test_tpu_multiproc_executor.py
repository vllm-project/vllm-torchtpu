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


def _make_executor(engine_override, worker_overrides):
    """Build an executor without running __init__, with a mocked engine
    cache_config and a collective_rpc that returns the per-worker overrides."""
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_copies_override_from_workers_when_engine_unset(mock_super):
    """Engine override is None -> copy the agreed worker value to the engine."""
    executor = _make_executor(engine_override=None,
                              worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_called_once_with(
        "get_num_gpu_blocks_override")
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 5428


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_does_not_overwrite_existing_engine_override(mock_super):
    """A user-supplied engine override (e.g. --num-gpu-blocks-override) wins;
    the worker value is never queried."""
    executor = _make_executor(engine_override=1234,
                              worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_not_called()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_raises_when_workers_disagree(mock_super):
    """Workers must agree on the override; a mismatch is a sizing bug, not
    something to silently pick one of. 5428 -> 6840 is 26%, far outside the
    relative tolerance."""
    executor = _make_executor(engine_override=None,
                              worker_overrides=[5428, 6840, 5428])

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
    executor = _make_executor(engine_override=None,
                              worker_overrides=[100, 102, 101])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 100


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_no_override_when_workers_return_none(mock_super):
    """If no worker computed an override (uniform sizing skipped), the engine
    stays None and vLLM falls back to its own num_blocks computation."""
    executor = _make_executor(engine_override=None,
                              worker_overrides=[None, None, None])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None


def _make_ray_executor(engine_override, worker_overrides):
    """Same shape as _make_executor but for the Ray multi-host executor."""
    from vllm_torchtpu.executors.ray_distributed_executor_v2 import \
        RayDistributedExecutorV2
    executor = RayDistributedExecutorV2.__new__(RayDistributedExecutorV2)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
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
    executor = _make_ray_executor(engine_override=None,
                                  worker_overrides=observed)
    with patch.object(RayExecutorV2, "get_kv_cache_specs",
                      return_value=_SPECS):
        specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    # Smallest budget wins, so no worker is asked for more than it measured.
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 6456


def test_ray_executor_raises_on_real_disagreement():
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    executor = _make_ray_executor(engine_override=None,
                                  worker_overrides=[6456, 8000] * 16)
    with patch.object(RayExecutorV2, "get_kv_cache_specs",
                      return_value=_SPECS):
        with pytest.raises(ValueError, match="workers disagree"):
            executor.get_kv_cache_specs()

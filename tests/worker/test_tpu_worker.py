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
"""Tests for TPUWorker"""

import os
from unittest.mock import MagicMock, patch

import torch

from vllm_torchtpu.worker.tpu_worker import (TPUWorker,
                                             _configure_tpu_process_env)


def test_configure_tpu_process_env_sets_world_size_only_for_multiple_ranks(
        monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")

    _configure_tpu_process_env(rank=0,
                               local_rank=0,
                               world_size=1,
                               local_world_size=1)

    assert os.environ["RANK"] == "0"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["LOCAL_WORLD_SIZE"] == "1"
    assert "WORLD_SIZE" not in os.environ

    _configure_tpu_process_env(rank=1,
                               local_rank=1,
                               world_size=8,
                               local_world_size=8)

    assert os.environ["RANK"] == "1"
    assert os.environ["LOCAL_RANK"] == "1"
    assert os.environ["LOCAL_WORLD_SIZE"] == "8"
    assert os.environ["WORLD_SIZE"] == "8"


def _make_vllm_config(profiler_torch_dir=None):
    """Build a minimal mock VllmConfig for TPUWorker.__init__."""
    cfg = MagicMock()
    cfg.model_config.dtype = torch.bfloat16
    cfg.cache_config.cache_dtype = "auto"
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.compilation_config.compile_ranges_endpoints = []
    cfg.profiler_config.torch_profiler_dir = profiler_torch_dir
    return cfg


def _build_worker(vllm_config, rank=0):
    """Construct a TPUWorker with heavy side-effects mocked out."""
    with patch(
            "vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches"), patch(
                "vllm_torchtpu.worker.tpu_worker.envs") as mock_envs, patch(
                    "vllm_torchtpu.worker.tpu_worker.WorkerBase.__init__",
                    return_value=None,
                ):
        mock_envs.MODEL_IMPL_TYPE = "vllm"

        worker = TPUWorker.__new__(TPUWorker)
        # Set attributes that WorkerBase.__init__ would normally set.
        worker.vllm_config = vllm_config
        worker.model_config = vllm_config.model_config
        worker.cache_config = vllm_config.cache_config
        worker.parallel_config = vllm_config.parallel_config
        worker.compilation_config = vllm_config.compilation_config
        worker.rank = rank
        worker.local_rank = 0

        # Now call __init__ (WorkerBase.__init__ is a no-op mock).
        worker.__init__(
            vllm_config=vllm_config,
            local_rank=0,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:29500",
            is_driver_worker=True,
        )
        return worker


class TestProfilerDir:
    """Verify profiler_config.torch_profiler_dir logic."""

    def test_uses_profiler_config_when_set(self):
        """profiler_config.torch_profiler_dir is used when set."""
        cfg = _make_vllm_config(profiler_torch_dir="/config/profiler/dir")
        worker = _build_worker(cfg)
        assert worker.profile_dir == "/config/profiler/dir"

    def test_no_profiler_when_not_set(self):
        """profile_dir is None when config is not set."""
        cfg = _make_vllm_config(profiler_torch_dir=None)
        worker = _build_worker(cfg)
        assert worker.profile_dir is None


def test_initialize_from_config_updates_num_gpu_blocks():
    worker = TPUWorker.__new__(TPUWorker)
    worker.cache_config = MagicMock(num_gpu_blocks=None)
    worker.vllm_config = MagicMock()
    worker.model_runner = MagicMock()
    kv_cache_config = MagicMock(num_blocks=2048)

    with patch("vllm_torchtpu.worker.tpu_worker."
               "ensure_kv_transfer_initialized"):
        worker.initialize_from_config(kv_cache_config)

    assert worker.cache_config.num_gpu_blocks == 2048
    worker.model_runner.initialize_kv_cache.assert_called_once_with(
        kv_cache_config)


@patch("vllm_torchtpu.worker.tpu_worker.get_pp_group")
@patch("vllm_torchtpu.worker.tpu_worker.get_tensor_model_parallel_rank",
       return_value=3)
@patch("vllm_torchtpu.worker.tpu_worker.get_kv_transfer_group")
@patch("vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group",
       return_value=True)
def test_kv_connector_handshake_metadata_uses_pp_tp_rank_key(
        _has_group, get_group, _get_tp_rank, get_pp_group):
    metadata = object()
    get_group.return_value.get_handshake_metadata.return_value = metadata
    get_pp_group.return_value.rank_in_group = 2
    worker = TPUWorker.__new__(TPUWorker)

    assert worker.get_kv_connector_handshake_metadata() == {(2, 3): metadata}

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
"""Shared fixtures for the vLLM model tests."""

import pytest
import torch.distributed as dist
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized


@pytest.fixture(scope="module")
def dist_ctx():
    """Single-rank gloo world and model-parallel groups for one test module.

    Destroys only the groups this fixture created, so later test modules
    start without a default process group. Groups that already existed when
    the fixture ran stay in place.
    """
    owns_world = not dist.is_initialized()
    owns_model_parallel = False
    if owns_world:
        import portpicker

        port = portpicker.pick_unused_port()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    try:
        owns_model_parallel = not model_parallel_is_initialized()
        # GroupCoordinator construction consults the current vLLM config.
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)
        yield
    finally:
        if owns_model_parallel:
            destroy_model_parallel()
        if owns_world:
            destroy_distributed_environment()

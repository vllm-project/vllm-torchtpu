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
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from .tpu_test_utils import prepare_tpu_environment, torchrun_worker_env


def test_prepare_tpu_environment(monkeypatch):
    """Verify prepare_tpu_environment dispatches to
    torch.tpu.distributed.set_tpu_launch_env.
    """
    mock_set_tpu_launch_env = MagicMock()
    mock_dist = SimpleNamespace(set_tpu_launch_env=mock_set_tpu_launch_env)
    mock_tpu = SimpleNamespace(distributed=mock_dist)

    monkeypatch.setattr(torch, "tpu", mock_tpu, raising=False)

    prepare_tpu_environment(world_size=8)
    mock_set_tpu_launch_env.assert_called_once_with(nproc_per_node=8)


def test_torchrun_worker_env_skips_without_torch_tpu(monkeypatch):
    """A missing TorchTPU skips the test instead of failing it."""
    # A None entry makes `import torch_tpu` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch_tpu", None)

    with pytest.raises(pytest.skip.Exception, match="TorchTPU is unavailable"):
        torchrun_worker_env(world_size=8)

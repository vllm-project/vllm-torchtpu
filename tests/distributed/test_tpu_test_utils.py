# SPDX-License-Identifier: Apache-2.0
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from .tpu_test_utils import prepare_tpu_environment


def test_prepare_tpu_environment_prefers_public_api(monkeypatch):
    """When torch.tpu.distributed.set_tpu_launch_env is available (torch_tpu >= 0.1.2),
    prepare_tpu_environment should dispatch to it.
    """
    mock_set_tpu_launch_env = MagicMock()
    mock_dist = SimpleNamespace(set_tpu_launch_env=mock_set_tpu_launch_env)
    mock_tpu = SimpleNamespace(distributed=mock_dist)

    monkeypatch.setattr(torch, "tpu", mock_tpu, raising=False)

    prepare_tpu_environment(world_size=8)
    mock_set_tpu_launch_env.assert_called_once_with(nproc_per_node=8)


def test_prepare_tpu_environment_falls_back_to_legacy_singlehost_wrapper(monkeypatch):
    """When running on torch_tpu <= 0.1.1, dispatch to legacy singlehost_wrapper."""
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(distributed=None), raising=False)

    mock_legacy_prepare = MagicMock()
    legacy_mod_name = "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
    mock_legacy_module = ModuleType(legacy_mod_name)
    mock_legacy_module.prepare_tpu_environment = mock_legacy_prepare

    with patch.dict("sys.modules", {legacy_mod_name: mock_legacy_module}):
        prepare_tpu_environment(world_size=8)
        mock_legacy_prepare.assert_called_once_with(world_size=8)

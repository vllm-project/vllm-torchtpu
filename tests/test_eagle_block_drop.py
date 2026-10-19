# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mamba_manager_test_utils import Geometry, make_kv_cache_manager, make_request

from vllm_torchtpu.core import eagle_block_drop
from vllm_torchtpu.gdn_pool_layout import PrefillCheckpointUnits


class _SpeculativeConfig:
    """The two SpeculativeConfig members the eagle block-drop bit involves."""

    def __init__(self, use_eagle: bool = True):
        self.use_eagle = use_eagle
        self.disable_eagle_block_drop = False

    def use_eagle_block_drop(self) -> bool:
        return self.use_eagle and not self.disable_eagle_block_drop


def test_override_writes_the_speculative_config_once(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
    config = SimpleNamespace(speculative_config=_SpeculativeConfig())
    eagle_block_drop.apply_eagle_block_drop_override(config)
    assert config.speculative_config.disable_eagle_block_drop is True
    assert config.speculative_config.use_eagle_block_drop() is False
    # Idempotent: check_and_update_config may run more than once.
    eagle_block_drop.apply_eagle_block_drop_override(config)
    assert config.speculative_config.disable_eagle_block_drop is True


@pytest.mark.parametrize(
    ("disable", "speculative"), [("0", True), ("", True), ("1", False)]
)
def test_override_leaves_the_config_alone_without_env_or_spec_decode(
    monkeypatch, disable, speculative
):
    if disable:
        monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", disable)
    else:
        monkeypatch.delenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", raising=False)
    speculative_config = _SpeculativeConfig() if speculative else None
    config = SimpleNamespace(speculative_config=speculative_config)
    eagle_block_drop.apply_eagle_block_drop_override(config)
    if speculative:
        assert config.speculative_config.disable_eagle_block_drop is False
    else:
        assert config.speculative_config is None


@pytest.mark.parametrize(("disable", "position"), [("0", 3072), ("1", 6144)])
def test_config_places_worker_and_manager_checkpoint_at_the_same_position(
    monkeypatch, disable, position
):
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", disable)
    config = SimpleNamespace(speculative_config=_SpeculativeConfig())
    eagle_block_drop.apply_eagle_block_drop_override(config)
    drop = config.speculative_config.use_eagle_block_drop()
    geometry = Geometry(3072, 3072, 1, drop, 16384, 0)
    manager = make_kv_cache_manager(geometry)
    request = make_request("r", list(range(8191)), 3072)
    assert manager.allocate_slots(request, 8191) is not None
    mamba = manager.coordinator.single_type_managers[1]
    assert mamba._checkpoints["r"] == (position, 1)
    units = PrefillCheckpointUnits(
        hash_block_size=3072,
        mamba_block_size=3072,
        alignment=64,
        drop_eagle_block=drop,
    )
    offsets = np.zeros(1, dtype=np.int32)
    units.fill_offsets(offsets, np.array([8191]), np.array([8191]))
    assert offsets.tolist() == [position]


def test_platform_discovery_imports_config_without_a_cycle():
    result = subprocess.run(
        [sys.executable, "-c", "import vllm.engine.arg_utils"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

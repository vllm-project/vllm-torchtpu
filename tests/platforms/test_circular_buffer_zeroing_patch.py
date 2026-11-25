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
"""Tests for circular_buffer_zeroing engine_core patch."""

import inspect

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from vllm_torchtpu import patch_registry
from vllm_torchtpu.core.circular_buffer_zeroing import (
    patch_circular_buffer_block_zeroing,
)

_TARGET = (
    "vllm_torchtpu.core.circular_buffer_zeroing:patch_circular_buffer_block_zeroing"
)


def _make_circular_spec(block_size: int = 64) -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def isolated_circular_zeroing_patch(monkeypatch):
    """Isolate engine_core patch registry and restore CircularBufferManager.__init__."""
    only_target = tuple(p for p in patch_registry.PATCHES if p.target == _TARGET)
    assert len(only_target) == 1
    monkeypatch.setattr(patch_registry, "PATCHES", only_target)
    applied = set(patch_registry._applied) - {_TARGET}
    monkeypatch.setattr(patch_registry, "_applied", applied)
    unwrapped_init = inspect.unwrap(CircularBufferManager.__init__)
    monkeypatch.setattr(CircularBufferManager, "__init__", unwrapped_init)
    return unwrapped_init


def test_engine_core_registry_applies_patch_idempotently(
    isolated_circular_zeroing_patch,
):
    unwrapped_init = isolated_circular_zeroing_patch

    patch_registry.apply("engine_core")
    patched_once = CircularBufferManager.__init__
    assert patched_once is not unwrapped_init
    # Second application must be idempotent rather than stacking wrappers.
    patch_circular_buffer_block_zeroing()
    assert CircularBufferManager.__init__ is patched_once


def test_records_and_drains_circular_buffer_blocks_when_zeroing_enabled(
    isolated_circular_zeroing_patch,
):
    del isolated_circular_zeroing_patch
    patch_registry.apply("engine_core")

    circular_spec = _make_circular_spec(block_size=64)
    fp8_full_spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
    )
    # Mixed-precision KV cache enables KVCacheConfig.needs_kv_cache_zeroing naturally.
    zeroed_cfg = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=fp8_full_spec),
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=circular_spec),
        ],
    )
    assert zeroed_cfg.needs_kv_cache_zeroing is True
    kv_cache_mgr = KVCacheManager(
        kv_cache_config=zeroed_cfg,
        max_model_len=2048,
        scheduler_block_size=64,
        hash_block_size=64,
        enable_caching=False,
    )
    coord = kv_cache_mgr.coordinator
    ring_mgr = coord.single_type_managers[1]
    assert isinstance(ring_mgr, CircularBufferManager)

    claimed_r0 = coord.allocate_new_blocks(
        "req-0", num_tokens=128, num_tokens_main_model=128
    )
    claimed_r1 = coord.allocate_new_blocks(
        "req-1", num_tokens=128, num_tokens_main_model=128
    )
    # Re-allocating for an active request claims 0 additional ring blocks.
    claimed_r0_again = coord.allocate_new_blocks(
        "req-0", num_tokens=64, num_tokens_main_model=64
    )
    assert claimed_r0_again[1] == []
    expected_ring_ids = [claimed_r0[1][0].block_id, claimed_r1[1][0].block_id]
    assert len(set(expected_ring_ids)) == 2
    assert ring_mgr.new_block_ids == expected_ring_ids

    # Step-level drain via KVCacheManager.take_new_block_ids() must include and
    # clear CircularBufferManager's recorded block IDs.
    drained = kv_cache_mgr.take_new_block_ids()
    for ring_block_id in expected_ring_ids:
        assert ring_block_id in drained
    assert ring_mgr.new_block_ids == []
    assert kv_cache_mgr.take_new_block_ids() == []


def test_skips_recording_when_zeroing_disabled(
    isolated_circular_zeroing_patch,
):
    del isolated_circular_zeroing_patch
    patch_registry.apply("engine_core")

    circular_spec = _make_circular_spec(block_size=64)
    unzeroed_cfg = KVCacheConfig(
        num_blocks=12,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=circular_spec)
        ],
    )
    assert unzeroed_cfg.needs_kv_cache_zeroing is False
    kv_cache_mgr = KVCacheManager(
        kv_cache_config=unzeroed_cfg,
        max_model_len=2048,
        scheduler_block_size=64,
        hash_block_size=64,
        enable_caching=False,
    )
    coord = kv_cache_mgr.coordinator
    coord.allocate_new_blocks("req-0", num_tokens=256, num_tokens_main_model=256)
    assert len(coord.get_blocks("req-0")[0]) == 1
    assert coord.single_type_managers[0].new_block_ids == []
    assert kv_cache_mgr.take_new_block_ids() == []


def test_wrapper_supports_future_upstream_init_signatures(
    monkeypatch,
    isolated_circular_zeroing_patch,
):
    del isolated_circular_zeroing_patch
    circular_spec = _make_circular_spec(block_size=64)
    forwarded = []

    def future_init(
        self,
        kv_cache_spec,
        needs_kv_cache_zeroing: bool = False,
        future_upstream_flag: bool = False,
    ) -> None:
        del kv_cache_spec
        self._record_new_block_ids = False
        forwarded.append((needs_kv_cache_zeroing, future_upstream_flag))

    monkeypatch.setattr(CircularBufferManager, "__init__", future_init)
    patch_circular_buffer_block_zeroing()
    probe_kw = CircularBufferManager(
        circular_spec,
        needs_kv_cache_zeroing=True,
        future_upstream_flag=True,
    )
    probe_pos = CircularBufferManager(circular_spec, True)
    probe_default = CircularBufferManager(circular_spec)
    assert forwarded == [(True, True), (True, False), (False, False)]
    assert probe_kw._record_new_block_ids is True
    assert probe_pos._record_new_block_ids is True
    assert probe_default._record_new_block_ids is False

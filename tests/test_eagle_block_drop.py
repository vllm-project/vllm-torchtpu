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
from types import SimpleNamespace

import vllm.v1.core.sched.scheduler as scheduler_module
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import eagle_block_drop

BLOCK = 1152


def _split_scheduler(use_eagle: bool):
    return SimpleNamespace(
        use_eagle=use_eagle,
        use_eagle_block_drop=use_eagle,
        block_size=BLOCK,
        cache_config=SimpleNamespace(block_size=BLOCK),
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=BLOCK,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False,
    )


def _request(prompt: int, computed: int = 0):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        shared_prefix_boundary=0,
    )


def _install(monkeypatch, fake_init=None):
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
    monkeypatch.setattr(
        Scheduler, "__init__", fake_init or (lambda self, *a, **k: None)
    )
    monkeypatch.setattr(
        Scheduler, "_mamba_block_aligned_split", Scheduler._mamba_block_aligned_split
    )
    eagle_block_drop.patch_disable_eagle_block_drop()


def test_split_back_off_matches_upstream_without_patch():
    split = Scheduler._mamba_block_aligned_split
    # Upstream: with EAGLE the chunk stops one block before the last boundary.
    assert split(_split_scheduler(True), _request(10_000), 16384) == 8 * BLOCK - 1152
    assert split(_split_scheduler(False), _request(10_000), 16384) == 8 * BLOCK


def test_patched_split_skips_eagle_back_off(monkeypatch):
    _install(monkeypatch)
    sched = _split_scheduler(True)
    assert Scheduler._mamba_block_aligned_split(sched, _request(10_000), 16384) == (
        8 * BLOCK
    )
    assert sched.use_eagle is True


def test_patched_init_builds_manager_without_eagle(monkeypatch):
    seen = {}

    def fake_manager(**kwargs):
        seen.update(kwargs)
        groups = [SpecGroup("full", [0], object, True)]
        return SimpleNamespace(
            use_eagle=kwargs["use_eagle"],
            coordinator=SimpleNamespace(
                eagle_group_ids={0},
                single_type_managers=[SimpleNamespace(use_eagle=True)],
                attention_groups=groups,
            ),
        )

    def fake_init(self):
        self.use_eagle = True
        self.block_size = BLOCK
        self.kv_cache_manager = scheduler_module.KVCacheManager(use_eagle=True)

    monkeypatch.setattr(scheduler_module, "KVCacheManager", fake_manager)
    _install(monkeypatch, fake_init)
    sched = Scheduler.__new__(Scheduler)
    Scheduler.__init__(sched)
    assert seen["use_eagle"] is False
    assert scheduler_module.KVCacheManager is fake_manager  # restored
    coordinator = sched.kv_cache_manager.coordinator
    assert coordinator.eagle_group_ids == set()
    assert not coordinator.single_type_managers[0].use_eagle
    assert not coordinator.attention_groups[0].use_eagle
    assert sched.use_eagle is True
    assert sched.use_eagle_block_drop is False


def test_patch_is_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "0")
    before = (Scheduler.__init__, Scheduler._mamba_block_aligned_split)
    eagle_block_drop.patch_disable_eagle_block_drop()
    assert (Scheduler.__init__, Scheduler._mamba_block_aligned_split) == before

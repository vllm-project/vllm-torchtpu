# SPDX-License-Identifier: Apache-2.0
from types import MethodType, SimpleNamespace

import pytest
import torch
import vllm.v1.core.sched.scheduler as scheduler_module
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
    SchedulerOffloadConfig,
)
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_offload.base import LookupResult

from vllm_torchtpu.core import eagle_block_drop
from vllm_torchtpu.core.eagle_block_drop import offload_config_without_eagle_groups
from vllm_torchtpu.offload import raiden_connector

BLOCK = 1152


def _split_scheduler(use_eagle: bool):
    return SimpleNamespace(
        use_eagle=use_eagle,
        block_size=BLOCK,
        cache_config=SimpleNamespace(block_size=BLOCK),
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=BLOCK,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
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


CHUNK = 4096


def _mtp_config() -> SchedulerOffloadConfig:
    spec = SimpleNamespace(
        tokens_per_block=[CHUNK],
        blocks_per_chunk=1,
        tokens_per_hash=CHUNK,
        offload_prompt_only=False,
    )
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=FullAttentionSpec(
            block_size=CHUNK, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        ),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
        parallel_config=SimpleNamespace(world_size=8, decode_context_parallel_size=16),
    )
    return SchedulerOffloadConfig.from_spec(
        spec, vllm_config, SimpleNamespace(kv_cache_groups=[group])
    )


def _hit_tokens(config, prompt_tokens: int, tier_chunks: int, local_chunks: int = 0):
    keys = [f"chunk-{i}" for i in range(prompt_tokens // CHUNK)]
    stored = set(keys[:tier_chunks])
    scheduler = SimpleNamespace(
        config=config,
        _sliding_window_groups=(),
        _lookup_groups=(0,),
        _mamba_align_size=None,
        _chunks_being_loaded=set(),
        manager=SimpleNamespace(
            lookup=lambda key, ctx: (
                LookupResult.HIT if key in stored else LookupResult.MISS
            )
        ),
        _events_tracker=SimpleNamespace(record_lookup=lambda *args: None),
    )
    scheduler._maximal_prefix_lookup = MethodType(
        OffloadingConnectorScheduler._maximal_prefix_lookup, scheduler
    )
    req_status = SimpleNamespace(
        num_locally_computed_tokens=local_chunks * CHUNK,
        req=SimpleNamespace(num_tokens=prompt_tokens, request_id="r"),
        group_states=[SimpleNamespace(offload_keys=keys)],
        req_context=None,
    )
    return OffloadingConnectorScheduler._lookup_complete_chunks(scheduler, req_status)


def test_upstream_marks_the_mla_group_as_a_draft_group_under_mtp():
    assert [g.is_eagle_group for g in _mtp_config().kv_group_configs] == [True]


def test_upstream_tier_hit_loads_one_chunk_short():
    assert _hit_tokens(_mtp_config(), 6 * CHUNK + 100, tier_chunks=6) == 5 * CHUNK


def test_cleared_config_loads_the_whole_tier_hit():
    config = offload_config_without_eagle_groups(_mtp_config())
    assert _hit_tokens(config, 6 * CHUNK + 100, tier_chunks=6) == 6 * CHUNK


def test_cleared_config_keeps_the_last_chunk_after_an_hbm_hit():
    upstream = _hit_tokens(_mtp_config(), 9 * CHUNK + 7, tier_chunks=7, local_chunks=3)
    cleared = _hit_tokens(
        offload_config_without_eagle_groups(_mtp_config()),
        9 * CHUNK + 7,
        tier_chunks=7,
        local_chunks=3,
    )
    assert (upstream, cleared) == (3 * CHUNK, 4 * CHUNK)


def test_one_chunk_tier_hit_is_loaded_instead_of_dropped():
    config = offload_config_without_eagle_groups(_mtp_config())
    assert _hit_tokens(_mtp_config(), 3 * CHUNK, tier_chunks=2, local_chunks=1) == 0
    assert _hit_tokens(config, 3 * CHUNK, tier_chunks=2, local_chunks=1) == CHUNK


def test_cleared_config_never_loads_past_the_stored_chunks():
    config = offload_config_without_eagle_groups(_mtp_config())
    assert _hit_tokens(config, 10 * CHUNK + 1, tier_chunks=4) == 4 * CHUNK
    assert _hit_tokens(config, 10 * CHUNK + 1, tier_chunks=0) == 0


def test_cleared_config_changes_only_the_draft_marks():
    upstream = _mtp_config()
    cleared = offload_config_without_eagle_groups(upstream)
    assert cleared.kv_group_configs[0] == upstream.kv_group_configs[0]._replace(
        is_eagle_group=False
    )
    assert cleared._replace(kv_group_configs=()) == upstream._replace(
        kv_group_configs=()
    )


def _storable_chunks(config, prompt_tokens: int, offloadable_tokens: int) -> int:
    request = SimpleNamespace(
        req=SimpleNamespace(num_prompt_tokens=prompt_tokens), config=config
    )
    return RequestOffloadState.storable_chunks(
        request,
        config.kv_group_configs[0],
        SimpleNamespace(block_ids=list(range(8))),
        offloadable_tokens,
    )


def test_cleared_config_stores_the_trailing_chunk_while_decoding():
    cleared = offload_config_without_eagle_groups(_mtp_config())
    decoding = (2 * CHUNK + 10, 3 * CHUNK + 5)
    assert _storable_chunks(_mtp_config(), *decoding) == 2
    assert _storable_chunks(cleared, *decoding) == 3


def test_prefill_stores_are_the_same_either_way():
    cleared = offload_config_without_eagle_groups(_mtp_config())
    prefill = (6 * CHUNK + 10, 5 * CHUNK)
    assert _storable_chunks(_mtp_config(), *prefill) == 5
    assert _storable_chunks(cleared, *prefill) == 5


@pytest.mark.parametrize("switch", ["0", "1"])
def test_raiden_scheduler_clears_draft_marks_only_when_switched_on(monkeypatch, switch):
    monkeypatch.setenv("VLLM_TPU_OFFLOAD_DISABLE_EAGLE_BLOCK_DROP", switch)

    def base_init(self, spec, vllm_config, kv_cache_config):
        self.config = _mtp_config()
        self.manager = object.__new__(raiden_connector.RaidenOffloadingManager)

    monkeypatch.setattr(OffloadingConnectorScheduler, "__init__", base_init)
    scheduler = raiden_connector.TPURaidenOffloadingScheduler(
        SimpleNamespace(extra_config={}), None, None
    )
    marks = [g.is_eagle_group for g in scheduler.config.kv_group_configs]
    assert marks == [switch == "0"]


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_OFFLOAD_DISABLE_EAGLE_BLOCK_DROP", raising=False)
    from vllm_torchtpu import envs

    assert envs.VLLM_TPU_OFFLOAD_DISABLE_EAGLE_BLOCK_DROP is False

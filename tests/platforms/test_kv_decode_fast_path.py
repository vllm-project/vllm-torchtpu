# SPDX-License-Identifier: Apache-2.0
"""Steady decode steps skip the per-group KV bookkeeping and change nothing.

See `vllm_torchtpu.core.kv_decode_fast_path`. Two scheduler-side KV cache
managers with the served Qwen3.8-Flash decode groups (65 groups) replay the
same async-scheduler trace in lock step, one with the fast path and one
without: admissions with local prefix hits and connector loads, MTP decode
with rejected drafts, finishes, preemption and re-prefill under memory
pressure. After every step the block tables, the per-group bookkeeping, the
free-block queue order, every block's reference count and hash, the
prefix-cache maps and the same-step unwritten set must be equal, and so must
every ``allocate_slots`` result.
"""

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from tests.platforms import kv_decode_replay as replay
from vllm_torchtpu.core import kv_decode_fast_path as fast_path

# Class attributes the engine-core patches below replace; restored after
# each test.
_PATCHED = (
    (BlockPool, "free_blocks"),
    (BlockPool, "_tpu_lifo_free_patch"),
    (BlockPool, "_insert_block_hash"),
    (BlockPool, "get_cached_block"),
    (Scheduler, "schedule"),
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
    (KVCacheManager, "allocate_slots"),
    (KVCacheManager, "cache_blocks"),
    (KVCacheManager, "free"),
    (KVCacheManager, "pop_blocks_for_free"),
    (KVCacheManager, "remove_skipped_blocks"),
)


@pytest.fixture
def served_patches(monkeypatch):
    """The engine core's KV patches as served, plus the fast path."""
    import vllm_torchtpu
    from vllm_torchtpu.core import mamba_checkpoint_retention

    for owner, name in _PATCHED:
        monkeypatch.setattr(owner, name, getattr(owner, name, False), raising=False)
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_KV_DECODE_FAST_PATH", "1")
    vllm_torchtpu._patch_vllm_block_pool_lifo_free()
    vllm_torchtpu._patch_vllm_same_step_prefix_hits()
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
    fast_path.patch_kv_decode_fast_path()
    if not getattr(KVCacheManager.allocate_slots, fast_path._PATCH_FLAG, False):
        pytest.skip(
            "KV decode fast path not installed: the installed vLLM is not the "
            "analysed release (see check_fingerprints)"
        )


def _count_coordinator_calls(manager) -> dict:
    """Counts the manager's per-group walks (instance attributes only)."""
    counts = {"cache_blocks": 0, "get_num_blocks_to_allocate": 0}
    coordinator = manager.coordinator
    for name in counts:
        method = getattr(coordinator, name)

        def counted(*args, _method=method, _name=name, **kwargs):
            counts[_name] += 1
            return _method(*args, **kwargs)

        setattr(coordinator, name, counted)
    return counts


@pytest.mark.parametrize(
    "seed,max_num_seqs,num_blocks,steps",
    [
        # Under memory pressure: preemption and local re-prefill.
        (1, 24, 9_000, 300),
        # No pressure: long steady decode runs and many finishes.
        (2, 32, 40_000, 400),
    ],
)
def test_fast_path_replay_matches_upstream_every_step(
    served_patches, monkeypatch, seed, max_num_seqs, num_blocks, steps
):
    config, scheduler_block, hash_block = replay.served_kv_cache_config(
        monkeypatch, num_blocks
    )
    worlds = []
    for enabled in (True, False):
        manager = replay.make_manager(config, scheduler_block, hash_block)
        if not enabled:
            manager.__dict__[fast_path._STATE] = None
        world = replay.World(
            manager,
            seed=seed,
            max_num_seqs=max_num_seqs,
            hash_block=hash_block,
            scheduler_block=scheduler_block,
        )
        worlds.append((world, _count_coordinator_calls(manager)))
    (fast, fast_calls), (base, base_calls) = worlds
    assert fast_path._state(fast.kv) is not None
    for step in range(steps):
        fast.step()
        base.step()
        fast_state = replay.snapshot(fast)
        base_state = replay.snapshot(base)
        if fast_state != base_state:
            diff = [key for key in base_state if fast_state[key] != base_state[key]]
            pytest.fail(f"step {step}: state differs in {diff}")
    assert base.finished > 0 and base.loads > 0
    if num_blocks < 10_000:
        assert base.preemptions > 0
    # The fast path took most steady calls.
    assert fast_calls["cache_blocks"] < base_calls["cache_blocks"] / 4
    assert (
        fast_calls["get_num_blocks_to_allocate"]
        < base_calls["get_num_blocks_to_allocate"] / 4
    )


def test_fingerprints_cover_every_function_the_thresholds_encode():
    names = set(fast_path._FINGERPRINTS)
    assert all(value for value in fast_path._FINGERPRINTS.values())
    for name in fast_path._analysed_functions():
        assert name in names


@pytest.mark.parametrize("prompt_len", [6_000, 6_144, 7_000])
def test_fast_path_single_token_steps_match_at_every_boundary(
    served_patches, monkeypatch, prompt_len
):
    """One-token steps reach every threshold exactly, tail on and off.

    6,144 tokens is a scheduler-block multiple: no prompt-tail entry. The
    others leave the prompt's last hash block inside a main-KV block.
    """
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request, RequestStatus

    config, scheduler_block, hash_block = replay.served_kv_cache_config(
        monkeypatch, 4_000
    )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(hash_block, sha256)
    managers, requests = [], []
    for enabled in (True, False):
        manager = replay.make_manager(config, scheduler_block, hash_block)
        if not enabled:
            manager.__dict__[fast_path._STATE] = None
        managers.append(manager)
        requests.append(
            Request(
                "r",
                list(range(prompt_len)),
                SamplingParams(max_tokens=5_000),
                None,
                block_hasher=hasher,
            )
        )
    counts = _count_coordinator_calls(managers[0])

    def check(where):
        world_states = []
        for manager in managers:
            world = replay.World.__new__(replay.World)
            world.kv, world.log = manager, []
            world_states.append(replay.snapshot(world))
        assert world_states[0] == world_states[1], where

    for manager, request in zip(managers, requests):
        start = 0
        while start < prompt_len:
            end = min(prompt_len, start + scheduler_block)
            blocks = manager.allocate_slots(
                request, end - start, num_lookahead_tokens=3
            )
            assert blocks is not None
            request.num_computed_tokens = start = end
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids(10_000)
    check("prefill")
    for step in range(2_300):
        results = []
        for manager, request in zip(managers, requests):
            blocks = manager.allocate_slots(request, 1, num_lookahead_tokens=3)
            results.append(tuple(tuple(b.block_id for b in g) for g in blocks.blocks))
            request.num_computed_tokens += 1
            request.append_output_token_ids(10_001 + step)
            manager.cache_blocks(request, request.num_computed_tokens)
        assert results[0] == results[1], step
        check(step)
    # Every threshold was crossed and most steps skipped the groups.
    assert counts["get_num_blocks_to_allocate"] < 2_300 / 4

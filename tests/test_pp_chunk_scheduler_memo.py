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
"""The pipeline chunk scheduler's prefix-hit memo on a real hybrid KV cache.

The scheduler is built over a real vLLM KVCacheManager (full attention and two
Mamba "align" groups, prefix caching on) as a Qwen3.5 MoE KV producer, so
hybrid-state retention and the cross-step memo work on the same BlockPool.
Every test starts from vLLM's own BlockPool and KVCacheManager methods with
the patches an engine process applies at start (main's same-step patch, then
the registry's prefix-cache epoch and change log), and restores the classes
afterwards.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

import vllm_torchtpu
from vllm_torchtpu.core import prefix_change_log
from vllm_torchtpu.core.pp_chunk_scheduler import (
    TpuPipelineChunkScheduler,
)
from vllm_torchtpu.core.pp_chunks import StepCostModel

HASH_BS = 2
BS = 4
_POOL_METHODS = (
    "_insert_block_hash",
    "_remove_cached_block_hashes",
    "reset_prefix_cache",
    "get_cached_block",
)


@pytest.fixture
def engine_block_pool():
    """vLLM's BlockPool and KVCacheManager methods with an engine's start-up
    patches, whatever earlier tests installed; restored afterwards."""
    saved = {name: BlockPool.__dict__[name] for name in _POOL_METHODS}
    saved_new_step = KVCacheManager.__dict__["new_step_starts"]
    for name in _POOL_METHODS:
        setattr(BlockPool, name, inspect.unwrap(saved[name]))
    KVCacheManager.new_step_starts = inspect.unwrap(saved_new_step)
    assert not prefix_change_log.epoch_installed()
    # The patch registry's order at engine start.
    vllm_torchtpu._patch_vllm_same_step_prefix_hits()
    prefix_change_log.install_epoch()
    assert prefix_change_log.install() is None
    init_none_hash(sha256)
    try:
        yield
    finally:
        for name, method in saved.items():
            setattr(BlockPool, name, method)
        KVCacheManager.new_step_starts = saved_new_step


def _manager(num_blocks: int = 64) -> KVCacheManager:
    mamba = MambaSpec(
        block_size=BS,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(["m0"], mamba),
            KVCacheGroupSpec(["m1"], mamba),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=BS,
        hash_block_size=HASH_BS,
        enable_caching=True,
        use_eagle=True,
    )


def _request(request_id: str, token_ids: list[int]) -> Request:
    params = SamplingParams(max_tokens=64)
    params.update_from_generation_config({}, eos_token_id=100_000)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BS, sha256),
    )


def _scheduler(kvm: KVCacheManager, options: dict) -> TpuPipelineChunkScheduler:
    """Runs TpuPipelineChunkScheduler.__init__ over a stub base __init__ on a
    Qwen3.5 MoE KV producer with ``kvm`` as its KV cache manager and
    hybrid-state retention on (as the Qwen3.5 config hook sets it)."""

    def base_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            additional_config={
                "tpu_pp_scheduler": {"hybrid_state_retention": True, **options}
            },
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_producer", is_kv_producer=True, is_kv_consumer=False
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=8),
            model_config=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"]
            ),
        )
        self.need_mamba_block_aligned_split = True
        self.kv_cache_manager = kvm
        self.connector = None
        self.running = []
        self.waiting = []
        self.kv_holding_waiting = []

    with patch.object(Scheduler, "__init__", base_init):
        return TpuPipelineChunkScheduler()


def _cache_prompt(kvm: KVCacheManager, token_ids: list[int]) -> None:
    """Prefills ``token_ids`` as one request, one block per step as the
    scheduler's Mamba-aligned chunks do, then finishes it, so its blocks stay
    cached and free. (One step for the whole prompt leaves no Mamba state an
    EAGLE lookup can hit.)"""
    request = _request("cached", token_ids)
    blocks, num_hit, _ = kvm.get_computed_blocks(request)
    done = 0
    while done < request.num_tokens:
        chunk = min(BS, request.num_tokens - done)
        if done == 0:
            assert kvm.allocate_slots(request, chunk, num_hit, blocks)
        else:
            assert kvm.allocate_slots(request, chunk)
        done += chunk
        request.num_computed_tokens = done
        kvm.new_step_starts()
    kvm.free(request)


def _fresh_hit(kvm: KVCacheManager, request: Request) -> int:
    return kvm.coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1
    )[1]


def _cached_prompt(kvm: KVCacheManager) -> list[int]:
    """Caches a 16-token prompt and returns it; a request that extends it
    hits it in the hybrid cache."""
    prompt = list(range(100, 116))
    _cache_prompt(kvm, prompt)
    assert _fresh_hit(kvm, _request("probe", prompt + [91, 92, 93, 94])) > 0
    return prompt


def test_a_prefix_cache_reset_clears_the_memo_with_retention_on(engine_block_pool):
    kvm = _manager()
    scheduler = _scheduler(kvm, {"prefix_aware": True})
    assert scheduler._retention is not None and scheduler._memo_xstep
    prompt = _cached_prompt(kvm)
    returning = _request("returning", prompt + [91, 92, 93, 94])
    assert scheduler._prefix_warmth(returning) == _fresh_hit(kvm, returning) > 0
    # vLLM's reset path: Scheduler.reset_prefix_cache -> KVCacheManager ->
    # coordinator -> the pool's reset_prefix_cache, which retention guards.
    assert kvm.reset_prefix_cache()
    assert _fresh_hit(kvm, returning) == 0
    assert scheduler._prefix_warmth(returning) == 0


def test_a_reset_through_retention_reaches_the_epoch_and_the_change_log(
    engine_block_pool,
):
    kvm = _manager()
    scheduler = _scheduler(kvm, {"prefix_aware": True})
    pool = kvm.block_pool
    epoch = pool.__dict__.get(prefix_change_log.EPOCH, 0)
    pool.__dict__[prefix_change_log.LOG].append((epoch, ("stale",)))
    assert pool.reset_prefix_cache()
    # Retention guards the pool's reset; the guard still reaches the
    # registry's epoch and change-log wrappers.
    assert pool.__dict__[prefix_change_log.EPOCH] == epoch + 1
    assert pool.__dict__[prefix_change_log.LOG] == []
    assert scheduler._retention is not None


def test_the_prefix_aware_lookup_answers_from_the_memo_as_vllm_does(
    engine_block_pool,
):
    # The local-hit lookup PrefixAwareScheduler and this scheduler share.
    kvm = _manager()
    scheduler = _scheduler(kvm, {"prefix_aware": True})
    prompt = _cached_prompt(kvm)
    returning = _request("returning", prompt + [91, 92, 93, 94])
    expected_blocks, expected_hit, _ = kvm.get_computed_blocks(returning)
    scheduler._prefix_warmth(returning)
    computes = scheduler._memo_computes

    def fresh_lookup(*args, **kwargs):
        raise AssertionError("the admission lookup recomputed the prefix hit")

    # Every fresh lookup, the scheduler's own and vLLM's get_computed_blocks,
    # goes through the coordinator: the answer must come from the memo.
    with patch.object(kvm.coordinator, "find_longest_cache_hit", fresh_lookup):
        blocks, hit, boundary, diverged = scheduler._get_local_prefix_cache_hit(
            returning
        )
    assert scheduler._memo_computes == computes
    assert hit == expected_hit > 0 and not diverged
    assert blocks.get_block_ids() == expected_blocks.get_block_ids()
    assert returning._tpu_local_hit_tokens == hit


def _model(granularity: int = 4) -> StepCostModel:
    buckets = [4, 8, 12, 16]
    linear = {b: 12.0 + 0.0109 * b for b in buckets}
    full = linear[16] + 1.5e-7 * 16**2 / 2
    return StepCostModel(buckets, linear, 1.5e-7, full * 1.1, granularity)


def test_quadratic_cost_reuses_prefix_hits_across_steps(engine_block_pool):
    kvm = _manager()
    scheduler = _scheduler(kvm, {"quadratic_cost": True})
    assert scheduler._memo_xstep and not scheduler._prefix_aware
    scheduler.install_cost_model(_model())
    prompt = _cached_prompt(kvm)
    scheduler.waiting = [
        _request(f"w{i}", prompt + [200 + i] * (3 + i)) for i in range(4)
    ]
    assert all(_fresh_hit(kvm, r) > 0 for r in scheduler.waiting)
    first = scheduler._pending_step_limit()
    computes = scheduler._memo_computes
    assert computes == len(scheduler.waiting)
    # Nothing in the cache changed: the next step reuses every lookup.
    assert scheduler._pending_step_limit() == first
    assert scheduler._memo_computes == computes
    # The memo answers what fresh lookups answer.
    for request in scheduler.waiting:
        assert scheduler._prefix_warmth(request) == _fresh_hit(kvm, request)

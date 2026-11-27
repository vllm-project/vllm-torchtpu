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
Mamba "align" groups, prefix caching on) as a KV producer, so hybrid-state
retention and the cross-step memo work on the same BlockPool.
Every test starts from vLLM's own BlockPool and KVCacheManager methods with
the patches an engine process applies at start (main's same-step patch, then
the registry's prefix-cache epoch and change log), and restores the classes
afterwards.
"""

import inspect
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus

import vllm_torchtpu
from tests.test_srpt_prefill_scheduler import (
    MODEL,
    _build_real_scheduler,
    _make_real_request,
)
from vllm_torchtpu.core import pp_chunk_scheduler, prefix_change_log
from vllm_torchtpu.core.pp_chunk_scheduler import (
    TpuPipelineChunkScheduler,
)
from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.scheduler_utils import KvRoles

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


def _manager(num_blocks: int = 64, prefill_checkpoint: bool = False) -> KVCacheManager:
    """A hybrid manager; ``prefill_checkpoint`` gives the Mamba groups one
    prefill checkpoint block, as pooled GDN declares them by default."""
    checkpoint = (
        {"num_prefill_checkpoint_blocks": 1, "prefill_checkpoint_alignment": HASH_BS}
        if prefill_checkpoint
        else {}
    )
    mamba = MambaSpec(
        block_size=BS,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        **checkpoint,
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


def _scheduler(
    kvm: KVCacheManager, options: dict, retention: bool = True
) -> TpuPipelineChunkScheduler:
    """Runs TpuPipelineChunkScheduler.__init__ over a stub base __init__ on a
    KV producer with ``kvm`` as its KV cache manager and, with ``retention``,
    hybrid-state retention on (as the Qwen3.5 config hook sets it)."""
    section = {"hybrid_state_retention": True, **options} if retention else options

    def base_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            additional_config={"tpu_pp_scheduler": section},
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_producer", is_kv_producer=True, is_kv_consumer=False
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=8),
        )
        self.need_mamba_block_aligned_split = True
        self.kv_cache_manager = kvm
        self.max_model_len = 4096
        self.num_lookahead_tokens = 0
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


def test_retention_on_a_small_pool_pins_at_most_5_percent_of_it(engine_block_pool):
    """On a real pool of 100 blocks the scheduler caps retention's cross-turn
    pins at 5 blocks, not the 360 measured on 8,675-block pools (retention
    evicts down to its cap: test_hybrid_retention)."""
    for num_blocks, cap in ((100, 5), (64, 3)):
        scheduler = _scheduler(_manager(num_blocks=num_blocks), {"prefix_aware": True})
        assert scheduler._retention is not None
        assert scheduler._retention.max_pinned_blocks == cap


@pytest.mark.parametrize("lookahead", [0, 4])
def test_the_fit_tier_counts_what_allocate_slots_takes(engine_block_pool, lookahead):
    """On a hybrid manager (full attention and two Mamba groups) a returning
    request's admission takes more blocks than its uncached tokens over the
    block size, the count the fit tier used before: one per Mamba group too,
    and a full-attention block for the speculative lookahead slots.
    ``_blocks_to_admit`` is exactly what allocate_slots then takes from the
    free blocks, hit blocks in the free queue included."""
    kvm = _manager()
    scheduler = _scheduler(kvm, {"prefix_aware": True})
    prompt = _cached_prompt(kvm)
    returning = _request("returning", prompt + list(range(200, 220)))
    warmth = scheduler._prefix_warmth(returning)
    without_lookahead = scheduler._blocks_to_admit(returning)
    scheduler.num_lookahead_tokens = lookahead
    need = scheduler._blocks_to_admit(returning)
    # 36 prompt tokens fill 9 blocks of 4; 4 lookahead slots need a 10th.
    assert need == without_lookahead + (1 if lookahead else 0)
    assert need > -(-(returning.num_tokens - warmth) // BS)
    blocks, num_hit, _ = kvm.get_computed_blocks(returning)
    assert num_hit == warmth > 0
    free = kvm.block_pool.get_num_free_blocks()
    assert (
        kvm.allocate_slots(
            returning,
            returning.num_tokens - num_hit,
            num_hit,
            blocks,
            num_lookahead_tokens=lookahead,
        )
        is not None
    )
    assert free - kvm.block_pool.get_num_free_blocks() == need


def test_the_fit_count_is_vllms_whole_sequence_check(engine_block_pool):
    """The count asks vLLM's coordinator what allocate_slots' whole-sequence
    check asks it: the hit's blocks with its tokens as computed, the prompt
    plus the lookahead slots, and apply_admission_cap=True."""
    kvm = _manager()
    scheduler = _scheduler(kvm, {"prefix_aware": True}, retention=False)
    scheduler.num_lookahead_tokens = 4
    prompt = _cached_prompt(kvm)
    returning = _request("returning", prompt + list(range(200, 220)))
    seen = []
    real = kvm.coordinator.get_num_blocks_to_allocate

    def recording(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    with patch.object(kvm.coordinator, "get_num_blocks_to_allocate", recording):
        scheduler._blocks_to_admit(returning)
    blocks, num_hit, _ = kvm.get_computed_blocks(returning)
    (call,) = seen
    assert call["total_computed_tokens"] == call["num_local_computed_tokens"] == num_hit
    assert num_hit > 0
    assert [[b.block_id for b in group] for group in call["new_computed_blocks"]] == [
        [b.block_id for b in group] for group in blocks.blocks
    ]
    assert call["num_tokens_main_model"] == returning.num_tokens
    assert call["num_tokens"] == returning.num_tokens + 4
    assert call["apply_admission_cap"] is True


def test_the_fit_tier_keeps_the_watermark_once_a_request_runs(engine_block_pool):
    """vLLM's allocate_slots keeps watermark_blocks free for a waiting request
    once another request is scheduled in the step, and it schedules running
    requests first: the count adds the watermark when a request runs, not on
    an idle step."""
    kvm = _manager()
    kvm.watermark_blocks = 3
    scheduler = _scheduler(kvm, {"prefix_aware": True}, retention=False)
    prompt = _cached_prompt(kvm)
    returning = _request("returning", prompt + list(range(200, 220)))
    idle = scheduler._blocks_to_admit(returning)
    scheduler.running = [_request("running", list(range(300, 310)))]
    assert scheduler._blocks_to_admit(returning) == idle + 3


def test_ranking_an_async_load_leaves_its_mamba_checkpoint_alone(engine_block_pool):
    """The warmest-first promotion ranks the requests whose async KV loads are
    in flight (kv_holding_waiting). Counting a request's blocks must not record
    a Mamba prefill checkpoint for its whole prompt: when its load completes,
    vLLM's cache_blocks would index that checkpoint past the blocks the request
    holds and assert, and the AssertionError would leave schedule()."""
    kvm = _manager(prefill_checkpoint=True)
    scheduler = _scheduler(kvm, {"prefix_aware": True}, retention=False)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    loads = []
    for request_id, base in (("head", 1000), ("second", 2000)):
        request = _request(request_id, list(range(base, base + 40)))
        # vLLM's waiting loop admits a host-tier load of the first 8 tokens.
        assert (
            kvm.allocate_slots(
                request,
                0,
                0,
                None,
                num_external_computed_tokens=8,
                delay_cache_blocks=True,
            )
            is not None
        )
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        request.num_computed_tokens = 8
        scheduler.kv_holding_waiting.add_request(request)
        loads.append(request)
    mamba = [
        m for m in kvm.coordinator.single_type_managers if hasattr(m, "_checkpoints")
    ]
    assert len(mamba) == 2
    assert all(m._checkpoints.get("head") is None for m in mamba)
    counted = []
    real = TpuPipelineChunkScheduler._blocks_to_admit

    def recording(self, request):
        counted.append(request.request_id)
        return real(self, request)

    with patch.object(TpuPipelineChunkScheduler, "_blocks_to_admit", recording):
        scheduler._step_wall_time = time.time()
        scheduler._select_waiting_queue_for_scheduling()
        scheduler._step_wall_time = None
    assert "head" in counted
    # The load completes, as vLLM's _update_waiting_for_remote_kv completes it
    # (with a checkpoint recorded by the count, this asserts in vLLM's Mamba
    # manager).
    kvm.cache_blocks(loads[0], loads[0].num_computed_tokens)
    assert all(m._checkpoints.get("head") is None for m in mamba)


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


# -- vLLM's own waiting loop -------------------------------------------------


@pytest.fixture(scope="module")
def opt_model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _real_pipeline_scheduler(model_config, prefix_aware: bool):
    """A real pipeline chunk scheduler (the SRPT tests' builder: vLLM's
    Scheduler.__init__, a real KVCacheManager), with a 16-token step budget,
    the KV producer role the waiting-queue sort runs on, and the promotion
    off, so only the sort can reorder the waiting queue."""
    options = {"tpu_pp_scheduler": {"prefix_aware": prefix_aware}}
    options["tpu_scheduler"] = {"promote_warmest": False}
    with patch.object(
        pp_chunk_scheduler,
        "kv_roles",
        lambda config: KvRoles(producer=True, consumer=False),
    ):
        return _build_real_scheduler(
            model_config,
            scheduler_cls=TpuPipelineChunkScheduler,
            num_blocks=64,
            max_num_seqs=4,
            max_num_batched_tokens=16,
            additional_config=options,
        )


@pytest.mark.parametrize("prefix_aware", [True, False])
def test_the_real_waiting_loop_admits_in_the_sorted_order(
    engine_block_pool, opt_model_config, prefix_aware
):
    """vLLM's schedule() on a real scheduler: with prefix_aware the sort puts
    a returning request whose prompt is cached (8 uncached tokens) ahead of
    an earlier cold one (40), vLLM's waiting loop admits it first, and its
    admission lookup is answered from the memo the sort filled. Without
    prefix_aware the loop admits in arrival order."""
    scheduler, hasher = _real_pipeline_scheduler(opt_model_config, prefix_aware)
    shared = list(range(500, 548))
    seed = _make_real_request("seed", 48, hasher, shared_prefix=shared)
    scheduler.add_request(seed)
    # Three 16-token steps prefill and cache the shared prompt.
    for _ in range(3):
        scheduler.schedule()
    assert seed.num_computed_tokens == seed.num_tokens
    scheduler.finish_requests("seed", RequestStatus.FINISHED_ABORTED)
    # An idle step: vLLM's step start marks the blocks the last step cached
    # as written, so they can be hit (a prefix cached turns ago, as a
    # returning request's is).
    scheduler.schedule()
    cold = _make_real_request("cold", 40, hasher, seed=7)
    warm = _make_real_request("warm", 56, hasher, shared_prefix=shared)
    warm.arrival_time = cold.arrival_time
    scheduler.add_request(cold)
    scheduler.add_request(warm)
    memo_answers = {}
    real_memo = pp_chunk_scheduler.memo_local_prefix_cache_hit

    def recording_memo(s, request, max_cache_hit_length):
        hit = real_memo(s, request, max_cache_hit_length)
        memo_answers[request.request_id] = hit
        return hit

    with patch.object(
        pp_chunk_scheduler, "memo_local_prefix_cache_hit", recording_memo
    ):
        out = scheduler.schedule()
    admitted = [r.req_id for r in out.scheduled_new_reqs]
    if prefix_aware:
        assert scheduler._paw_sort and scheduler._paw_sort_reordered == 1
        assert admitted[0] == "warm"
        # vLLM's admission took the memoised hit: 48 cached tokens.
        assert memo_answers["warm"] is not None and memo_answers["warm"][1] == 48
        assert out.num_scheduled_tokens["warm"] == 8
    else:
        assert not scheduler._paw_sort and memo_answers == {}
        assert admitted[0] == "cold"

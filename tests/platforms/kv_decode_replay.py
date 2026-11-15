# SPDX-License-Identifier: Apache-2.0
"""Drives a KVCacheManager the way the async scheduler does, on the CPU.

Used by test_kv_decode_fast_path.py and by host microbenchmarks. A ``World``
holds one engine core's scheduler-side KV cache manager and its requests and
replays the KV calls of vLLM's ``AsyncScheduler`` for a P/D decode engine
with MTP: running requests first (``allocate_slots`` with lookahead, then
preemption from the tail on failure), then waiting ones (local prefix hits
plus a connector load for the rest, landed some steps later, or a local
re-prefill after preemption), then the previous step's outputs (rejected
drafts roll back, ``cache_blocks`` after each output, finished requests go
through ``remove_skipped_blocks`` and ``free``). Every decision comes from a
seeded generator and the world's own state, so two worlds with equal states
take equal decisions.

``served_kv_cache_config`` builds the Qwen3.8-Flash decode's groups as served
(TP8, FP8 KV, FP32 SSM, main-KV DCP4, packed compressed keys, MTP K=3): 65
groups, scheduler block 2048, hash block 512.
"""

import random
import time
from collections import deque
from types import SimpleNamespace as NS

import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request, RequestStatus

SERVED_GROUP_SIZES = [512] * 37 + [2048] * 12 + [512] * 2 + [4] * 13 + [512]
NUM_SPEC = 3
# P(accepted drafts = 0..3): 1.35 accepted + 1 bonus per step, close to the
# golden 2.32 tokens per step.
ACCEPT_WEIGHTS = (0.25, 0.3, 0.3, 0.15)


def served_kv_cache_config(monkeypatch, num_blocks: int):
    """Returns (scheduler KVCacheConfig, scheduler block, hash block)."""
    from vllm.config.attention import AttentionConfig
    from vllm.config.cache import CacheConfig
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
        resolve_dcp_kv_block_size,
        resolve_kv_cache_block_sizes,
    )

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import ensure_kv_group_size_rule
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    for name, value in (
        ("QWEN4_EXP_PACK_COMPRESSED_K", "1"),
        ("QWEN4_EXP_PACK_MAIN_KV", "0"),
        ("VLLM_TPU_QSA_MAIN_KV_DCP", "4"),
    ):
        monkeypatch.setenv(name, value)
    cache = CacheConfig(
        block_size=128,
        cache_dtype="fp8",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float32",
    )
    cache.kv_cache_layout = "NHD"
    cfg = NS(
        additional_config={},
        attention_config=AttentionConfig(),
        cache_config=cache,
        model_config=NS(
            hf_text_config=q4exp_text_config(mtp_num_hidden_layers=1),
            max_model_len=262144,
            dtype=torch.bfloat16,
        ),
        parallel_config=NS(
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False, max_num_seqs=128),
        speculative_config=NS(
            num_speculative_tokens=NUM_SPEC,
            method="mtp",
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: False,
        ),
        kv_transfer_config=NS(),
    )
    ensure_kv_group_size_rule()
    layout, specs = pooled_cache_specs(cfg)
    cache.block_size = layout.block_size
    cache.mamba_block_size = layout.block_size
    groups = get_kv_cache_groups(cfg, specs)
    native = get_kv_cache_config_from_groups(cfg, groups, 1024**3)
    config = generate_scheduler_kv_cache_config([native])
    config.num_blocks = num_blocks
    config.prefix_cache_retention_interval = 0
    sizes = [
        resolve_dcp_kv_block_size(g.kv_cache_spec, 1) for g in config.kv_cache_groups
    ]
    assert sizes == SERVED_GROUP_SIZES, sizes
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(config, cfg)
    assert (scheduler_block, hash_block) == (2048, 512)
    return config, scheduler_block, hash_block


def make_manager(config, scheduler_block: int, hash_block: int, max_model_len=262144):
    """The decode engine's scheduler-side manager, built as Scheduler does."""
    from vllm_torchtpu.core.eagle_block_drop import clear_eagle_block_drop

    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=max_model_len,
        scheduler_block_size=scheduler_block,
        hash_block_size=hash_block,
        # 2 concurrent batches x 2048 batched tokens (async scheduling).
        max_in_flight_tokens=4096,
        enable_caching=True,
        # VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 builds it without the drop.
        use_eagle=False,
        num_prefill_lookahead=1,
        dcp_world_size=1,
        pcp_world_size=1,
    )
    clear_eagle_block_drop(manager)
    return manager


class Workload:
    """Seeded multi-turn sessions: each turn extends the previous context."""

    def __init__(
        self,
        seed: int,
        first_prompt=(2_000, 40_000),
        user_tokens=(100, 3_000),
        output_tokens=(50, 600),
        long_output_tokens=(2_000, 9_000),
        long_output_share=0.25,
        max_turns=4,
    ):
        self.rng = random.Random(seed)
        self.first_prompt = first_prompt
        self.user_tokens = user_tokens
        self.output_tokens = output_tokens
        # Long outputs cross several scheduler blocks during decode.
        self.long_output_tokens = long_output_tokens
        self.long_output_share = long_output_share
        self.max_turns = max_turns
        self.sessions: list[dict] = []
        self.next_token = 1

    def _tokens(self, n: int) -> list[int]:
        start = self.next_token
        self.next_token += n
        return list(range(start, start + n))

    def next_request(self, step: int) -> tuple[str, list[int], int]:
        """(request id, prompt, max_tokens) for a new turn."""
        rng = self.rng
        ready = [
            s
            for s in self.sessions
            if s["ready"] <= step and s["turn"] < self.max_turns
        ]
        if ready and rng.random() < 0.6:
            session = rng.choice(ready)
            prompt = session["context"] + self._tokens(rng.randint(*self.user_tokens))
        else:
            session = {"id": len(self.sessions), "turn": 0, "context": []}
            self.sessions.append(session)
            prompt = self._tokens(rng.randint(*self.first_prompt))
        session["turn"] += 1
        session["ready"] = 1 << 60
        session["prompt"] = prompt
        request_id = f"s{session['id']}-t{session['turn']}"
        if rng.random() < self.long_output_share:
            return request_id, prompt, rng.randint(*self.long_output_tokens)
        return request_id, prompt, rng.randint(*self.output_tokens)

    def finished(self, request_id: str, output: list[int], step: int) -> None:
        session = self.sessions[int(request_id[1:].split("-")[0])]
        session["context"] = session["prompt"] + output
        session["ready"] = step + self.rng.randint(1, 20)


class World:
    """One engine core's KV cache manager and the requests it serves."""

    def __init__(
        self,
        manager: KVCacheManager,
        seed: int,
        max_num_seqs: int,
        hash_block: int,
        token_budget: int = 2048,
        scheduler_block: int = 2048,
        workload: Workload | None = None,
        timed: bool = False,
    ):
        init_none_hash(sha256)
        self.kv = manager
        self.rng = random.Random(seed + 1)
        self.workload = workload or Workload(seed)
        self.hasher = get_request_block_hasher(hash_block, sha256)
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.scheduler_block = scheduler_block
        self.max_model_len = manager.max_model_len
        self.running: list[Request] = []
        self.waiting: deque[Request] = deque()
        self.landing: dict[str, int] = {}
        self.in_flight: list[tuple[Request, int, int]] = []
        self.step_index = 0
        self.log: list[tuple] = []
        self.timed = timed
        # Seconds and call counts of steady running-request calls.
        self.alloc_time = 0.0
        self.alloc_calls = 0
        self.cache_time = 0.0
        self.cache_calls = 0
        self.preemptions = 0
        self.finished = 0
        self.loads = 0

    # ------------------------------------------------------------ helpers
    def _new_request(self) -> Request:
        request_id, prompt, max_tokens = self.workload.next_request(self.step_index)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        return Request(request_id, prompt, params, None, block_hasher=self.hasher)

    def _ids(self, blocks) -> tuple | None:
        if blocks is None:
            return None
        return tuple(tuple(b.block_id for b in group) for group in blocks.blocks)

    def _preempt(self, request: Request) -> None:
        self.kv.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_output_placeholders = 0
        request.spec_token_ids = []
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_preemptions += 1
        self.waiting.appendleft(request)
        self.preemptions += 1
        self.log.append(("preempt", request.request_id))

    def _finish(self, request: Request) -> None:
        self.running.remove(request)
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        self.kv.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )
        self.kv.free(request)
        self.workload.finished(
            request.request_id, list(request.output_token_ids), self.step_index
        )
        self.finished += 1
        self.log.append(("finish", request.request_id))

    def _aligned_chunk(self, request: Request, start: int, num_new: int) -> int:
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new
        end = start + num_new
        if end < prefill_end:
            aligned = end // self.scheduler_block * self.scheduler_block
            if aligned > start:
                return aligned - start
        return num_new

    def _allocate(self, request: Request, num_new: int, **kwargs):
        kv = self.kv
        timed = (
            self.timed
            and kwargs.keys() == {"num_lookahead_tokens"}
            and request.status == RequestStatus.RUNNING
        )
        if timed:
            t0 = time.perf_counter()
        blocks = kv.allocate_slots(request, num_new, **kwargs)
        if timed:
            self.alloc_time += time.perf_counter() - t0
            self.alloc_calls += 1
        self.log.append(("alloc", request.request_id, num_new, self._ids(blocks)))
        return blocks

    # ------------------------------------------------------------ one step
    def schedule(self) -> dict[str, int]:
        kv = self.kv
        pool = kv.block_pool
        # vllm_torchtpu._patch_vllm_same_step_prefix_hits: schedule() starts
        # a new pass.
        pool.__dict__.setdefault("_tpu_unwritten_block_ids", set()).clear()
        kv.new_step_starts()
        budget = self.token_budget
        scheduled: dict[str, int] = {}
        spec: dict[str, int] = {}
        preempted = False
        index = 0
        while index < len(self.running) and budget > 0:
            request = self.running[index]
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                index += 1
                continue
            num_new = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            num_new = min(
                num_new, budget, self.max_model_len - request.num_computed_tokens - 1
            )
            num_new = self._aligned_chunk(request, request.num_computed_tokens, num_new)
            if num_new <= 0:
                index += 1
                continue
            while True:
                blocks = self._allocate(request, num_new, num_lookahead_tokens=NUM_SPEC)
                if blocks is not None:
                    break
                victim = self.running.pop()
                if victim.request_id in scheduled:
                    budget += scheduled.pop(victim.request_id)
                    spec.pop(victim.request_id, None)
                self._preempt(victim)
                preempted = True
                if victim is request:
                    break
            if blocks is None:
                break
            scheduled[request.request_id] = num_new
            budget -= num_new
            index += 1
            if request.spec_token_ids:
                num_spec = (
                    num_new
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_spec > 0:
                    spec[request.request_id] = min(
                        num_spec, len(request.spec_token_ids)
                    )
                request.spec_token_ids = []

        # Top up the waiting queue with new turns.
        while len(self.waiting) + len(self.running) < self.max_num_seqs + 8:
            self.waiting.append(self._new_request())
        if not preempted:
            self._schedule_waiting(scheduled, budget)
        # _update_after_schedule (Scheduler and AsyncScheduler).
        outputs = []
        for request in self.running:
            num = scheduled.get(request.request_id)
            if num is None:
                continue
            request.num_computed_tokens += num
            request.num_in_flight_tokens += num
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            num_spec = spec.get(request.request_id, 0)
            if not request.is_prefill_chunk:
                request.num_output_placeholders += 1 + num_spec
                request.spec_token_ids = [-1] * NUM_SPEC
            outputs.append((request, num, num_spec, request.is_prefill_chunk))
        previous, self.in_flight = self.in_flight, outputs
        self.step_index += 1
        return previous

    def _schedule_waiting(self, scheduled: dict[str, int], budget: int) -> None:
        kv = self.kv
        skipped = deque()
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            request = self.waiting[0]
            request_id = request.request_id
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                if self.landing[request_id] > self.step_index:
                    skipped.append(self.waiting.popleft())
                    continue
                # _update_waiting_for_remote_kv
                del self.landing[request_id]
                kv.cache_blocks(request, request.num_computed_tokens)
                self.log.append(("land", request_id))
                if request.num_computed_tokens == request.num_tokens:
                    request.num_computed_tokens = request.num_tokens - 1
                request.status = (
                    RequestStatus.PREEMPTED
                    if request.num_preemptions
                    else RequestStatus.WAITING
                )
            if request.num_stale_output_tokens:
                skipped.append(self.waiting.popleft())
                continue
            if request.num_computed_tokens == 0:
                hit_blocks, num_hit = kv.get_computed_blocks(request)[:2]
                if (
                    request.status == RequestStatus.WAITING
                    and request.num_preemptions == 0
                ):
                    # A new P/D request: the connector loads the rest.
                    num_external = request.num_tokens - num_hit
                    blocks = self._allocate(
                        request,
                        0,
                        num_new_computed_tokens=num_hit,
                        new_computed_blocks=hit_blocks,
                        num_lookahead_tokens=0,
                        num_external_computed_tokens=num_external,
                        delay_cache_blocks=True,
                        reserved_blocks=self.rng.choice((0, 0, 8)),
                        has_scheduled_reqs=bool(self.running),
                    )
                    if blocks is None:
                        break
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    request.num_computed_tokens = num_hit + num_external
                    self.landing[request_id] = self.step_index + self.rng.randint(1, 4)
                    self.loads += 1
                    skipped.append(self.waiting.popleft())
                    continue
                # A preempted request re-prefills locally from its hit.
                num_new = min(request.num_tokens - num_hit, budget)
                num_new = self._aligned_chunk(request, num_hit, num_new)
                blocks = self._allocate(
                    request,
                    num_new,
                    num_new_computed_tokens=num_hit,
                    new_computed_blocks=hit_blocks,
                    num_lookahead_tokens=NUM_SPEC,
                    has_scheduled_reqs=bool(self.running),
                )
                num_computed = num_hit
            else:
                num_new = min(request.num_tokens - request.num_computed_tokens, budget)
                blocks = self._allocate(
                    request,
                    num_new,
                    num_lookahead_tokens=NUM_SPEC,
                    has_scheduled_reqs=bool(self.running),
                )
                num_computed = request.num_computed_tokens
            if blocks is None:
                break
            self.waiting.popleft()
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed
            scheduled[request_id] = num_new
            budget -= num_new
        self.waiting.extendleft(reversed(skipped))

    def update_from_output(self, previous) -> None:
        """Applies the outputs of the step scheduled before the last one."""
        kv = self.kv
        rng = self.rng
        for request, num, num_spec, prefill_chunk in previous:
            request.num_in_flight_tokens -= num
            if request.num_stale_output_tokens > 0:
                request.num_stale_output_tokens -= num
                continue
            if request.status != RequestStatus.RUNNING or prefill_chunk:
                continue
            accepted = rng.choices(range(NUM_SPEC + 1), ACCEPT_WEIGHTS)[0]
            accepted = min(accepted, num_spec)
            rejected = num_spec - accepted
            if request.num_computed_tokens > 0:
                request.num_computed_tokens -= rejected
            if request.num_output_placeholders > 0:
                request.num_output_placeholders -= rejected
            remaining = request.max_tokens - request.num_output_tokens
            new = [rng.randint(1, 100_000) for _ in range(accepted + 1)][:remaining]
            request.append_output_token_ids(new)
            request.num_output_placeholders -= len(new)
            if self.timed:
                t0 = time.perf_counter()
            kv.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
            if self.timed:
                self.cache_time += time.perf_counter() - t0
                self.cache_calls += 1
            if request.num_output_tokens >= request.max_tokens:
                self._finish(request)

    def step(self) -> None:
        previous = self.schedule()
        self.update_from_output(previous)


def snapshot(world: World, full_pool: bool = True) -> dict:
    """The manager's observable state: tables, caches, pool and free queue."""
    kv = world.kv
    pool = kv.block_pool
    groups = []
    for manager in kv.coordinator.single_type_managers:
        state = {
            "tables": {
                rid: tuple(b.block_id for b in blocks)
                for rid, blocks in manager.req_to_blocks.items()
            },
            "cached": dict(manager.num_cached_block),
            "partial": {
                rid: (idx, b.block_id)
                for rid, (idx, b) in manager._partial_hit_reqs.items()
            },
            "new_ids": list(manager.new_block_ids),
            "cow": [(a.block_id, b.block_id) for a, b in manager._pending_cow_copies],
            "offloads": [
                (rid, gid, b.block_id, n)
                for rid, gid, b, n in manager._pending_boundary_state_offloads
            ],
        }
        if hasattr(manager, "last_state_block_idx"):
            state.update(
                last=dict(manager.last_state_block_idx),
                retired=dict(manager._num_retired_blocks),
                allocated=set(manager._allocated_block_reqs),
                checkpoints=dict(manager._checkpoints),
                this_step=set(manager.cached_blocks_this_step),
                own={
                    rid: {bid: (b.block_id, d) for bid, (b, d) in own.items()}
                    for rid, own in manager.__dict__.get(
                        "_tpu_own_checkpoints", {}
                    ).items()
                },
                pinned={
                    rid: [b.block_id for b in blocks]
                    for rid, blocks in manager.__dict__.get(
                        "_tpu_pinned_checkpoints", {}
                    ).items()
                },
            )
        groups.append(state)
    snap = {
        "groups": groups,
        "free": pool.get_num_free_blocks(),
        "unwritten": set(pool.__dict__.get("_tpu_unwritten_block_ids", ())),
        "log": list(world.log),
    }
    if full_pool:
        snap["queue"] = [
            b.block_id for b in pool.free_block_queue.get_all_free_blocks()
        ]
        snap["blocks"] = [
            (b.ref_cnt, b.block_hash, b.block_hash_num_tokens) for b in pool.blocks
        ]
        snap["hash_map"] = {
            key: (
                value.block_id if not isinstance(value, dict) else tuple(sorted(value))
            )
            for key, value in pool.cached_block_hash_to_block._cache.items()
        }
        snap["extra_hashes"] = {
            bid: frozenset(keys)
            for bid, keys in pool.cached_block_hashes_by_block.items()
        }
    world.log.clear()
    return snap

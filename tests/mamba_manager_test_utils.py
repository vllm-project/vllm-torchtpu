# SPDX-License-Identifier: Apache-2.0
"""Runs the installed vLLM KV cache manager and a model of the TPU worker in
lockstep, for the GDN internal prefill checkpoint tests."""

import inspect
import random
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.gdn_pool_layout import PrefillCheckpointUnits
from vllm_torchtpu.layers.core.attention_metadata import _gather_prefill_ckpt_ids

MAMBA_GROUP = 1
# The GDN layers declare this alignment: any checkpoint row is exportable.
PREFILL_CHECKPOINT_ALIGNMENT = 1


def pin_upstream_mamba_manager(monkeypatch) -> None:
    """Restores unpatched MambaManager and BlockPool methods for one test.

    The BlockPool lookups are unwrapped because the same-step-hit patch hides
    blocks hashed since the last schedule(), which these tests never call.
    """
    monkeypatch.setattr(
        MambaManager, "cache_blocks", inspect.unwrap(MambaManager.cache_blocks)
    )
    for name in ("_insert_block_hash", "get_cached_block"):
        monkeypatch.setattr(BlockPool, name, inspect.unwrap(getattr(BlockPool, name)))


def upstream_has_prefill_end_gate() -> bool:
    """Returns True if MambaManager only reserves checkpoints on the final
    prefill chunk when prefix_cache_retention_interval=0 (vLLM #59175)."""
    return (
        "prefill_end"
        in inspect.signature(MambaManager._needs_internal_checkpoint).parameters
    )


@dataclass(frozen=True)
class Geometry:
    """One engine configuration, in the scheduler's logical tokens."""

    block: int
    hash_block: int
    spec_blocks: int
    eagle: bool
    budget: int
    retention: int | None

    def reserves_every_chunk(self) -> bool:
        """Whether the manager reserves a checkpoint on every chunk that has
        a valid one, like the worker rule, rather than on the last only."""
        return self.retention != 0 or not upstream_has_prefill_end_gate()


def make_kv_cache_manager(geometry: Geometry, num_blocks: int = 4096) -> KVCacheManager:
    init_none_hash(sha256)
    mamba = MambaSpec(
        block_size=geometry.block,
        shapes=((1,), (1,)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="align",
        num_speculative_blocks=geometry.spec_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=PREFILL_CHECKPOINT_ALIGNMENT,
    )
    attention = FullAttentionSpec(
        block_size=geometry.hash_block, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
        prefix_cache_retention_interval=geometry.retention,
    )
    return KVCacheManager(
        config,
        max_model_len=131072,
        scheduler_block_size=geometry.block,
        hash_block_size=geometry.hash_block,
        enable_caching=True,
        # The scheduler passes use_eagle_block_drop, True under MTP.
        use_eagle=geometry.eagle,
        enable_kv_cache_events=True,
    )


def make_request(request_id: str, token_ids: list[int], hash_block: int) -> Request:
    return Request(
        request_id,
        token_ids,
        SamplingParams(max_tokens=8),
        None,
        block_hasher=get_request_block_hasher(hash_block, sha256),
    )


class Lockstep:
    """Runs KVCacheManager and a model of the TPU worker in lockstep.

    Each step schedules a prefill chunk, updates the worker's append-only Mamba
    block table, computes checkpoint offsets via PrefillCheckpointUnits, and
    gathers checkpoint block IDs via _gather_prefill_ckpt_ids. Assertions
    verify that the worker only writes blocks owned by the request, writes
    checkpoints into the blocks reserved by MambaManager, and leaves every
    cached Mamba block and boundary-state offload matching its prefix hash.
    """

    def __init__(self, geometry: Geometry, split, *, allow_unreserved=False):
        self.geometry = geometry
        self.split = split
        self.allow_unreserved = allow_unreserved
        self.manager = make_kv_cache_manager(geometry)
        self.mamba = self.manager.coordinator.single_type_managers[MAMBA_GROUP]
        self.pool = self.manager.block_pool
        self.scheduler = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=geometry.block),
            block_size=geometry.block,
            hash_block_size=geometry.hash_block,
            use_eagle=geometry.eagle,
            use_eagle_block_drop=geometry.eagle,
            max_num_scheduled_tokens=geometry.budget,
            scheduler_config=SimpleNamespace(
                long_prefill_token_threshold=geometry.budget
            ),
            mamba_has_prefill_checkpoint_blocks=True,
            mamba_prefill_checkpoint_alignment=PREFILL_CHECKPOINT_ALIGNMENT,
            mamba_partial_cache_hit=(
                geometry.hash_block < geometry.block
                and self.manager.coordinator.enable_partial_hash_hits
            ),
            mamba_shared_prefix_checkpoint=False,
        )
        self.units = PrefillCheckpointUnits(
            hash_block_size=geometry.hash_block,
            mamba_block_size=geometry.block,
            alignment=PREFILL_CHECKPOINT_ALIGNMENT,
            drop_eagle_block=geometry.eagle,
        )
        self.requests: dict[str, Request] = {}
        self.tables: dict[str, list[int]] = {}
        self._admitted: dict[str, tuple] = {}
        self.content: dict[int, object] = {}
        # Partial-hit copies: block id -> (request, hit length).
        self.partial_hit_copies: dict[int, tuple[Request, int]] = {}
        self.stats: Counter = Counter()

    def tag(self, request: Request, num_tokens: int) -> object:
        """Returns the expected tag for a block holding state after num_tokens."""
        hash_block = self.geometry.hash_block
        if num_tokens > 0 and num_tokens % hash_block == 0:
            return request.block_hashes[num_tokens // hash_block - 1]
        return ("unhashable", request.request_id, num_tokens)

    def admit(self, request: Request, *, full_sequence_must_fit=False) -> int:
        """Looks up the prefix cache for a new request; returns the hit."""
        blocks, num_hit, _ = self.manager.get_computed_blocks(request)
        self.requests[request.request_id] = request
        self._admitted[request.request_id] = (blocks, num_hit, full_sequence_must_fit)
        return num_hit

    def step(self, requests: list[Request], *, decode=False) -> list[tuple]:
        """Schedules one chunk per request, runs the worker, checks invariants,
        and returns (request, start, end) tuples."""
        scheduled = []
        for request in requests:
            scheduled.append(self._schedule(request, decode=decode))
        for source, dest in self.mamba.take_pending_cow_copies():
            self.content[dest.block_id] = self.content.get(source.block_id)
        for request, start, end, checkpoint in scheduled:
            self._run_worker(request, start, end, checkpoint)
        self._check_prefix_cache()
        for request, start, end, _ in scheduled:
            request.num_computed_tokens = end
        self.pool.take_events()
        self.manager.new_step_starts()
        return [(request, start, end) for request, start, end, _ in scheduled]

    def prefill(self, request: Request, **admit_kwargs) -> list[tuple[int, int]]:
        """Admits a request and runs its whole prompt; returns the chunks."""
        self.admit(request, **admit_kwargs)
        chunks = []
        while request.request_id in self._admitted or (
            request.num_computed_tokens < request.num_tokens
        ):
            ((_, start, end),) = self.step([request])
            chunks.append((start, end))
        return chunks

    def decode(self, request: Request, num_steps: int) -> None:
        for i in range(num_steps):
            request.append_output_token_ids([1 + i])
            self.step([request], decode=True)

    def free(self, request: Request) -> None:
        self.manager.free(request)
        del self.tables[request.request_id]

    def _schedule(self, request: Request, *, decode: bool) -> tuple:
        request_id = request.request_id
        admitted = self._admitted.pop(request_id, None)
        hit_blocks, num_hit, must_fit = admitted or (None, 0, False)
        start = request.num_computed_tokens + num_hit
        num_new_tokens = request.num_tokens - start
        if not decode:
            num_new_tokens = self.split(
                self.scheduler,
                request,
                min(self.geometry.budget, num_new_tokens),
                num_hit,
            )
        assert num_new_tokens > 0
        num_copies = len(self.mamba._pending_cow_copies)
        new_blocks = self.manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_hit,
            new_computed_blocks=hit_blocks,
            full_sequence_must_fit=must_fit,
        )
        assert new_blocks is not None
        if admitted is not None and start % self.geometry.block:
            for _, dest in self.mamba._pending_cow_copies[num_copies:]:
                self.partial_hit_copies[dest.block_id] = (request, start)
        if admitted is not None:
            self.tables[request_id] = list(
                self.manager.get_block_ids(request_id)[MAMBA_GROUP]
            )
            request.num_computed_tokens = start
        else:
            self.tables[request_id] += new_blocks.get_block_ids()[MAMBA_GROUP]
        assert len(self.tables[request_id]) == len(self.mamba.req_to_blocks[request_id])
        return (
            request,
            start,
            start + num_new_tokens,
            self.mamba._checkpoints.get(request_id),
        )

    def _run_worker(self, request: Request, start: int, end: int, checkpoint) -> None:
        block = self.geometry.block
        request_id = request.request_id
        table = self.tables[request_id]
        owned = {
            b.block_id for b in self.mamba.req_to_blocks[request_id] if not b.is_null
        }
        state_col = (end - 1) // block
        state_id = table[state_col]
        if start > 0:
            assert self.content.get(table[(start - 1) // block]) == self.tag(
                request, start
            ), (request_id, start, end)

        offsets = np.zeros(1, dtype=np.int32)
        self.units.fill_offsets(
            offsets, np.array([end]), np.array([end - start], dtype=np.int32)
        )
        offset = int(offsets[0])
        # A padded device row: stale ids from earlier occupants past the
        # request's own columns.
        width = len(table) + 4
        row = torch.tensor(
            table + [max(table) + 1 + i for i in range(width - len(table))],
            dtype=torch.int32,
        )
        ckpt_id = int(
            _gather_prefill_ckpt_ids(
                row.unsqueeze(0),
                torch.tensor([state_col], dtype=torch.int32),
                torch.tensor([offset], dtype=torch.int32),
                self.geometry.spec_blocks,
            )[0]
        )

        if checkpoint is not None:
            position, column = checkpoint
            reserved_id = self.mamba.req_to_blocks[request_id][column].block_id
            assert start + offset == position, (request_id, start, end, offset)
            assert column == state_col - 1
            assert table[column] == reserved_id != 0
            assert ckpt_id == reserved_id
            self.stats["reserved"] += 1
        elif offset > 0:
            self._classify_unreserved(table, state_col, owned)
            assert self.allow_unreserved or ckpt_id < 0, (request_id, start, end)

        writes = []
        if ckpt_id >= 0:
            assert ckpt_id in owned, (request_id, ckpt_id)
            assert checkpoint is not None or ckpt_id == state_id, (
                "checkpoint written into a block nothing overwrites this step",
                request_id,
                start,
                end,
            )
            writes.append((ckpt_id, self.tag(request, start + offset)))
        assert state_id in owned and state_id != 0
        # Both GDN kernels land the final state after the checkpoint.
        writes.append((state_id, self.tag(request, end)))
        for block_id, content in writes:
            self.content[block_id] = content
            self.partial_hit_copies.pop(block_id, None)

    def _classify_unreserved(self, table, state_col, owned) -> None:
        """Counts what the worker table holds at the checkpoint column of a
        chunk the manager did not reserve."""
        block_id = table[state_col - 1]
        tail = table[state_col : state_col + 1 + self.geometry.spec_blocks]
        if block_id == 0:
            kind = "null"
        elif block_id not in owned:
            kind = "not owned"
        elif block_id == table[state_col]:
            kind = "own block, moved to the state column"
        elif block_id in tail:
            kind = "own block, moved to a speculative column"
        else:
            kind = "own block, elsewhere"
        self.stats[f"unreserved: {kind}"] += 1

    def _check_prefix_cache(self) -> None:
        for entry in self.mamba.take_pending_boundary_state_offloads():
            request_id, _, block, boundary_tokens = entry
            expected = self.tag(self.requests[request_id], boundary_tokens)
            if self.content.get(block.block_id) != expected:
                assert self._is_partial_hit_copy(block.block_id, expected), (
                    "offload offer over the wrong state",
                    entry,
                )
        for key, blocks in self.pool.cached_block_hash_to_block._cache.items():
            if get_group_id(key) != MAMBA_GROUP:
                continue
            for block in (
                [blocks] if isinstance(blocks, KVCacheBlock) else blocks.values()
            ):
                expected = get_block_hash(key)
                if self.content.get(block.block_id) != expected:
                    assert self._is_partial_hit_copy(block.block_id, expected), (
                        "prefix-cache entry over the wrong state",
                        block.block_id,
                    )

    def _is_partial_hit_copy(self, block_id: int, expected: object) -> bool:
        """Returns True if a cache mismatch is caused by upstream vLLM keying a
        partial-hit CoW copy at the next block boundary.

        When hash_block < mamba_block and a request hits mid-block, vLLM copies
        the hit state into a private block in that column. If an internal
        checkpoint lets the chunk run past that block boundary without
        splitting, vLLM v0.31.x hashes the copy under the boundary key even
        though the boundary state was never written. It involves neither the
        checkpoint block nor the scratch blocks under test, so it is counted,
        not failed.
        """
        copy = self.partial_hit_copies.get(block_id)
        if copy is None:
            return False
        request, start = copy
        boundary = (start // self.geometry.block + 1) * self.geometry.block
        if self.content.get(block_id) != self.tag(request, start) or (
            expected != self.tag(request, boundary)
        ):
            return False
        self.stats["upstream: partial-hit copy keyed at its block boundary"] += 1
        return True


def run_scenario(lockstep: Lockstep, seed: int) -> None:
    """Prefills a prompt, two copies at once, and prefix-hit resends."""
    rng = random.Random(seed)
    geometry = lockstep.geometry
    block, hash_block, budget = geometry.block, geometry.hash_block, geometry.budget
    # Seed 0 is a long prompt, seed 1 a medium one, seed 2 a short one.
    low, high = [
        (2 * budget, 3 * budget),
        (budget, 2 * budget),
        (hash_block + 1, 2 * block),
    ][seed % 3]
    length = rng.randint(low, max(low, high))
    if rng.random() < 0.3:
        # End at or just before a block boundary, so decode crosses it.
        length = max(length // block * block - rng.randint(0, 2), hash_block + 1)
    base = [rng.randint(1, 30000) for _ in range(length + block)]
    counter = iter(range(1000))

    def new_request(num_tokens: int, tokens=None) -> Request:
        token_ids = (tokens or base)[:num_tokens]
        return make_request(f"r{next(counter)}", token_ids, hash_block)

    # Two copies of one prompt admitted together, so neither hits the other.
    first, twin = new_request(length), new_request(length)
    lockstep.admit(first, full_sequence_must_fit=rng.random() < 0.5)
    lockstep.admit(twin)
    pending = [first, twin]
    while pending:
        lockstep.step(pending)
        pending = [r for r in pending if r.num_computed_tokens < r.num_tokens]
    lockstep.decode(first, 3)
    lockstep.free(first)
    lockstep.free(twin)

    # Resends: the same prompt, a prefix, an extension, and another prompt.
    for num_tokens in (
        length,
        rng.randint(min(length, max(hash_block + 1, length // 2)), length),
        length + rng.randint(1, block),
    ):
        request = new_request(num_tokens)
        lockstep.prefill(request)
        lockstep.decode(request, 1)
        lockstep.free(request)
    other = [rng.randint(1, 30000) for _ in range(length)]
    request = new_request(length, other)
    lockstep.prefill(request)
    lockstep.free(request)

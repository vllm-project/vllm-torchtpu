# SPDX-License-Identifier: Apache-2.0
"""Verifies that worker GDN prefill checkpoints match MambaManager reservations."""

from types import SimpleNamespace

import numpy as np
import pytest
from mamba_manager_test_utils import (
    Geometry,
    Lockstep,
    make_request,
    pin_upstream_mamba_manager,
    run_scenario,
)
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import _patch_vllm_mamba_split_scheduler_block_size
from vllm_torchtpu.core import mamba_manager_patches
from vllm_torchtpu.gdn_pool_layout import (
    PrefillCheckpointUnits,
    prefill_checkpoint_enabled,
)

# PP4xPCP8 with --block-size 384: one logical block is 384 x 8 tokens.
RANK_BLOCK = 384
BLOCK = 3072
BUDGET = 16384
MAMBA_GROUP = 1


def _offsets(seq_lens, query_lens, *, drop_eagle_block, alignment=64):
    units = PrefillCheckpointUnits(
        hash_block_size=BLOCK,
        mamba_block_size=BLOCK,
        alignment=alignment,
        drop_eagle_block=drop_eagle_block,
    )
    out = np.zeros(len(seq_lens), dtype=np.int32)
    units.fill_offsets(out, np.array(seq_lens), np.array(query_lens))
    return out


@pytest.mark.parametrize(("drop_eagle_block", "offset"), [(True, 3072), (False, 6144)])
def test_one_chunk_prompt_checkpoints_below_its_last_full_block(
    drop_eagle_block, offset
):
    offsets = _offsets([8191], [8191], drop_eagle_block=drop_eagle_block)
    assert offsets.dtype == np.int32
    assert offsets.tolist() == [offset]


def test_each_chunk_ending_on_a_valid_position_checkpoints():
    # 19456 tokens in 16K-budget chunks, no MTP: 15360 then 4096. The first
    # chunk has a valid checkpoint at 12288 as well; whether the manager
    # reserves it is checked against the block table (see the lockstep tests).
    offsets = _offsets([15360, 19456], [15360, 4096], drop_eagle_block=False)
    assert offsets.tolist() == [12288, 3072]


def test_rows_without_a_valid_checkpoint_get_zero():
    offsets = _offsets(
        # A decode row, a chunk starting mid-block, a chunk too short to hold
        # a checkpoint, and a fresh 8191-token prefill.
        [9001, 4000, 6000, 8191],
        [1, 2000, 2928, 8191],
        drop_eagle_block=True,
    )
    assert offsets.tolist() == [0, 0, 0, 3072]


def test_no_alignment_means_no_checkpoints():
    offsets = _offsets(
        [8191, 19456], [8191, 4096], drop_eagle_block=False, alignment=None
    )
    assert offsets.tolist() == [0, 0]


def _vllm_config(
    mamba_cache_mode="align",
    architecture="Qwen3_5MoeForConditionalGeneration",
    decode_context_parallel_size=1,
    prefix_match_unit=None,
    mamba_block_size=384,
    prefill_context_parallel_size=8,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode=mamba_cache_mode,
            prefix_match_unit=prefix_match_unit,
            mamba_block_size=mamba_block_size,
        ),
        model_config=SimpleNamespace(architecture=architecture),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
        ),
    )


@pytest.fixture
def default_pool(monkeypatch):
    """Leaves the unified-pool choice to the architecture."""
    monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    return monkeypatch


def test_checkpoints_are_on_for_unified_pool_gdn(default_pool):
    assert prefill_checkpoint_enabled(_vllm_config())


@pytest.mark.parametrize(
    "config",
    [
        _vllm_config(mamba_cache_mode="none"),
        _vllm_config(mamba_cache_mode="all"),
        _vllm_config(decode_context_parallel_size=2),
        # Hybrid families on the per-layer KV caches.
        _vllm_config(architecture="NemotronHForCausalLM"),
        _vllm_config(architecture="FalconH1ForCausalLM"),
    ],
)
def test_checkpoints_need_align_mode_the_unified_pool_and_no_dcp(default_pool, config):
    assert not prefill_checkpoint_enabled(config)


@pytest.mark.parametrize(
    ("prefix_match_unit", "mamba_block_size", "pcp", "enabled"),
    [
        (None, 384, 8, True),
        (3072, 384, 8, True),
        (1536, 384, 8, False),
        (128, 1024, 1, False),
        (1024, 1024, 1, True),
    ],
)
def test_checkpoints_need_a_hash_block_covering_the_mamba_block(
    default_pool, prefix_match_unit, mamba_block_size, pcp, enabled
):
    config = _vllm_config(
        prefix_match_unit=prefix_match_unit,
        mamba_block_size=mamba_block_size,
        prefill_context_parallel_size=pcp,
    )
    assert prefill_checkpoint_enabled(config) is enabled


def test_checkpoints_follow_the_unified_pool_override(default_pool):
    default_pool.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    assert not prefill_checkpoint_enabled(_vllm_config())


# Lockstep tests: the patched scheduler split selects each chunk, KVCacheManager
# allocates blocks with the scratch-hash shim installed, and the worker exports
# checkpoints from its append-only block table into the reserved blocks.


@pytest.fixture
def scheduler_split(monkeypatch):
    monkeypatch.setattr(
        Scheduler, "_mamba_block_aligned_split", Scheduler._mamba_block_aligned_split
    )
    _patch_vllm_mamba_split_scheduler_block_size()
    return Scheduler._mamba_block_aligned_split


@pytest.fixture
def shimmed(monkeypatch):
    pin_upstream_mamba_manager(monkeypatch)
    mamba_manager_patches.patch_mamba_checkpoint_hashes()
    yield


def _geometry(drop_eagle_block: bool, num_speculative_blocks: int) -> Geometry:
    return Geometry(
        block=BLOCK,
        hash_block=BLOCK,
        spec_blocks=num_speculative_blocks,
        eagle=drop_eagle_block,
        budget=BUDGET,
        retention=0,
    )


def _token_ids(num_tokens: int) -> list[int]:
    return [i % 30000 + 1 for i in range(num_tokens)]


def _expected_reservations(geometry: Geometry, worker_positions: list) -> list:
    """The manager reserves every worker position, or under vLLM #59175 and
    retention 0 only the prefill-end chunk's."""
    if geometry.reserves_every_chunk():
        return worker_positions
    return [None] * (len(worker_positions) - 1) + worker_positions[-1:]


def _reservations(lockstep: Lockstep, request) -> tuple[list, list]:
    """Prefills request and returns its chunk lengths and reserved positions."""
    chunks, reserved = [], []
    lockstep.admit(request)
    mamba = lockstep.mamba
    while request.request_id in lockstep._admitted or (
        request.num_computed_tokens < request.num_tokens
    ):
        ((_, start, end),) = lockstep.step([request])
        chunks.append(end - start)
        checkpoint = mamba._checkpoints.get(request.request_id)
        reserved.append(None if checkpoint is None else checkpoint[0])
    return chunks, reserved


@pytest.mark.parametrize(
    (
        "num_tokens",
        "drop_eagle_block",
        "num_speculative_blocks",
        "chunks",
        "worker_positions",
    ),
    [
        (8191, True, 1, [8191], [3072]),
        (8191, False, 0, [8191], [6144]),
        (19456, False, 0, [15360, 4096], [12288, 18432]),
        (19456, True, 1, [15360, 4096], [9216, None]),
        # The second chunk's checkpoint takes the first chunk's speculative
        # block, which the worker table also holds one column lower.
        (22000, True, 1, [15360, 6640], [9216, 18432]),
        # The checkpoint keeps the first chunk's second speculative block in
        # place.
        (22000, True, 2, [15360, 6640], [9216, 18432]),
    ],
)
def test_worker_offsets_match_manager_reservations(
    scheduler_split,
    shimmed,
    num_tokens,
    drop_eagle_block,
    num_speculative_blocks,
    chunks,
    worker_positions,
):
    geometry = _geometry(drop_eagle_block, num_speculative_blocks)
    lockstep = Lockstep(geometry, scheduler_split)
    request = make_request("r", _token_ids(num_tokens), BLOCK)
    scheduled, reserved = _reservations(lockstep, request)
    assert scheduled == chunks
    ends = np.cumsum(chunks).tolist()
    starts = [0, *ends[:-1]]
    offsets = _offsets(ends, chunks, drop_eagle_block=drop_eagle_block).tolist()
    assert [s + o if o else None for s, o in zip(starts, offsets)] == (worker_positions)
    assert reserved == _expected_reservations(geometry, worker_positions)


@pytest.mark.parametrize(
    ("drop_eagle_block", "num_speculative_blocks", "hit_tokens", "checkpoint"),
    [(False, 0, 21504, 27648), (True, 1, 18432, 24576), (True, 2, 18432, 24576)],
)
def test_worker_table_matches_manager_after_a_prefix_hit(
    scheduler_split,
    shimmed,
    drop_eagle_block,
    num_speculative_blocks,
    hit_tokens,
    checkpoint,
):
    lockstep = Lockstep(
        _geometry(drop_eagle_block, num_speculative_blocks), scheduler_split
    )
    first = make_request("r", _token_ids(22000), BLOCK)
    lockstep.prefill(first)
    lockstep.free(first)
    # A longer prompt with the same prefix resumes from the cached checkpoint
    # and reserves one of its own in the one remaining chunk.
    resend = make_request("resend", _token_ids(30000), BLOCK)
    assert lockstep.admit(resend) == hit_tokens
    ((_, start, end),) = lockstep.step([resend])
    assert (start, end) == (hit_tokens, 30000)
    assert lockstep.mamba._checkpoints["resend"][0] == checkpoint


@pytest.mark.parametrize("num_speculative_blocks", [1, 2])
def test_worker_skips_chunks_the_manager_did_not_reserve(
    scheduler_split, shimmed, num_speculative_blocks
):
    """Under vLLM #59175 with retention=0, MambaManager only reserves a
    checkpoint on the final prefill chunk. From the second chunk on, the
    checkpoint column in the worker's append-only table still holds the prior
    chunk's relocated speculative block, which the worker must ignore."""
    geometry = Geometry(
        block=BLOCK,
        hash_block=BLOCK,
        spec_blocks=num_speculative_blocks,
        eagle=False,
        budget=8192,
        retention=0,
    )
    lockstep = Lockstep(geometry, scheduler_split)
    chunks = lockstep.prefill(make_request("r", _token_ids(26000), BLOCK))
    assert chunks == [(0, 6144), (6144, 12288), (12288, 18432), (18432, 26000)]
    moved = sum(
        count for kind, count in lockstep.stats.items() if "own block, moved" in kind
    )
    assert moved == (0 if geometry.reserves_every_chunk() else 2)


GEOMETRIES = [
    Geometry(block, hash_block, spec_blocks, eagle, budget, retention)
    for block, hash_block in [(3072, 3072), (1024, 128), (2048, 1024)]
    for spec_blocks in (0, 1, 2)
    for eagle in (True, False)
    for budget in (16384, 8192, 4096)
    for retention in (0, None)
]


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=str)
def test_worker_and_manager_agree_in_random_prefills(
    scheduler_split, shimmed, geometry
):
    """Fuzzes prompts, two copies at once, decode steps and prefix-hit
    resends (same prompt, a prefix, an extension, another prompt). Every step
    the worker writes only blocks the request owns, every checkpoint into the
    block the manager reserved, and every Mamba prefix-cache entry and
    boundary-state offload offer names a block holding its key's state."""
    for seed in range(2):
        run_scenario(Lockstep(geometry, scheduler_split), seed)

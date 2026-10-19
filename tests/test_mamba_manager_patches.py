# SPDX-License-Identifier: Apache-2.0
"""Tests for Mamba prefill checkpoint hash cleanup (vLLM #59759).

Verifies that unwritten speculative blocks from prior chunks are not hashed or
offered for offload when a prefill checkpoint is reserved. With eagle block
drop one of them would take the checkpoint's own key.
"""

import hashlib
import inspect

import pytest
import torch
from mamba_manager_test_utils import pin_upstream_mamba_manager
from packaging.version import Version
from vllm.distributed.kv_events import BlockRemoved
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import mamba_manager_patches

# The logical PP4xPCP8 block: 384 rank-local tokens times 8 PCP ranks.
BLOCK = 3072
MAMBA_GROUP = 1


@pytest.fixture
def upstream(monkeypatch):
    if Version("0.31.1rc0") <= mamba_manager_patches._VLLM_VERSION:
        pytest.skip("The installed vLLM relocates scratch blocks.")
    pin_upstream_mamba_manager(monkeypatch)
    yield


@pytest.fixture
def shimmed(monkeypatch):
    """Installs the checkpoint hash shim over the pinned vLLM manager."""
    if Version("0.31.1rc0") <= mamba_manager_patches._VLLM_VERSION:
        pytest.skip("The installed vLLM relocates scratch blocks.")
    pin_upstream_mamba_manager(monkeypatch)
    mamba_manager_patches.patch_mamba_checkpoint_hashes()
    installed = getattr(MambaManager.cache_blocks, mamba_manager_patches._MARKER, False)
    assert installed
    yield


def _kv_cache_manager(num_speculative_blocks: int = 1) -> KVCacheManager:
    init_none_hash(sha256)
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=((1,), (1,)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=64,
    )
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=65536,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
        # The scheduler passes use_eagle_block_drop, True under MTP.
        use_eagle=True,
        enable_kv_cache_events=True,
    )


def _request(request_id: str, num_tokens: int) -> Request:
    return Request(
        request_id,
        [i % 30000 + 1 for i in range(num_tokens)],
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _schedule(manager: KVCacheManager, request: Request, end: int) -> None:
    num_new_tokens = end - request.num_computed_tokens
    assert manager.allocate_slots(request, num_new_tokens) is not None


def _run_chunks(manager: KVCacheManager, request: Request, chunk_ends: list[int]):
    for end in chunk_ends:
        _schedule(manager, request, end)
        request.num_computed_tokens = end
        manager.new_step_starts()


def _prefill(manager: KVCacheManager, request: Request, chunk_ends: list[int]):
    """Schedules the given chunks; returns the last checkpoint and the table."""
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP]
    _run_chunks(manager, request, chunk_ends)
    position, column = mamba._checkpoints[request.request_id]
    blocks = mamba.req_to_blocks[request.request_id]
    return position, column, blocks


def _resend_hit(manager: KVCacheManager, num_tokens: int):
    blocks, hit_tokens, _ = manager.get_computed_blocks(_request("resend", num_tokens))
    mamba_hit = [b for b in blocks.blocks[MAMBA_GROUP] if not b.is_null]
    return hit_tokens, mamba_hit


def _cached(manager: KVCacheManager, request: Request, position: int):
    return manager.block_pool.get_cached_block(
        request.block_hashes[position // BLOCK - 1], [MAMBA_GROUP]
    )


# Eagle drop, one speculative block, retention 0: the final chunk checkpoints
# at the column after the first chunk's speculative block, which the
# full-block pass hashes under the checkpoint's key.
STALE_SPEC_CASES = [
    (19000, [12288, 19000], 15360, 5),
    (22000, [15360, 22000], 18432, 6),
]


@pytest.mark.parametrize(
    ("num_tokens", "chunk_ends", "position", "column"), STALE_SPEC_CASES
)
def test_upstream_serves_an_unwritten_spec_block_for_the_checkpoint(
    upstream, num_tokens, chunk_ends, position, column
):
    manager = _kv_cache_manager()
    request = _request("r", num_tokens)
    got_position, got_column, blocks = _prefill(manager, request, chunk_ends)
    assert (got_position, got_column) == (position, column)
    assert _cached(manager, request, position) != [blocks[column]]
    manager.free(request)
    hit_tokens, mamba_hit = _resend_hit(manager, num_tokens)
    assert hit_tokens == position
    assert mamba_hit != [blocks[column]]


@pytest.mark.parametrize(
    ("num_tokens", "chunk_ends", "position", "column"), STALE_SPEC_CASES
)
def test_shim_keys_the_checkpoint_hash_to_the_checkpoint_block(
    shimmed, num_tokens, chunk_ends, position, column
):
    manager = _kv_cache_manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP]
    request = _request("r", num_tokens)
    _run_chunks(manager, request, chunk_ends[:1])
    mamba.take_pending_boundary_state_offloads()
    manager.block_pool.take_events()
    stale = mamba.req_to_blocks["r"][column - 1]
    _schedule(manager, request, chunk_ends[1])

    position_and_column = mamba._checkpoints["r"]
    blocks = mamba.req_to_blocks["r"]
    checkpoint = blocks[column]
    key = make_block_hash_with_group_id(
        request.block_hashes[position // BLOCK - 1], MAMBA_GROUP
    )
    assert position_and_column == (position, column)
    # The state block stays at [-(S + 1)] with the checkpoint right before it.
    assert len(blocks) - 2 == column + 1
    assert _cached(manager, request, position) == [checkpoint]
    # Same-step hits on the key stay deferred: the checkpoint block holds it.
    assert key in mamba.cached_blocks_this_step
    offers = mamba.take_pending_boundary_state_offloads()
    assert ("r", MAMBA_GROUP, checkpoint, position) in offers
    removed_hashes = {
        block_hash
        for event in manager.block_pool.take_events()
        if isinstance(event, BlockRemoved) and event.group_idx == MAMBA_GROUP
        for block_hash in event.block_hashes
    }
    assert blocks[column - 1] is stale
    assert stale.block_hash is None
    assert all(entry[2] is not stale for entry in offers)
    assert maybe_convert_block_hash(get_block_hash(key)) not in removed_hashes

    request.num_computed_tokens = chunk_ends[1]
    manager.new_step_starts()
    manager.free(request)
    hit_tokens, mamba_hit = _resend_hit(manager, num_tokens)
    assert hit_tokens == position
    assert mamba_hit == [checkpoint]


def test_shim_leaves_another_requests_entry_under_the_key(shimmed):
    # Two copies admitted in one step, so neither hits the other's blocks.
    num_tokens, chunk_ends, position, column = STALE_SPEC_CASES[1]
    manager = _kv_cache_manager()
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP]
    first, twin = _request("first", num_tokens), _request("twin", num_tokens)
    for end in chunk_ends:
        for request in (first, twin):
            _schedule(manager, request, end)
            request.num_computed_tokens = end
        manager.new_step_starts()
    checkpoints = [mamba.req_to_blocks[r.request_id][column] for r in (first, twin)]
    key = make_block_hash_with_group_id(
        first.block_hashes[position // BLOCK - 1], MAMBA_GROUP
    )
    entry = manager.block_pool.cached_block_hash_to_block._cache[key]
    assert sorted(entry) == sorted(block.block_id for block in checkpoints)
    assert _cached(manager, first, position) == [checkpoints[0]]


@pytest.mark.parametrize("shim", [False, True])
@pytest.mark.parametrize(
    ("num_tokens", "chunk_ends", "position", "column"),
    [
        # One chunk: no scratch block yet.
        (8191, [8191], 3072, 1),
        # The first chunk's speculative block is below the natural column of
        # the checkpoint, and retention 0 does not hash its column.
        (25000, [15360, 25000], 21504, 7),
    ],
)
def test_checkpoint_hash_is_correct_either_way(
    upstream, shim, num_tokens, chunk_ends, position, column
):
    if shim:
        mamba_manager_patches.patch_mamba_checkpoint_hashes()
    manager = _kv_cache_manager()
    request = _request("r", num_tokens)
    got_position, got_column, blocks = _prefill(manager, request, chunk_ends)
    assert (got_position, got_column) == (position, column)
    assert _cached(manager, request, position) == [blocks[column]]


# Port of upstream vLLM 6fa67afed9 (#59759)
# tests/v1/core/test_single_type_kv_cache_manager.py.
@pytest.mark.parametrize("num_speculative_blocks", [1, 2, 3])
@pytest.mark.parametrize("first_chunk_blocks", [2, 4])
@pytest.mark.parametrize("retention_interval", [0, None])
def test_mamba_checkpoint_hash_maps_only_to_checkpoint_block(
    shimmed, num_speculative_blocks, first_chunk_blocks, retention_interval
):
    """A checkpoint step must not leave a never-written speculative block
    cached."""
    block_size = 128
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=16,
    )
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=128)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    # MTP/EAGLE moves the checkpoint one block below the last full block.
    manager.drop_eagle_checkpoint_block = True
    init_none_hash(sha256)
    prompt_tokens = 6 * block_size + 50
    request = Request(
        request_id="r",
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )

    def step(start, end):
        estimate = manager.get_num_blocks_to_allocate("r", end, [], start, start, end)
        free_before = pool.get_num_free_blocks()
        manager.allocate_new_blocks("r", end, end)
        assert free_before - pool.get_num_free_blocks() == estimate
        manager.cache_blocks(
            request,
            end,
            retention_interval=retention_interval,
            replay_boundaries=[5 * block_size],
        )
        request.num_computed_tokens = end

    step(0, first_chunk_blocks * block_size)
    speculative_blocks = manager.req_to_blocks["r"][-num_speculative_blocks:]
    manager.new_step_starts()
    step(first_chunk_blocks * block_size, prompt_tokens)

    checkpoint_position, checkpoint_idx = manager._checkpoints["r"]
    checkpoint_block = manager.req_to_blocks["r"][checkpoint_idx]
    assert all(
        b.block_hash is None or b is checkpoint_block for b in speculative_blocks
    )
    block_hash = request.block_hashes[checkpoint_position // block_size - 1]
    assert pool.get_cached_block(block_hash, [0]) == [checkpoint_block]
    offered = {entry[2].block_id for entry in manager._pending_boundary_state_offloads}
    assert not offered & {
        b.block_id for b in speculative_blocks if b is not checkpoint_block
    }


def test_shim_installs_once(upstream):
    original = MambaManager.cache_blocks
    mamba_manager_patches.patch_mamba_checkpoint_hashes()
    installed = MambaManager.cache_blocks
    assert installed is not original
    assert installed.__wrapped__ is original
    mamba_manager_patches.patch_mamba_checkpoint_hashes()
    assert MambaManager.cache_blocks is installed


@pytest.mark.parametrize(
    ("method", "digest"),
    [
        (
            "get_num_blocks_to_allocate",
            "17013b3340343b8a50f0c0f23fffe647a419162cd28f5ea620202f7b0cc832d5",
        ),
        (
            "allocate_new_blocks",
            "4d1a5f2b49b9cd558ae97e5697d7b21a4cd7989e0fb8dfe66629bd24a13a3319",
        ),
        (
            "cache_blocks",
            "626bd2566400890033ecd9c5bf939af51f0ed991b1209cab6f9d708dccfc6134",
        ),
    ],
)
def test_scratch_hash_shim_matches_pinned_manager(method, digest):
    assert Version("0.31.1rc0") > mamba_manager_patches._VLLM_VERSION, (
        "Remove mamba_manager_patches and its registry entry: the vLLM pin "
        "includes #59759. Keep the checkpoint lockstep tests."
    )
    source = inspect.getsource(inspect.unwrap(getattr(MambaManager, method)))
    assert hashlib.sha256(source.encode()).hexdigest() == digest


@pytest.mark.parametrize("version", ["0.31.1rc0", "0.31.1", "0.32.0"])
def test_upstream_fix_skips_the_shim(monkeypatch, version):
    monkeypatch.setattr(mamba_manager_patches, "_VLLM_VERSION", Version(version))
    original = inspect.unwrap(MambaManager.cache_blocks)
    monkeypatch.setattr(MambaManager, "cache_blocks", original)
    mamba_manager_patches.patch_mamba_checkpoint_hashes()
    assert MambaManager.cache_blocks is original

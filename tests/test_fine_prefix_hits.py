# SPDX-License-Identifier: Apache-2.0
"""TPU_FINE_PREFIX_HIT_TOKENS on vLLM's KVCacheManager with DeepSeek-V4-shaped
groups: two 128-token sliding-window groups, one 1024-token full-attention
group, and 16/32-token state groups with 8/128-token windows."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import checkpoint_free_order as cfo
from vllm_torchtpu.core import fine_prefix_hits as fph
from vllm_torchtpu.core import hca_state_checkpoint as hsc

FULL = 1024
FULL_GROUP = 2
UNIT = 128


def _swa(block_size, window, head_size):
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        sliding_window=window,
    )


def _manager(num_blocks, tokens_per_state=128, drop_hca=False):
    # TPU_KV_DROP_HCA_STATE_CHECKPOINT declares the 128-ratio compressor state
    # (st32) not prefix-cacheable.
    hca_cls = hsc.state_spec_cls(128, 128, FULL, drop_hca)
    groups = [
        KVCacheGroupSpec(["swa0"], _swa(128, 128, 512)),
        KVCacheGroupSpec(["swa1"], _swa(128, 128, 512)),
        KVCacheGroupSpec(
            ["mla"],
            MLAAttentionSpec(
                block_size=FULL,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.uint8,
                tokens_per_state=tokens_per_state,
            ),
        ),
        KVCacheGroupSpec(["st16"], _swa(16, 8, 4096)),
        KVCacheGroupSpec(
            ["st32"],
            hca_cls(
                block_size=32,
                num_kv_heads=1,
                head_size=2048,
                dtype=torch.uint8,
                sliding_window=128,
            ),
        ),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=32768,
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=FULL,
        hash_block_size=16,
        enable_caching=True,
        use_eagle=False,
    )


init_none_hash(sha256)
_HASHER = get_request_block_hasher(16, sha256)
_ids = iter(range(1, 1 << 30))


def _request(tokens, max_tokens):
    return Request(
        f"r{next(_ids)}",
        list(tokens),
        SamplingParams(max_tokens=max_tokens),
        None,
        block_hasher=_HASHER,
    )


def _prefill(mgr, req, step=128):
    blocks, hit, _ = mgr.get_computed_blocks(req)
    first = True
    while req.num_computed_tokens < req.num_prompt_tokens:
        local = hit if first else 0
        n = min(step, req.num_prompt_tokens - req.num_computed_tokens - local)
        out = mgr.allocate_slots(
            req,
            n,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks if first else None,
        )
        assert out is not None
        req.num_computed_tokens += n + local
        first = False
    return hit


def _decode(mgr, req, steps, accept=4):
    for _ in range(steps):
        for _ in range(accept):
            req.append_output_token_ids(7_000_000 + len(req.output_token_ids))
        assert mgr.allocate_slots(req, accept, num_lookahead_tokens=6) is not None
        req.num_computed_tokens += accept


def _run(mgr, tokens, decode_steps=50):
    req = _request(tokens, decode_steps * 4)
    hit = _prefill(mgr, req)
    _decode(mgr, req, decode_steps)
    mgr.free(req)
    return hit


@pytest.fixture(params=[False, True], ids=["block-hits", "fine-hits"])
def fine(request, monkeypatch):
    if request.param:
        fph.install(UNIT, monkeypatch.setattr)
    return request.param


def _hold(monkeypatch):
    monkeypatch.setattr(KVCacheManager, "free", cfo._free)
    monkeypatch.setattr(KVCacheManager, "pop_blocks_for_free", cfo._pop_blocks_for_free)
    monkeypatch.setattr(
        SlidingWindowManager,
        "remove_skipped_blocks",
        cfo._remove_skipped_blocks_holding_checkpoint,
    )


@pytest.fixture
def hold(monkeypatch):
    _hold(monkeypatch)


# 40,333-token first turn: replay boundary 40,320 = 39 blocks + 384 tokens.
TURN1 = list(range(1, 40_334))
# The next turn shares all but the last two prompt tokens (the generation
# prompt), as a chat template renders it.
TURN2 = TURN1[:-2] + list(range(50_000_000, 50_003_000))


@pytest.mark.parametrize("decode_steps", [10, 300], ids=["tail-open", "tail-full"])
def test_next_turn_hits_inside_the_block(fine, decode_steps):
    """tail-full: decode fills the prompt-tail block before the next turn."""
    mgr = _manager(400)
    _run(mgr, TURN1, decode_steps)
    blocks, hit, _ = mgr.get_computed_blocks(_request(TURN2, 10))
    assert hit == (fph.fine_boundary(len(TURN1), UNIT) if fine else 39 * FULL)
    assert hit == 40_320 or not fine
    for group, manager in enumerate(mgr.coordinator.single_type_managers):
        assert len(blocks.blocks[group]) == -(-hit // manager.block_size)


def test_fine_hit_copies_only_the_full_attention_tail(fine):
    mgr = _manager(400)
    _run(mgr, TURN1)
    req = _request(TURN2, 10)
    blocks, hit, _ = mgr.get_computed_blocks(req)
    src = blocks.blocks[FULL_GROUP][-1].block_id
    assert mgr.allocate_slots(
        req, 128, num_new_computed_tokens=hit, new_computed_blocks=blocks
    )
    copies, _ = mgr.take_kv_cache_block_copies()
    if not fine:
        assert copies == []
        return
    assert len(copies) == 1
    assert copies[0].src_block_id == src
    own = mgr.coordinator.get_blocks(req.request_id)[FULL_GROUP]
    assert own[hit // FULL].block_id == copies[0].dst_block_id != src
    # The windowed groups hit on their own block boundaries: nothing to copy.
    for group, manager in enumerate(mgr.coordinator.single_type_managers):
        if group != FULL_GROUP:
            assert hit % manager.block_size == 0


def test_hit_after_churn_with_held_checkpoint(fine, hold):
    mgr = _manager(130)
    _run(mgr, TURN1)
    req = _request(TURN2, 1300)
    hit2 = _prefill(mgr, req)
    assert hit2 == (40_320 if fine else 39 * FULL)
    for step in range(300):
        _decode(mgr, req, 1)
        if step % 10 == 0:
            base = 10_000_000 + step * 100_000
            _run(mgr, range(base, base + 3000), decode_steps=5)
    mgr.free(req)
    turn3 = TURN2[:-2] + list(range(60_000_000, 60_002_000))
    _, hit, _ = mgr.get_computed_blocks(_request(turn3, 10))
    assert hit == (fph.fine_boundary(len(TURN2), UNIT) if fine else 42 * FULL)


def test_block_aligned_prompt_resumes_at_its_last_unit(fine):
    """A 20-block prompt: the replay boundary is 19 blocks, or 20 blocks less
    one unit with fine hits."""
    mgr = _manager(400)
    turn1 = list(range(1, 20_481))
    _run(mgr, turn1)
    _, hit, _ = mgr.get_computed_blocks(_request(turn1 + [9] * 900, 4))
    assert hit == (20 * FULL - UNIT if fine else 19 * FULL)


def test_divergence_inside_a_block_falls_back_to_a_checkpoint(fine):
    """Nothing is registered inside a block another prompt merely passed
    through; the hit is the retention checkpoint either way."""
    mgr = _manager(400)
    _run(mgr, TURN1)
    _, hit, _ = mgr.get_computed_blocks(_request(TURN1[:35_000] + [5] * 900, 4))
    assert hit == 32 * FULL


@pytest.mark.parametrize("held", [False, True], ids=["free", "hold"])
def test_divergence_between_the_two_checkpoints_keeps_the_block_hit(
    fine, held, monkeypatch
):
    """A prompt that leaves turn 1 between 39,936 and the fine boundary 40,320
    (a truncated history) still resumes at the block-aligned checkpoint."""
    if held:
        _hold(monkeypatch)
    mgr = _manager(400)
    _run(mgr, TURN1, decode_steps=300)
    _, hit, _ = mgr.get_computed_blocks(_request(TURN1[:40_100] + [3] * 900, 4))
    assert hit == 39 * FULL


@pytest.mark.parametrize("unit", [64, 96, 1024])
def test_rejects_units_the_groups_cannot_hit_at(unit, monkeypatch):
    fph.install(unit, monkeypatch.setattr)
    with pytest.raises(ValueError, match="TPU_FINE_PREFIX_HIT_TOKENS"):
        _manager(64)


def test_rejects_unit_below_compression_ratio(monkeypatch):
    fph.install(UNIT, monkeypatch.setattr)
    with pytest.raises(ValueError, match="compression ratio 256"):
        _manager(64, tokens_per_state=256)


@pytest.mark.parametrize("decode_steps", [10, 300], ids=["tail-open", "tail-full"])
def test_fine_hits_with_dropped_hca_state_checkpoint(fine, decode_steps):
    """TPU_KV_DROP_HCA_STATE_CHECKPOINT (served on rc7) leaves the 128-ratio
    state group out of the lookup; fine hits are whole 128-token runs, so the
    hit is unchanged and that group still caches nothing."""
    mgr = _manager(400, drop_hca=True)
    _run(mgr, TURN1, decode_steps)
    req = _request(TURN2, 10)
    blocks, hit, _ = mgr.get_computed_blocks(req)
    assert hit == (40_320 if fine else 39 * FULL)
    assert blocks.blocks[4] == []
    assert mgr.allocate_slots(
        req, 128, num_new_computed_tokens=hit, new_computed_blocks=blocks
    )
    copies, _ = mgr.take_kv_cache_block_copies()
    assert len(copies) == (1 if fine else 0)
    assert not any(
        blk.block_hash is not None
        for blk in mgr.coordinator.single_type_managers[4].req_to_blocks[
            req.request_id
        ]
    )

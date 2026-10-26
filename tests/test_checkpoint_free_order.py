# SPDX-License-Identifier: Apache-2.0
"""TPU_KV_HOLD_REPLAY_CHECKPOINT on vLLM's KVCacheManager with DeepSeek-V4-shaped
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

FULL = 1024
FULL_GROUP = 2


def _swa(block_size, window, head_size):
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        sliding_window=window,
    )


def _manager(num_blocks):
    groups = [
        KVCacheGroupSpec(["swa0"], _swa(128, 128, 512)),
        KVCacheGroupSpec(["swa1"], _swa(128, 128, 512)),
        KVCacheGroupSpec(
            ["mla"],
            MLAAttentionSpec(
                block_size=FULL, num_kv_heads=1, head_size=64, dtype=torch.uint8
            ),
        ),
        KVCacheGroupSpec(["st16"], _swa(16, 8, 4096)),
        KVCacheGroupSpec(["st32"], _swa(32, 128, 2048)),
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


@pytest.fixture(params=[False, True], ids=["dense-free", "hold"])
def hold(request, monkeypatch):
    if request.param:
        monkeypatch.setattr(KVCacheManager, "free", cfo._free)
        monkeypatch.setattr(
            KVCacheManager, "pop_blocks_for_free", cfo._pop_blocks_for_free
        )
        monkeypatch.setattr(
            SlidingWindowManager,
            "remove_skipped_blocks",
            cfo._remove_skipped_blocks_holding_checkpoint,
        )
    return request.param


def test_next_turn_survives_churn_during_a_long_decode(hold):
    mgr = _manager(130)
    turn1 = list(range(1, 40_001))
    _run(mgr, turn1)
    turn2 = turn1 + list(range(50_000_000, 50_003_000))
    req = _request(turn2, 1300)
    assert _prefill(mgr, req) == 39 * FULL
    # Other requests churn the pool while turn 2 decodes.
    for step in range(300):
        _decode(mgr, req, 1)
        if step % 10 == 0:
            base = 10_000_000 + step * 100_000
            _run(mgr, range(base, base + 3000), decode_steps=5)
    mgr.free(req)
    turn3 = turn2 + list(range(60_000_000, 60_002_000))
    _, hit, _ = mgr.get_computed_blocks(_request(turn3, 10))
    # Turn 2's whole prompt is still cached; only the checkpoint decides.
    assert hit == (41 * FULL if hold else 0)


def test_hold_keeps_only_the_replay_checkpoint(hold):
    if not hold:
        pytest.skip("hold-only invariant")
    mgr = _manager(400)
    req = _request(range(1, 10_001), 400)  # replay boundary 9216
    _prefill(mgr, req)
    _decode(mgr, req, 100)
    blocks = mgr.coordinator.get_blocks(req.request_id)
    for group, manager in enumerate(mgr.coordinator.single_type_managers):
        if not isinstance(manager, SlidingWindowManager):
            continue
        hold_range = cfo.replay_checkpoint_range(manager, req.num_prompt_tokens)
        assert len(hold_range) > 0 and hold_range.stop == 9216 // manager.block_size
        window_start = (
            req.num_computed_tokens - manager.sliding_window
        ) // manager.block_size
        for idx in range(window_start - 1):
            assert blocks[group][idx].is_null == (idx not in hold_range), (group, idx)


def test_free_order_tail_checkpoint_then_head(hold):
    if not hold:
        pytest.skip("hold-only invariant")
    mgr = _manager(400)
    req = _request(range(1, 10_001), 400)
    _prefill(mgr, req)
    _decode(mgr, req, 100)
    chain = list(mgr.coordinator.get_blocks(req.request_id)[FULL_GROUP])
    ckpt = {
        b.block_id
        for g, m in enumerate(mgr.coordinator.single_type_managers)
        if isinstance(m, SlidingWindowManager)
        for i, b in enumerate(mgr.coordinator.get_blocks(req.request_id)[g])
        if i in cfo.replay_checkpoint_range(m, req.num_prompt_tokens)
    }
    order = [b.block_id for b in cfo.chain_ordered_blocks(mgr, req)]
    split = 9216 // FULL
    tail = [b.block_id for b in chain[split:][::-1]]
    head = [b.block_id for b in chain[:split][::-1]]
    n = len(order)
    assert order[n - len(head) :] == head
    assert set(order[n - len(head) - len(ckpt) : n - len(head)]) == ckpt
    assert (
        order[n - len(head) - len(ckpt) - len(tail) : n - len(head) - len(ckpt)] == tail
    )

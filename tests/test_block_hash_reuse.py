# SPDX-License-Identifier: Apache-2.0
"""A reused block hash equals the hash upstream computes for that block."""

import random

import pytest
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_torchtpu.core import block_hash_reuse
from vllm_torchtpu.core.block_hash_reuse import (
    KeptPrompt,
    RecentPrompts,
    packed,
    reusing_block_hasher,
)

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def none_hash():
    init_none_hash(sha256)


def _request(request_id, tokens, hasher, cache_salt=None):
    return Request(
        request_id,
        tokens,
        SamplingParams(max_tokens=8),
        None,
        cache_salt=cache_salt,
        block_hasher=hasher,
    )


def _hashers(capacity=64):
    upstream = get_request_block_hasher(BLOCK_SIZE, sha256)
    return upstream, reusing_block_hasher(BLOCK_SIZE, sha256, upstream, capacity)


def _conversation_prompts(rng, conversations=12, turns=10):
    """Interleaved multi-turn prompts: shared system prompt, growing turns, some edits and resets."""
    system = [rng.randrange(1000, 150000) for _ in range(5 * BLOCK_SIZE + 3)]
    prompts = {c: list(system) for c in range(conversations)}
    order = [c for c in range(conversations) for _ in range(turns)]
    rng.shuffle(order)
    for c in order:
        prompt = prompts[c]
        roll = rng.random()
        if roll < 0.1 and len(prompt) > len(system) + BLOCK_SIZE:
            cut = rng.randrange(len(system), len(prompt))
            prompt[cut] = rng.randrange(1000, 150000)
        elif roll < 0.15:
            del prompt[rng.randrange(len(system), len(prompt) + 1) :]
        prompt.extend(rng.randrange(1000, 150000) for _ in range(rng.randrange(1, 6 * BLOCK_SIZE)))
        yield f"{c}", list(prompt)


def test_longest_shared_matches_a_block_by_block_search():
    rng = random.Random(0)
    for _ in range(500):
        recent = RecentPrompts(capacity=8)
        stem = [rng.randrange(3) for _ in range(rng.randrange(1, 6) * 4)]
        kept = []
        for _ in range(rng.randrange(1, 6)):
            tokens = stem + [rng.randrange(3) for _ in range(rng.randrange(0, 6) * 4)]
            hashes = [hash(tuple(tokens[: (b + 1) * 4])) for b in range(len(tokens) // 4)]
            recent.keep(KeptPrompt(packed(tokens, len(tokens) // 4, 4), hashes), None)
            kept.append(tokens)
        new = list(rng.choice(kept)) + [rng.randrange(3) for _ in range(rng.randrange(0, 4) * 4)]
        if rng.random() < 0.5:
            new[rng.randrange(len(new))] += 1
        new = new[: len(new) // 4 * 4]
        if not new:
            continue
        longest_shared, longest_extended = 0, 0
        for tokens in kept:
            if tokens[:4] != new[:4]:
                continue
            shared = 0
            while (shared + 1) * 4 <= min(len(tokens), len(new)) and (
                tokens[shared * 4 : (shared + 1) * 4] == new[shared * 4 : (shared + 1) * 4]
            ):
                shared += 1
            longest_shared = max(longest_shared, shared)
            if shared * 4 == len(tokens):
                longest_extended = max(longest_extended, shared)
        expected = longest_extended or longest_shared
        key, shared, extends = recent.longest_shared(
            packed(new, len(new) // 4, 4), hash(tuple(new[:4])), len(new) // 4, 16
        )
        assert shared == expected
        if key is not None:
            assert recent.hashes(key)[:shared] == [
                hash(tuple(new[: (b + 1) * 4])) for b in range(shared)
            ]
            assert extends == (shared == len(recent.hashes(key)))


def test_reused_hashes_equal_upstream_on_interleaved_conversations():
    upstream, reusing = _hashers()
    for request_id, prompt in _conversation_prompts(random.Random(1)):
        assert _request(request_id, prompt, reusing).block_hashes == (
            _request(request_id, prompt, upstream).block_hashes
        )
    assert reusing.counts.reused_blocks > 0.6 * reusing.counts.blocks


def test_freshly_decoded_ids_reuse_as_shared_ones_do():
    upstream, reusing = _hashers()
    rng = random.Random(4)
    previous = [rng.randrange(1000, 150000) for _ in range(9 * BLOCK_SIZE)]
    prompt = [int(str(token)) for token in previous] + [rng.randrange(1000, 150000) for _ in range(2 * BLOCK_SIZE)]
    _request("previous", previous, reusing)
    assert _request("r", prompt, reusing).block_hashes == _request("r", prompt, upstream).block_hashes
    assert reusing.counts.reused_blocks == 9


def test_output_blocks_after_a_reused_prompt_equal_upstream():
    upstream, reusing = _hashers()
    rng = random.Random(2)
    prompt = [rng.randrange(1000, 150000) for _ in range(7 * BLOCK_SIZE + 5)]
    _request("previous", prompt[: 4 * BLOCK_SIZE], reusing)
    outputs = [rng.randrange(1000, 150000) for _ in range(3 * BLOCK_SIZE)]
    reused, plain = _request("r", prompt, reusing), _request("r", prompt, upstream)
    for token in outputs:
        reused.append_output_token_ids(token)
        plain.append_output_token_ids(token)
    assert reusing.counts.reused_blocks == 4
    assert reused.block_hashes == plain.block_hashes
    assert len(plain.block_hashes) == 10


def test_cache_salt_keeps_prompts_apart():
    upstream, reusing = _hashers()
    prompt = list(range(1000, 1000 + 6 * BLOCK_SIZE))
    for salt in ("a", "b", None, "a"):
        assert _request(salt or "-", prompt, reusing, salt).block_hashes == (
            _request(salt or "-", prompt, upstream, salt).block_hashes
        )
    assert reusing.counts.reused_blocks == 6


def test_a_prompt_sharing_only_the_system_prompt_keeps_the_other_conversation():
    _, reusing = _hashers(capacity=2)
    rng = random.Random(3)
    system = [rng.randrange(1000, 150000) for _ in range(2 * BLOCK_SIZE)]
    first = system + [rng.randrange(1000, 150000) for _ in range(6 * BLOCK_SIZE)]
    second = system + [rng.randrange(1000, 150000) for _ in range(6 * BLOCK_SIZE)]
    _request("a1", first, reusing)
    _request("b1", second, reusing)
    _request("a2", first + [7] * BLOCK_SIZE, reusing)
    assert reusing.counts.reused_blocks == 2 + 8
    assert len(reusing.recent) == 2


def test_the_check_fails_when_tokens_are_not_compared(monkeypatch):
    def longest_by_length(tokens, candidates, blocks, block_size):
        fitting = [(len(kept.hashes), key) for key, kept in candidates if len(kept.hashes) <= blocks]
        return (max(fitting)[1], max(fitting)[0]) if fitting else (None, 0)

    monkeypatch.setattr(block_hash_reuse, "longest_extended", longest_by_length)
    upstream, reusing = _hashers()
    differing = sum(
        _request(request_id, prompt, reusing).block_hashes
        != _request(request_id, prompt, upstream).block_hashes
        for request_id, prompt in _conversation_prompts(random.Random(1))
    )
    assert differing > 0


def test_patch_installs_only_when_enabled(monkeypatch):
    from vllm.v1.engine import core

    from vllm_torchtpu import envs

    monkeypatch.setattr(kv_cache_utils, "get_request_block_hasher", get_request_block_hasher)
    monkeypatch.setattr(core, "get_request_block_hasher", get_request_block_hasher)
    monkeypatch.setenv("VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS", "0")
    block_hash_reuse.patch_block_hash_reuse()
    assert core.get_request_block_hasher is get_request_block_hasher

    monkeypatch.setenv("VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS", "8")
    assert envs.VLLM_TPU_BLOCK_HASH_REUSE_PROMPTS == 8
    block_hash_reuse.patch_block_hash_reuse()
    hasher = core.get_request_block_hasher(BLOCK_SIZE, sha256)
    assert kv_cache_utils.get_request_block_hasher is core.get_request_block_hasher
    assert hasher.recent.capacity == 8

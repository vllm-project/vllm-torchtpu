# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for Stage 1: SessionTurnTracker and 6-category token breakdown."""

import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.telemetry.token_breakdown import (
    BoundedLRU,
    SessionTurnTracker,
    TokenBreakdown,
)


def test_env_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_AGENTIC_TELEMETRY", raising=False)
    assert envs.environment_variables["VLLM_TPU_AGENTIC_TELEMETRY"]() is False

    monkeypatch.setenv("VLLM_TPU_AGENTIC_TELEMETRY", "1")
    assert envs.environment_variables["VLLM_TPU_AGENTIC_TELEMETRY"]() is True


def test_bounded_lru_eviction_and_recency():
    lru: BoundedLRU[str, int] = BoundedLRU(maxsize=2)
    assert lru.put("a", 1) is None
    assert lru.put("b", 2) is None
    assert lru.get("a") == 1  # refreshes 'a', so 'b' becomes LRU
    evicted = lru.put("c", 3)
    assert evicted == ("b", 2)
    assert lru.get("b") is None
    assert len(lru) == 2


def test_turn1_cold_start_and_cross_session_opportunity():
    tracker = SessionTurnTracker(block_size=4)
    # Session A runs Turn 1 with 8 tokens (2 blocks) + 4 output tokens
    prompt_a = [10, 11, 12, 13, 20, 21, 22, 23]
    b_a = tracker.classify_prefill(
        prompt_a, num_cached_tokens=0, session_id="sess-a", turn_id=1
    )
    assert b_a == TokenBreakdown(cache_hit=0, new_input=8)
    assert b_a.total_tokens == len(prompt_a)
    tracker.record_turn_completion("sess-a", prompt_a, [30, 31, 32, 33], turn_id=1)

    # Session B starts Turn 1 sharing the first 4-token block (system prompt)
    # when that block was evicted from HBM (num_cached_tokens=0).
    prompt_b = [10, 11, 12, 13, 90, 91, 92, 93]
    b_b = tracker.classify_prefill(
        prompt_b, num_cached_tokens=0, session_id="sess-b", turn_id=1
    )
    assert b_b == TokenBreakdown(
        cache_hit=0,
        cross_session_opportunity=4,
        new_input=4,
    )
    assert b_b.total_tokens == len(prompt_b)


def test_turn2_full_hit_and_capacity_eviction():
    tracker = SessionTurnTracker(block_size=4)
    t1_prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    t1_output = [9, 10, 11, 12]
    tracker.record_turn_completion("sess-1", t1_prompt, t1_output, turn_id=1)

    # Turn 2 appends 4 new user tokens -> total 16 tokens, 12 from Turn 1.
    t2_prompt = t1_prompt + t1_output + [13, 14, 15, 16]

    # Case A: All 12 prior tokens hit in HBM
    full_hit = tracker.classify_prefill(
        t2_prompt, num_cached_tokens=12, session_id="sess-1", turn_id=2
    )
    assert full_hit == TokenBreakdown(cache_hit=12, new_input=4)
    assert full_hit.total_tokens == len(t2_prompt)

    # Case B: Only 4 tokens hit in HBM; remaining 8 prior tokens were evicted
    evicted = tracker.classify_prefill(
        t2_prompt, num_cached_tokens=4, session_id="sess-1", turn_id=2
    )
    assert evicted == TokenBreakdown(
        cache_hit=4,
        affinity_honored_eviction=8,
        new_input=4,
    )
    assert evicted.total_tokens == len(t2_prompt)


def test_turn2_prefix_mutation_and_offset():
    tracker = SessionTurnTracker(block_size=4)
    t1_prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    t1_output = [9, 10, 11, 12]
    tracker.record_turn_completion("sess-mut", t1_prompt, t1_output, turn_id=1)

    # Turn 2 mutates token at index 6 (from 7 -> 999), invalidating [6..12)
    t2_mutated = [1, 2, 3, 4, 5, 6, 999, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    res = tracker.classify_prefill(
        t2_mutated, num_cached_tokens=4, session_id="sess-mut", turn_id=2
    )
    # [0..4) = cache_hit (4)
    # [4..12) = block 1 [4..8) and block 2 [8..12) invalidated by mutation at idx 6 -> prefix_mutation_loss (8)
    # [12..16) = new_input (4)
    assert res == TokenBreakdown(
        cache_hit=4,
        affinity_honored_eviction=0,
        prefix_mutation_loss=8,
        new_input=4,
        mutation_offset_token_idx=6,
    )
    assert res.total_tokens == len(t2_mutated)
    attrs = res.to_span_attributes()
    assert attrs["gen_ai.usage.mutation_offset_token_idx"] == 6
    assert attrs["gen_ai.usage.input_tokens.prefix_mutation_loss"] == 8


def test_routing_loss_and_subagent_fork_inheritance():
    tracker = SessionTurnTracker(block_size=4)
    # Cold replica receives Turn 3 of an unknown session
    routed = tracker.classify_prefill(
        [1, 2, 3, 4, 5, 6, 7, 8],
        num_cached_tokens=0,
        session_id="unknown-sess",
        turn_id=3,
    )
    assert routed == TokenBreakdown(cache_hit=0, routing_loss=8)

    # Parent session completes Turn 1
    parent_seq = [10, 20, 30, 40, 50, 60, 70, 80]
    tracker.record_turn_completion("parent-sess", parent_seq, [], turn_id=1)

    # Child sub-agent forks from parent-sess on Turn 1, missing 4 tokens due to eviction
    child_prompt = parent_seq + [90, 91, 92, 93]
    fork_res = tracker.classify_prefill(
        child_prompt,
        num_cached_tokens=4,
        session_id="child-sess",
        turn_id=1,
        parent_session_id="parent-sess",
    )
    assert fork_res == TokenBreakdown(
        cache_hit=4,
        affinity_honored_eviction=4,
        new_input=4,
        parent_session_id="parent-sess",
    )
    assert fork_res.to_span_attributes()["gen_ai.session.parent_id"] == "parent-sess"


def test_tier1_to_tier2_demotion_preserves_attribution():
    tracker = SessionTurnTracker(block_size=4, tier1_maxsize=1, tier2_maxsize=10)
    # Record sess-1, then sess-2 to force sess-1 into Tier 2
    tracker.record_turn_completion(
        "sess-1", [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], turn_id=1
    )
    tracker.record_turn_completion("sess-2", [100, 101, 102, 103], [], turn_id=1)
    assert tracker.tier1.peek("sess-1") is None
    assert tracker.tier2.peek("sess-1") is not None

    # Classify sess-1 Turn 2 from Tier 2 block hashes (with mutation in block 1)
    t2_prompt = [1, 2, 3, 4, 5, 999, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    res = tracker.classify_prefill(
        t2_prompt, num_cached_tokens=4, session_id="sess-1", turn_id=2
    )
    assert res == TokenBreakdown(
        cache_hit=4,
        affinity_honored_eviction=0,
        prefix_mutation_loss=8,
        new_input=4,
        mutation_offset_token_idx=4,
    )
    assert res.total_tokens == len(t2_prompt)


def test_unreused_reply_loss_and_thinking_suffix_boundary():
    # 1. With 4-token empty thinking suffix and 64-token cache blocks:
    # Turn 1 has 153 user tokens + 4-token empty thinking suffix (prompt=157) + 24 output tokens (total=181).
    # Blocks 0..1 [0..128) are full; Block 2 starts at 128 and shares 25 user tokens [128..153) with <think> [153..157) + reply [157..181).
    tracker = SessionTurnTracker(block_size=64)
    t1_prompt = list(range(153)) + [151667, 271, 151668, 271]
    t1_output = list(range(500, 524))
    tracker.record_turn_completion("sess-think", t1_prompt, t1_output, turn_id=1)

    t2_prompt = list(range(153)) + list(range(800, 828)) + list(range(900, 995))
    res = tracker.classify_prefill(
        t2_prompt, num_cached_tokens=128, session_id="sess-think", turn_id=2
    )
    # 25 tokens in [128..153) + 4 <think> tokens + 24 reply tokens = 53 unreused_reply_loss
    assert res == TokenBreakdown(
        cache_hit=128,
        unreused_reply_loss=53,
        prefix_mutation_loss=0,
        new_input=95,
        mutation_offset_token_idx=None,
    )
    assert res.total_tokens == len(t2_prompt)
    assert "gen_ai.usage.mutation_offset_token_idx" not in res.to_span_attributes()

    # 2. Fallback when previous prompt does NOT end with empty think block (thinking on / other models):
    # Reply boundary falls back to end of previous prompt (len(t1_no_think)=150).
    t1_no_think = list(range(150))
    tracker.record_turn_completion("sess-no-think", t1_no_think, t1_output, turn_id=1)
    t2_no_think = list(range(150)) + list(range(800, 824)) + list(range(900, 950))
    res_no_think = tracker.classify_prefill(
        t2_no_think, num_cached_tokens=128, session_id="sess-no-think", turn_id=2
    )
    # 22 prompt tokens [128..150) sharing block 2 + 24 reply tokens [150..174) = 46 unreused_reply_loss
    assert res_no_think == TokenBreakdown(
        cache_hit=128,
        unreused_reply_loss=46,
        prefix_mutation_loss=0,
        new_input=50,
        mutation_offset_token_idx=None,
    )
    assert res_no_think.total_tokens == len(t2_no_think)


def test_negative_category_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        TokenBreakdown(cache_hit=-1)

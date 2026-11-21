# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Unit tests for SessionTurnTracker and 7-category prefill token breakdown."""

import pytest

from vllm_torchtpu.telemetry.token_breakdown import (
    BoundedLRU,
    ChatTurnMarkers,
    SessionTurnTracker,
    TokenBreakdown,
    resolve_turn_markers_from_tokenizer,
)

_CHATML_MARKERS = ChatTurnMarkers(
    im_start_token_id=151644,
    im_end_token_id=151645,
    assistant_role_token_id=77091,
    think_start_token_id=151667,
    think_end_token_id=151668,
)


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
    # [4..12) = block 1 [4..8) and block 2 [8..12) invalidated by mutation at idx 6 ->
    # prefix_mutation_loss (8)
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
    tracker = SessionTurnTracker(block_size=4, turn_markers=_CHATML_MARKERS)
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

    # Multi-turn server hop with ChatML <|im_end|> (151645) tokens:
    # [sys .. 151645, user_1 .. 151645, assistant_1 .. 151645, user_2_tool_output ..
    # 151645]
    # Second-to-last <|im_end|> closes assistant_1 at index 12 (so [0..12) is prior
    # history = 12 tokens),
    # and [12..20) is new tool output (8 tokens).
    hop_prompt = [
        10,
        11,
        12,
        151645,
        20,
        21,
        22,
        151645,
        30,
        31,
        32,
        151645,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        151645,
    ]
    hop_res = tracker.classify_prefill(
        hop_prompt,
        num_cached_tokens=4,
        session_id="hopped-sess",
        turn_id=2,
    )
    assert hop_res == TokenBreakdown(
        cache_hit=4,
        routing_loss=8,
        new_input=8,
    )
    assert hop_res.total_tokens == len(hop_prompt)

    # Server hop with multiple trailing user messages after the last assistant turn:
    # [sys .. 151645, user_1 .. 151645, <|im_start|>assistant (151644, 77091) .. 151645,
    #  user_2_tool_output .. 151645, user_2_extra_msg .. 151645]
    # Last completed assistant reply ends at index 14 -> [0..14) is prior history (4
    # cache_hit + 10 routing_loss),
    # and both trailing user messages [14..26) are new_input (12 tokens).
    hop_multi_user_prompt = [
        10,
        11,
        12,
        151645,
        20,
        21,
        22,
        151645,
        151644,
        77091,
        30,
        31,
        32,
        151645,
        40,
        41,
        42,
        43,
        44,
        151645,
        50,
        51,
        52,
        53,
        54,
        151645,
    ]
    hop_multi_res = tracker.classify_prefill(
        hop_multi_user_prompt,
        num_cached_tokens=4,
        session_id="hopped-multi-user-sess",
        turn_id=2,
    )
    assert hop_multi_res == TokenBreakdown(
        cache_hit=4,
        routing_loss=10,
        new_input=12,
    )
    assert hop_multi_res.total_tokens == len(hop_multi_user_prompt)


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
    # Turn 1 has 153 user tokens + 4-token empty thinking suffix (prompt=157) + 24
    # output tokens (total=181).
    # Blocks 0..1 [0..128) are full; Block 2 starts at 128 and shares 25 user tokens
    # [128..153) with <think> [153..157) + reply [157..181).
    tracker = SessionTurnTracker(block_size=64, turn_markers=_CHATML_MARKERS)
    t1_prompt = list(range(153)) + [151667, 271, 151668, 271]
    t1_output = list(range(500, 524))
    tracker.record_turn_completion("sess-think", t1_prompt, t1_output, turn_id=1)

    t2_prompt = list(range(153)) + list(range(800, 828)) + list(range(900, 995))
    res = tracker.classify_prefill(
        t2_prompt, num_cached_tokens=128, session_id="sess-think", turn_id=2
    )
    # 25 tokens in [128..153) + 4 <think> tokens + 24 reply tokens = 53
    # unreused_reply_loss
    assert res == TokenBreakdown(
        cache_hit=128,
        unreused_reply_loss=53,
        prefix_mutation_loss=0,
        new_input=95,
        mutation_offset_token_idx=None,
    )
    assert res.total_tokens == len(t2_prompt)
    assert "gen_ai.usage.mutation_offset_token_idx" not in res.to_span_attributes()

    # 2. Fallback when previous prompt does NOT end with empty think block (thinking on
    # / other models):
    # Reply boundary falls back to end of previous prompt (len(t1_no_think)=150).
    t1_no_think = list(range(150))
    tracker.record_turn_completion("sess-no-think", t1_no_think, t1_output, turn_id=1)
    t2_no_think = list(range(150)) + list(range(800, 824)) + list(range(900, 950))
    res_no_think = tracker.classify_prefill(
        t2_no_think, num_cached_tokens=128, session_id="sess-no-think", turn_id=2
    )
    # 22 prompt tokens [128..150) sharing block 2 + 24 reply tokens [150..174) = 46
    # unreused_reply_loss
    assert res_no_think == TokenBreakdown(
        cache_hit=128,
        unreused_reply_loss=46,
        prefix_mutation_loss=0,
        new_input=50,
        mutation_offset_token_idx=None,
    )
    assert res_no_think.total_tokens == len(t2_no_think)

    # 3. When reasoning tokens are dropped on Turn 2 (carried-back reply shorter than
    # generated reply):
    # unreused_reply_loss stops at the first <|im_end|> (151645) after the divergence
    # point
    # so subsequent new user/tool tokens are attributed to new_input rather than
    # unreused_reply_loss.
    t1_think_on = list(range(150))
    t1_long_reasoning_output = list(
        range(2000, 2220)
    )  # 220 generated tokens (e.g. 200 reasoning + 20 reply)
    tracker.record_turn_completion(
        "sess-dropped-reasoning", t1_think_on, t1_long_reasoning_output, turn_id=1
    )
    # Turn 2 carries back only 19 assistant tokens + <|im_end|> (151645) = 20 tokens
    # [150..170),
    # followed by 120 new user/tool tokens [170..290).
    t2_carried_reply = list(range(3000, 3019)) + [151645]
    t2_new_user_turn = list(range(4000, 4120))
    t2_dropped_prompt = t1_think_on + t2_carried_reply + t2_new_user_turn
    res_dropped = tracker.classify_prefill(
        t2_dropped_prompt,
        num_cached_tokens=128,
        session_id="sess-dropped-reasoning",
        turn_id=2,
    )
    # 22 prompt tokens [128..150) sharing block 2 + 20 carried-back reply tokens
    # [150..170) = 42 unreused_reply_loss
    assert res_dropped == TokenBreakdown(
        cache_hit=128,
        unreused_reply_loss=42,
        prefix_mutation_loss=0,
        new_input=120,
        mutation_offset_token_idx=None,
    )
    assert res_dropped.total_tokens == len(t2_dropped_prompt)

    # 4. On an edited turn (system prompt mutated at token 5) with dropped reasoning
    # tokens,
    # prefix_mutation_loss also stops at the old reply's <|im_end|> (170) so new tokens
    # after
    # the old reply (120) count as new_input rather than prefix_mutation_loss.
    tracker.record_turn_completion(
        "sess-edited-dropped", t1_think_on, t1_long_reasoning_output, turn_id=1
    )
    t2_mutated_prefix = list(range(150))
    t2_mutated_prefix[5] = 999999
    t2_edited_dropped_prompt = t2_mutated_prefix + t2_carried_reply + t2_new_user_turn
    res_edited_dropped = tracker.classify_prefill(
        t2_edited_dropped_prompt,
        num_cached_tokens=0,
        session_id="sess-edited-dropped",
        turn_id=2,
    )
    assert res_edited_dropped == TokenBreakdown(
        cache_hit=0,
        prefix_mutation_loss=170,
        unreused_reply_loss=0,
        new_input=120,
        mutation_offset_token_idx=5,
    )
    assert res_edited_dropped.total_tokens == len(t2_edited_dropped_prompt)


def test_history_compaction_and_dropped_messages_preserve_new_input():
    """Compacting earlier history (even below reply_start_idx) keeps new_input."""
    im_start, im_end, nl = 151644, 151645, 198
    sys_role, user_role, asst_role = 8948, 872, 77091
    think_block = [151667, 271, 151668, 271]
    gen = [im_start, asst_role, nl]

    def msg(role: int, content: list[int]) -> list[int]:
        return [im_start, role, nl] + content + [im_end, nl]

    sys_c = list(range(10000, 10300))
    user_long = list(range(20000, 20500))
    reply_rec = list(range(60000, 60083))
    tool_new = list(range(70000, 70400))

    # Case 1: Intra-message compaction shrinking user_1 by 200 tokens (total >
    # prior.reply_start_idx).
    t1_prompt = msg(sys_role, sys_c) + msg(user_role, user_long) + gen + think_block
    t1_out = list(range(50000, 50200))
    user_short = user_long[:300]
    t2_compacted = (
        msg(sys_role, sys_c)
        + msg(user_role, user_short)
        + gen
        + reply_rec
        + [im_end, nl]
        + msg(user_role, tool_new)
        + gen
        + think_block
    )

    for tier1_cap in (4096, 1):
        tracker = SessionTurnTracker(
            block_size=128, tier1_maxsize=tier1_cap, turn_markers=_CHATML_MARKERS
        )
        tracker.record_turn_completion("sess-cmp", t1_prompt, t1_out, turn_id=1)
        if tier1_cap == 1:
            tracker.record_turn_completion("evictor", [1, 2, 3, 4], [], turn_id=1)
            assert tracker.tier2.peek("sess-cmp") is not None
        res = tracker.classify_prefill(
            t2_compacted, num_cached_tokens=512, session_id="sess-cmp", turn_id=2
        )
        assert res.cache_hit == 512
        assert res.prefix_mutation_loss == 697 - 512
        assert res.new_input == 413
        assert res.unreused_reply_loss == 0
        assert res.total_tokens == len(t2_compacted)
        if tier1_cap > 1:
            assert res.mutation_offset_token_idx == 608

    # Case 2: Extreme multi-turn compaction (drops 2 old messages AND shrinks total <
    # prior.reply_start_idx).
    big_user_1 = list(range(20000, 22000))
    old_asst_1 = list(range(30000, 30200))
    old_user_2 = list(range(40000, 41500))
    prior_multi = (
        msg(sys_role, sys_c)
        + msg(user_role, big_user_1)
        + msg(asst_role, old_asst_1)
        + msg(user_role, old_user_2)
        + gen
        + think_block
    )
    summary_user = list(range(80000, 80100))
    compacted_multi = (
        msg(sys_role, sys_c)
        + msg(user_role, summary_user)
        + gen
        + reply_rec
        + [im_end, nl]
        + msg(user_role, tool_new)
        + gen
        + think_block
    )
    assert len(compacted_multi) < len(prior_multi) - 4  # total < prior.reply_start_idx
    tracker_m = SessionTurnTracker(block_size=128, turn_markers=_CHATML_MARKERS)
    tracker_m.record_turn_completion("sess-drop-msgs", prior_multi, t1_out, turn_id=2)
    res_m = tracker_m.classify_prefill(
        compacted_multi, num_cached_tokens=256, session_id="sess-drop-msgs", turn_id=3
    )
    assert res_m.cache_hit == 256
    assert res_m.new_input == 413
    assert res_m.prefix_mutation_loss == len(compacted_multi) - 256 - 413
    assert res_m.total_tokens == len(compacted_multi)


def test_tier2_partial_last_block_when_new_prompt_shorter_than_prior():
    """Tier 2 matches Tier 1 on partial last block when total < prior.seq_len."""
    im_start, im_end, nl = 151644, 151645, 198
    sys_role, user_role, asst_role = 8948, 872, 77091
    think_block = [151667, 271, 151668, 271]
    gen = [im_start, asst_role, nl]

    def msg(role: int, content: list[int]) -> list[int]:
        return [im_start, role, nl] + content + [im_end, nl]

    # Construct Turn 1 so reply_start_idx = 1026 (inside block 8 [1024..1152)),
    # with 500 generated tokens -> prior.seq_len = 1530.
    sys_c = list(range(1000, 1500))  # msg = 505 tokens
    user_c = list(
        range(2000, 2513)
    )  # msg = 518 tokens; + gen(3) + think(4) = 1030 -> reply_start = 1026
    t1_prompt = msg(sys_role, sys_c) + msg(user_role, user_c) + gen + think_block
    assert len(t1_prompt) == 1030
    t1_long_out = list(range(50000, 50500))  # prior.seq_len = 1530

    # Turn 2 carries back a 15-token reply + 40-token tool output -> total = 1026 + 15 +
    # 2 + 45 + 3 + 4 = 1095
    # which lies inside block 8 [1024..1152) and has total (1095) < prior.seq_len
    # (1530).
    short_reply = list(range(60000, 60015))
    short_tool = list(range(70000, 70040))
    t2_prompt = (
        msg(sys_role, sys_c)
        + msg(user_role, user_c)
        + gen
        + short_reply
        + [im_end, nl]
        + msg(user_role, short_tool)
        + gen
        + think_block
    )
    assert 1024 < len(t2_prompt) < 1152

    t_tier1 = SessionTurnTracker(
        block_size=128, tier1_maxsize=16, turn_markers=_CHATML_MARKERS
    )
    t_tier1.record_turn_completion("s-tail", t1_prompt, t1_long_out, turn_id=1)
    b_tier1 = t_tier1.classify_prefill(
        t2_prompt, num_cached_tokens=1024, session_id="s-tail", turn_id=2
    )

    t_tier2 = SessionTurnTracker(
        block_size=128,
        tier1_maxsize=1,
        tier2_maxsize=16,
        turn_markers=_CHATML_MARKERS,
    )
    t_tier2.record_turn_completion("s-tail", t1_prompt, t1_long_out, turn_id=1)
    t_tier2.record_turn_completion("other", [1, 2, 3, 4], [], turn_id=1)
    assert t_tier2.tier1.peek("s-tail") is None
    assert t_tier2.tier2.peek("s-tail") is not None
    b_tier2 = t_tier2.classify_prefill(
        t2_prompt, num_cached_tokens=1024, session_id="s-tail", turn_id=2
    )

    assert b_tier2 == b_tier1
    assert b_tier2.new_input == 53
    assert b_tier2.affinity_honored_eviction == 0
    assert b_tier2.unreused_reply_loss == len(t2_prompt) - 1024 - 53


def test_revisit_bounce_back_hop_tier1_tier2_and_subsequent_turn():
    """Revisit hop r_A -> r_B -> r_A classifies T3 as routing_loss and T4 normally."""
    im_start, im_end, nl = 151644, 151645, 198
    sys_role, user_role, asst_role = 8948, 872, 77091
    think_block = [151667, 271, 151668, 271]
    gen = [im_start, asst_role, nl]

    def msg(role: int, content: list[int]) -> list[int]:
        return [im_start, role, nl] + content + [im_end, nl]

    sys_c = list(range(1000, 1200))
    u1 = list(range(2000, 2200))
    a1_rec = list(range(3000, 3080))
    tool2 = list(range(4000, 4300))
    a2_rec = list(range(5000, 5090))
    tool3 = list(range(6000, 6250))  # T3 new tool output: msg = 255 tokens
    a3_rec = list(range(7000, 7060))
    tool4 = list(range(8000, 8180))  # T4 new tool output: msg = 185 tokens

    p1 = msg(sys_role, sys_c) + msg(user_role, u1) + gen + think_block
    p2 = (
        msg(sys_role, sys_c)
        + msg(user_role, u1)
        + gen
        + a1_rec
        + [im_end, nl]
        + msg(user_role, tool2)
        + gen
        + think_block
    )
    p3 = (
        msg(sys_role, sys_c)
        + msg(user_role, u1)
        + gen
        + a1_rec
        + [im_end, nl]
        + msg(user_role, tool2)
        + gen
        + a2_rec
        + [im_end, nl]
        + msg(user_role, tool3)
        + gen
        + think_block
    )
    p4 = (
        msg(sys_role, sys_c)
        + msg(user_role, u1)
        + gen
        + a1_rec
        + [im_end, nl]
        + msg(user_role, tool2)
        + gen
        + a2_rec
        + [im_end, nl]
        + msg(user_role, tool3)
        + gen
        + a3_rec
        + [im_end, nl]
        + msg(user_role, tool4)
        + gen
        + think_block
    )
    # Trailing new_input after closing <|im_end|> of prior assistant reply:
    # [nl] (1) + msg(user_role, tool_k) (len(tool_k)+5) + gen (3) + think_block (4)
    want_t3_new = 1 + (len(tool3) + 5) + 3 + 4  # 263
    want_t4_new = 1 + (len(tool4) + 5) + 3 + 4  # 193

    for tier1_cap in (4096, 1):
        r_a = SessionTurnTracker(
            block_size=128,
            tier1_maxsize=tier1_cap,
            tier2_maxsize=64,
            turn_markers=_CHATML_MARKERS,
        )
        r_b = SessionTurnTracker(
            block_size=128,
            tier1_maxsize=tier1_cap,
            tier2_maxsize=64,
            turn_markers=_CHATML_MARKERS,
        )

        # Turn 1 runs on r_A.
        b1 = r_a.classify_prefill(
            p1, num_cached_tokens=0, session_id="sess-bounce", turn_id=1
        )
        assert b1.new_input == len(p1)
        r_a.record_turn_completion(
            "sess-bounce", p1, list(range(90000, 90120)), turn_id=1
        )

        if tier1_cap == 1:
            # Demote sess-bounce on r_A into Tier 2 while it runs Turn 2 elsewhere.
            r_a.record_turn_completion("evictor", [1, 2, 3, 4], [], turn_id=1)
            assert r_a.tier1.peek("sess-bounce") is None
            assert r_a.tier2.peek("sess-bounce") is not None

        # Turn 2 hops to r_B (first-visit hop on r_B).
        b2 = r_b.classify_prefill(
            p2, num_cached_tokens=128, session_id="sess-bounce", turn_id=2
        )
        assert b2.routing_loss > 0 and b2.affinity_honored_eviction == 0
        r_b.record_turn_completion(
            "sess-bounce", p2, list(range(91000, 91150)), turn_id=2
        )

        # Turn 3 bounces back to r_A (revisit hop: r_A holds stale Turn 1 state, both
        # hit=0 and hit=256).
        b3_cold = r_a.classify_prefill(
            p3, num_cached_tokens=0, session_id="sess-bounce", turn_id=3
        )
        assert b3_cold == TokenBreakdown(
            cache_hit=0,
            routing_loss=len(p3) - want_t3_new,
            new_input=want_t3_new,
        )
        assert b3_cold.affinity_honored_eviction == 0

        b3_partial = r_a.classify_prefill(
            p3, num_cached_tokens=256, session_id="sess-bounce", turn_id=3
        )
        assert b3_partial == TokenBreakdown(
            cache_hit=256,
            routing_loss=len(p3) - want_t3_new - 256,
            new_input=want_t3_new,
        )

        # Edge check: if hit exceeds last_reply_end on a hop, routing_loss clamps to 0
        # and sum == total.
        b3_deep_hit = r_a.classify_prefill(
            p3, num_cached_tokens=len(p3) - 10, session_id="sess-bounce", turn_id=3
        )
        assert b3_deep_hit == TokenBreakdown(
            cache_hit=len(p3) - 10,
            routing_loss=0,
            new_input=10,
        )
        assert b3_deep_hit.total_tokens == len(p3)

        # Record Turn 3 completion on r_A, then run consecutive Turn 4 on r_A.
        r_a.record_turn_completion(
            "sess-bounce", p3, list(range(92000, 92100)), turn_id=3
        )
        b4 = r_a.classify_prefill(
            p4, num_cached_tokens=1024, session_id="sess-bounce", turn_id=4
        )
        assert b4.routing_loss == 0
        assert b4.prefix_mutation_loss == 0
        assert b4.unreused_reply_loss > 0
        assert b4.new_input == want_t4_new
        assert b4.total_tokens == len(p4)


def test_negative_category_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        TokenBreakdown(cache_hit=-1)


def test_aborted_turn_verified_match_cache_stays_bounded():
    tracker = SessionTurnTracker(block_size=4, tier1_maxsize=2)
    for idx in range(5):
        sid = f"sess-abort-{idx}"
        tracker.record_turn_completion(sid, [1, 2, 3, 4], [5, 6, 7, 8], turn_id=1)
        # Classify Turn 2 without calling record_turn_completion (simulating abort).
        tracker.classify_prefill(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            num_cached_tokens=4,
            session_id=sid,
            turn_id=2,
        )
    assert len(tracker._verified_match_len) <= 2
    tracker.discard_pending("sess-abort-4")
    assert tracker._verified_match_len.peek("sess-abort-4") is None


def test_mutation_below_num_cached_tokens_detected_and_hashes_uncorrupted():
    """Mutation inside `[0, hit)` is caught and does not corrupt Tier-2 block hashes."""
    for tier1_cap in (16, 1):
        tracker = SessionTurnTracker(
            block_size=4, tier1_maxsize=tier1_cap, tier2_maxsize=16
        )
        tracker.record_turn_completion(
            "sess-m0", [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], turn_id=1
        )
        if tier1_cap == 1:
            tracker.record_turn_completion("evict-1", [42, 43, 44, 45], [], turn_id=1)

        # Another session cached [999, 2, 3, 4], so hit=4 despite edit at index 0.
        t2_prompt = [999, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        b2 = tracker.classify_prefill(
            t2_prompt, num_cached_tokens=4, session_id="sess-m0", turn_id=2
        )
        assert b2 == TokenBreakdown(
            cache_hit=4,
            affinity_honored_eviction=0,
            prefix_mutation_loss=8,
            new_input=4,
            mutation_offset_token_idx=0,
        )
        tracker.record_turn_completion(
            "sess-m0", t2_prompt, [17, 18, 19, 20], turn_id=2
        )

        # Force demotion to Tier 2 and verify Turn 3 sees exact prefix match.
        if tier1_cap == 1:
            tracker.record_turn_completion("evict-2", [52, 53, 54, 55], [], turn_id=1)
        t3_prompt = t2_prompt + [17, 18, 19, 20, 21, 22, 23, 24]
        b3 = tracker.classify_prefill(
            t3_prompt, num_cached_tokens=8, session_id="sess-m0", turn_id=3
        )
        assert b3 == TokenBreakdown(
            cache_hit=8,
            affinity_honored_eviction=12,
            prefix_mutation_loss=0,
            new_input=4,
        )


def test_token_and_block_budgets_bound_tier_memory():
    """Tier 1 and Tier 2 evict oldest entries when token/block budgets are exceeded."""
    tracker = SessionTurnTracker(
        block_size=4,
        tier1_maxsize=100,
        tier2_maxsize=100,
        tier1_max_tokens=16,
        tier2_max_blocks=6,
    )
    # Each session stores 12 tokens (3 blocks). Two sessions = 24 tokens > 16 budget,
    # so s-0 demotes to Tier 2 while s-1 stays in Tier 1.
    for idx in range(4):
        base = idx * 100
        tracker.record_turn_completion(
            f"s-{idx}",
            list(range(base, base + 8)),
            list(range(base + 8, base + 12)),
            turn_id=1,
        )
    assert tracker._tier1_total_tokens <= 16
    assert len(tracker.tier1) == 1
    # Tier 2 budget is 6 blocks (at most 2 demoted 3-block sessions: s-1 and s-2).
    assert tracker._tier2_total_blocks <= 6
    assert tracker.tier2.peek("s-0") is None
    assert tracker.tier2.peek("s-2") is not None


def test_tokenizer_turn_markers_and_no_false_match_on_gemma_or_repetition():
    """Tokenizer-driven turn markers support ChatML/Llama/Gemma and avoid false hits."""

    class FakeTokenizer:
        def __init__(self, vocab: dict[str, int], unk_token_id: int = 0) -> None:
            self._vocab = vocab
            self.unk_token_id = unk_token_id

        def get_vocab(self) -> dict[str, int]:
            return self._vocab

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [self._vocab[text]] if text in self._vocab else [99, 100]

    llama_tok = FakeTokenizer(
        {"<|start_header_id|>": 128006, "<|eot_id|>": 128009, "assistant": 78191}
    )
    assert resolve_turn_markers_from_tokenizer(llama_tok) == ChatTurnMarkers(
        im_start_token_id=128006,
        im_end_token_id=128009,
        assistant_role_token_id=78191,
    )

    gemma_tok = FakeTokenizer(
        {"<start_of_turn>": 106, "<end_of_turn>": 107, "model": 2516}
    )
    gemma_markers = resolve_turn_markers_from_tokenizer(gemma_tok)
    assert gemma_markers == ChatTurnMarkers(
        im_start_token_id=106,
        im_end_token_id=107,
        assistant_role_token_id=2516,
    )

    # On Gemma (256k vocab), token IDs 151644/151645 are ordinary vocabulary words,
    # and trailing [16, 198, 17, 198] ("1\n2\n") does NOT trigger <think> suffix logic.
    gemma_tracker = SessionTurnTracker(block_size=4)
    gemma_tracker.configure_from_tokenizer(gemma_tok)
    p1_gemma = [151644, 77091, 151645, 50, 16, 198, 17, 198]
    gemma_tracker.record_turn_completion(
        "gemma-s", p1_gemma, [200, 201, 202, 203], turn_id=1
    )
    assert gemma_tracker.tier1.peek("gemma-s").reply_start_idx == len(p1_gemma)

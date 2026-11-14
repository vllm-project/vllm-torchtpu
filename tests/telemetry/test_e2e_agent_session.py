# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Stage 4 end-to-end 5-turn agent session verification for CUJ-2 telemetry."""

from types import SimpleNamespace
from typing import Any

from vllm_torchtpu.telemetry import (
    CATEGORIES,
    AgenticTelemetryService,
    compute_rolling_block_hashes,
)


def _run_turn(
    svc: AgenticTelemetryService,
    *,
    request_id: str,
    prompt_tokens: list[int],
    output_tokens: list[int],
    num_cached_tokens: int,
    session_id: str,
    turn_id: int,
    parent_session_id: str | None = None,
    now: float = 100.0,
) -> dict[str, Any]:
    headers = {
        "x-agent-session-id": session_id,
        "x-agent-turn-id": str(turn_id),
    }
    if parent_session_id:
        headers["x-agent-parent-session-id"] = parent_session_id

    req_state = SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_tokens,
        num_cached_tokens=num_cached_tokens,
        is_prefilling=True,
        detokenizer=SimpleNamespace(output_token_ids=output_tokens),
    )
    out = SimpleNamespace(
        request_id=request_id,
        num_cached_tokens=num_cached_tokens,
        new_token_ids=output_tokens,
        finish_reason="stop",
        trace_headers=headers,
        kv_transfer_params=None,
        num_external_computed_tokens=0,
    )
    svc.on_prefill_output(req_state, out)
    attrs = svc.build_span_attributes(req_state, out, now=now)

    # Invariant: 6 categories must sum to exact prompt token count on every turn.
    cat_sum = sum(attrs[f"gen_ai.usage.input_tokens.{c}"] for c in CATEGORIES)
    assert cat_sum == len(prompt_tokens)
    return attrs


def test_five_turn_agent_session_and_subagent_fork() -> None:
    """Verify 5-turn agent workflow: cold start, cache hit, mutation, eviction, and sub-agent fork."""
    svc = AgenticTelemetryService(block_size=4, replica_id="tpu-v6e-node-1")

    # Turn 1 (Cold start): 8 prompt tokens, 0 cached -> 8 new_input; generates 4 tokens.
    t1_prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    t1_out = [9, 10, 11, 12]
    attrs1 = _run_turn(
        svc,
        request_id="req-t1",
        prompt_tokens=t1_prompt,
        output_tokens=t1_out,
        num_cached_tokens=0,
        session_id="agent-main",
        turn_id=1,
        now=10.0,
    )
    assert attrs1["gen_ai.usage.input_tokens.new_input"] == 8
    assert attrs1["gen_ai.usage.input_tokens.cache_hit"] == 0

    # Turn 2 (Prefix hit + new tool output): 12 prior tokens cached + 4 new tokens.
    t2_prompt = t1_prompt + t1_out + [13, 14, 15, 16]
    t2_out = [17, 18, 19, 20]
    attrs2 = _run_turn(
        svc,
        request_id="req-t2",
        prompt_tokens=t2_prompt,
        output_tokens=t2_out,
        num_cached_tokens=12,
        session_id="agent-main",
        turn_id=2,
        now=20.0,
    )
    assert attrs2["gen_ai.usage.input_tokens.cache_hit"] == 12
    assert attrs2["gen_ai.usage.input_tokens.new_input"] == 4

    # Turn 3 (Dynamic system prompt mutation at token index 4):
    # First 4 tokens hit cache; token 4 mutates (999 instead of 5);
    # prior sequence length was 20, so tokens 4..20 (16 tokens) are prefix_mutation_loss,
    # plus 4 new tokens at the tail.
    t3_prompt = [1, 2, 3, 4, 999, 6, 7, 8] + t1_out + [13, 14, 15, 16] + t2_out + [
        21,
        22,
        23,
        24,
    ]
    t3_out = [25, 26, 27, 28]
    attrs3 = _run_turn(
        svc,
        request_id="req-t3",
        prompt_tokens=t3_prompt,
        output_tokens=t3_out,
        num_cached_tokens=4,
        session_id="agent-main",
        turn_id=3,
        now=30.0,
    )
    assert attrs3["gen_ai.usage.input_tokens.cache_hit"] == 4
    assert attrs3["gen_ai.usage.input_tokens.prefix_mutation_loss"] == 16
    assert attrs3["gen_ai.usage.mutation_offset_token_idx"] == 4
    assert attrs3["gen_ai.usage.input_tokens.new_input"] == 4

    # Turn 4 (HBM capacity eviction on same replica):
    # Prefix matches Turn 3's 28 tokens, but only 8 tokens remain in HBM (20 evicted at t=38.5s).
    t3_full = t3_prompt + t3_out
    hashes = compute_rolling_block_hashes(t3_full, block_size=4)
    for blk_hash in hashes[2:]:
        svc.on_block_evicted(blk_hash, now=38.5)

    t4_prompt = t3_full + [29, 30, 31, 32]
    t4_out = [33, 34, 35, 36]
    attrs4 = _run_turn(
        svc,
        request_id="req-t4",
        prompt_tokens=t4_prompt,
        output_tokens=t4_out,
        num_cached_tokens=8,
        session_id="agent-main",
        turn_id=4,
        now=40.0,
    )
    assert attrs4["gen_ai.usage.input_tokens.cache_hit"] == 8
    assert attrs4["gen_ai.usage.input_tokens.affinity_honored_eviction"] == 20
    assert attrs4["gen_ai.usage.input_tokens.new_input"] == 4
    assert attrs4["vllm.kv_cache.time_since_eviction_ms"] == 1500.0

    # Turn 5 (Sub-agent fork inheriting from `agent-main`):
    # Sub-agent shares the 36-token history of `agent-main` (16 cached in HBM, 20 evicted) + 4 new tokens.
    t4_full = t4_prompt + t4_out
    t5_prompt = t4_full + [37, 38, 39, 40]
    attrs5 = _run_turn(
        svc,
        request_id="req-t5-sub",
        prompt_tokens=t5_prompt,
        output_tokens=[41, 42],
        num_cached_tokens=16,
        session_id="agent-sub-1",
        turn_id=1,
        parent_session_id="agent-main",
        now=45.0,
    )
    assert attrs5["gen_ai.session.parent_id"] == "agent-main"
    assert attrs5["gen_ai.usage.input_tokens.cache_hit"] == 16
    assert attrs5["gen_ai.usage.input_tokens.affinity_honored_eviction"] == 20
    assert attrs5["gen_ai.usage.input_tokens.new_input"] == 4

    # Cumulative category totals across all 5 turns match sum of all prompt lengths.
    total_prompts = (
        len(t1_prompt)
        + len(t2_prompt)
        + len(t3_prompt)
        + len(t4_prompt)
        + len(t5_prompt)
    )
    assert sum(svc.category_totals.values()) == total_prompts

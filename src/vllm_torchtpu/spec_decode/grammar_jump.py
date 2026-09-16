# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar jump-forward decoding via the speculative verification graph.

In structured output, long stretches of the response are fully determined by
the grammar: constant JSON keys, punctuation, fixed enum values. Generating
them one token per decode step wastes a full model forward on each.

On TPU the way to skip those forwards is *not* a bespoke variable-length
prefill operator. The speculative verification graph is already a statically
shaped ``[B, 1 + K]`` mini-prefill: it is compiled, it writes ``K`` KV entries
per row in one dispatch, and the runner already applies the grammar bitmask to
its target logits. Feeding deterministic grammar tokens into it as draft tokens
therefore needs no new kernel and no runner change.

``vllm_torchtpu`` supports exactly two drafters (see
``SpeculativeDecodingManager.propose_draft_token_ids``): ``ngram``, which is
CPU-only and needs no draft model, and an async eagle3 drafter. This module
does not add a third. It tops up the speculative slots the configured proposer
left empty, which reaches the same result without touching the runner.

The hook has to live in the engine-core process: the grammar FSM belongs to the
scheduler, while drafters run in the worker. ``Scheduler.update_draft_token_ids``
is the one seam that can see both.

Enable with ``VLLM_TPU_JUMP_DECODING=1``. Requires speculative decoding to be
configured. Pair it with vLLM's own ``disable_any_whitespace`` structured-output
option: with optional whitespace in the grammar, almost no JSON position has a
single outgoing transition and there is nothing to jump.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("vllm." + __name__)

# Emit an aggregate counter line every this many top-up attempts. The counters
# are the only way to tell "no speedup because the feature never engaged" apart
# from "no speedup because this schema has nothing to jump", and in a
# multiprocess engine the logs are the only channel out of the engine-core
# process.
_LOG_EVERY = 1000

# (tokens, flags) Drafter Protocol Constants:
# Flag encoding: G=2, D=1, empty=0.
# Ordered row invariant: every row is monotonically non-increasing G^a D^b 0^c.
DRAFT_FLAG_EMPTY = 0
DRAFT_FLAG_MODEL = 1
DRAFT_FLAG_GRAMMAR = 2


@dataclass
class JumpStats:
    """Opportunity and yield counters for the jump top-up."""

    calls: int = 0
    """Requests the top-up was consulted for (had budget and a live grammar)."""

    hits: int = 0
    """Calls that produced at least one jump token."""

    tokens_proposed: int = 0
    """Jump tokens handed to the speculative verifier."""

    truncations: int = 0
    """Jump strings that did not fit in the remaining speculative slots."""

    fsm_rejections: int = 0
    """Jump strings whose tokenization the FSM did not fully accept."""

    def summary(self) -> str:
        hit_rate = (self.hits / self.calls) if self.calls else 0.0
        per_hit = (self.tokens_proposed / self.hits) if self.hits else 0.0
        return (f"calls={self.calls} hits={self.hits} "
                f"hit_rate={hit_rate:.3f} tokens={self.tokens_proposed} "
                f"tokens_per_hit={per_hit:.2f} "
                f"truncations={self.truncations} "
                f"fsm_rejections={self.fsm_rejections}")


_STATS = JumpStats()


def get_stats() -> JumpStats:
    """Returns the process-local counters.

    Only meaningful in the process that runs the scheduler. With a multiprocess
    engine that is the engine-core child, so tests that read this directly must
    run with ``VLLM_ENABLE_V1_MULTIPROCESSING=0``; otherwise read the periodic
    log line instead.
    """
    return _STATS


def propose_jump_topup(
    grammar: Any,
    tokenizer: Any,
    validated_draft: list[int],
    num_spec_tokens: int,
    stats: JumpStats | None = None,
) -> list[int]:
    """Returns deterministic grammar tokens to append after ``validated_draft``.

    The FSM is walked past ``validated_draft`` first, because the jump string
    has to describe the position the returned tokens will actually occupy. The
    matcher is restored to its original state before returning: this is a peek,
    and the tokens are only committed if the target model's verification
    accepts them.

    The budget is derived from ``num_spec_tokens`` rather than accepted from the
    caller. xgrammar caps rollback at ``max_rollback_tokens``, which vLLM sets to
    ``num_speculative_tokens``; deriving the budget here makes
    ``len(validated_draft) + len(result) <= num_spec_tokens`` structural, so the
    restore can never exceed what the matcher can undo.

    Args:
      grammar: a ``StructuredOutputGrammar`` exposing ``matcher`` and
        ``is_terminated()``.
      tokenizer: used to turn the jump string into token ids.
      validated_draft: draft tokens the proposer produced, already checked
        against the grammar.
      num_spec_tokens: total speculative slots for this request.
      stats: optional counters to update.

    Returns:
      Tokens to append to the draft; empty if nothing can be jumped.
    """
    budget = num_spec_tokens - len(validated_draft)
    if budget <= 0:
        return []
    if grammar.is_terminated():
        return []

    matcher = grammar.matcher
    if stats is not None:
        stats.calls += 1

    advanced = 0
    try:
        for token in validated_draft:
            if not matcher.accept_token(token):
                # The draft was validated against this grammar moments ago, so
                # this means the FSM moved underneath us. Propose nothing.
                return []
            advanced += 1

        jump_str = matcher.find_jump_forward_string()
        if not jump_str:
            return []

        candidate = tokenizer.encode(jump_str, add_special_tokens=False)
        if not candidate:
            return []
        if len(candidate) > budget:
            candidate = candidate[:budget]
            if stats is not None:
                stats.truncations += 1

        # The jump string is text; its tokenization does not have to line up
        # with the token boundaries the FSM expects. Accept token by token and
        # keep only the prefix that actually lands.
        jumped = 0
        for token in candidate:
            if not matcher.accept_token(token):
                break
            jumped += 1
        advanced += jumped

        if stats is not None:
            if jumped < len(candidate):
                stats.fsm_rejections += 1
            if jumped:
                stats.hits += 1
                stats.tokens_proposed += jumped
        return candidate[:jumped]
    finally:
        if advanced:
            matcher.rollback(advanced)


def validate_ordered_row_invariant(flags: list[int] | Any) -> bool:
    """Validates that a row of draft flags follows the ordered invariant (G^a D^b 0^c).

    Under the GaaD contract, Grammar tokens (2) must precede Model tokens (1),
    which must precede Empty slots (0). Numerically, flags must be monotonically
    non-increasing: flags[i] >= flags[i+1] for all i, with each flag in {0, 1, 2}.
    """
    valid_flags = {DRAFT_FLAG_EMPTY, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR}
    for i, flag in enumerate(flags):
        if flag not in valid_flags:
            return False
        if i > 0 and flag > flags[i - 1]:
            return False
    return True


def propose_jump_topup_with_flags(
    grammar: Any,
    tokenizer: Any,
    validated_draft: list[int],
    num_spec_tokens: int,
    stats: JumpStats | None = None,
) -> tuple[list[int], list[int]]:
    """Returns deterministic grammar tokens and their draft flags.

    Wraps ``propose_jump_topup`` to conform to the (tokens, flags) drafter
    protocol. Every generated jump token is assigned ``DRAFT_FLAG_GRAMMAR`` (2).

    Returns:
        (tokens, flags): A tuple containing the list of jump token IDs and a
        matching list of DRAFT_FLAG_GRAMMAR flags.
    """
    tokens = propose_jump_topup(
        grammar=grammar,
        tokenizer=tokenizer,
        validated_draft=validated_draft,
        num_spec_tokens=num_spec_tokens,
        stats=stats,
    )
    flags = [DRAFT_FLAG_GRAMMAR] * len(tokens)
    return tokens, flags


def _topup_draft_token_ids(scheduler: Any,
                           draft_token_ids: Any,
                           scheduler_output: Any = None) -> None:
    """Tops up speculative draft token lists with deterministic grammar jump tokens."""
    if draft_token_ids is None:
        return
    manager = getattr(scheduler, "structured_output_manager", None)
    if manager is None:
        return
    tokenizer = getattr(manager, "tokenizer", None)
    if tokenizer is None:
        return

    req_ids = getattr(draft_token_ids, "req_ids", [])
    token_lists = getattr(draft_token_ids, "draft_token_ids", [])
    num_spec_tokens = getattr(scheduler, "num_spec_tokens", 0)
    if num_spec_tokens <= 0:
        return

    for req_id, spec_tokens in zip(req_ids, token_lists):
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished() or getattr(
                request, "is_prefill_chunk", False):
            continue
        if hasattr(manager,
                   "should_advance") and not manager.should_advance(request):
            continue
        structured_req = getattr(request, "structured_output_request", None)
        if structured_req is None:
            continue
        grammar = getattr(structured_req, "grammar", None)
        if grammar is None:
            continue

        orig_budget = num_spec_tokens
        if scheduler_output is not None:
            sched_spec = getattr(scheduler_output,
                                 "scheduled_spec_decode_tokens", {})
            placeholder = sched_spec.get(req_id)
            if placeholder:
                orig_budget = len(placeholder)

        remaining = request.max_tokens - len(request.output_token_ids) - 1
        effective_spec_tokens = min(orig_budget,
                                    len(spec_tokens) + max(0, remaining))

        validated_draft = grammar.validate_tokens(list(spec_tokens))
        topup = propose_jump_topup(grammar, tokenizer, validated_draft,
                                   effective_spec_tokens, _STATS)
        if topup:
            spec_tokens.extend(topup)


def _topup_running_requests(scheduler: Any) -> None:
    """Fills each structured request's unused speculative slots."""
    num_spec_tokens = getattr(scheduler, "num_spec_tokens", 0)
    if num_spec_tokens <= 0:
        return

    manager = getattr(scheduler, "structured_output_manager", None)
    if manager is None:
        return
    tokenizer = getattr(manager, "tokenizer", None)
    if tokenizer is None:
        return

    for request in scheduler.running:
        if request.is_finished() or getattr(request, "is_prefill_chunk",
                                            False):
            continue
        if not hasattr(request, "spec_token_ids"):
            continue
        if hasattr(manager,
                   "should_advance") and not manager.should_advance(request):
            continue
        structured_req = getattr(request, "structured_output_request", None)
        if structured_req is None:
            continue
        grammar = getattr(structured_req, "grammar", None)
        if grammar is None:
            continue

        draft = request.spec_token_ids
        # Never speculate past the request's own output allowance; the bonus
        # token of this step consumes one of the remaining slots.
        remaining = request.max_tokens - len(request.output_token_ids) - 1
        effective_spec_tokens = min(num_spec_tokens,
                                    len(draft) + max(0, remaining))
        topup = propose_jump_topup(grammar, tokenizer, draft,
                                   effective_spec_tokens, _STATS)
        if topup:
            request.spec_token_ids = draft + topup

    if _STATS.calls and (_STATS.calls % 10 == 0 or _STATS.hits > 0):
        logger.info("TPU grammar jump decoding: %s", _STATS.summary())


def _patch_vllm_grammar_jump_decoding() -> None:
    """Wraps ``Scheduler.update_draft_token_ids`` and ``Scheduler.update_draft_token_ids_in_output``

    with the grammar jump draft top-up.
    """
    from vllm_torchtpu import envs
    if not envs.VLLM_TPU_JUMP_DECODING:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_grammar_jump_patched", False):
        return

    if not hasattr(Scheduler, "update_draft_token_ids"):
        raise AttributeError(
            "TPU grammar jump decoding expects Scheduler.update_draft_token_ids; "
            "vLLM's speculative decoding interface has changed and this "
            "patch needs updating.")

    original_update = Scheduler.update_draft_token_ids
    original_update_in_output = getattr(Scheduler,
                                        "update_draft_token_ids_in_output",
                                        None)

    def update_draft_token_ids(self, draft_token_ids) -> None:
        _topup_draft_token_ids(self, draft_token_ids)
        original_update(self, draft_token_ids)
        _topup_running_requests(self)

    def update_draft_token_ids_in_output(self, draft_token_ids,
                                         scheduler_output) -> None:
        _topup_draft_token_ids(self, draft_token_ids, scheduler_output)
        if original_update_in_output is not None:
            original_update_in_output(self, draft_token_ids, scheduler_output)

    Scheduler.update_draft_token_ids = update_draft_token_ids
    if original_update_in_output is not None:
        Scheduler.update_draft_token_ids_in_output = update_draft_token_ids_in_output

    Scheduler._tpu_grammar_jump_patched = True
    logger.info("Applied TPU patch: grammar jump decoding draft top-up.")

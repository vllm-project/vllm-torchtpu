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
# Flag encoding: GRAMMAR=2, MODEL=1, EMPTY=0.
# Row invariant (see validate_draft_row_invariant): grammar and model slots may
# interleave freely; only the EMPTY slots must form a suffix.
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


def validate_draft_row_invariant(flags: list[int] | Any) -> bool:
    """Validates one row of draft flags against the hardware contract.

    The rejection kernel ORs ``is_grammar`` into ``accepted`` *before* the
    first-mismatch min-reduce, so a grammar token sitting after a model token is
    handled correctly and gets truncated if an earlier model token is rejected.
    Grammar and model slots may therefore interleave freely -- and they do:
    dense JSON such as ``,"a":1`` yields rows like ``[2, 2, 1, 2, 2]``.

    What the kernel does require is that the row has no holes: every flag is in
    ``{0, 1, 2}``, and the EMPTY slots (which carry the ``-1`` token padding)
    form a suffix. A zero in the middle would mark a live draft slot as padding.
    """
    valid_flags = {DRAFT_FLAG_EMPTY, DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR}
    seen_empty = False
    for flag in flags:
        if flag not in valid_flags:
            return False
        if flag == DRAFT_FLAG_EMPTY:
            seen_empty = True
        elif seen_empty:
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


@dataclass
class CompositeDraftPlan:
    """Represents an integrated speculative draft plan combining Pre-Jump, Model Draft, and Post-TopUp."""

    tokens: list[int]
    """[budget] Draft token IDs, padded with -1."""

    flags: list[int]
    """[budget] Per-slot draft flags (0=EMPTY, 1=MODEL, 2=GRAMMAR)."""

    num_invalid: int
    """Count of empty/padding slots (-1) at the tail of the budget."""


class CompositeDraftPlanner:
    """Universal Speculative Drafter Planner.

    Unifies every grammar/speculative combination into a single pass that emits
    one row of ``budget`` slots, each slot carrying a *provenance* flag:

    - ``DRAFT_FLAG_GRAMMAR`` (2): the grammar FSM is deterministic at this
      position, so the token is forced-accepted during verification.
    - ``DRAFT_FLAG_MODEL`` (1): an ordinary speculative guess, verified normally.
    - ``DRAFT_FLAG_EMPTY`` (0): padding (``-1``) up to the static budget K, so
      the TPU graph shape never changes.

    Provenance is a per-slot property, not a positional one. Grammar and model
    slots interleave freely -- dense JSON like ``,"a":1`` really does produce
    ``[2, 2, 1, 2, 2]``. The only structural rule is the one enforced by
    :func:`validate_draft_row_invariant`: the EMPTY slots form a suffix.

    Phases:
    1. Pre-Jump: with no model draft to absorb, walk the FSM's deterministic
       forward string straight into the row.
    2. Model absorption: trim the incoming draft at the first grammar
       violation. Flags already established for those tokens are carried over,
       and any leading run that matches the FSM's forced string at the current
       position is upgraded MODEL -> GRAMMAR (the drafter simply guessed what
       the grammar was going to force anyway).
    3. Post-TopUp: fill the remaining slots with deterministic grammar tokens.
    4. Padding: ``-1`` / flag 0 up to the budget.
    """

    @classmethod
    def plan(
        cls,
        grammar: Any,
        tokenizer: Any,
        existing_model_draft: list[int] | None = None,
        budget: int = 7,
        max_remaining_tokens: int | None = None,
        stats: JumpStats | None = None,
        existing_model_flags: list[int] | None = None,
        standalone: bool = False,
    ) -> CompositeDraftPlan:
        effective_budget = budget
        if max_remaining_tokens is not None and max_remaining_tokens >= 0:
            effective_budget = min(budget, max_remaining_tokens)

        tokens: list[int] = []
        flags: list[int] = []
        grammar_usable = (grammar is not None and tokenizer is not None
                          and not grammar.is_terminated())

        if standalone:
            existing_model_draft = None

        # Phase 1: Pre-Jump (used for standalone GaaD or cold start)
        if not existing_model_draft and grammar_usable:
            pre_jump, pre_flags = propose_jump_topup_with_flags(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
                stats=stats,
            )
            if pre_jump:
                tokens.extend(pre_jump)
                flags.extend(pre_flags)

        # Phase 2: Model Draft absorption & FSM validation
        if existing_model_draft:
            remaining_for_model = effective_budget - len(tokens)
            if remaining_for_model > 0:
                raw_draft = list(existing_model_draft[:remaining_for_model])
                if grammar is not None:
                    valid_draft = grammar.validate_tokens(raw_draft)
                else:
                    valid_draft = raw_draft
                draft_flags = cls._resolve_model_flags(
                    grammar=grammar,
                    tokenizer=tokenizer,
                    grammar_usable=grammar_usable,
                    valid_draft=valid_draft,
                    existing_model_flags=existing_model_flags,
                    effective_budget=effective_budget,
                )
                tokens.extend(valid_draft)
                flags.extend(draft_flags)

        # Phase 3: Post-TopUp (fill vacant speculative slots with grammar tokens)
        remaining_for_topup = effective_budget - len(tokens)
        if remaining_for_topup > 0 and grammar_usable:
            topup, topup_flags = propose_jump_topup_with_flags(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=tokens,
                num_spec_tokens=effective_budget,
                stats=stats,
            )
            if topup:
                tokens.extend(topup)
                flags.extend(topup_flags)

        # Phase 4: Static padding to budget K for TPU SIMD alignment
        num_valid = len(tokens)
        pad_len = max(0, budget - num_valid)
        if pad_len > 0:
            tokens.extend([-1] * pad_len)
            flags.extend([DRAFT_FLAG_EMPTY] * pad_len)

        return CompositeDraftPlan(
            tokens=tokens,
            flags=flags,
            num_invalid=pad_len,
        )

    @staticmethod
    def _resolve_model_flags(
        grammar: Any,
        tokenizer: Any,
        grammar_usable: bool,
        valid_draft: list[int],
        existing_model_flags: list[int] | None,
        effective_budget: int,
    ) -> list[int]:
        """Assigns a provenance flag to every absorbed model-draft token.

        Two things can make a slot GRAMMAR rather than MODEL:

        * It was already proven grammar-forced on an earlier pass. The live
          scheduler path re-feeds last step's plan as ``existing_model_draft``,
          so without carrying the flags forward a token would silently lose its
          forced acceptance on every round-trip.
        * The drafter independently produced the string the FSM is about to
          force anyway. Peeking the jump-forward string at the *current*
          position and matching it against the leading run of the draft costs
          one FSM query and upgrades those slots for free -- no change to the
          drafter, and none to Eagle3, whose hidden states for those positions
          do not exist yet.

        Only a *leading* run can be upgraded: the jump string describes the
        position the row opens at, so the match has to start at slot 0.
        """
        flags = [DRAFT_FLAG_MODEL] * len(valid_draft)
        if not valid_draft:
            return flags

        if existing_model_flags:
            for i, flag in enumerate(existing_model_flags[:len(valid_draft)]):
                if flag == DRAFT_FLAG_GRAMMAR:
                    flags[i] = DRAFT_FLAG_GRAMMAR

        if grammar_usable:
            # Peek only: propose_jump_topup restores the matcher before it
            # returns. stats stay untouched because these tokens were proposed
            # by the drafter, not by GaaD -- counting them would inflate
            # tokens_proposed.
            forced = propose_jump_topup(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
                stats=None,
            )
            for i in range(min(len(forced), len(valid_draft))):
                if forced[i] != valid_draft[i]:
                    break
                flags[i] = DRAFT_FLAG_GRAMMAR

        return flags


def _topup_scheduled_spec_decode_tokens(scheduler: Any,
                                        scheduler_output: Any) -> None:
    """Tops up validated async speculative draft tokens in scheduler_output using CompositeDraftPlanner."""
    if scheduler_output is None:
        return
    sched_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens",
                         None)
    if not sched_spec:
        return
    manager = getattr(scheduler, "structured_output_manager", None)
    tokenizer = getattr(manager, "tokenizer", None) if manager else None

    if getattr(scheduler_output, "scheduled_spec_decode_flags", None) is None:
        scheduler_output.scheduled_spec_decode_flags = {}
    sched_flags = scheduler_output.scheduled_spec_decode_flags
    num_invalid_map = getattr(scheduler_output, "num_invalid_spec_tokens",
                              None)
    if num_invalid_map is None:
        num_invalid_map = {}
        scheduler_output.num_invalid_spec_tokens = num_invalid_map

    for req_id, spec_tokens in sched_spec.items():
        orig_budget = len(spec_tokens)
        num_invalid = num_invalid_map.get(req_id, 0)
        valid_len = max(0, orig_budget - num_invalid)
        raw_draft = list(spec_tokens[:valid_len])

        grammar = None
        remaining = None
        request = scheduler.requests.get(req_id) if hasattr(
            scheduler, "requests") else None
        if (request is not None and not request.is_finished()
                and not getattr(request, "is_prefill_chunk", False)
                and len(request.output_token_ids) > 0):
            can_advance = (not hasattr(manager, "should_advance")
                           or manager.should_advance(request))
            structured_req = getattr(request, "structured_output_request",
                                     None)
            if can_advance and structured_req is not None:
                grammar = getattr(structured_req, "grammar", None)
                remaining = request.max_tokens - len(
                    request.output_token_ids) - 1

        # Read before the write below clobbers it: a token proven
        # grammar-forced last pass must keep its GRAMMAR provenance.
        prior_flags = sched_flags.get(req_id)

        plan = CompositeDraftPlanner.plan(
            grammar=grammar,
            tokenizer=tokenizer,
            existing_model_draft=raw_draft,
            budget=orig_budget,
            max_remaining_tokens=remaining,
            stats=_STATS,
            existing_model_flags=prior_flags,
        )

        spec_tokens[:] = plan.tokens
        sched_flags[req_id] = plan.flags
        num_invalid_map[req_id] = plan.num_invalid


def _topup_running_requests(scheduler: Any) -> None:
    """Fills each structured request's unused speculative slots using CompositeDraftPlanner."""
    if getattr(getattr(scheduler, "scheduler_config", None),
               "async_scheduling", False):
        return
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
        if (request.is_finished()
                or getattr(request, "is_prefill_chunk", False)
                or len(request.output_token_ids) == 0):
            continue
        if not hasattr(request,
                       "spec_token_ids") or request.spec_token_ids is None:
            request.spec_token_ids = []
        if hasattr(manager,
                   "should_advance") and not manager.should_advance(request):
            continue
        structured_req = getattr(request, "structured_output_request", None)
        if structured_req is None:
            continue
        grammar = getattr(structured_req, "grammar", None)
        if grammar is None:
            continue

        raw_draft = list(request.spec_token_ids)
        remaining = request.max_tokens - len(request.output_token_ids) - 1
        plan = CompositeDraftPlanner.plan(
            grammar=grammar,
            tokenizer=tokenizer,
            existing_model_draft=raw_draft,
            budget=num_spec_tokens,
            max_remaining_tokens=remaining,
            stats=_STATS,
            existing_model_flags=getattr(request, "spec_token_flags", None),
        )

        valid_count = num_spec_tokens - plan.num_invalid
        request.spec_token_ids = plan.tokens[:valid_count]
        request.spec_token_flags = plan.flags[:valid_count]

    if _STATS.calls and (_STATS.calls % 10 == 0 or _STATS.hits > 0):
        logger.info("TPU grammar jump decoding: %s", _STATS.summary())


def _patch_vllm_grammar_jump_decoding() -> None:
    """Wraps ``Scheduler.update_draft_token_ids``, ``Scheduler.update_draft_token_ids_in_output``,
    and ``Scheduler.schedule`` with the grammar jump draft top-up and flag propagation.
    """
    from vllm_torchtpu import envs
    if not envs.VLLM_TPU_JUMP_DECODING:
        return

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    if not hasattr(SchedulerOutput, "scheduled_spec_decode_flags"):
        SchedulerOutput.scheduled_spec_decode_flags = None

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
    original_schedule = getattr(Scheduler, "schedule", None)

    def update_draft_token_ids(self, draft_token_ids, *args, **kwargs) -> None:
        original_update(self, draft_token_ids, *args, **kwargs)
        _topup_running_requests(self)

    def update_draft_token_ids_in_output(self, draft_token_ids,
                                         scheduler_output, *args,
                                         **kwargs) -> None:
        if original_update_in_output is not None:
            original_update_in_output(self, draft_token_ids, scheduler_output,
                                      *args, **kwargs)
        _topup_scheduled_spec_decode_tokens(self, scheduler_output)

    def schedule(self, *args, **kwargs):
        req_flags = {}
        for r in getattr(self, "running", ()):
            flags = getattr(r, "spec_token_flags", None)
            if flags:
                req_flags[r.request_id] = list(flags)
                r.spec_token_flags = None
        out = original_schedule(self, *args, **kwargs)
        if req_flags and hasattr(out, "scheduled_spec_decode_tokens"):
            sched_flags = {}
            for req_id, spec_tokens in out.scheduled_spec_decode_tokens.items(
            ):
                if req_id in req_flags:
                    f = req_flags[req_id][:len(spec_tokens)]
                    if len(f) < len(spec_tokens):
                        f.extend([DRAFT_FLAG_MODEL] *
                                 (len(spec_tokens) - len(f)))
                    sched_flags[req_id] = f
            if sched_flags:
                out.scheduled_spec_decode_flags = sched_flags
        return out

    Scheduler.update_draft_token_ids = update_draft_token_ids
    if original_update_in_output is not None:
        Scheduler.update_draft_token_ids_in_output = update_draft_token_ids_in_output
    if original_schedule is not None:
        Scheduler.schedule = schedule

    Scheduler._tpu_grammar_jump_patched = True
    logger.info("Applied TPU patch: grammar jump decoding draft top-up.")

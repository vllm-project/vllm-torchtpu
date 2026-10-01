# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar-as-a-Drafter (GaaD) speculative decoding subsystem.

Encapsulates all GaaD logic in a single self-contained module:
1. Constants & Flags (EMPTY=0, MODEL=1, GRAMMAR=2) + validate_draft_row_invariant()
2. FSMBridge (deterministic lookahead byte extraction & BPE boundary validation)
3. GaadProposer (CPU/scheduler-native proposer aligned with NgramProposer)
4. Scheduler Top-Up Seam (Lifecycle Hook)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger("vllm." + __name__)

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
    hits: int = 0
    tokens_proposed: int = 0
    truncations: int = 0
    fsm_rejections: int = 0

    def summary(self) -> str:
        hit_rate = (self.hits / self.calls) if self.calls else 0.0
        per_hit = (self.tokens_proposed / self.hits) if self.hits else 0.0
        return (
            f"calls={self.calls} hits={self.hits} "
            f"hit_rate={hit_rate:.3f} tokens={self.tokens_proposed} "
            f"tokens_per_hit={per_hit:.2f} "
            f"truncations={self.truncations} "
            f"fsm_rejections={self.fsm_rejections}"
        )


_STATS = JumpStats()


def get_stats() -> JumpStats:
    """Returns the process-local counters."""
    return _STATS


def validate_draft_row_invariant(flags: list[int] | Any) -> bool:
    """Validates one row of draft flags against the hardware contract.

    The rejection kernel ORs ``is_grammar`` into ``accepted`` *before* the
    first-mismatch min-reduce, so a grammar token sitting after a model token is
    handled correctly and gets truncated if an earlier model token is rejected.
    Grammar and model slots may therefore interleave freely.

    What the kernel does require is that the row has no holes: every flag is in
    ``{0, 1, 2}``, and the EMPTY slots (which carry the ``-1`` token padding)
    form a suffix.
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


class FSMBridge:
    """Bridge to extract deterministic lookahead tokens from grammar FSM."""

    def __init__(self, stats: JumpStats | None = None):
        self.stats = stats

    def extract_deterministic_jump(
        self,
        grammar: Any,
        tokenizer: Any,
        validated_draft: list[int] | None = None,
        num_spec_tokens: int = 0,
    ) -> tuple[list[int], list[int]]:
        """Extracts deterministic jump tokens and flags, validating token boundary.

        The FSM is walked past ``validated_draft`` first, because the jump string
        has to describe the position the returned tokens will actually occupy.
        The matcher is restored to its original state before returning (peek semantics).
        """
        if validated_draft is None:
            validated_draft = []
        budget = num_spec_tokens - len(validated_draft)
        if budget <= 0:
            return [], []
        if grammar is None or grammar.is_terminated():
            return [], []

        matcher = getattr(grammar, "matcher", None)
        if matcher is None:
            return [], []

        if self.stats is not None:
            self.stats.calls += 1

        advanced = 0
        try:
            for token in validated_draft:
                if not matcher.accept_token(token):
                    return [], []
                advanced += 1

            jump_str = matcher.find_jump_forward_string()
            if not jump_str:
                return [], []

            candidate = tokenizer.encode(jump_str, add_special_tokens=False)
            if not candidate:
                return [], []
            if len(candidate) > budget:
                candidate = candidate[:budget]
                if self.stats is not None:
                    self.stats.truncations += 1

            # Validate token boundaries by accepting token by token into FSM
            jumped = 0
            for token in candidate:
                if not matcher.accept_token(token):
                    break
                jumped += 1
            advanced += jumped

            if self.stats is not None:
                if jumped < len(candidate):
                    self.stats.fsm_rejections += 1
                if jumped:
                    self.stats.hits += 1
                    self.stats.tokens_proposed += jumped

            tokens = candidate[:jumped]
            flags = [DRAFT_FLAG_GRAMMAR] * jumped
            return tokens, flags
        finally:
            if advanced:
                matcher.rollback(advanced)


@dataclass
class CompositeDraftPlan:
    """Integrated speculative draft plan combining Pre-Jump, Model Draft,
    and Post-TopUp.
    """

    tokens: list[int]
    flags: list[int]
    num_invalid: int


class CompositeDraftPlanner:
    """Universal Speculative Drafter Planner."""

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
        grammar_usable = (
            grammar is not None
            and tokenizer is not None
            and not grammar.is_terminated()
        )
        bridge = FSMBridge(stats=stats)

        if standalone:
            existing_model_draft = None

        # Phase 1: Pre-Jump
        if not existing_model_draft and grammar_usable:
            pre_jump, pre_flags = bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
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

        # Phase 3: Post-TopUp
        remaining_for_topup = effective_budget - len(tokens)
        if remaining_for_topup > 0 and grammar_usable:
            topup, topup_flags = bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=tokens,
                num_spec_tokens=effective_budget,
            )
            if topup:
                tokens.extend(topup)
                flags.extend(topup_flags)

        # Phase 4: Static padding to budget K
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
        flags = [DRAFT_FLAG_MODEL] * len(valid_draft)
        if not valid_draft:
            return flags

        if existing_model_flags:
            for i, flag in enumerate(existing_model_flags[: len(valid_draft)]):
                if flag == DRAFT_FLAG_GRAMMAR:
                    flags[i] = DRAFT_FLAG_GRAMMAR

        if grammar_usable:
            bridge = FSMBridge()
            forced, _ = bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
            )
            for i in range(min(len(forced), len(valid_draft))):
                if forced[i] != valid_draft[i]:
                    break
                flags[i] = DRAFT_FLAG_GRAMMAR

        return flags


class GaadProposer:
    """Grammar-as-a-Drafter (GaaD) proposer running in Engine-Core/CPU.

    Architecturally aligned with NgramProposer:
    1. Zero neural weights / zero accelerator forward passes.
    2. Operates entirely on CPU/Host within the scheduler loop.
    3. Traverses FSM deterministic lookaheads to emit jump draft tokens.
    """

    def __init__(self, vllm_config: VllmConfig):
        spec_config = getattr(vllm_config, "speculative_config", None)
        self.k = getattr(spec_config, "num_speculative_tokens", 0) if spec_config else 0
        self.bridge = FSMBridge(stats=_STATS)

    def propose(
        self,
        requests: list[Any],
        tokenizer: Any,
    ) -> list[list[int]]:
        """Proposes deterministic jump draft tokens for structured requests.

        Returns:
            list[list[int]]: Per-request draft token IDs (up to self.k).
            Returns [] for unconstrained requests or when FSM has no jump.
        """
        draft_token_ids: list[list[int]] = []
        for req in requests:
            structured_req = getattr(req, "structured_output_request", None)
            grammar = (
                getattr(structured_req, "grammar", None) if structured_req else None
            )
            if (
                req.is_finished()
                or getattr(req, "is_prefill_chunk", False)
                or len(getattr(req, "output_token_ids", ())) == 0
                or grammar is None
            ):
                draft_token_ids.append([])
                continue

            remaining = req.max_tokens - len(req.output_token_ids) - 1
            effective_budget = min(self.k, remaining) if remaining >= 0 else self.k
            if effective_budget <= 0:
                draft_token_ids.append([])
                continue

            tokens, _ = self.bridge.extract_deterministic_jump(
                grammar=grammar,
                tokenizer=tokenizer,
                validated_draft=[],
                num_spec_tokens=effective_budget,
            )
            draft_token_ids.append(tokens)

        return draft_token_ids


def topup_scheduled_spec_decode_tokens(scheduler: Any, scheduler_output: Any) -> None:
    """Tops up validated async speculative draft tokens in scheduler_output."""
    if scheduler_output is None:
        return
    sched_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
    if not sched_spec:
        return
    manager = getattr(scheduler, "structured_output_manager", None)
    tokenizer = getattr(manager, "tokenizer", None) if manager else None

    if getattr(scheduler_output, "scheduled_spec_decode_flags", None) is None:
        scheduler_output.scheduled_spec_decode_flags = {}
    sched_flags = scheduler_output.scheduled_spec_decode_flags
    num_invalid_map = getattr(scheduler_output, "num_invalid_spec_tokens", None)
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
        request = (
            scheduler.requests.get(req_id) if hasattr(scheduler, "requests") else None
        )

        if (
            request is not None
            and not request.is_finished()
            and not getattr(request, "is_prefill_chunk", False)
            and len(request.output_token_ids) > 0
        ):
            can_advance = not hasattr(
                manager, "should_advance"
            ) or manager.should_advance(request)
            structured_req = getattr(request, "structured_output_request", None)
            if can_advance and structured_req is not None:
                grammar = getattr(structured_req, "grammar", None)
                remaining = request.max_tokens - len(request.output_token_ids) - 1

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


def topup_running_requests(scheduler: Any) -> None:
    """Fills each structured request's unused speculative slots."""
    if getattr(getattr(scheduler, "scheduler_config", None), "async_scheduling", False):
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
        if (
            request.is_finished()
            or getattr(request, "is_prefill_chunk", False)
            or len(request.output_token_ids) == 0
        ):
            continue
        if not hasattr(request, "spec_token_ids") or request.spec_token_ids is None:
            request.spec_token_ids = []
        if hasattr(manager, "should_advance") and not manager.should_advance(request):
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


def patch_vllm_grammar_jump_decoding() -> None:
    """Wraps Scheduler draft update and scheduling with GaaD top-up and flags."""
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
            "patch needs updating."
        )

    original_update = Scheduler.update_draft_token_ids
    original_update_in_output = getattr(
        Scheduler, "update_draft_token_ids_in_output", None
    )
    original_schedule = getattr(Scheduler, "schedule", None)

    def update_draft_token_ids(self, draft_token_ids, *args, **kwargs) -> None:
        original_update(self, draft_token_ids, *args, **kwargs)
        topup_running_requests(self)

    def update_draft_token_ids_in_output(
        self, draft_token_ids, scheduler_output, *args, **kwargs
    ) -> None:
        if original_update_in_output is not None:
            original_update_in_output(
                self, draft_token_ids, scheduler_output, *args, **kwargs
            )
        topup_scheduled_spec_decode_tokens(self, scheduler_output)

    Scheduler.update_draft_token_ids = update_draft_token_ids
    if original_update_in_output is not None:
        Scheduler.update_draft_token_ids_in_output = update_draft_token_ids_in_output

    if original_schedule is not None:

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
                for req_id, spec_tokens in out.scheduled_spec_decode_tokens.items():
                    if req_id in req_flags:
                        f = req_flags[req_id][: len(spec_tokens)]
                        if len(f) < len(spec_tokens):
                            f.extend([DRAFT_FLAG_MODEL] * (len(spec_tokens) - len(f)))
                        sched_flags[req_id] = f
                if sched_flags:
                    out.scheduled_spec_decode_flags = sched_flags
            return out

        Scheduler.schedule = schedule

    Scheduler._tpu_grammar_jump_patched = True
    logger.info("Applied TPU patch: grammar jump decoding draft top-up.")


def patch_vllm_speculative_config_gaad() -> None:
    """Allows SpeculativeConfig to accept method='gaad' without raising ValueError."""
    try:
        from vllm.config.speculative import SpeculativeConfig
    except (ImportError, ModuleNotFoundError):
        return

    if getattr(SpeculativeConfig, "_tpu_gaad_patched", False):
        return

    original_init = SpeculativeConfig.__init__

    def patched_init(self, *args, **kwargs):
        method = kwargs.get("method")
        if method == "gaad":
            kwargs["method"] = "ngram"
            original_init(self, *args, **kwargs)
            self.method = "gaad"
        else:
            original_init(self, *args, **kwargs)

    SpeculativeConfig.__init__ = patched_init
    SpeculativeConfig._tpu_gaad_patched = True
    logger.info("Applied TPU patch: SpeculativeConfig supports method='gaad'.")

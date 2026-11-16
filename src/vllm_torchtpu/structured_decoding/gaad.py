# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Grammar-as-a-Draft (GaaD): grammar-forced tokens in speculative draft rows.

Where a structured-output request's grammar admits exactly one continuation
string, the tokens of that string are known before the model runs. GaaD plans
them into the request's draft row with flag ``GRAMMAR``; the rejection sampler
accepts a ``GRAMMAR`` slot without comparing it with the target, and a rejected
``MODEL`` slot ahead of it still rejects it, as it rejects every later slot.

Each row is planned in ``Scheduler.schedule()``, before the step is dispatched,
from the grammar state after the request's last token and from the drafter's
proposal for this step. With async scheduling both are on the host only once
the step in flight is committed, so while a running request is one GaaD plans
rows for, each step runs synchronously: the engine first commits the step in
flight (``patch_engine_core``), and the runner reads the scheduler's tokens and
drafts instead of the device's. Every other step stays async; a batch without
structured output never waits.

GaaD is on when ``--additional-config`` has a ``gaad`` entry, for example
``{"gaad": {"draft_tokens": 3}}``. A row has ``num_speculative_tokens`` slots.
The model drafter fills at most ``draft_tokens`` of them (all of them when
unset) and the grammar may force tokens into the rest, so a wide row costs the
drafter nothing. The runner computes target logits only for ``MODEL`` slots and
the bonus slot. Without a drafter, the speculative config names
``GAAD_PROPOSER`` as a ``custom_class`` method and GaaD fills every row.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig

logger = init_logger(__name__)

DRAFT_FLAG_EMPTY = 0
DRAFT_FLAG_MODEL = 1
DRAFT_FLAG_GRAMMAR = 2

# The `custom_class` speculative model that selects GaaD without a drafter.
GAAD_PROPOSER = "vllm_torchtpu.structured_decoding.gaad.GrammarOnlyProposer"


class GrammarOnlyProposer:
    """Names GaaD without a drafter; it proposes nothing.

    vLLM's runner base class builds one from the `custom_class` config; the
    TPU runner then replaces it with no drafter.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def propose(self, *args: Any, **kwargs: Any) -> list[list[int]]:
        del args, kwargs
        return []


def is_standalone(speculative_config: SpeculativeConfig | None) -> bool:
    """Whether the speculative config selects GaaD without a drafter."""
    return (
        speculative_config is not None
        and speculative_config.method == "custom_class"
        and speculative_config.model == GAAD_PROPOSER
    )


def _options(vllm_config: VllmConfig) -> Any:
    """The ``gaad`` entry of ``additional_config``; None when GaaD is off."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return None
    return additional_config.get("gaad")


def is_enabled(vllm_config: VllmConfig) -> bool:
    """Whether ``additional_config`` turns GaaD on."""
    return _options(vllm_config) is not None


def draft_limit(vllm_config: VllmConfig) -> int:
    """How many slots of a row the model drafter fills."""
    assert vllm_config.speculative_config is not None
    k = vllm_config.speculative_config.num_speculative_tokens
    options = _options(vllm_config)
    if options is None:
        return k
    return options.get("draft_tokens", k)


def check_config(vllm_config: VllmConfig) -> None:
    """Validates a GaaD configuration.

    Async scheduling stays on with a drafter that drafts on the device; GaaD
    makes only its own steps synchronous (``patch_engine_core``). Without one
    (GaaD alone, or a host drafter such as ngram) the runner has no device
    drafts to feed async steps, so async scheduling is turned off.
    """
    speculative_config = vllm_config.speculative_config
    options = _options(vllm_config)
    if options is None:
        if is_standalone(speculative_config):
            raise ValueError(
                f"Speculative model {GAAD_PROPOSER} needs a gaad entry in "
                "additional_config, e.g. --additional-config '{\"gaad\": {}}'."
            )
        return
    if not isinstance(options, dict):
        raise ValueError(f'additional_config["gaad"] must be a dict, got {options!r}.')
    unknown = sorted(set(options) - {"draft_tokens"})
    if unknown:
        raise ValueError(
            f'Unknown additional_config["gaad"] keys {unknown}; the only key '
            "is draft_tokens."
        )
    if speculative_config is None:
        raise ValueError(
            'additional_config["gaad"] needs a speculative config: a drafter, '
            f'or {{"model": "{GAAD_PROPOSER}", "num_speculative_tokens": K}}.'
        )
    k = speculative_config.num_speculative_tokens
    draft_tokens = options.get("draft_tokens", k)
    if (
        isinstance(draft_tokens, bool)
        or not isinstance(draft_tokens, int)
        or not 1 <= draft_tokens <= k
    ):
        raise ValueError(
            'additional_config["gaad"]["draft_tokens"] must be an integer from '
            f"1 to num_speculative_tokens ({k}), got {draft_tokens!r}."
        )
    scheduler_config = vllm_config.scheduler_config
    if scheduler_config.async_scheduling and not speculative_config.use_eagle():
        logger.warning(
            "GaaD without a drafter that drafts on the device runs every step "
            "synchronously; turning async scheduling off."
        )
        scheduler_config.async_scheduling = False


class GrammarCursor:
    """Walks a grammar matcher along a draft row; ``rollback`` restores it.

    The forced token is the next token of the canonical encoding of the
    matcher's jump-forward string. The encoding is used whole or not at all: a
    jump whose encoding the grammar rejects anywhere forces nothing. It is
    encoded once per jump run and looked up again after a token off the run.
    """

    def __init__(self, matcher: Any, tokenizer: Any):
        self._matcher = matcher
        self._tokenizer = tokenizer
        self._advanced = 0
        # The rest of the current jump run, or None before it is looked up.
        self._run: list[int] | None = None

    def forced_next(self) -> int | None:
        """The only token the grammar lets the row continue with, or None."""
        if self._run is None:
            self._run = self._jump_tokens()
        return self._run[0] if self._run else None

    def advance(self, token: int) -> bool:
        """Advances past ``token`` if the grammar allows it."""
        if not self._matcher.accept_token(token):
            return False
        self._advanced += 1
        if self._run and self._run[0] == token:
            # An exhausted run is looked up again: the grammar may force more.
            self._run = self._run[1:] or None
        else:
            self._run = None
        return True

    def rollback(self) -> None:
        if self._advanced:
            self._matcher.rollback(self._advanced)
        self._advanced = 0
        self._run = None

    def _jump_tokens(self) -> list[int]:
        if self._matcher.is_terminated():
            return []
        jump = self._matcher.find_jump_forward_string()
        if not jump:
            return []
        tokens = list(self._tokenizer.encode(jump, add_special_tokens=False))
        probe = self._matcher.fork()
        if not all(probe.accept_token(token) for token in tokens):
            return []
        return tokens


def plan_draft_row(
    model_draft: Sequence[int], cursor: GrammarCursor, k: int
) -> tuple[list[int], list[int]]:
    """Merges ``model_draft`` with the tokens the grammar forces, ``k`` at most.

    1. A model token that is the forced token becomes a ``GRAMMAR`` slot; one
       the grammar allows where it forces nothing stays a ``MODEL`` slot.
    2. The model draft is cut at its first token the grammar rejects or that
       differs from the forced token: its later tokens continue from a
       tokenization the forced path does not take.
    3. Forced tokens fill the remaining slots while the grammar forces one.

    Advances ``cursor`` past every planned token. The row has no ``EMPTY``
    padding: the scheduler schedules exactly the planned tokens.
    """
    tokens: list[int] = []
    flags: list[int] = []
    for token in model_draft[:k]:
        forced = cursor.forced_next()
        if (forced is not None and forced != token) or not cursor.advance(token):
            break
        tokens.append(token)
        flags.append(DRAFT_FLAG_GRAMMAR if forced == token else DRAFT_FLAG_MODEL)
    while len(tokens) < k:
        forced = cursor.forced_next()
        if forced is None or not cursor.advance(forced):
            break
        tokens.append(forced)
        flags.append(DRAFT_FLAG_GRAMMAR)
    return tokens, flags


def _has_penalties(sampling_params: Any) -> bool:
    return (
        sampling_params.presence_penalty != 0.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
    )


def _grammar_cursor(
    request: Any, structured_output_manager: Any, k: int
) -> tuple[GrammarCursor, int] | None:
    """The cursor at ``request``'s committed grammar state and its row budget.

    None for a request GaaD skips: one still prefilling, whose grammar does
    not constrain its next token (it has none, it is not xgrammar, or
    reasoning has not ended), that reports logprobs or applies penalties (a
    ``GRAMMAR`` slot has no target logits row to report or penalize), or that
    has no room left for a draft.
    """
    if request.is_prefill_chunk:
        return None
    structured = request.structured_output_request
    if structured is None:
        return None
    # Reasoning tokens before the grammar starts are unconstrained.
    bounds = structured_output_manager._get_constraint_bounds(request, ())
    if bounds.constraint_start != 0:
        return None
    grammar = structured.grammar
    matcher = getattr(grammar, "matcher", None)
    if matcher is None or grammar.is_terminated():
        return None
    params = request.sampling_params
    if params is None or params.logprobs is not None or _has_penalties(params):
        return None
    budget = min(k, request.max_tokens - request.num_output_tokens - 1)
    if budget <= 0:
        return None
    return GrammarCursor(matcher, structured_output_manager.tokenizer), budget


def plan_request(
    request: Any, structured_output_manager: Any, k: int, limit: int | None = None
) -> list[int] | None:
    """Replaces ``request.spec_token_ids`` with its GaaD row; returns its flags.

    The row keeps at most ``limit`` of the drafter's tokens (all when None).

    Returns None, leaving the drafts as they are, for a request GaaD skips
    (``_grammar_cursor``) and for one with tokens in flight, whose grammar
    state lags its tokens.
    """
    if request.num_output_placeholders:
        return None
    start = _grammar_cursor(request, structured_output_manager, k)
    if start is None:
        return None
    cursor, budget = start
    model_draft = request.spec_token_ids[:limit]
    try:
        tokens, flags = plan_draft_row(model_draft, cursor, budget)
    finally:
        cursor.rollback()
    request.spec_token_ids = tokens
    return flags


def is_planned(request: Any, structured_output_manager: Any, k: int) -> bool:
    """Whether GaaD plans ``request``'s rows, its tokens in flight aside."""
    return _grammar_cursor(request, structured_output_manager, k) is not None


def has_grammar_drafts(scheduler_output: Any) -> bool:
    """Whether a scheduled step carries a ``GRAMMAR`` draft."""
    flags = getattr(scheduler_output, "scheduled_spec_decode_flags", None) or {}
    return any(DRAFT_FLAG_GRAMMAR in row for row in flags.values())


def follows_commit(scheduler_output: Any) -> bool:
    """Whether a step was scheduled with no step in flight to substitute from.

    So is a step the engine scheduled right after committing the step in
    flight (``patch_engine_core``): the scheduler's counts are exact, not
    ahead by the drafts of a step in flight, and its drafts are on the host.
    A step with grammar drafts always is one.
    """
    return bool(
        getattr(scheduler_output, "gaad_follows_commit", False)
    ) or has_grammar_drafts(scheduler_output)


def needs_sync_step(scheduler: Any) -> bool:
    """Whether the step in flight is committed before the next is scheduled.

    As in rvllm, per batch: it is while a running request is one GaaD plans
    rows for. Its next row may carry grammar drafts as soon as the grammar
    forces a token, which can start within the step in flight; judging it on
    the state before those tokens would leave the first step of each forced
    run without grammar drafts, and a JSON-like grammar starts one after
    nearly every value. A batch without such a request stays async.
    """
    manager = scheduler.structured_output_manager
    k = scheduler.num_spec_tokens
    return any(is_planned(request, manager, k) for request in scheduler.running)


def plan_running_requests(scheduler: Any) -> dict[str, list[int]]:
    """Plans every running request's draft row; returns flags by request id."""
    manager = scheduler.structured_output_manager
    k = scheduler.num_spec_tokens
    flags_by_req: dict[str, list[int]] = {}
    limit = draft_limit(scheduler.vllm_config)
    for request in scheduler.running:
        flags = plan_request(request, manager, k, limit)
        if flags is not None:
            flags_by_req[request.request_id] = flags
        elif len(request.spec_token_ids) > limit:
            # A request GaaD skips keeps only the drafter's share of the row.
            request.spec_token_ids = request.spec_token_ids[:limit]
    return flags_by_req


def scheduled_flags(
    flags_by_req: dict[str, list[int]],
    scheduled_spec_decode_tokens: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Trims each planned row's flags to the drafts the scheduler scheduled."""
    return {
        req_id: flags_by_req[req_id][: len(tokens)]
        for req_id, tokens in scheduled_spec_decode_tokens.items()
        if req_id in flags_by_req
    }


def patch_scheduler() -> None:
    """Plans GaaD rows in ``Scheduler.schedule`` in the engine-core process.

    The scheduler sends the flags of each scheduled row to the runner as
    ``SchedulerOutput.scheduled_spec_decode_flags``; a request without flags
    has only ``MODEL`` drafts. ``SchedulerOutput.gaad_follows_commit`` marks
    the first step after the engine committed the step in flight. Whether GaaD
    is on comes from each scheduler's config, so the patch is installed always
    and does nothing when it is off.
    """
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_gaad_patched", False):
        return
    SchedulerOutput.scheduled_spec_decode_flags = None
    SchedulerOutput.gaad_follows_commit = False
    original_schedule = Scheduler.schedule

    def schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_enabled(self.vllm_config):
            return original_schedule(self, *args, **kwargs)
        flags_by_req = plan_running_requests(self)
        # Any: vLLM's SchedulerOutput does not declare the fields added
        # above; the runner reads them with getattr.
        output: Any = original_schedule(self, *args, **kwargs)
        if flags_by_req:
            output.scheduled_spec_decode_flags = scheduled_flags(
                flags_by_req, output.scheduled_spec_decode_tokens
            )
        # The first step the runner executes after a commit of the engine's.
        if output.total_num_scheduled_tokens and getattr(
            self, "_tpu_gaad_committed", False
        ):
            output.gaad_follows_commit = True
            self._tpu_gaad_committed = False
        return output

    Scheduler.schedule = schedule
    Scheduler._tpu_gaad_patched = True
    logger.info("Applied TPU patch: GaaD draft rows in Scheduler.schedule.")


def patch_engine_core() -> None:
    """Commits the step in flight before a step GaaD may plan grammar drafts for.

    The engine then schedules the next step with nothing in flight, so GaaD
    plans it from the committed grammar state and the drafter's host drafts,
    as with async scheduling off. Async scheduling only: without it every step
    is committed before the next anyway. Whether GaaD is on comes from each
    engine's config, so the patch is installed always and does nothing when it
    is off.
    """
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_gaad_patched", False):
        return
    original_step = EngineCore.step_with_batch_queue

    def step_with_batch_queue(self: Any) -> Any:
        queue = self.batch_queue
        # The queue holds at most the one step in flight here.
        if (
            not queue
            or not is_enabled(self.vllm_config)
            or not needs_sync_step(self.scheduler)
        ):
            return original_step(self)
        in_flight = queue[-1][1]
        scheduler = self.scheduler
        # With nothing to schedule, the original only waits for the step in
        # flight and commits it.
        scheduler.has_requests = lambda: False
        try:
            outputs, _ = original_step(self)
        finally:
            del scheduler.has_requests
        scheduler._tpu_gaad_committed = True
        # The runner keeps a host copy of a structured step's drafts.
        if self.check_for_draft_tokens and in_flight.has_structured_output_requests:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                scheduler.update_draft_token_ids(draft_token_ids)
        # The step ran a model; False would make the engine loop sleep.
        return outputs, in_flight.total_num_scheduled_tokens > 0

    EngineCore.step_with_batch_queue = step_with_batch_queue
    EngineCore._tpu_gaad_patched = True
    logger.info("Applied TPU patch: GaaD synchronous steps in EngineCore.")

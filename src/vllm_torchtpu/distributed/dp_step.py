# SPDX-License-Identifier: Apache-2.0
"""EP-DP lockstep shapes, agreed two steps ahead by a background all-reduce.

Under expert parallelism the DP ranks reduce partial expert outputs across one
another, so every rank must enter a step's collectives with the same shapes:
the token bucket, the number of forward chunks, the per-chunk request count
that sizes the draft forward, and the draft and encoder flags. Each rank's
scheduler decides its own work, so the ranks agree these on vLLM's CPU DP
group. A blocking all-reduce costs a host-bound decode step 1.45 ms for 16
ranks on two hosts, plus the wait for the slowest rank, on every step.

So no step waits on its own agreement. After step k's last launch every rank
contributes a bound on its step k+2 (`StepAgreement.end`) to an asynchronous
all-reduce, which completes while the host runs step k+1; step k+2 reads it
before its first launch and runs at the agreed bound once a local check
confirms that its work fits (`begin`). A rank that cannot bound that step says
so, and the step agrees its exact shape before its launch: it is the one step
with a blocking all-reduce. The ranks' hosts drift at most two steps apart.

The bound comes from the rank's scheduler, which alone knows its next steps
(`gated_schedule`). Once every running request has finished its prompt, vLLM
schedules each at most 1 + its draft count, at most 1 + num_spec_tokens,
whatever the drafts' acceptance. A finished request only shrinks a later
step, and its rows become padding. New work is what breaks the bound, so the
scheduler announces it two schedules ahead: it admits waiting requests only on
a step that neither of the two previous schedules bounded, and holds a running
request rolled back to a prefill between schedules (a failed KV load) on a
bounded step as well. A new request therefore enters at most two steps after
it becomes admissible, on a step whose shape is agreed before its launch.

Two premises of vLLM's DP engine loop carry the bound. Each step's bound is
the rank's latest schedule's (`_dp_next_bound`): a step that schedules
nothing, or an idle engine iteration that runs no schedule, publishes the
bound of the schedule before it, whose admissions the gate history still
holds. And an engine that empties its last request schedules once more with
nothing running, so the bound an idle rank publishes is the empty one.
"""

import collections
import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch

# Steps between a step's closing all-reduce and the step whose shape it agrees.
AHEAD = 2

# Set by `gated_schedule` on every SchedulerOutput: per running request, the
# most tokens either of the next AHEAD schedules gives it; None when one of
# those steps may carry new work (an admission, a prompt chunk).
NEXT_STEP_TOKENS = "tpu_dp_next_step_tokens"

# Forward-context key (`additional_kwargs`): the token rows every DP rank runs
# for each stage of this forward, agreed ahead. Absent on a step whose
# shape was agreed before its launch, where a stage that narrows its rows must
# agree its own count.
AGREED_ROWS = "tpu_dp_agreed_rows"


@dataclasses.dataclass(frozen=True)
class StepShape:
    """The collective shape of one lockstep step.

    A reduced shape holds the maximum over ranks of each field, and the minimum
    over ranks in `min_tokens` and `min_reqs`; a rank's own shape holds its own
    counts in both.
    """

    tokens: int
    chunks: int
    reqs: int
    encoder: bool
    proposals: bool
    context: bool
    min_tokens: int
    min_reqs: int

    @classmethod
    def local(
        cls,
        tokens: int,
        chunks: int,
        reqs: int,
        *,
        encoder: bool = False,
        proposals: bool = False,
        context: bool = False,
    ) -> "StepShape":
        """One rank's shape."""
        return cls(tokens, chunks, reqs, encoder, proposals, context, tokens, reqs)

    def fits(self, agreed: "StepShape") -> bool:
        """Whether this rank's step runs inside the `agreed` collectives."""
        return (
            self.tokens <= agreed.tokens
            and self.chunks <= agreed.chunks
            and self.reqs <= agreed.reqs
            and self.encoder <= agreed.encoder
            and self.proposals <= agreed.proposals
            and self.context <= agreed.context
        )

    def encode(self) -> list[int]:
        """The MAX all-reduce operand; minima ride as negated maxima."""
        return [
            self.tokens,
            self.chunks,
            self.reqs,
            int(self.encoder),
            int(self.proposals),
            int(self.context),
            -self.min_tokens,
            -self.min_reqs,
        ]

    @classmethod
    def decode(cls, values: Sequence[int]) -> "StepShape":
        """The shape `encode` reduced over the ranks."""
        tokens, chunks, reqs, encoder, proposals, context, neg_tokens, neg_reqs = (
            int(v) for v in values
        )
        return cls(
            tokens,
            chunks,
            reqs,
            bool(encoder),
            bool(proposals),
            bool(context),
            -neg_tokens,
            -neg_reqs,
        )


_IDLE = StepShape.local(0, 0, 0)
_WIDTH = len(_IDLE.encode())


@dataclasses.dataclass(frozen=True)
class _Reduction:
    """An all-reduce in flight; Gloo reads and writes `operand` until done."""

    operand: torch.Tensor
    work: Any

    def wait(self) -> list[int]:
        self.work.wait()
        return self.operand.tolist()


class StepAgreement:
    """One rank's side of the agreement AHEAD steps ahead.

    Every step calls `begin` before its first launch and `end` after its last,
    in the same order on every rank.
    """

    def __init__(self, group: Any) -> None:
        self._group = group
        # The closing all-reduces of the last AHEAD steps, oldest first.
        self._in_flight: collections.deque[_Reduction] = collections.deque()
        # Whether the last `begin` agreed its shape before launch.
        self.synchronous = True
        # The expert re-layout fields of the reduction the last `begin` read.
        self.relayout: tuple[int, ...] = (0, 0, 0)

    def begin(self, local: StepShape) -> StepShape:
        """This step's DP-wide shape.

        Raises:
            RuntimeError: this rank's step exceeds the shape agreed AHEAD
                steps earlier; its collectives would not pair with its peers'.
        """
        agreed = None
        if len(self._in_flight) == AHEAD:
            values = self._in_flight.popleft().wait()
            self.relayout = tuple(values[1 + _WIDTH :])
            if not values[0]:
                agreed = StepShape.decode(values[1 : 1 + _WIDTH])
        self.synchronous = agreed is None
        if agreed is None:
            return StepShape.decode(self._all_reduce(local.encode()).wait())
        if not local.fits(agreed):
            raise RuntimeError(
                f"DP step {local} exceeds the shape agreed {AHEAD} steps "
                f"earlier {agreed}"
            )
        return agreed

    def end(self, bound: StepShape | None, relayout: Sequence[int]) -> None:
        """Start agreeing the step AHEAD steps on from every rank's `bound`
        (None: unbounded); the all-reduce runs while the host goes on."""
        values = [int(bound is None), *(bound or _IDLE).encode(), *relayout]
        self._in_flight.append(self._all_reduce(values))

    def _all_reduce(self, values: list[int]) -> _Reduction:
        operand = torch.tensor(values, dtype=torch.int64)
        work = torch.distributed.all_reduce(
            operand,
            op=torch.distributed.ReduceOp.MAX,
            group=self._group,
            async_op=True,
        )
        return _Reduction(operand, work)


def next_step_bound(
    tokens: Sequence[int] | None,
    max_tokens: int,
    count_chunks: Callable[[Sequence[int]], tuple[int, int]],
) -> StepShape | None:
    """A rank's bound on its next step from its scheduler's `NEXT_STEP_TOKENS`.

    Args:
        tokens: the scheduler's per-request token bounds, or None.
        max_tokens: the largest token bucket; the scheduler's token budget
            never exceeds it.
        count_chunks: (forward chunks, largest chunk's request count) for a
            batch of per-request token counts.
    """
    if tokens is None:
        return None
    chunks, reqs = count_chunks(tokens)
    busy = bool(tokens)
    return StepShape.local(
        min(sum(tokens), max_tokens), chunks, reqs, proposals=busy, context=busy
    )


def lockstep_enabled(parallel_config: Any) -> bool:
    """Whether DP ranks pair their expert-parallel collectives every step."""
    from vllm_torchtpu import utils

    return (
        utils.get_dp_size(parallel_config) > 1
        and parallel_config.enable_expert_parallel
    )


def _next_step_tokens(scheduler: Any) -> list[int] | None:
    """Per running request, the most tokens a later schedule gives it.

    vLLM schedules a running request num_tokens_with_spec +
    num_output_placeholders - num_computed_tokens tokens (before budget
    clamps); a decoding request keeps that at 1 + its draft count, at most
    1 + num_spec_tokens. None when a request is in its prompt or was rolled
    back further: its next chunk may carry any token count and encoder inputs.
    """
    limit = 1 + scheduler.num_spec_tokens
    for request in scheduler.running:
        count = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if request.num_computed_tokens < request.num_prompt_tokens or count > limit:
            return None
    return [limit] * len(scheduler.running)


def _admissible(scheduler: Any) -> bool:
    """Whether the next schedule could take a request out of waiting."""
    from vllm.v1.request import RequestStatus

    active = len(scheduler.running) + scheduler.num_waiting_for_streaming_input
    if active >= scheduler.max_num_active_reqs:
        return False
    for queue in (scheduler.kv_holding_waiting, scheduler.waiting):
        for request in queue:
            if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                continue
            if (
                request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
                or request.request_id in scheduler.finished_recving_kv_req_ids
            ):
                return True
    return False


def gated_schedule(
    scheduler: Any, schedule: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run `schedule`, admitting new work only on an announced step.

    A step is announced when none of the AHEAD schedules before it published
    a bound (`NEXT_STEP_TOKENS`); the first steps are announced. An
    unannounced step schedules running decodes only; if a running request was
    rolled back to a prefill it schedules nothing, and announces a later one.
    """
    from vllm.v1.core.sched.interface import PauseState

    # Whether each of the last AHEAD schedules published a bound, oldest first.
    bounded = getattr(scheduler, "_tpu_dp_bounded", (False,) * AHEAD)
    announced = not any(bounded)
    held = not announced and _next_step_tokens(scheduler) is None
    previous = scheduler._pause_state
    if held:
        scheduler._pause_state = PauseState.PAUSED_ALL
    elif not announced and previous == PauseState.UNPAUSED:
        scheduler._pause_state = PauseState.PAUSED_NEW
    try:
        output = schedule(scheduler, *args, **kwargs)
    finally:
        scheduler._pause_state = previous
    tokens = None if held or _admissible(scheduler) else _next_step_tokens(scheduler)
    scheduler._tpu_dp_bounded = (*bounded[1:], tokens is not None)
    setattr(output, NEXT_STEP_TOKENS, tokens)
    return output


def install_admission_gate() -> None:
    """Gate vLLM's Scheduler.schedule on an EP-DP lockstep engine core."""
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.schedule
    if getattr(original, "_tpu_dp_gate", False):
        return

    @functools.wraps(original)
    def schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not lockstep_enabled(self.vllm_config.parallel_config):
            return original(self, *args, **kwargs)
        return gated_schedule(self, original, *args, **kwargs)

    schedule._tpu_dp_gate = True  # type: ignore[attr-defined]
    Scheduler.schedule = schedule

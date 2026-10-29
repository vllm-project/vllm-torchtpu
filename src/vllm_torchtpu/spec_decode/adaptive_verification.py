# Copyright 2026 Google LLC
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
"""Adaptive draft verification for DSpark on TPU.

The TPU counterpart of Model Runner V2's
``vllm/v1/worker/gpu/spec_decode/adaptive_verification.py``. Every draft slot
is scored by its survival probability, the running product of the drafter's
per-position confidences; a step verifies the globally best ``budget`` slots,
with the budget chosen to maximize expected emitted tokens per millisecond
under a step-cost table profiled at startup. Survival only falls along a
request, so each request keeps a prefix of its drafts.

The TPU runner lays every step out on the host from V1-shaped scheduler
output, so the budget and the per-request split are both decided on the host,
before the step's DP token sync. Under async scheduling the drafts a step
verifies are still being proposed on the device at that point, so the
decision reads each request's previous confidences. Selection never looks at
the trimmed tokens, so verification stays lossless. The scheduler scheduled
the trimmed drafts and rolls them back as rejected. Under DP lockstep every
rank pads to the largest rank's bucket, so once the sync settles the bucket a
rank re-admits its best trimmed drafts into the padding (``expand``).
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Seconds between two "verified x% of drafts" log lines.
_LOG_INTERVAL_S = 30.0
# Smoothing of the measured per-step time the cost table leaves out.
_OVERHEAD_EMA = 0.05
# A gap between two decode steps this many times longer than predicted was
# a stall (a compile, a host hiccup), and one this many times shorter was the
# host catching up after one; neither is a step to calibrate against.
_STALL_FACTOR = 4.0


def validate_adaptive_verification_config(vllm_config: VllmConfig) -> None:
    """TPU's replacement for ``VllmConfig._validate_adaptive_verification``,
    whose full-CUDA-graph requirement has no TPU counterpart."""
    spec = vllm_config.speculative_config
    if spec is None or not spec.enable_adaptive_verification:
        return
    disable = "Pass enable_adaptive_verification=false in the speculative config"
    if vllm_config.lora_config is not None:
        raise ValueError(
            f"Adaptive verification is not compatible with LoRA on TPU. {disable}."
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size > 1:
        # Confidences and step costs exist only on the last stage, and every
        # stage must lay out the same trimmed batch.
        raise ValueError(
            "Adaptive verification is not compatible with pipeline "
            f"parallelism on TPU. {disable}."
        )
    if parallel_config.prefill_context_parallel_size > 1:
        # TODO: size the cost table in per-rank tokens and trim the batch
        # before it is partitioned.
        raise NotImplementedError(
            "Adaptive verification with prefill context parallelism is not "
            f"supported on TPU yet. {disable}."
        )
    if vllm_config.model_config.enforce_eager:
        raise ValueError(
            "Adaptive verification profiles its step costs during warmup, "
            "which enforce_eager skips. Drop enforce_eager, or pass "
            "enable_adaptive_verification=false in the speculative config."
        )


@dataclass(frozen=True)
class StepCostTable:
    """Profiled step costs in ms.

    The target's verify forward is timed on decode-shaped batches: for each
    request count of ``request_grid``, at token counts spanning what a
    decode step of that many requests can schedule, at both ends of every
    token bucket the span crosses. A step's cost follows its real tokens as
    well as its bucket, since the MoE skips padding rows, so within a bucket
    the cost is interpolated linearly in tokens, and between grid entries
    linearly in requests. The draft forward is timed by padded draft-token
    bucket."""

    token_buckets: np.ndarray
    request_grid: np.ndarray
    # Per grid entry: ascending token counts and their costs.
    verify_tokens: tuple[np.ndarray, ...]
    verify_ms: tuple[np.ndarray, ...]
    draft_buckets: np.ndarray
    draft_ms: np.ndarray

    @classmethod
    def from_samples(
        cls,
        token_buckets: Sequence[int],
        verify: Sequence[tuple[int, int, float]],
        draft: Sequence[tuple[int, float]],
    ) -> StepCostTable:
        """``verify`` holds (requests, tokens, ms), ``draft`` (tokens, ms)."""

        def curve(samples: Iterable[tuple[int, float]]):
            sizes, ms = np.asarray(sorted(samples), dtype=np.float64).T
            return sizes.astype(np.int64), ms

        grid = sorted({num_reqs for num_reqs, _, _ in verify})
        # Verify costs are kept as measured: more tokens can cost less. A
        # bucket's full verify windows take the static query-length
        # attention pass, cheaper than the mixed pass its partial windows
        # take, so a full step can undercut a trimmed one in its bucket.
        curves = [
            curve((t, ms) for num_reqs, t, ms in verify if num_reqs == g) for g in grid
        ]
        draft_buckets, draft_ms = curve(draft)
        # A larger draft bucket never costs less; flatten timing noise.
        draft_ms = np.maximum.accumulate(draft_ms)
        return cls(
            np.asarray(sorted(token_buckets), dtype=np.int64),
            np.asarray(grid, dtype=np.int64),
            tuple(t for t, _ in curves),
            tuple(ms for _, ms in curves),
            draft_buckets,
            draft_ms,
        )

    @property
    def max_tokens(self) -> int:
        return int(max(t[-1] for t in self.verify_tokens))

    def verify_cost(self, num_reqs: int, num_tokens: np.ndarray) -> np.ndarray:
        """Cost of verifying ``num_tokens`` tokens of ``num_reqs`` requests."""
        shape = np.shape(num_tokens)
        num_tokens = np.atleast_1d(np.asarray(num_tokens))
        grid = self.request_grid
        hi = int(np.searchsorted(grid, num_reqs))
        if hi == 0 or hi == len(grid) or grid[hi] == num_reqs:
            cost = self._curve(min(hi, len(grid) - 1), num_tokens)[0]
            return cost.reshape(shape)
        lo = hi - 1
        w = (num_reqs - grid[lo]) / (grid[hi] - grid[lo])
        cost_lo, valid_lo = self._curve(lo, num_tokens)
        cost_hi, valid_hi = self._curve(hi, num_tokens)
        # A grid entry covers only the token counts its own request count
        # can schedule; where one of the two does not, the other answers.
        # Their spans overlap as long as neighbors differ by at most 1 + K.
        cost = np.where(
            valid_lo & valid_hi,
            (1 - w) * cost_lo + w * cost_hi,
            np.where(valid_hi, cost_hi, cost_lo),
        )
        return cost.reshape(shape)

    def _curve(self, i: int, num_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid entry ``i``'s cost at ``num_tokens``, and where it was profiled."""
        tokens, ms = self.verify_tokens[i], self.verify_ms[i]
        point_bucket = np.searchsorted(self.token_buckets, tokens)
        query_bucket = np.searchsorted(self.token_buckets, num_tokens)
        cost = np.interp(num_tokens, tokens, ms)
        for b in np.unique(query_bucket):
            in_bucket = point_bucket == b
            if in_bucket.any():
                # Interpolate within the bucket only: crossing a bucket edge
                # is a step, not a slope.
                q = query_bucket == b
                cost[q] = np.interp(num_tokens[q], tokens[in_bucket], ms[in_bucket])
        valid = (num_tokens >= tokens[0]) & (num_tokens <= tokens[-1])
        return cost, valid

    def draft_cost(self, num_draft_tokens: int) -> float:
        idx = np.searchsorted(self.draft_buckets, num_draft_tokens)
        return float(self.draft_ms[min(int(idx), len(self.draft_ms) - 1)])


def survival_probabilities(
    confidences: np.ndarray, num_drafts: np.ndarray
) -> np.ndarray:
    """``[num_reqs, K]`` running product of confidences, ``-inf`` past each
    request's scheduled drafts."""
    survival = np.cumprod(confidences.astype(np.float64), axis=1)
    steps = np.arange(confidences.shape[1])
    return np.where(steps[None, :] < num_drafts[:, None], survival, -np.inf)


def choose_draft_budget(
    survival: np.ndarray,
    num_sampling_reqs: int,
    num_non_draft_tokens: int,
    fixed_ms: float,
    costs: StepCostTable,
) -> int:
    """Number of draft slots that maximizes expected tokens per ms."""
    scores = np.sort(survival[np.isfinite(survival)])[::-1]
    budgets = np.arange(scores.size + 1)
    expected = num_sampling_reqs + np.concatenate(([0.0], np.cumsum(scores)))
    step_ms = fixed_ms + costs.verify_cost(
        num_sampling_reqs, num_non_draft_tokens + budgets
    )
    # A calibrated overhead below zero must not flip the ratio's sign.
    return int(np.argmax(expected / np.maximum(step_ms, 1e-3)))


def allocate_draft_budget(survival: np.ndarray, budget: int) -> np.ndarray:
    """Per-request count of the globally best ``budget`` draft slots."""
    if budget == 0:
        return np.zeros(survival.shape[0], dtype=np.int64)
    flat = survival.ravel()
    winners = np.argpartition(-flat, budget - 1)[:budget]
    admitted = np.zeros(flat.shape, dtype=bool)
    admitted[winners] = True
    return admitted.reshape(survival.shape).sum(axis=1)


def trim_scheduled_drafts(
    scheduler_output: SchedulerOutput, admitted: dict[str, int]
) -> SchedulerOutput:
    """A copy of ``scheduler_output`` scheduling only the first
    ``admitted[req_id]`` drafts of each listed request.

    The original is left untouched: under a uniprocess executor the scheduler
    holds the same object and must see what it scheduled.
    """
    num_scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)
    spec_tokens = dict(scheduler_output.scheduled_spec_decode_tokens)
    total = scheduler_output.total_num_scheduled_tokens
    for req_id, num_admitted in admitted.items():
        drafts = spec_tokens[req_id]
        num_trimmed = len(drafts) - num_admitted
        if num_trimmed <= 0:
            continue
        num_scheduled_tokens[req_id] -= num_trimmed
        total -= num_trimmed
        if num_admitted:
            spec_tokens[req_id] = drafts[:num_admitted]
        else:
            del spec_tokens[req_id]
    return dataclasses.replace(
        scheduler_output,
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=spec_tokens,
        total_num_scheduled_tokens=total,
    )


@dataclass
class _StepPlan:
    """A trimmed step's inputs, kept until the step's bucket is known."""

    scheduler_output: SchedulerOutput
    req_ids: list[str]
    num_drafts: np.ndarray
    survival: np.ndarray
    budget: int


@dataclass
class _PendingConfidences:
    req_ids: list[str]
    chunks_cpu: list[torch.Tensor]
    real_lens: list[int]
    ready_event: torch.Event | None

    def wait(self) -> Iterable[tuple[str, np.ndarray]]:
        if self.ready_event is not None:
            self.ready_event.synchronize()
        rows = np.concatenate(
            [t[:n].numpy() for t, n in zip(self.chunks_cpu, self.real_lens)]
        )
        return zip(self.req_ids, rows)


class AdaptiveVerificationManager:
    """Chooses each decode step's verification budget from the drafter's
    confidences and trims the scheduler output to it."""

    def __init__(
        self,
        num_speculative_tokens: int,
        draft_block_size: int,
        async_scheduling: bool,
        calibrate_overhead: bool,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.draft_block_size = draft_block_size
        # Copies a decision may not read yet: under async scheduling the
        # newest copy holds the confidences of drafts still being proposed.
        self._confidence_lag = 1 if async_scheduling else 0
        self._pending: deque[_PendingConfidences] = deque()
        self._confidences: dict[str, np.ndarray] = {}
        self._default_confidences = np.ones(num_speculative_tokens)
        self.cost_table: StepCostTable | None = None
        # Startup warmup only: verify exactly this many drafts per step.
        self.forced_budget: int | None = None
        # Drafts trimmed from each request in the current step.
        self.step_trimmed: dict[str, int] = {}
        self._plan: _StepPlan | None = None
        # The table times the target and draft forwards only; sampling,
        # rejection, host work and the DP sync come on top, and its profiled
        # context length need not match the live one, so the remainder can
        # fall either side of zero. Back-to-back decode steps measure it: the
        # async loop runs at the device's pace. Every TP rank must budget
        # alike, and their clocks differ, so only an engine without TP peers
        # calibrates.
        self.calibrate_overhead = calibrate_overhead
        self.overhead_ms = 0.0
        self._clock = time.monotonic
        # (start time, request count, real tokens) of the last decided step.
        self._last_step: tuple[float, int, int] | None = None
        self._num_decisions = 0
        self._num_scheduled = 0
        self._num_verified = 0
        self._num_trimmed_steps = 0
        self._last_log_time = time.monotonic()

    def record_confidences(
        self,
        req_ids: Sequence[str],
        chunks: list[torch.Tensor],
        real_lens: list[int],
    ) -> None:
        """Start the D2H copy of one proposal's ``[rows, K]`` confidences;
        row ``i`` of the concatenated real rows belongs to ``req_ids[i]``."""
        assert sum(real_lens) == len(req_ids), (real_lens, len(req_ids))
        chunks_cpu = [t.to("cpu", non_blocking=True) for t in chunks]
        ready_event = None
        if chunks[0].device.type != "cpu":
            ready_event = torch.tpu.Event()
            ready_event.record()
        self._pending.append(
            _PendingConfidences(list(req_ids), chunks_cpu, real_lens, ready_event)
        )

    def reset(self) -> None:
        self._drain(keep=0)
        self._confidences.clear()
        self.step_trimmed = {}
        self._plan = None
        self._last_step = None

    def _drain(self, keep: int) -> None:
        while len(self._pending) > keep:
            self._confidences.update(self._pending.popleft().wait())

    def trim(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """``scheduler_output`` with the drafts outside this step's budget
        removed; the same object when nothing is trimmed."""
        self.step_trimmed = {}
        self._plan = None
        now = self._clock()
        last_step, self._last_step = self._last_step, None
        num_scheduled = scheduler_output.num_scheduled_tokens
        self._drain(keep=self._confidence_lag)
        # A request left out of a step, finished ones included, forgets its
        # confidences; a later step scores it as certain.
        self._confidences = {
            r: c for r, c in self._confidences.items() if r in num_scheduled
        }
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if (
            not spec_tokens
            or scheduler_output.has_structured_output_requests
            or (self.cost_table is None and self.forced_budget is None)
        ):
            return scheduler_output
        # Decode-only steps: every request is a verify window or one token.
        if any(
            n != 1 + len(spec_tokens.get(req_id, ()))
            for req_id, n in num_scheduled.items()
        ):
            return scheduler_output
        if (
            self.cost_table is not None
            and scheduler_output.total_num_scheduled_tokens > self.cost_table.max_tokens
        ):
            return scheduler_output

        req_ids = list(spec_tokens)
        num_drafts = np.fromiter(
            (len(spec_tokens[r]) for r in req_ids), dtype=np.int64, count=len(req_ids)
        )
        confidences = np.stack(
            [self._confidences.get(r, self._default_confidences) for r in req_ids]
        )
        survival = survival_probabilities(confidences, num_drafts)
        total_drafts = int(num_drafts.sum())
        num_reqs = len(num_scheduled)
        if self.forced_budget is not None:
            budget = min(self.forced_budget, total_drafts)
        else:
            assert self.cost_table is not None
            if self.calibrate_overhead and last_step is not None:
                self._calibrate(now, *last_step)
            budget = choose_draft_budget(
                survival,
                num_sampling_reqs=num_reqs,
                num_non_draft_tokens=num_reqs,
                fixed_ms=self.cost_table.draft_cost(num_reqs * self.draft_block_size)
                + self.overhead_ms,
                costs=self.cost_table,
            )
            self._last_step = (now, num_reqs, num_reqs + budget)
        self._observe(total_drafts, budget)
        if budget == total_drafts:
            return scheduler_output
        self._plan = _StepPlan(scheduler_output, req_ids, num_drafts, survival, budget)
        return self._apply(self._plan)[0]

    def expand(self, num_tokens: int) -> tuple[SchedulerOutput, list[str]] | None:
        """Re-admit up to ``num_tokens`` of this step's trimmed drafts.

        For a step whose padded token bucket turned out larger than its own
        trimmed size (under DP lockstep, the largest rank's): the bucket's
        padding rows can carry drafts instead. Returns the new copy and the
        requests whose draft count changed, or None when nothing changes.
        Startup warmup keeps its forced budget."""
        plan = self._plan
        if plan is None or num_tokens <= 0 or self.forced_budget is not None:
            return None
        budget = min(plan.budget + num_tokens, int(plan.num_drafts.sum()))
        if budget == plan.budget:
            return None
        if self._last_step is not None:
            started, num_reqs, step_tokens = self._last_step
            self._last_step = (started, num_reqs, step_tokens + budget - plan.budget)
        before = allocate_draft_budget(plan.survival, plan.budget)
        self._num_verified += budget - plan.budget
        self._num_trimmed_steps -= budget == int(plan.num_drafts.sum())
        plan.budget = budget
        scheduler_output, admitted = self._apply(plan)
        changed = [r for r, a, b in zip(plan.req_ids, admitted, before) if a != b]
        return scheduler_output, changed

    def _apply(self, plan: _StepPlan) -> tuple[SchedulerOutput, np.ndarray]:
        admitted = allocate_draft_budget(plan.survival, plan.budget)
        self.step_trimmed = {
            r: int(d - a)
            for r, d, a in zip(plan.req_ids, plan.num_drafts, admitted)
            if a < d
        }
        return (
            trim_scheduled_drafts(
                plan.scheduler_output, dict(zip(plan.req_ids, admitted.tolist()))
            ),
            admitted,
        )

    def _calibrate(
        self, now: float, started: float, num_reqs: int, num_tokens: int
    ) -> None:
        assert self.cost_table is not None
        expected_ms = self.cost_table.draft_cost(
            num_reqs * self.draft_block_size
        ) + float(self.cost_table.verify_cost(num_reqs, num_tokens))
        predicted_ms = expected_ms + self.overhead_ms
        elapsed_ms = (now - started) * 1e3
        if not (
            predicted_ms / _STALL_FACTOR <= elapsed_ms <= _STALL_FACTOR * predicted_ms
        ):
            return
        # Signed: clamping at zero would leave a table that overestimates
        # the step uncorrected, and steer the budget toward trimming.
        sample = elapsed_ms - expected_ms
        self.overhead_ms += _OVERHEAD_EMA * (sample - self.overhead_ms)

    def _observe(self, num_scheduled: int, num_verified: int) -> None:
        if self.forced_budget is not None:
            return
        # Report the steps before this one: `expand` may still change it.
        now = time.monotonic()
        if self._num_decisions and now - self._last_log_time >= _LOG_INTERVAL_S:
            logger.info(
                "Adaptive verification: verified %d of %d scheduled drafts "
                "(%.1f%%) in %d decode steps, %d of them trimmed; step time "
                "beyond the cost table %+.2f ms.",
                self._num_verified,
                self._num_scheduled,
                100.0 * self._num_verified / max(self._num_scheduled, 1),
                self._num_decisions,
                self._num_trimmed_steps,
                self.overhead_ms,
            )
            self._num_decisions = self._num_trimmed_steps = 0
            self._num_scheduled = self._num_verified = 0
            self._last_log_time = now
        self._num_decisions += 1
        self._num_scheduled += num_scheduled
        self._num_verified += num_verified
        self._num_trimmed_steps += num_verified < num_scheduled

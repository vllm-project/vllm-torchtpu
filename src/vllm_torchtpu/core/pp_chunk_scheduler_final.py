# SPDX-License-Identifier: Apache-2.0
"""The final lineage's PP8 chunk scheduler (final-p2 10a2d5209, fp2q-r5 0447ff58b), as an
opt-in flavour on ix-unified-port.

Selected with TPU_PP_SCHEDULER=final (default: unified's TpuPipelineChunkScheduler). The
final trees subclassed PrefixAwareScheduler, so admission (PAW sort, SRPF, memo, resend trim,
retention and single-step splits) is PrefixAwareScheduler's own schedule() and
_mamba_block_aligned_split, with the cold-prefill cap and the mid-block stop skip turned off,
and the time-budget chunking below on top:

- _pending_step_limit: pending work counted from num_computed_tokens, each spread costed as
  steps x step_ms(bucket, 0, share^2): no pending pairs, no in-flight step history.
- _mamba_block_aligned_split: StepCostModel.chunk with no live target ceiling; the first
  prefill chunk of a step always advances (min step max(granularity, block_size) under the
  Mamba split), and when the Mamba alignment declines that first chunk it is retried at the
  minimum step instead of waiting.

The method bodies are the final-p2 code (core/pp_chunk_scheduler.py there), unchanged except
for names. Unified's flavour differs in all three points (it borrows PAS methods onto vLLM's
Scheduler, adds warmest promotion, in-flight costing, PR #2243 quad cost and a retention split
without the first-chunk retry).
"""

from __future__ import annotations

import math
from typing import Any

from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


class TpuPipelineChunkSchedulerFinal(PrefixAwareScheduler):
    def _cap_cold_prefill_tokens(
        self,
        request: Any,
        start_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        # Pipeline chunk sizing uses StepCostModel in _mamba_block_aligned_split.
        return num_new_tokens

    def _midblock_stop_skip_tokens(self, request: Any, start: int, num_new_tokens: int) -> int:
        return 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cost: StepCostModel | None = None
        self._step_limit: int | None = None
        self._pp_size = 1
        self._mamba_split = self.need_mamba_block_aligned_split
        self.need_mamba_block_aligned_split = True
        self._step_tokens = 0
        self._step_pairs = 0.0
        self._step_squares = 0.0
        self._step_prefill_tokens = 0
        self._step_schedule = 0
        self._seeded: set[str] = set()
        self._clips = 0
        self._waits = 0
        logger.info(
            "Pipeline chunk scheduler (final-lineage flavour, TPU_PP_SCHEDULER=final): "
            "waiting for the step cost model"
        )

    def install_cost_model(self, model: StepCostModel) -> None:
        self._cost = model
        self._pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        logger.info("Pipeline chunk scheduler: %s", model.describe())

    def _account(self, tokens: int, prefix: int) -> None:
        self._step_tokens += tokens
        self._step_pairs += float(tokens) * prefix
        self._step_squares += float(tokens) * tokens

    def schedule(self, throttle_prefills: bool = False):
        self._step_tokens = 0
        self._step_pairs = 0.0
        self._step_squares = 0.0
        self._step_prefill_tokens = 0
        self._step_schedule = 0
        self._seeded = set()
        self._step_limit = self._pending_step_limit()
        # Every running request past its prefill with a token left to compute takes it
        # this step; count them before the loop sizes any prefill chunk.
        for request in self.running:
            computed = request.num_computed_tokens
            if computed >= _prefill_end(request) and request.num_tokens > computed:
                self._account(1, computed)
                self._seeded.add(request.request_id)
        return super().schedule(throttle_prefills)

    def step_estimate_ms(self, scheduler_output: Any) -> float | None:
        """Modelled time on one stage of the step the last ``schedule`` produced, in ms."""
        if self._cost is None:
            return None
        pairs = squares = 0.0
        for req_id, tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            prefix = request.num_computed_tokens - tokens if request is not None else 0
            pairs += float(tokens) * prefix
            squares += float(tokens) * tokens
        buckets = self._cost.buckets
        total = scheduler_output.total_num_scheduled_tokens
        bucket = next((b for b in buckets if b >= total), buckets[-1])
        return self._cost.step_ms(bucket, pairs, squares)

    def _pending_step_limit(self) -> int | None:
        """The bucket a step is held to (final-p2): the pending prefill work spread over up
        to as many steps as the pipeline has stages, the soonest spread taken."""
        if self._cost is None:
            return None
        pending = 0
        for request in self.running:
            pending += max(_prefill_end(request) - request.num_computed_tokens, 0)
        for request in self.waiting:
            pending += max(_prefill_end(request) - request.num_computed_tokens, 0)
        if pending == 0:
            return None
        granularity = self._cost.granularity
        buckets = self._cost.buckets
        limit, soonest = buckets[-1], math.inf
        for spread in range(1, self._pp_size + 1):
            share = -(-pending // spread)
            share = -(-share // granularity) * granularity
            share = min(buckets[-1], max(granularity, share))
            bucket = next(b for b in buckets if b >= share)
            steps = -(-pending // share) + self._pp_size - 1
            done_ms = steps * self._cost.step_ms(bucket, 0.0, float(share) * share)
            if done_ms < soonest:
                limit, soonest = share, done_ms
        return limit

    def _mamba_block_aligned_split(
        self,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        if start >= _prefill_end(request):
            if request.request_id not in self._seeded:
                self._account(num_new_tokens, start)
            return num_new_tokens
        orig_new_tokens = num_new_tokens
        if self._cost is not None and num_new_tokens > 0:
            cap = self._cost.chunk(
                start,
                num_new_tokens,
                self._step_tokens,
                self._step_pairs,
                self._step_squares,
                self._step_limit,
                self._step_schedule,
            )
            if cap == 0 and self._step_prefill_tokens == 0:
                # The first prefill chunk of a step always advances.
                min_step = (
                    max(self._cost.granularity, self.block_size)
                    if self._mamba_split
                    else self._cost.granularity
                )
                cap = min(num_new_tokens, min_step)
            if cap == 0:
                self._waits += 1
            elif cap < num_new_tokens:
                self._clips += 1
            num_new_tokens = cap
        if self._mamba_split and num_new_tokens > 0:
            aligned = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
            if (
                aligned == 0
                and self._step_prefill_tokens == 0
                and orig_new_tokens >= self.block_size
            ):
                min_step = (
                    max(self.block_size, self._cost.granularity)
                    if self._cost is not None
                    else self.block_size
                )
                aligned = super()._mamba_block_aligned_split(
                    request,
                    min(orig_new_tokens, min_step),
                    num_new_local_computed_tokens,
                    num_external_computed_tokens,
                )
            num_new_tokens = aligned
        self._account(num_new_tokens, start)
        self._step_prefill_tokens += num_new_tokens
        if self._cost is not None:
            self._step_schedule += self._cost.pairs(num_new_tokens, start)
        return num_new_tokens

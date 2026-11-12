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
"""Scheduler that sizes prefill chunks for a pipeline.

vLLM's scheduler fills every step up to a token budget and clips each
request's chunk at one point, ``_mamba_block_aligned_split``. This subclass
routes every request through that point, clips prefill chunks to the
step's limits (``core/pp_chunks.py``) and then applies the base scheduler's
mamba block alignment when the model has a mamba cache. Decode requests
count toward the step before any prefill chunk is sized, whatever their
position in the running queue.

Two policies set the limits, each behind its own env var. With lone-prompt
chunking (``TPU_PP_LONE_PROMPT_CHUNKS``), when the pending prefill work
cannot fill every stage with a full step, the step shrinks to that work
spread over the pipeline, so a lone prompt runs as one small step per stage
instead of trailing itself through the pipeline one full step at a time; a
busy server keeps full steps and first-come order. With equal-cost chunking
(``TPU_PP_EQUAL_COST_CHUNKS``), prefill chunks are clipped to what the
step's remaining time holds under a cost model the stages time at startup.
Either way a step holds at most as many (query block, KV block) attention
pairs as the batched attention kernel can schedule, which bounds a chunk
deep in a very long prompt so the kernel never overruns its SMEM. The
limits arrive after the engine core has built its executor and scheduler;
until then chunks are cut by token count alone.
"""

import math
from typing import Any, cast

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import envs
from vllm_torchtpu.core.pp_chunks import StepCostModel, chunk_pairs, step_limit
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = "vllm_torchtpu.core.pp_chunk_scheduler.TpuPipelineChunkScheduler"
PREFIX_AWARE_PP_SCHEDULER_CLS = (
    "vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwarePipelineChunkScheduler"
)
# Schedulers that size prefill chunks for a pipeline: the platform keeps a
# configured one and installs SCHEDULER_CLS when none is configured.
DYNAMIC_CHUNK_SCHEDULER_CLASSES = (SCHEDULER_CLS, PREFIX_AWARE_PP_SCHEDULER_CLS)


def uses_dynamic_chunks(vllm_config: Any) -> bool:
    """Whether this configuration sizes pipeline chunks; the platform
    installs the scheduler class only when it does."""
    return (
        vllm_config is not None
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.scheduler_config.scheduler_cls
        in DYNAMIC_CHUNK_SCHEDULER_CLASSES
    )


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


class TpuPipelineChunkScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lone_prompt = envs.TPU_PP_LONE_PROMPT_CHUNKS
        self._equal_cost = envs.TPU_PP_EQUAL_COST_CHUNKS
        self._cost: StepCostModel | None = None
        self._granularity: int | None = None
        # (pairs, bq, bkv) the batched attention kernel schedules per step.
        self._schedule: tuple[int, int, int] | None = None
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
            "Pipeline chunk scheduler (lone-prompt %s, equal-cost %s): waiting "
            "for the chunk limits",
            self._lone_prompt,
            self._equal_cost,
        )

    def install_chunk_limits(
        self, granularity: int, schedule: tuple[int, int, int] | None
    ) -> None:
        """Cut chunks in ``granularity`` tokens and keep a step within the
        attention ``schedule`` capacity, without a cost model."""
        self._granularity = granularity
        self._schedule = schedule
        self._pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        text = f"granularity {granularity} tokens"
        if schedule is not None:
            pairs, bq, bkv = schedule
            text += f", at most {pairs} attention pairs ({bq}x{bkv}) per step"
        logger.info("Pipeline chunk scheduler: %s", text)

    def install_cost_model(self, model: StepCostModel) -> None:
        self._cost = model
        self._granularity = model.granularity
        self._schedule = model.schedule
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
        # Every running request past its prefill with a token left to
        # compute takes it this step; count them before the loop sizes any
        # prefill chunk. A request whose last chunk is still in flight has
        # nothing to compute yet.
        for request in self.running:
            computed = request.num_computed_tokens
            if computed >= _prefill_end(request) and request.num_tokens > computed:
                self._account(1, computed)
                self._seeded.add(request.request_id)
        return super().schedule(throttle_prefills)

    def step_estimate_ms(self, scheduler_output: Any) -> float | None:
        """Modelled time on one stage of the step the last ``schedule``
        produced, in ms; None until the cost model is installed."""
        if self._cost is None:
            return None
        pairs = squares = 0.0
        for req_id, tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            # schedule() already counted the step's tokens as computed.
            prefix = request.num_computed_tokens - tokens if request is not None else 0
            pairs += float(tokens) * prefix
            squares += float(tokens) * tokens
        buckets = self._cost.buckets
        total = scheduler_output.total_num_scheduled_tokens
        bucket = next((b for b in buckets if b >= total), buckets[-1])
        return self._cost.step_ms(bucket, pairs, squares)

    def _pending_step_limit(self) -> int | None:
        """The bucket a step is held to under lone-prompt chunking: the
        prefill work still to schedule spread over up to as many steps as the
        pipeline has stages, rounded up to a granule, at most a full step.
        With a cost model, of those spreads the one that gets the work
        through the pipeline soonest is taken: a step of half the tokens
        that takes as long as the whole one only adds steps. Without one the
        work is spread over every stage (`step_limit`). None with lone-prompt
        chunking off, until the limits are installed, or when no prefill
        work is pending."""
        granularity = self._granularity
        if not self._lone_prompt or granularity is None:
            return None
        pending = 0
        for request in self.running:
            pending += max(_prefill_end(request) - request.num_computed_tokens, 0)
        # Requests holding KV blocks (including async KV loads) wait in a
        # separate queue. Count both queues before the base scheduler scans
        # them. deferred_waiting tracks a subset of these requests, so adding
        # it separately would double-count their remaining prompt work.
        for queue in (self.waiting, self.kv_holding_waiting):
            for request in queue:
                pending += max(_prefill_end(request) - request.num_computed_tokens, 0)
        if self._cost is None:
            return step_limit(
                pending,
                self._pp_size,
                granularity,
                self.scheduler_config.max_num_batched_tokens,
            )
        if pending == 0:
            return None
        buckets = self._cost.buckets
        limit, soonest = buckets[-1], math.inf
        for spread in range(1, self._pp_size + 1):
            share = -(-pending // spread)
            share = -(-share // granularity) * granularity
            share = min(buckets[-1], max(granularity, share))
            bucket = next(b for b in buckets if b >= share)
            # The last of the work leaves the pipeline a stage count minus
            # one steps after its step enters.
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
        granularity = self._granularity
        if granularity is not None and num_new_tokens > 0:
            cap = self._chunk_cap(start, num_new_tokens, granularity)
            if cap == 0 and self._step_prefill_tokens == 0:
                # The first prefill chunk of a step always advances.
                cap = min(num_new_tokens, granularity)
            if cap == 0:
                self._waits += 1
            elif cap < num_new_tokens:
                self._clips += 1
                logger.debug(
                    "Pipeline chunk: request %s prefix %d, %d -> %d tokens "
                    "(step holds %d)",
                    request.request_id,
                    start,
                    num_new_tokens,
                    cap,
                    self._step_tokens,
                )
            num_new_tokens = cap
        if self._mamba_split and num_new_tokens > 0:
            num_new_tokens = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        self._account(num_new_tokens, start)
        self._step_prefill_tokens += num_new_tokens
        self._step_schedule += chunk_pairs(self._schedule, num_new_tokens, start)
        return num_new_tokens

    def _chunk_cap(self, start: int, num_new_tokens: int, granularity: int) -> int:
        """Largest chunk of a prefill behind ``start`` computed tokens that
        the step's limits hold: the cost model's time budget with equal-cost
        chunking, else the lone-prompt step limit; either way within the
        attention schedule capacity and leaving the step total on a granule
        multiple when clipped. 0 when nothing fits."""
        if self._cost is not None:
            return self._cost.chunk(
                start,
                num_new_tokens,
                self._step_tokens,
                self._step_pairs,
                self._step_squares,
                self._step_limit,
                self._step_schedule,
            )
        cap = num_new_tokens
        if self._step_limit is not None:
            cap = min(cap, self._step_limit - self._step_tokens)
        if self._schedule is not None:
            room = self._schedule[0] - self._step_schedule
            while cap > 0 and chunk_pairs(self._schedule, cap, start) > room:
                # Back off a granule, keeping the step total aligned.
                total = self._step_tokens + cap - 1
                cap = total - total % granularity - self._step_tokens
        return max(cap, 0)


def install_chunk_limits_from_workers(scheduler: Any, executor: Any) -> None:
    """Hand the scheduler its chunk limits from every stage: the largest
    granule, the smallest attention schedule capacity and, with equal-cost
    chunking, a step cost model fit to each stage's timed forwards."""
    if not isinstance(scheduler, TpuPipelineChunkScheduler):
        return
    method = (
        "profile_pipeline_chunks" if scheduler._equal_cost else "pipeline_chunk_limits"
    )
    reports = executor.collective_rpc(method)
    granularity = max(int(report["granularity"]) for report in reports)
    schedules = [
        tuple(int(x) for x in report["schedule"])
        for report in reports
        if report.get("schedule") is not None
    ]
    # Each worker reports its kernel schedule as (pairs, bq, bkv).
    schedule = cast(tuple[int, int, int], min(schedules)) if schedules else None
    if not scheduler._equal_cost:
        scheduler.install_chunk_limits(granularity, schedule)
        return
    model = StepCostModel.fit(
        [[tuple(s) for s in report["samples"]] for report in reports],
        envs.TPU_PP_CHUNK_SLACK,
        granularity,
        schedule,
    )
    scheduler.install_cost_model(model)


def patch_engine_core_for_pp_chunks(vllm_config: Any) -> None:
    """Have the engine core collect the chunk limits once its executor and
    scheduler exist. Installed from the platform's config check (in-process
    engines) and from the engine-core process entry point; idempotent."""
    if not uses_dynamic_chunks(vllm_config):
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_pp_chunks_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        install_chunk_limits_from_workers(self.scheduler, self.model_executor)

    EngineCore.__init__ = __init__
    EngineCore._tpu_pp_chunks_patch = True

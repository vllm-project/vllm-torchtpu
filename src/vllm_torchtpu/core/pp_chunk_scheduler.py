# SPDX-License-Identifier: Apache-2.0
"""Scheduler that sizes prefill chunks against a per-step time budget.

vLLM's scheduler fills every step up to a token budget and clips each
request's chunk at one point, ``_mamba_block_aligned_split``. This subclass
routes every request through that point, clips prefill chunks to what the
step's remaining time holds (``core/pp_chunks.py``) and then applies the
base scheduler's mamba block alignment when the model has a mamba cache.
Decode requests count toward the step before any prefill chunk is sized,
whatever their position in the running queue. When the pending prefill
work cannot fill every stage with a full step, the step shrinks to that
work spread over the pipeline, so a lone prompt runs as one small step per
stage instead of trailing itself through the pipeline one full step at a
time; a busy server keeps full steps and first-come order. A step also holds
at most as many (query block, KV block) attention pairs as the batched
attention kernel can schedule, which bounds a chunk deep in a very long
prompt so the kernel never overruns its SMEM. The cost model
arrives after the engine core has built its executor and scheduler; until
then chunks are cut by token count alone.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import envs
from vllm_torchtpu.core.pp_chunks import StepCostModel

logger = init_logger(__name__)

SCHEDULER_CLS = "vllm_torchtpu.core.pp_chunk_scheduler.TpuPipelineChunkScheduler"


def uses_dynamic_chunks(vllm_config: Any) -> bool:
    """Whether this configuration sizes pipeline steps by time; the platform
    installs the scheduler class only when it does."""
    return (
        vllm_config is not None
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.scheduler_config.scheduler_cls == SCHEDULER_CLS
    )


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


class TpuPipelineChunkScheduler(Scheduler):
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
        logger.info("Pipeline chunk scheduler: waiting for the step cost model")

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

    def _pending_step_limit(self) -> int | None:
        """The bucket a step is held to: the prefill work still to schedule
        spread over the pipeline's stages, rounded up to a granule, at most
        a full step. None until the cost model is installed or when no
        prefill work is pending."""
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
        share = -(-pending // self._pp_size)
        share = -(-share // granularity) * granularity
        return min(self._cost.buckets[-1], max(granularity, share))

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
                cap = min(num_new_tokens, self._cost.granularity)
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
        if self._cost is not None:
            self._step_schedule += self._cost.pairs(num_new_tokens, start)
        return num_new_tokens


def profile_and_install_cost_model(scheduler: Any, executor: Any) -> None:
    """Time every stage's forwards and hand the scheduler the model."""
    if not isinstance(scheduler, TpuPipelineChunkScheduler):
        return
    reports = executor.collective_rpc("profile_pipeline_chunks")
    granularity = max(int(report["granularity"]) for report in reports)
    schedules = [
        tuple(int(x) for x in report["schedule"])
        for report in reports
        if report.get("schedule") is not None
    ]
    model = StepCostModel.fit(
        [[tuple(s) for s in report["samples"]] for report in reports],
        envs.TPU_PP_CHUNK_SLACK,
        granularity,
        min(schedules) if schedules else None,
    )
    scheduler.install_cost_model(model)


def patch_engine_core_for_pp_chunks(vllm_config: Any) -> None:
    """Have the engine core time the stages once its executor and scheduler
    exist. Installed from the platform's config check (in-process engines)
    and from the engine-core process entry point; idempotent."""
    if not uses_dynamic_chunks(vllm_config):
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_pp_chunks_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        profile_and_install_cost_model(self.scheduler, self.model_executor)

    EngineCore.__init__ = __init__
    EngineCore._tpu_pp_chunks_patch = True

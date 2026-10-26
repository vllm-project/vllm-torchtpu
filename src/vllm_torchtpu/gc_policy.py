# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Keep CPython's cyclic collector out of a data-parallel decode step.

Every DP rank's step waits on the others at a per-step collective, so a pause
on one rank is paid by all of them: at DP32 a single rank's generation-1
collection of 15-34 ms lengthens that step on all 32 ranks, and with 32 ranks
drawing independently such a collection lands in about a third of the steps
(run 8x3wdpfv, DSv4.1-Flash C768).

After warmup the heap is static, so ``arm()`` freezes it out of the collector's
reach and turns automatic collection off; the cycles a step does create are
reclaimed by explicit collections placed where every rank is at the same point,
so the slice pays the pause once instead of once per rank.

The policy belongs in **both** of a rank's processes. The model runner is one;
its cadence is the DP-coordinated step. The other is the engine core, whose
scheduler holds every request's block tables and allocates freely each step: a
collection there costs 25-33 ms, runs jax's ``_xla_gc_callback`` on top of the
scan, and lands inside ``Scheduler.schedule`` (run 2vajbn6k, job j-5d0028e8,
80 of 101 sampled stacks). vLLM already freezes the engine core's startup heap,
but the request state is built later and automatic collection stays on.

The engine core's cadence is vLLM's own DP ``step_counter``, not a local count
of loop iterations: an idle rank runs loop iterations its peers do not (one per
input message), so a local count drifts apart between ranks and each rank would
collect on a different step, 32 pauses per cadence instead of one. The counter
vLLM synchronizes its DP state on is identical on every rank and restarts with
each wave, and a wave's end, when every rank goes idle together, is where most
of the engine core's collections land.
"""

import functools
import gc
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class StepGCPolicy:
    """Freeze the post-warmup heap; collect on a step cadence, not on demand."""

    def __init__(self, collect_every: int, name: str = "runner") -> None:
        if collect_every < 0:
            raise ValueError(f"collect_every must not be negative: {collect_every}")
        self.collect_every = collect_every
        self.name = name
        self.armed = False
        self.steps = 0
        self.collections = 0
        self.collect_ns = 0

    def arm(self) -> None:
        """Called once warmup is complete; everything alive now is permanent."""
        if self.armed:
            raise RuntimeError("gc policy already armed")
        gc.collect()
        gc.freeze()
        gc.disable()
        self.armed = True
        logger.info(
            "gc policy (%s): froze %d objects, automatic collection off, "
            "explicit collection every %d coordinated steps",
            self.name,
            gc.get_freeze_count(),
            self.collect_every,
        )

    def step(self) -> None:
        """Called once per DP-coordinated step, before the step's collective."""
        if not self.armed:
            return
        self.steps += 1
        if self.collect_every and self.steps % self.collect_every == 0:
            self.collect(f"step {self.steps}")

    def lockstep(self, step_counter: int, running: bool) -> None:
        """Called once per DP engine-core iteration with vLLM's step counter.

        ``step_counter`` is the same on every DP rank and ``running`` is False
        only on the iteration where every rank ends the wave together.
        """
        if not self.armed:
            return
        self.steps += 1
        if not running:
            self.collect("wave end")
        elif self.collect_every and step_counter % self.collect_every == 0:
            self.collect(f"wave step {step_counter}")

    def collect(self, at: str) -> None:
        started = time.perf_counter_ns()
        collected = gc.collect()
        elapsed = time.perf_counter_ns() - started
        self.collections += 1
        self.collect_ns += elapsed
        logger.info(
            "gc policy (%s): %s, explicit collection %d freed %d objects in "
            "%.1f ms (%.3f ms/step amortized)",
            self.name,
            at,
            self.collections,
            collected,
            elapsed / 1e6,
            self.collect_ns / 1e6 / max(self.steps, 1),
        )


_POLICY: StepGCPolicy | None = None
_ENGINE_POLICY: StepGCPolicy | None = None


def _make(mode: str, collect_every: int, name: str) -> StepGCPolicy | None:
    if mode == "default":
        return None
    if mode != "freeze":
        raise ValueError(f"Unknown TPU_GC_POLICY: {mode}")
    return StepGCPolicy(collect_every, name)


def install(mode: str, collect_every: int) -> StepGCPolicy | None:
    """Create the policy named by ``TPU_GC_POLICY``, or none for 'default'."""
    global _POLICY
    _POLICY = _make(mode, collect_every, "runner")
    return _POLICY


def arm_after_warmup() -> None:
    from vllm_torchtpu import envs

    policy = install(envs.TPU_GC_POLICY, envs.TPU_GC_COLLECT_EVERY)
    if policy is not None:
        policy.arm()


def on_coordinated_step() -> None:
    if _POLICY is not None:
        _POLICY.step()


def install_engine_core() -> None:
    """Apply the policy in the engine-core process, keyed to its DP lockstep.

    Arming waits for the busy loop's first iteration, after the block pool and
    the scheduler exist. A DP engine core collects on vLLM's DP step counter
    (``_has_global_unfinished_reqs``, called once per iteration on every rank);
    any other engine core collects on its own iteration count.
    """
    global _ENGINE_POLICY
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

    from vllm_torchtpu import envs

    if getattr(EngineCoreProc._process_engine_step, "_tpu_gc_policy", False):
        return
    policy = _make(envs.TPU_GC_POLICY, envs.TPU_GC_COLLECT_EVERY, "engine core")
    if policy is None:
        return
    _ENGINE_POLICY = policy

    sync = getattr(DPEngineCoreProc, "_has_global_unfinished_reqs", None)
    has_lockstep = sync is not None
    if not has_lockstep:
        logger.warning(
            "gc policy (engine core): this vLLM has no "
            "DPEngineCoreProc._has_global_unfinished_reqs; DP ranks will count "
            "their own iterations and may collect on different steps"
        )

    original_step = EngineCoreProc._process_engine_step

    @functools.wraps(original_step)
    def process_engine_step(self, *args, **kwargs):
        if not policy.armed:
            policy.arm()
        if not (has_lockstep and isinstance(self, DPEngineCoreProc)):
            policy.step()
        return original_step(self, *args, **kwargs)

    process_engine_step._tpu_gc_policy = True
    EngineCoreProc._process_engine_step = process_engine_step

    if has_lockstep:

        @functools.wraps(sync)
        def has_global_unfinished_reqs(self, *args, **kwargs):
            running = sync(self, *args, **kwargs)
            policy.lockstep(self.step_counter, running)
            return running

        has_global_unfinished_reqs._tpu_gc_policy = True
        DPEngineCoreProc._has_global_unfinished_reqs = has_global_unfinished_reqs

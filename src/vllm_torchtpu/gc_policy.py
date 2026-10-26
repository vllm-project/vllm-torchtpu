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
reclaimed by one explicit collection every ``collect_every`` coordinated steps.
Every rank reaches that step at the same time, so the slice pays the pause once
instead of once per rank.
"""

import gc
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class StepGCPolicy:
    """Freeze the post-warmup heap; collect on a step cadence, not on demand."""

    def __init__(self, collect_every: int) -> None:
        if collect_every < 0:
            raise ValueError(f"collect_every must not be negative: {collect_every}")
        self.collect_every = collect_every
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
            "gc policy: froze %d objects, automatic collection off, "
            "explicit collection every %d coordinated steps",
            gc.get_freeze_count(),
            self.collect_every,
        )

    def step(self) -> None:
        """Called once per DP-coordinated step, before the step's collective."""
        if not self.armed:
            return
        self.steps += 1
        if not self.collect_every or self.steps % self.collect_every:
            return
        started = time.perf_counter_ns()
        collected = gc.collect()
        elapsed = time.perf_counter_ns() - started
        self.collections += 1
        self.collect_ns += elapsed
        logger.info(
            "gc policy: step %d, explicit collection %d freed %d objects in "
            "%.1f ms (%.3f ms/step amortized)",
            self.steps,
            self.collections,
            collected,
            elapsed / 1e6,
            self.collect_ns / 1e6 / self.steps,
        )


_POLICY: StepGCPolicy | None = None


def install(mode: str, collect_every: int) -> StepGCPolicy | None:
    """Create the policy named by ``TPU_GC_POLICY``, or none for 'default'."""
    global _POLICY
    if mode == "default":
        return None
    if mode != "freeze":
        raise ValueError(f"Unknown TPU_GC_POLICY: {mode}")
    _POLICY = StepGCPolicy(collect_every)
    return _POLICY


def arm_after_warmup() -> None:
    from vllm_torchtpu import envs

    policy = install(envs.TPU_GC_POLICY, envs.TPU_GC_COLLECT_EVERY)
    if policy is not None:
        policy.arm()


def on_coordinated_step() -> None:
    if _POLICY is not None:
        _POLICY.step()

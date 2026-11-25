# SPDX-License-Identifier: Apache-2.0
"""Decode host-cycle diet members and their in-process A/B (q38 SV-HOSTDIET).

Every key is off by default, host only and compile-ignored; no device program,
collective or scheduler decision changes with any of them.

- ``VLLM_TPU_HD_MARKS=1`` (HD-0, log only): extra ``host_phases`` marks that
  split existing phases (the copy wait out of ``dp_entry``, and sub-phases of
  ``update_states``, ``prep_host_and_uploads``, ``prepare_inputs``,
  ``seed_reset_subst``, ``adapter_prepare_inputs`` and ``forward_dispatch``).
  An old phase equals the sum of its new sub-phases plus its remainder. Takes
  effect only with ``VLLM_TPU_HOST_PHASES``.
- ``VLLM_TPU_HD_STAGE_CPU=1`` (HD-5): the runner's stage timings keep CPU
  samples only, so the serving thread records no ``TpuEvent`` pair per stage
  (four stages a decode step) and the collector thread synchronizes none.
- ``VLLM_TPU_HD_REQVEC=1`` (HD-1): per-request Python loops of the input
  preparation are replaced by numpy over the step's requests, with
  bitwise-identical results: the async substitution indices, the rejected
  seq / pos index arrays and the previous-results commit; only on steps with
  at least REQVEC_MIN_REQS requests (the stock loops are cheaper below).
- ``VLLM_TPU_HD_AB=<N>`` (default 0): the members above (not the marks) are
  on only in odd windows of N worker steps and off in even ones. The step is
  ``execute_model``'s call count, which every rank advances at the same call
  (right after ``host_phases.begin_step``), so every rank flips in the same
  step. Each rank logs every flip.

Lines, plain per-rank logging (never ``info_once``): ``q38 HD engaged ...``
at the first worker step (warm compile-cache starts too), ``q38 HD AB ...`` at
each window flip, and ``q38 HD rank=...`` counters every ``STATS_EVERY``
steps.
"""

import os

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MARKS = envs.VLLM_TPU_HD_MARKS
STAGE_CPU = envs.VLLM_TPU_HD_STAGE_CPU
REQVEC = envs.VLLM_TPU_HD_REQVEC
AB = envs.VLLM_TPU_HD_AB
MEMBERS = tuple(
    name for name, on in (("stage_cpu", STAGE_CPU), ("reqvec", REQVEC)) if on
)
ANY = bool(MEMBERS) or MARKS
STATS_EVERY = 2000
# REQVEC acts only on steps with at least this many requests: below it numpy
# overhead exceeds the per-request loop (dev1 j-958db05f microbench: B1 ~93 live
# -146 us on _prepare_inputs, B3 16-28 live +50 us).
REQVEC_MIN_REQS = 48

if AB and not MEMBERS:
    raise ValueError(
        "VLLM_TPU_HD_AB is set but no SV-HOSTDIET member is on "
        "(VLLM_TPU_HD_STAGE_CPU / VLLM_TPU_HD_REQVEC)"
    )


class _State:
    def __init__(self) -> None:
        self.steps = 0
        self.on = not AB
        self.window = 0
        self.on_steps = 0
        self.engaged = False
        self.rank = -1
        self.tp_rank = -1
        self.counts: dict[str, int] = {}


_S = _State()


def _tp_rank() -> int:
    try:
        from vllm.distributed import parallel_state

        return int(parallel_state.get_tensor_model_parallel_rank())
    except Exception:
        return -1


def set_rank(rank: int) -> None:
    _S.rank = int(rank)


def step() -> None:
    """One worker step (``execute_model`` entry). Advances the shared step
    count and the A/B window; no-op with every key off."""
    if not ANY:
        return
    _S.steps += 1
    if not _S.engaged:
        _S.engaged = True
        _S.tp_rank = _tp_rank()
        logger.info(
            "q38 HD engaged rank=%d tp_rank=%d pid=%d members=%s ab=%d marks=%d",
            _S.rank,
            _S.tp_rank,
            os.getpid(),
            ",".join(MEMBERS) or "-",
            AB,
            int(MARKS),
        )
    if AB:
        window = _S.steps // AB
        on = window % 2 == 1
        if window != _S.window or _S.steps == 1:
            _S.window = window
            _S.on = on
            logger.info(
                "q38 HD AB rank=%d tp_rank=%d pid=%d step=%d window=%d on=%d",
                _S.rank,
                _S.tp_rank,
                os.getpid(),
                _S.steps,
                window,
                int(on),
            )
    if _S.on:
        _S.on_steps += 1
    if _S.steps % STATS_EVERY == 0:
        logger.info(
            "q38 HD rank=%d tp_rank=%d pid=%d steps=%d on_steps=%d members=%s %s",
            _S.rank,
            _S.tp_rank,
            os.getpid(),
            _S.steps,
            _S.on_steps,
            ",".join(MEMBERS) or "-",
            " ".join(f"{k}={v}" for k, v in sorted(_S.counts.items())) or "-",
        )


def on(member: str) -> bool:
    """Whether ``member`` acts in the current step."""
    if member == "stage_cpu":
        return STAGE_CPU and _S.on
    if member == "reqvec":
        return REQVEC and _S.on
    raise KeyError(member)


def count(name: str) -> None:
    _S.counts[name] = _S.counts.get(name, 0) + 1


def _reset_for_tests() -> None:
    global _S
    _S = _State()

# SPDX-License-Identifier: Apache-2.0
"""Per-rank counts of model steps by padded token bucket (VLLM_TPU_LOG_STEP_BUCKETS=N).

``count(num_tokens_padded)`` adds one step to that bucket; every N steps the rank logs
the cumulative counts, so a log shows which compiled token buckets a run exercised
(for example, whether short-prompt evals reach the 3,200-token prefill bucket that a
tile change acts on). Host-side only; with N=0 (the default) every call returns at once.
"""

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_EVERY = envs.VLLM_TPU_LOG_STEP_BUCKETS
_counts: dict[int, int] = {}
_steps = 0


def count(num_tokens_padded: int) -> None:
    global _steps
    if not _EVERY:
        return
    n = int(num_tokens_padded)
    _counts[n] = _counts.get(n, 0) + 1
    _steps += 1
    if _steps % _EVERY == 0:
        _report()


def _report() -> None:
    logger.info(
        "STEPBUCKETS steps=%d %s",
        _steps,
        " ".join(f"{k}:{v}" for k, v in sorted(_counts.items())),
    )

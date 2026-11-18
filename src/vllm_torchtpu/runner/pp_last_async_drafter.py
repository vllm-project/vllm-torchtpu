# SPDX-License-Identifier: Apache-2.0
"""The prefill-only MTP drafter under PP last async output (q38 PPARF;
``VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER``, default off).

``pp_last_async_output.decide`` turns the last stage's unwaited return off
whenever a speculative config is set ("speculative decoding"), because a
verifying drafter takes its drafts from host tokens after the wait. A
pipeline producer's prefill-only MTP drafter never verifies: it seeds on the
just-sampled device tokens (``handoff_seed_tokens``), and the pipeline
executor never takes its drafts (``pp_push._install_take_draft_token_ids``).
Yet with the flag off every such prefill runs without PP last async output,
and its drafts come back through a blocking host copy.

With the flag on, three things change, and nothing else:

1. ``decide`` admits a runner whose drafter is a prefill-only MTP producer
   (the runner's ``_mtp_prefill_only``). A verifying drafter is still
   refused.
2. Under PP last async output, ``sample_tokens`` asks the drafter for its
   device drafts (``return_device=True``), so the drafter's graph is not
   forced through a blocking host copy. The drafter's graph is the same;
   only the moment it is enqueued moves.
3. The copy-gate fence (port of q38 HIGHC-PPAE1's ``TPU_PP_COPYGATE_FENCE``
   mode 1, q38-highc-stage7 21fe0655b + 02f57e92f). The sampled-token copy's
   ready event (``AsyncTPUCopyState.copy_ready_event``) is the copy gate: the
   worker's output thread replies with a step's tokens only after it, and a
   register-ahead consumer may pull the step's pages as soon as the first
   token is out. ``fence_pending_device_work`` re-records the event, which
   covers only work ENQUEUED before it (a deferred PyTorch TPU eager graph is
   not). So the runner (a) fences whenever the producer registers ahead (every
   pipeline producer, otherwise by TPU_RAIDEN_STAGE3_REGISTER_AHEAD: a superset
   of the connector's rule, so it can over-fence by one event record but never
   under-fence), with or without a drafter; and (b) enqueues, in ONE
   non-blocking ``synchronize_tensors`` call, the draft tokens plus every root
   registered for the step (``register_root``; cleared at execute_model
   start), and only then re-records the gate. The fence sits before the
   unwaited return. The runner's own commit reads the scheduler's tokens
   (``apply_scheduled_token_ids``), so the output thread is the gate's only
   consumer.

Host-side only: no traced program reads the flag, so it is not a compile-key
factor (tpu_compiler._TPU_COMPILE_ENV_IGNORED).
"""

import os
from collections.abc import Callable, Sequence
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER"

# Engagement (the reader's markers): one line when the runner's decision is
# made, then one at the first gate re-arm of the process and every
# REPORT_EVERY re-arms after it.
REPORT_EVERY = 4096
_COUNT = {"fences": 0, "roots": 0}
_DECISION = {"logged": False}


def enabled() -> bool:
    return bool(envs.VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER)


def registers_ahead(*, is_kv_producer: bool, pipeline_parallel_size: int, register_ahead_flag: bool) -> bool:
    """Whether this producer's Stage-3 send can be pulled before its finish:
    every pipeline producer, otherwise by TPU_RAIDEN_STAGE3_REGISTER_AHEAD."""
    return bool(is_kv_producer) and (int(pipeline_parallel_size) > 1 or bool(register_ahead_flag))


def register_root(runner: Any, tensor: Any) -> None:
    """Records a device tensor whose pending graph writes KV pages of the
    current step, so the gate fence enqueues it."""
    if tensor is None:
        return
    runner.__dict__.setdefault("_copygate_roots", []).append(tensor)


def take_roots(runner: Any) -> list[Any]:
    roots = runner.__dict__.get("_copygate_roots") or []
    runner.__dict__["_copygate_roots"] = []
    return list(roots)


def fence(
    copy_state: Any,
    roots: Sequence[Any],
    *,
    synchronize: Callable[[list[Any]], None],
    skip_dispatch: bool = False,
    skip_fence: bool = False,
) -> None:
    """Enqueue every root in one call, then re-arm the copy gate behind it.
    ``skip_dispatch`` / ``skip_fence`` mirror the source's probe controls; this
    tree has no probe, so the runner passes False for both."""
    roots = [r for r in roots if r is not None]
    if roots and not skip_dispatch:
        synchronize(roots)
    if not skip_fence:
        copy_state.fence_pending_device_work()
    _note(0 if skip_dispatch else len(roots), not skip_fence)


def _note(nroots: int, fenced: bool) -> None:
    _COUNT["roots"] += nroots
    if not fenced:
        return
    _COUNT["fences"] += 1
    n = _COUNT["fences"]
    if n == 1 or n % REPORT_EVERY == 0:
        logger.info(
            "q38 PPARF copy-gate fence (%s=%s): %d gate re-arm(s), %d root(s) enqueued in this process so far",
            ENV, os.environ.get(ENV), n, _COUNT["roots"],
        )


def note_decision(on: bool, reason: str, *, is_last_stage: bool, prefill_only_drafter: bool) -> None:
    """The one-time engagement line, on the ranks that sample and draft (the
    last stage of a prefill-only MTP producer). The runner calls this only with
    the flag on, so with the flag off nothing is printed."""
    if _DECISION["logged"] or not (is_last_stage and prefill_only_drafter):
        return
    _DECISION["logged"] = True
    if on:
        logger.info(
            "q38 PPARF on: prefill-only drafter admitted under PP last async output (%s=%s)",
            ENV, os.environ.get(ENV),
        )
    else:
        logger.info(
            "q38 PPARF NOT engaged: PP last async output is off (%s) (%s=%s)",
            reason, ENV, os.environ.get(ENV),
        )

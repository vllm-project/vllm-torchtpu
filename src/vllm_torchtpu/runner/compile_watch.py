# SPDX-License-Identifier: Apache-2.0
"""Say what compiles after warmup, under ``VLLM_XLA_CHECK_RECOMPILATION=1``.

Every program a server runs is meant to be compiled by the warmup. One that
compiles mid-serving stalls every lockstep rank and allocates HBM scratch
next to a full KV cache, but the server log only names it (``Start
TpuCompiler::Compile for tt_jit_...``): no shapes, no caller. With the
recompilation check on, once warmup is done this turns on, for the rest of
the process:

- one WARNING per dynamo compile, with its compile id, duration and the
  vllm_torchtpu frame that triggered it;
- torch's ``recompiles`` log, which names the guard that failed (the input
  whose shape, stride or value changed);
- torch-tpu's StableHLO dump on a compilation-cache miss, which carries the
  argument shapes of the eager ``tt_jit_*`` programs dynamo never sees.

All three only fire on a compile, so a warmed server pays nothing per step.
"""

import time
import traceback

import torch
import vllm.envs as vllm_envs

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_installed = False


def _trigger_frame() -> str:
    for frame in reversed(traceback.extract_stack()[:-2]):
        path = frame.filename
        if "/vllm_torchtpu/" in path and not path.endswith("compile_watch.py"):
            return f"{path.rsplit('/', 1)[-1]}:{frame.lineno} {frame.name}"
    return "?"


def install_post_warmup_compile_log() -> None:
    """Turn the post-warmup compile reports on; a no-op unless
    ``VLLM_XLA_CHECK_RECOMPILATION`` is set, and idempotent."""
    global _installed
    if _installed or not vllm_envs.VLLM_XLA_CHECK_RECOMPILATION:
        return
    from torch._dynamo.callback import callback_handler

    state = {"count": 0, "start": 0.0}

    def on_start(args) -> None:
        state["start"] = time.perf_counter()

    def on_end(args) -> None:
        state["count"] += 1
        logger.warning(
            "Post-warmup dynamo compile #%d (%s, %.0f ms) from %s",
            state["count"],
            getattr(args, "compile_id", "?"),
            (time.perf_counter() - state["start"]) * 1e3,
            _trigger_frame(),
        )

    callback_handler.register_start_callback(on_start)
    callback_handler.register_end_callback(on_end)
    torch._logging.set_logs(recompiles=True, graph_breaks=True)
    tpu = getattr(torch, "tpu", None)
    if tpu is not None and hasattr(tpu, "_set_dump_on_cache_miss"):
        tpu._set_dump_on_cache_miss(True)
    _installed = True
    logger.info(
        "Post-warmup compile reports on: dynamo compiles, recompile guards, "
        "StableHLO dumps on compilation-cache misses"
    )

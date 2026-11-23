# SPDX-License-Identifier: Apache-2.0
"""Report torch.compile (dynamo) compiles that happen after warmup.

Every graph the server runs is meant to be compiled by the warmup; a compile
afterwards is a stall on that rank (hundreds of ms) that every lockstep rank
waits out. Server logs say nothing about them by default, so once warmup is
done each later compile logs one WARNING line naming its count, duration
and the vllm_torchtpu frame that triggered it:

    Post-warmup compile #3 (12/76, 219 ms) from tpu_runner.py:2746
    _apply_async_token_substitution

Off by default; ``VLLM_TPU_LOG_POST_WARMUP_COMPILES=1`` turns it on.
"""

import sys
import time

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_installed = False


def _arg_shape(v) -> str | None:
    """`(shape):dtype` of a tensor arg; `[n x (first)..(last):dtype]` of a
    non-empty list/tuple of tensors (e.g. VERIFY_RO_INGRAPH's slot groups)."""
    if isinstance(v, torch.Tensor):
        return f"{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if (
        isinstance(v, (list, tuple))
        and v
        and all(isinstance(t, torch.Tensor) for t in v)
    ):
        return (
            f"[{len(v)}x{tuple(v[0].shape)}..{tuple(v[-1].shape)}:"
            f"{str(v[0].dtype).replace('torch.', '')}]"
        )
    return None


def _trigger_frame() -> str:
    """The innermost vllm_torchtpu frame, with the shapes of its tensor locals
    (which name the bucket that was not warmed)."""
    frame = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename
        if "/vllm_torchtpu/" in path and not path.endswith("compile_watch.py"):
            loc = frame.f_locals
            shapes = [
                f"{name}={tuple(value.shape)}"
                for name, value in loc.items()
                if isinstance(value, torch.Tensor)
            ][:8]
            label = f"{path.rsplit('/', 1)[-1]}:{frame.f_lineno} {frame.f_code.co_name}"
            fn = loc.get("fn")
            if fn is not None and isinstance(loc.get("args"), tuple):
                # A generic wrapper (e.g. eagle3._static_shape_args): name the
                # wrapped function and its tensor args' shapes/dtypes.
                label += f" -> {getattr(fn, '__qualname__', repr(fn))}"
                shapes = [
                    f"a{i}={_arg_shape(v)}"
                    for i, v in enumerate(loc["args"])
                    if _arg_shape(v) is not None
                ][:8]
            return label + (f" [{' '.join(shapes)}]" if shapes else "")
        frame = frame.f_back
    return "?"


def install_post_warmup_compile_log() -> None:
    global _installed
    if _installed or not envs.VLLM_TPU_LOG_POST_WARMUP_COMPILES:
        return
    from torch._dynamo.callback import callback_handler

    state = {"count": 0, "start": 0.0}

    def on_start(args) -> None:
        state["start"] = time.perf_counter()

    def on_end(args) -> None:
        state["count"] += 1
        logger.warning(
            "Post-warmup compile #%d (%s, %.0f ms) from %s",
            state["count"],
            getattr(args, "compile_id", "?"),
            (time.perf_counter() - state["start"]) * 1e3,
            _trigger_frame(),
        )

    callback_handler.register_start_callback(on_start)
    callback_handler.register_end_callback(on_end)
    _installed = True

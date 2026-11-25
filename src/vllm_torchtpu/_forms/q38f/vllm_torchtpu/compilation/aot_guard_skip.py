# SPDX-License-Identifier: Apache-2.0
"""Skip the per-call Dynamo guard check of a freshly AOT-compiled model when
vLLM does not evaluate guards (VLLM_TPU_AOT_SKIP_GUARD_CHECK).

Ported for q38 L4 D98 from MM's mm/code/vllm-torchtpu-host-rb
src/vllm_torchtpu/compilation/aot_guard_skip.py (unchanged skip logic), and
carried onto the D14 serve tree (q38-L1-d14serve 965b3c6e7) as q38 L4
X-H14-AOTSKIP: there the freshly compiled whole-step call (model_mk_whole),
the draft's forward0 and the fused draft loop take the skip.

vLLM already disables it for an AOT artifact loaded from the cache
(`decorators._try_load_aot_compiled_fn`, when
`dynamic_shapes_config.evaluate_guards` is off), but not for the one it has
just compiled, which then binds the forward's arguments with
`inspect.Signature.bind` and runs the guard manager on every call: ~60 us of
host time per backbone launch on the serving critical path (MM, measured).
Guards are not evaluated for the cached artifact either, so skipping them here
changes nothing but that cost.

On this tree the flag is part of the hashed compile key (lead ruling 18:3xZ
11-13): an arm that sets it gets its own vLLM AOT artifact directory, so it
compiles its own forwards (and takes the skip) instead of loading a twin arm's
artifact warm from a shared in-pod cache, which would measure nothing.

Engagement (one line per AOT artifact, tagged with the process rank, read by
tests/q38_aotskip/q38_aotskip_engage.py):
- ``AOT_GUARD_SKIP engaged: rank=R cls=C guard_check_enabled=False``: a fresh
  compile whose artifact really has its guard check off after the skip;
- ``AOT_GUARD_SKIP cache-loaded: rank=R cls=C guard_check_enabled=B``: an
  artifact loaded from the cache (vLLM's own skip; B must be False);
- ``AOT_GUARD_SKIP not engaged: rank=R cls=C (...)``: the skip did not apply.
"""

import functools
import os

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", "-1") or -1)


def _check_state(fn) -> str:
    return str(getattr(fn, "_guard_check_enabled", "unknown"))


def patch_aot_guard_skip() -> None:
    if not envs.VLLM_TPU_AOT_SKIP_GUARD_CHECK:
        return
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    if not getattr(TorchCompileWithNoGuardsWrapper.aot_compile, "_tpu_guard_skip", False):
        original = TorchCompileWithNoGuardsWrapper.aot_compile

        @functools.wraps(original)
        def aot_compile(self, *args, **kwargs):
            compiled = original(self, *args, **kwargs)
            config = getattr(self, "compilation_config", None)
            evaluate = getattr(
                getattr(config, "dynamic_shapes_config", None), "evaluate_guards", True
            )
            if not evaluate and hasattr(compiled, "disable_guard_check"):
                compiled.disable_guard_check()
                state = _check_state(compiled)
                if state == "False":
                    logger.info(
                        "AOT_GUARD_SKIP engaged: rank=%d cls=%s guard_check_enabled=%s",
                        _rank(),
                        type(self).__name__,
                        state,
                    )
                else:
                    logger.info(
                        "AOT_GUARD_SKIP not engaged: rank=%d cls=%s (disable_guard_check "
                        "left guard_check_enabled=%s)",
                        _rank(),
                        type(self).__name__,
                        state,
                    )
            else:
                logger.info(
                    "AOT_GUARD_SKIP not engaged: rank=%d cls=%s (evaluate_guards=%s, "
                    "disable_guard_check=%s)",
                    _rank(),
                    type(self).__name__,
                    evaluate,
                    hasattr(compiled, "disable_guard_check"),
                )
            return compiled

        aot_compile._tpu_guard_skip = True
        TorchCompileWithNoGuardsWrapper.aot_compile = aot_compile

    load = getattr(decorators, "_try_load_aot_compiled_fn", None)
    if load is not None and not getattr(load, "_tpu_guard_skip", False):

        @functools.wraps(load)
        def _try_load_aot_compiled_fn(model, *args, **kwargs):
            loaded = load(model, *args, **kwargs)
            if loaded is not None:
                logger.info(
                    "AOT_GUARD_SKIP cache-loaded: rank=%d cls=%s guard_check_enabled=%s",
                    _rank(),
                    type(model).__name__,
                    _check_state(loaded),
                )
            return loaded

        _try_load_aot_compiled_fn._tpu_guard_skip = True
        decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn
    logger.info("AOT-compiled models skip the per-call guard check")

# SPDX-License-Identifier: Apache-2.0
"""Skip the per-call Dynamo guard check of a freshly AOT-compiled model when
vLLM does not evaluate guards (VLLM_TPU_AOT_SKIP_GUARD_CHECK).

vLLM already disables it for an AOT artifact loaded from the cache
(`decorators._try_load_aot_compiled_fn`, when
`dynamic_shapes_config.evaluate_guards` is off), but not for the one it has
just compiled, which then binds the forward's arguments with
`inspect.Signature.bind` and runs the guard manager on every call: ~60 us of
host time per backbone launch on the serving critical path. Guards are not
evaluated for the cached artifact either, so skipping them here changes
nothing but that cost.
"""

import functools

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_aot_guard_skip() -> None:
    if not envs.VLLM_TPU_AOT_SKIP_GUARD_CHECK:
        return
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    if getattr(TorchCompileWithNoGuardsWrapper.aot_compile, "_tpu_guard_skip", False):
        return
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
        return compiled

    aot_compile._tpu_guard_skip = True
    TorchCompileWithNoGuardsWrapper.aot_compile = aot_compile
    logger.info("AOT-compiled models skip the per-call guard check")

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Skip the per-call Dynamo guard check of a freshly AOT-compiled model when
vLLM does not evaluate guards.

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

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_aot_guard_skip() -> None:
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

    aot_compile.__dict__["_tpu_guard_skip"] = True
    TorchCompileWithNoGuardsWrapper.aot_compile = aot_compile
    logger.info("AOT-compiled models skip the per-call guard check")

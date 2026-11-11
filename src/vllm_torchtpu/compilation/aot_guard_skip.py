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
"""Skip per-call guard checks on guard-free AOT-compiled forwards."""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_installed = False
_active = False


def guards_dropped(compilation_config: Any) -> bool:
    from vllm.config import CompilationMode

    return not compilation_config.dynamic_shapes_config.evaluate_guards and (
        compilation_config.mode != CompilationMode.STOCK_TORCH_COMPILE
    )


def should_enable(vllm_config: Any) -> bool:
    model_cfg = getattr(vllm_config, "model_config", None)
    cache_cfg = getattr(vllm_config, "cache_config", None)
    par_cfg = getattr(vllm_config, "parallel_config", None)
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    if model_cfg is None or cache_cfg is None or par_cfg is None or comp_cfg is None:
        return False
    return (
        not bool(
            getattr(
                model_cfg,
                "supports_multimodal_inputs",
                getattr(model_cfg, "is_multimodal_model", False),
            )
        )
        and not bool(getattr(model_cfg, "enforce_eager", False))
        and not bool(getattr(cache_cfg, "enable_prefix_caching", True))
        and getattr(vllm_config, "lora_config", None) is None
        and getattr(vllm_config, "speculative_config", None) is None
        and getattr(vllm_config, "kv_transfer_config", None) is None
        and getattr(par_cfg, "data_parallel_size", 1) == 1
        and guards_dropped(comp_cfg)
    )


def disable(fn: Any) -> bool:
    if fn is None or not hasattr(fn, "disable_guard_check"):
        return False
    fn.disable_guard_check()
    return True


def set_active(active: bool) -> None:
    global _active
    _active = active


def install() -> bool:
    global _installed, _active
    _active = True
    if _installed:
        return True
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    original = TorchCompileWithNoGuardsWrapper.aot_compile

    def aot_compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        fn = original(self, *args, **kwargs)
        if _active and should_enable(self.vllm_config) and disable(fn):
            logger.info("Disabled empty AOT guard check for %s", type(self).__name__)
        return fn

    aot_compile._tpu_original = original  # type: ignore[attr-defined]
    TorchCompileWithNoGuardsWrapper.aot_compile = aot_compile
    _installed = True
    return True

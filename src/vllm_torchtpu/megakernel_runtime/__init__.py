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
"""PyTorch/vLLM runtime bridge, weight collector, and driver for `megakernel`."""

from __future__ import annotations

from typing import Any

from .hooks import StepContext, is_suppressed, suppressed

_DRIVER_EXPORTS = frozenset(
    {
        "MegakernelDriver",
        "TorchStepMetaOps",
        "prepare_step_inputs_eager",
    }
)

_BRIDGE_EXPORTS = frozenset(
    {
        "PooledStepExport",
        "build_pooled_step_op",
        "execute_pooled_step_inplace",
        "export_pooled_step",
        "make_typed_step_wrapper",
        "sharded_jax_callable",
    }
)

_COLLECTOR_EXPORTS = frozenset(
    {
        "TorchArrayOps",
        "VLLMWeightSource",
        "WEIGHT_FLIPPED_ATTR",
        "apply_tp_moe_flags",
        "extract_ecopy_layer",
        "find_backbone",
        "get_routed_experts",
        "load_vllm_weights",
        "reshard_experts_ep_to_tp_local",
        "reshard_layer_experts_ep_to_tp",
    }
)


def _is_megakernel_requested(runner: Any = None) -> bool:
    import vllm_torchtpu.envs as envs

    raw_env = str(envs.VLLM_TPU_MEGAKERNEL or "").strip().lower()
    if raw_env and raw_env not in ("0", "false", "no", "off", "none"):
        return True
    if runner is not None:
        vllm_cfg = getattr(runner, "vllm_config", None)
        add_cfg = getattr(vllm_cfg, "additional_config", None)
        if isinstance(add_cfg, dict):
            mk_cfg = add_cfg.get("megakernel")
            if isinstance(mk_cfg, dict):
                en = mk_cfg.get("enabled", True)
                if isinstance(en, str):
                    return en.strip().lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                        "off",
                        "none",
                    )
                return bool(en)
            if isinstance(mk_cfg, str):
                return mk_cfg.strip().lower() not in (
                    "",
                    "0",
                    "false",
                    "no",
                    "off",
                    "none",
                )
            if mk_cfg:
                return True
    return False


def maybe_create(runner: Any) -> Any:
    """Create a megakernel driver for `runner` if enabled, importing lazily."""
    if not _is_megakernel_requested(runner):
        return None
    from .driver import maybe_create as _driver_maybe_create

    return _driver_maybe_create(runner)


def __getattr__(name: str) -> Any:
    if name in _DRIVER_EXPORTS:
        from . import driver

        return getattr(driver, name)
    if name in _BRIDGE_EXPORTS:
        from . import exported_bridge

        return getattr(exported_bridge, name)
    if name in _COLLECTOR_EXPORTS:
        from . import weight_collector

        return getattr(weight_collector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MegakernelDriver",
    "PooledStepExport",
    "StepContext",
    "TorchArrayOps",
    "TorchStepMetaOps",
    "VLLMWeightSource",
    "WEIGHT_FLIPPED_ATTR",
    "apply_tp_moe_flags",
    "build_pooled_step_op",
    "execute_pooled_step_inplace",
    "export_pooled_step",
    "extract_ecopy_layer",
    "find_backbone",
    "get_routed_experts",
    "is_suppressed",
    "load_vllm_weights",
    "make_typed_step_wrapper",
    "maybe_create",
    "prepare_step_inputs_eager",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
    "sharded_jax_callable",
    "suppressed",
]

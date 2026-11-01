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

from .errors import MegakernelUnsupported
from .hooks import StepContext, is_suppressed, suppressed

_DRIVER_EXPORTS = frozenset({"ProgramDriver"})
_STEP_META_EXPORTS = frozenset({"TorchStepMetaOps"})

# Removed names -> their replacement (an `AttributeError` says where to go).
_MOVED = {
    "MegakernelDriver": (
        "megakernel_runtime.ProgramDriver with "
        "vllm_torchtpu.models.qwen3_5.megakernel.adapter.Qwen35Program"
    ),
    "prepare_step_inputs_eager": (
        "vllm_torchtpu.models.qwen3_5.megakernel.adapter.prepare_step_inputs_eager"
    ),
}

_BRIDGE_EXPORTS = frozenset(
    {
        "PooledStepExport",
        "build_pooled_step_op",
        "execute_pooled_step_inplace",
        "export_pooled_step",
    }
)
_JAX_BRIDGE_EXPORTS = frozenset(
    {"build_jax_step_op", "make_typed_step_wrapper", "sharded_jax_callable"}
)

_WEIGHT_SOURCE_EXPORTS = frozenset(
    {"TorchArrayOps", "VLLMWeightSource", "WEIGHT_FLIPPED_ATTR", "load_vllm_weights"}
)
_COLLECTOR_EXPORTS = frozenset(
    {
        "apply_tp_moe_flags",
        "extract_ecopy_layer",
        "get_routed_experts",
        "reshard_experts_ep_to_tp_local",
        "reshard_layer_experts_ep_to_tp",
    }
)


def _is_megakernel_requested(runner: Any = None) -> bool:
    import vllm_torchtpu.envs as envs

    from .config import collect_config

    v_cfg = getattr(runner, "vllm_config", None) if runner is not None else None
    add_cfg = getattr(v_cfg, "additional_config", None)
    return collect_config(envs.VLLM_TPU_MEGAKERNEL, add_cfg).enabled


def guard_weight_update(runner: Any) -> None:
    """Raise `MegakernelUnsupported("weight_update")` while a megakernel serves
    `runner`: its packed weights would go stale."""
    if getattr(getattr(runner, "_megakernel", None), "serving", False) is True:
        raise MegakernelUnsupported("weight_update")


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
    if name in _STEP_META_EXPORTS:
        from . import step_meta_ops

        return getattr(step_meta_ops, name)
    if name == "find_backbone":
        from .backbone import find_backbone

        return find_backbone
    if name in _JAX_BRIDGE_EXPORTS:
        from . import jax_bridge

        return getattr(jax_bridge, name)
    if name in _BRIDGE_EXPORTS:
        from . import exported_bridge

        return getattr(exported_bridge, name)
    if name in _WEIGHT_SOURCE_EXPORTS:
        from . import weight_source

        return getattr(weight_source, name)
    if name in _COLLECTOR_EXPORTS:
        from . import weight_collector

        return getattr(weight_collector, name)
    if name in _MOVED:
        raise AttributeError(
            f"module {__name__!r} no longer exports {name!r}; use {_MOVED[name]}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MegakernelUnsupported",
    "PooledStepExport",
    "ProgramDriver",
    "StepContext",
    "TorchArrayOps",
    "TorchStepMetaOps",
    "VLLMWeightSource",
    "WEIGHT_FLIPPED_ATTR",
    "apply_tp_moe_flags",
    "build_jax_step_op",
    "build_pooled_step_op",
    "execute_pooled_step_inplace",
    "export_pooled_step",
    "extract_ecopy_layer",
    "find_backbone",
    "get_routed_experts",
    "guard_weight_update",
    "is_suppressed",
    "load_vllm_weights",
    "make_typed_step_wrapper",
    "maybe_create",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
    "sharded_jax_callable",
    "suppressed",
]

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

_DRIVER_EXPORTS = frozenset(
    {
        "MKABCalibrator",
        "MegakernelDriver",
        "Qwen35Megakernel",
        "parse_megakernel_config",
        "parse_mk_buckets",
        "parse_mk_verify_buckets",
        "prepare_step_inputs_eager",
        "prepare_step_metadata",
        "register_megakernel_driver",
        "select_bucket",
    }
)

_BRIDGE_EXPORTS = frozenset(
    {
        "MegakernelConfigError",
        "PooledStepExport",
        "build_pooled_step_op",
        "execute_pooled_step_inplace",
        "export_pooled_step",
        "make_typed_step_wrapper",
        "sharded_jax_callable",
        "verify_pooled_step_hlo",
        "verify_torchtpu_compiled_step_ir",
    }
)

_COLLECTOR_EXPORTS = frozenset(
    {
        "apply_tp_moe_flags",
        "extract_ecopy_layer",
        "extract_kn_fp8_weight",
        "find_backbone",
        "get_routed_experts",
        "pack_att_in_layer",
        "pack_att_out_layer",
        "pack_expert_views_layer",
        "pack_lin_in_layer",
        "pack_lin_out_layer",
        "pack_router_and_shared_layer",
        "pack_smalls_layer",
        "pack_weights",
        "quantize_ba_weight",
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
    try:
        from . import qwen38_driver  # noqa: F401  (registers Qwen4ExpForCausalLM)
    except Exception:  # fail closed: a broken optional driver must not stop serving
        from vllm_torchtpu.logger import init_logger

        init_logger(__name__).exception("Qwen3.8 megakernel driver unavailable; stock path")
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
    "MKABCalibrator",
    "MegakernelConfigError",
    "MegakernelDriver",
    "PooledStepExport",
    "Qwen35Megakernel",
    "apply_tp_moe_flags",
    "build_pooled_step_op",
    "execute_pooled_step_inplace",
    "export_pooled_step",
    "extract_ecopy_layer",
    "extract_kn_fp8_weight",
    "find_backbone",
    "get_routed_experts",
    "make_typed_step_wrapper",
    "maybe_create",
    "pack_att_in_layer",
    "pack_att_out_layer",
    "pack_expert_views_layer",
    "pack_lin_in_layer",
    "pack_lin_out_layer",
    "pack_router_and_shared_layer",
    "pack_smalls_layer",
    "pack_weights",
    "parse_megakernel_config",
    "parse_mk_buckets",
    "parse_mk_verify_buckets",
    "prepare_step_inputs_eager",
    "prepare_step_metadata",
    "quantize_ba_weight",
    "register_megakernel_driver",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
    "select_bucket",
    "sharded_jax_callable",
    "verify_pooled_step_hlo",
    "verify_torchtpu_compiled_step_ir",
]

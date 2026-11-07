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
"""Config checks and the Loop / mHC execution plan of SenseNova-Loop (no model
imports, so register_models can load it while the platform registers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vllm.model_executor.models.config import Qwen3_5ForCausalLMConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

LAYER_TYPES = ("linear_attention", "full_attention")
# Set by SenseNovaLoopConfig once hf_text_config describes the visit layers.
_PHYSICAL_LAYER_TYPES = "sensenova_physical_layer_types"


# ----------------------------------------------------------------------------
# Execution plan (checkpoint `loop_settings` and `mhc`)
# ----------------------------------------------------------------------------
def _setting(hf_config: Any, name: str) -> dict | None:
    value = getattr(hf_config, name, None)
    return value if value is None or isinstance(value, dict) else dict(value)


def execution_plan(hf_config: Any, num_layers: int) -> tuple[int, ...]:
    """Physical layer indices in execution order (remote code `execution_plan`)."""
    loop = _setting(hf_config, "loop_settings")
    if loop is None:
        raise ValueError("SenseNova-Loop needs an explicit loop_settings")
    start, end, count = (loop.get(key) for key in ("start", "end", "eval_count"))
    if any(type(value) is not int for value in (start, end, count)) or not (
        0 <= start < end <= num_layers and count >= 1
    ):
        raise ValueError(f"Invalid Loop interval or eval_count: {loop}")
    if loop.get("counts") != [count]:
        raise ValueError("Only fixed-count Loop checkpoints are supported")
    loop_visits = tuple(range(start, end)) * count
    return tuple(range(start)) + loop_visits + tuple(range(end, num_layers))


def mhc_settings(hf_config: Any, num_layers: int) -> dict | None:
    """Validated `mhc` settings, or None for a checkpoint without mHC."""
    mhc = _setting(hf_config, "mhc")
    if mhc is None:
        return None
    allowed = {"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"}
    if set(mhc) - allowed:
        raise ValueError(f"Unknown mHC fields: {sorted(set(mhc) - allowed)}")
    start, end, streams, iterations = (
        mhc.get(key) for key in ("start", "end", "streams", "iterations")
    )
    if any(type(v) is not int for v in (start, end, streams, iterations)) or not (
        0 <= start < end <= num_layers and streams in (2, 4) and iterations >= 1
    ):
        raise ValueError(f"Invalid mHC range, streams or iterations: {mhc}")
    loop = cast(dict, _setting(hf_config, "loop_settings"))
    bounds = (0, loop["start"], loop["end"], num_layers)
    if not any(lo <= start < end <= hi for lo, hi in zip(bounds, bounds[1:])):
        raise ValueError("mHC cannot cross a Loop phase boundary")
    # The fixed-identity mix and the branch pre-scale are valid checkpoints,
    # but no released one uses them and the TPU path is not tested for them.
    if mhc.get("fixed_identity", False) or mhc.get("pre_scale", 1.0) != 1.0:
        raise NotImplementedError(
            f"mHC fixed_identity / pre_scale != 1.0 are not supported on TPU: {mhc}"
        )
    return {"start": start, "end": end, "streams": streams, "iterations": iterations}


def physical_layer_types(text_config: Any) -> list[str]:
    """The checkpoint's per-physical-layer types, before the visit extension."""
    return list(getattr(text_config, _PHYSICAL_LAYER_TYPES, text_config.layer_types))


def visit_layer_physicals(plan: tuple[int, ...]) -> list[int]:
    """Physical index of each repeat-visit module, in execution order."""
    seen: set[int] = set()
    repeats = []
    for physical in plan:
        if physical in seen:
            repeats.append(physical)
        seen.add(physical)
    return repeats


class SenseNovaLoopConfig(Qwen3_5ForCausalLMConfig):
    """Qwen3.5 text checks (fp32 SSM state from mamba_ssm_dtype, 1-D RoPE:
    text positions make interleaved mRoPE equal plain RoPE), plus the Loop
    plan validated up front and hf_text_config extended to the visit layers,
    so layer counts by block type see every cache-owning layer."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_5ForCausalLMConfig.verify_and_update_config(vllm_config)
        model_config = vllm_config.model_config
        # The upstream check strips mRoPE from the text config only; the
        # top-level (vision-language) config must not re-enable it either.
        rope_parameters = getattr(model_config.hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "SenseNova-Loop on TPU does not support speculative decoding yet."
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "SenseNova-Loop on TPU does not support context parallelism yet."
            )
        text_config = model_config.hf_text_config
        if getattr(text_config, "layer_scale", False):
            raise NotImplementedError("SenseNova-Loop layer_scale is not supported")
        types = physical_layer_types(text_config)
        if set(types) - set(LAYER_TYPES):
            raise NotImplementedError(f"Unsupported layer types {set(types)}")
        plan = execution_plan(model_config.hf_config, len(types))
        mhc_settings(model_config.hf_config, len(types))
        if hasattr(text_config, _PHYSICAL_LAYER_TYPES):
            return  # already extended (config re-verified)
        setattr(text_config, _PHYSICAL_LAYER_TYPES, list(types))
        visit_types = types + [types[p] for p in visit_layer_physicals(plan)]
        text_config.layer_types = visit_types
        text_config.num_hidden_layers = len(visit_types)

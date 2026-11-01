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
"""Startup guards: a spec's declared fields (chips, TP sizes, collective
domain, parallelism, features, required server settings) against the runner's
config, before `create`. Config and local device facts only: no collective."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from vllm_torchtpu import _OFFLOADING_CONNECTORS
from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_TP,
    PARALLELISM_UNSUPPORTED,
    REQUIRES_KEYS,
    REQUIRES_UNMET,
    UNSUPPORTED_TP_SIZE,
    UNSUPPORTED_TPU_CHIP,
    EligibilityResult,
)
from vllm_torchtpu.megakernel_runtime.backbone import find_backbone
from vllm_torchtpu.megakernel_runtime.plugin import RunnerView


def _ep_group_size() -> int:
    """vLLM's expert-parallel group size (1 when EP is off or unavailable)."""
    try:
        from vllm.distributed.parallel_state import get_ep_group
    except ImportError:
        return 1
    try:
        group = get_ep_group()
    except (AssertionError, AttributeError, RuntimeError):
        return 1
    return int(getattr(group, "world_size", 1) or 1) if group is not None else 1


_OFFLOAD_CONNECTORS = _OFFLOADING_CONNECTORS


def _runner_cfg(runner: Any, name: str) -> Any:
    v_cfg = getattr(runner, "vllm_config", None)
    cfg = getattr(v_cfg, name, None) if v_cfg is not None else None
    return cfg if cfg is not None else getattr(runner, name, None)


def kv_connector_names(kv_transfer_config: Any) -> tuple[str, ...]:
    """The KV connector names in effect (a `MultiConnector`'s members)."""
    if kv_transfer_config is None:
        return ()
    name = str(getattr(kv_transfer_config, "kv_connector", None) or "")
    if name != "MultiConnector":
        return (name,)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    members = extra.get("connectors", ()) if isinstance(extra, dict) else ()
    names = tuple(
        str(
            (m.get("kv_connector") if isinstance(m, dict) else None)
            or getattr(m, "kv_connector", None)
            or ""
        )
        for m in members
    )
    return names or (name,)


def kv_connector_is_offloading(kv_transfer_config: Any) -> bool:
    """Offloading connectors only (a `MultiConnector` only if every member is)."""
    names = kv_connector_names(kv_transfer_config)
    return bool(names) and all(n in _OFFLOAD_CONNECTORS for n in names)


def _aux_hidden_layers(backbone: Any) -> Any:
    return getattr(backbone, "aux_hidden_state_layers", ()) if backbone else ()


def _runner_backbone(runner: Any) -> Any:
    try:
        return find_backbone(getattr(runner, "model", None))
    except ValueError:
        return None


def _decline(code: str, detail: str) -> EligibilityResult:
    return EligibilityResult(False, code, f"Megakernel {detail}; skipping.")


def _mm_limit_count(limit: Any) -> int:
    """Item count of one `limit_mm_per_prompt` entry (an int, or vLLM's
    per-modality options object / dict with a `count`)."""
    if isinstance(limit, Mapping):
        limit = limit.get("count", 1)
    else:
        limit = getattr(limit, "count", limit)
    try:
        return int(limit)
    except (TypeError, ValueError):
        return 1


def _multimodal_inputs_possible(runner: Any) -> bool:
    """Whether a request on this server can carry multimodal inputs.

    `runner.supports_mm_inputs` alone over-reports: vLLM 0.32 turns on
    `enable_mm_embeds` for every KV connector with a consumer role (CPU KV
    offload uses `kv_role="kv_both"`), which makes a language-model-only
    server look multimodal. Inputs can't occur when the model config is
    language-model-only or every `limit_mm_per_prompt` entry is 0; the
    per-step `inputs_embeds` check (`ProgramDriver._text_ctx`) stays as the
    safety net either way.
    """
    if not bool(getattr(runner, "supports_mm_inputs", False)):
        return False
    vcfg = getattr(runner, "vllm_config", None)
    m_cfg = getattr(vcfg, "model_config", None) or getattr(runner, "model_config", None)
    mm_cfg = getattr(m_cfg, "multimodal_config", None)
    if mm_cfg is None:
        return True
    if bool(getattr(mm_cfg, "language_model_only", False)):
        return False
    limits = getattr(mm_cfg, "limit_per_prompt", None)
    if isinstance(limits, Mapping) and limits:
        return any(_mm_limit_count(v) > 0 for v in limits.values())
    return True


_PARALLELISM_DETAIL = {
    "dp": "agrees across the TP group only: data_parallel_size > 1 is unsupported",
    "ep_wider_than_tp": (
        "agrees across the TP group only: an EP group wider than TP is unsupported"
    ),
    "dp_lockstep": "does not support EP-DP lockstep",
    "pp": "requires a single-stage pipeline (PP=1)",
    "dcp": "does not support decode context parallelism (DCP)",
    "pcp": "does not support prefill context parallelism (PCP)",
    "ep_in_tp": "does not support expert parallelism inside the TP group",
}


def detect_parallelism(runner: Any, tp_size: int) -> list[str]:
    """The parallelism kinds (beyond "tp") the runner uses, in check order.
    Reads config and group sizes only: no collective."""
    p_cfg = _runner_cfg(runner, "parallel_config")
    dp = int(os.environ.get("TORCH_TPU_DP_SIZE", "0") or 0) or int(
        getattr(p_cfg, "data_parallel_size", 1) or 1
    )
    ep_wide = _ep_group_size() > max(1, int(tp_size))
    dp_fn = getattr(runner, "_dp_lockstep_enabled", None)
    pp_multi = not (
        getattr(runner, "_pp_is_first", True) and getattr(runner, "_pp_is_last", True)
    )
    dcp = (
        bool(getattr(runner, "_dcp_enabled", False))
        or int(getattr(p_cfg, "decode_context_parallel_size", 1) or 1) > 1
    )
    pcp = (
        bool(getattr(runner, "_pcp_enabled", False))
        or int(getattr(p_cfg, "prefill_context_parallel_size", 1) or 1) > 1
    )
    ep_in_tp = bool(getattr(p_cfg, "enable_expert_parallel", False)) and not ep_wide
    flags = (
        ("dp", dp > 1),
        ("ep_wider_than_tp", ep_wide),
        ("dp_lockstep", bool(callable(dp_fn) and dp_fn())),
        ("pp", pp_multi),
        ("dcp", dcp),
        ("pcp", pcp),
        ("ep_in_tp", ep_in_tp),
    )
    return [kind for kind, on in flags if on]


def _requires_value(runner: Any, key: str) -> str:
    """The server's value for a `requires` key, as a lowercase string."""
    section, _, attr = REQUIRES_KEYS[key].partition(".")
    value = getattr(_runner_cfg(runner, section), attr, None)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower().removeprefix("torch.")


def check_spec(runner: Any, spec: Any, *, view: RunnerView) -> EligibilityResult | None:
    """Enforce a spec's declarative fields before `create`: `chips`,
    `tp_sizes`, `collective_domain`, `parallelism`, `features`, then
    `requires`. Returns the first
    named decline (`None` if all pass). Reads config and local device facts
    only, identical on every rank: no collective."""
    tp_size = view.tp_size
    if spec.chips:
        kind = view.device_kind
        # Only TPU chips are checked; a CPU (interpret mode) device isn't.
        if kind.upper().startswith("TPU") and not any(
            kind.lower().startswith(c.lower()) for c in spec.chips
        ):
            msg = (
                f"Megakernel model folder supports chips {spec.chips}, not "
                f"{kind!r}; skipping."
            )
            return EligibilityResult(False, UNSUPPORTED_TPU_CHIP, msg)
    if spec.tp_sizes and tp_size not in spec.tp_sizes:
        msg = (
            f"Megakernel model folder supports TP sizes {spec.tp_sizes}, not "
            f"{tp_size}; skipping."
        )
        return EligibilityResult(False, UNSUPPORTED_TP_SIZE, msg)
    domain = spec.collective_domain
    if domain != DOMAIN_TP:
        return _decline(
            f"{PARALLELISM_UNSUPPORTED}:{domain}",
            f"agrees across the TP group only: collective_domain {domain!r} "
            "is unsupported",
        )
    for kind in detect_parallelism(runner, tp_size):
        if kind not in spec.parallelism:
            return _decline(
                f"{PARALLELISM_UNSUPPORTED}:{kind}",
                f"{_PARALLELISM_DETAIL[kind]} (parallelism {kind!r} undeclared)",
            )
    el = check_runner_features(runner, spec.features)
    if el is not None:
        return el
    for key, want in sorted(spec.requires.items()):
        got = _requires_value(runner, key)
        if got != want.lower():
            msg = (
                f"Megakernel model folder requires {key}={want} (this server: "
                f"{got}); pass --{key.replace('_', '-')} {want}; skipping."
            )
            return EligibilityResult(False, f"{REQUIRES_UNMET}:{key}", msg)
    return None


def check_runner_features(
    runner: Any, features: frozenset[str]
) -> EligibilityResult | None:
    """The first framework feature that is on but not declared, as a named
    decline (`None` if all are covered). Config only: no collective."""
    caps = frozenset(features)
    rules: list[tuple[bool, str, str]] = [
        (
            _multimodal_inputs_possible(runner) and "multimodal" not in caps,
            "multimodal_inputs",
            "requires text-only input_ids (multimodal inputs)",
        ),
        (
            getattr(runner, "lora_config", None) is not None and "lora" not in caps,
            "lora_enabled",
            "does not support LoRA adapters",
        ),
        (
            bool(_aux_hidden_layers(_runner_backbone(runner)))
            and "aux_hidden_states" not in caps,
            "aux_hidden_states",
            "does not support aux_hidden_state_layers",
        ),
    ]
    for bad, code, detail in rules:
        if bad:
            return _decline(code, detail)
    spec_cfg = getattr(runner, "speculative_config", None)
    c_cfg = _runner_cfg(runner, "cache_config")
    kv_cfg = _runner_cfg(runner, "kv_transfer_config")
    par_cfg = _runner_cfg(runner, "parallel_config")
    m_cfg = _runner_cfg(runner, "model_config")
    offload = kv_cfg is not None and kv_connector_is_offloading(kv_cfg)
    other_kv = next(
        (n for n in kv_connector_names(kv_cfg) if n not in _OFFLOAD_CONNECTORS),
        None,
    )
    rules = [
        (
            int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0) > 0
            and "spec_decode" not in caps,
            "spec_decode_unsupported",
            "plugin does not declare speculative decoding",
        ),
        (
            bool(getattr(runner, "uses_mrope", False)) and "mrope" not in caps,
            "mrope_unsupported",
            "plugin does not declare M-RoPE",
        ),
        (
            bool(getattr(c_cfg, "enable_prefix_caching", False))
            and "prefix_caching" not in caps,
            "prefix_caching_unsupported",
            "plugin does not declare prefix caching",
        ),
        (
            offload and "kv_offload" not in caps,
            "kv_offload_unsupported",
            "plugin does not declare KV offloading",
        ),
        (
            kv_cfg is not None and not offload and "kv_transfer" not in caps,
            f"kv_connector_unsupported:{other_kv}",
            f"plugin does not declare KV transfer (connector {other_kv})",
        ),
        (
            bool(getattr(par_cfg, "enable_eplb", False)) and "eplb_dynamic" not in caps,
            "eplb_dynamic_unsupported",
            "plugin does not declare dynamic EPLB",
        ),
        # The attribute name on the pinned vLLM is checked on the serving image.
        (
            bool(getattr(m_cfg, "enable_sleep_mode", False))
            and "sleep_mode" not in caps,
            "sleep_mode_unsupported",
            "plugin does not declare sleep mode",
        ),
    ]
    for bad, code, detail in rules:
        if bad:
            return _decline(code, detail)
    return None

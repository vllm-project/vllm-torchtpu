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

from typing import Any

from vllm_torchtpu import _OFFLOADING_CONNECTORS
from vllm_torchtpu.distributed.kv_transfer.connector_names import (
    MULTI_CONNECTORS,
)
from vllm_torchtpu.distributed.kv_transfer.connector_names import (
    kv_connector_names as _connector_names,
)
from vllm_torchtpu.megakernel.adapter.backbone import find_backbone
from vllm_torchtpu.megakernel.adapter.plugin import RunnerProtocol, RunnerView
from vllm_torchtpu.megakernel.core.runtime.contract import (
    DOMAIN_TP,
    FEATURE_KV_TRANSFER,
    FEATURE_KV_TRANSFER_CONSUMER,
    PARALLELISM_UNSUPPORTED,
    REQUIRES_KEYS,
    REQUIRES_UNMET,
    UNSUPPORTED_TP_SIZE,
    UNSUPPORTED_TPU_CHIP,
    EligibilityResult,
)


def _without_vllm(exc: ModuleNotFoundError) -> bool:
    return (exc.name or "").partition(".")[0] == "vllm"


def _ep_group_size() -> int:
    """vLLM's expert-parallel group size (`ep_mesh.get_ep_group`; 1 when EP is
    off, or without vLLM, as in a CPU test of a plugin)."""
    try:
        from vllm_torchtpu.distributed.ep_mesh import get_ep_group
    except ModuleNotFoundError as exc:
        if not _without_vllm(exc):
            raise
        return 1
    group = get_ep_group()
    return group.world_size if group is not None else 1


def _dp_size(parallel_config: Any) -> int:
    """The data-parallel size by the runner's rule (`utils.get_dp_size`; 1
    without vLLM, which runs no data-parallel engines then)."""
    try:
        from vllm_torchtpu.utils import get_dp_size
    except ModuleNotFoundError as exc:
        if not _without_vllm(exc):
            raise
        return 1
    return get_dp_size(parallel_config)


def kv_connector_names(kv_transfer_config: Any) -> tuple[str, ...]:
    """The KV connector names in effect: the configured connector, or the
    children of a `MultiConnector` / `TPUMultiConnector` (the multi-connector
    itself when it lists none). Empty without a config or a connector name."""
    if kv_transfer_config is None:
        return ()
    if names := _connector_names(kv_transfer_config):
        return tuple(names)
    name = kv_transfer_config.kv_connector
    return (name,) if name else ()


def connector_class(name: str) -> Any:
    """The KV connector class registered in vLLM under `name` (vllm-torchtpu
    registers its connectors there), or `None`: unknown, or no vLLM."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
    except ModuleNotFoundError as exc:
        if not _without_vllm(exc):
            raise
        return None
    try:
        return KVConnectorFactory.get_connector_class_by_name(name)
    except Exception:  # unregistered or failing import: unknown, so declined
        return None


def _child_kv_roles(kv_transfer_config: Any) -> dict[str, Any]:
    """`kv_role` per child connector of a multi-connector (`None` when a child
    sets none); empty for a single connector."""
    if kv_transfer_config.kv_connector not in MULTI_CONNECTORS:
        return {}
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return {
        str(c["kv_connector"]): c.get("kv_role") for c in extra.get("connectors", [])
    }


def kv_transfer_decline(
    kv_transfer_config: Any, features: frozenset[str]
) -> tuple[str, str] | None:
    """`(code, detail)` declining the config's KV transfer connectors, or `None`.

    Offloading connectors are `kv_offload`'s. `kv_transfer` serves any
    transfer connector. `kv_transfer_consumer` serves them only when this
    engine is a KV consumer (`kv_role == "kv_consumer"`, and no child of a
    multi-connector says otherwise) and every transfer connector's class
    declares `layerwise_transfer = False`: such a connector loads KV before
    the scheduler runs the request and has no per-layer hooks for the
    megakernel to skip. Producers, `kv_both`, layer-wise connectors and
    connectors that declare nothing are declined (fail closed).
    """
    names = kv_connector_names(kv_transfer_config)
    transfer = [n for n in names if n not in _OFFLOADING_CONNECTORS]
    if not transfer or FEATURE_KV_TRANSFER in features:
        return None
    code = f"kv_connector_unsupported:{transfer[0]}"
    if FEATURE_KV_TRANSFER_CONSUMER not in features:
        return code, f"plugin does not declare KV transfer (connector {transfer[0]})"
    role = kv_transfer_config.kv_role
    if role != "kv_consumer":
        return code, (
            f"serves a KV transfer connector only on a KV consumer (connector "
            f"{transfer[0]}, kv_role {role!r})"
        )
    child_roles = _child_kv_roles(kv_transfer_config)
    for name in transfer:
        child_role = child_roles.get(name)
        if child_role not in (None, "kv_consumer"):
            return f"kv_connector_unsupported:{name}", (
                f"serves a KV transfer connector only as a consumer (connector "
                f"{name}, kv_role {child_role!r})"
            )
        cls = connector_class(name)
        layerwise = getattr(cls, "layerwise_transfer", None) if cls else None
        if layerwise is not False:
            known = "unknown connector" if cls is None else f"{layerwise=}"
            return f"kv_connector_unsupported:{name}", (
                "serves only KV transfer connectors that declare no layer-wise "
                f"load or save (connector {name}: {known})"
            )
    return None


def _aux_hidden_layers(backbone: Any) -> Any:
    return getattr(backbone, "aux_hidden_state_layers", ()) if backbone else ()


def _runner_backbone(runner: RunnerProtocol) -> Any:
    try:
        return find_backbone(runner.model)
    except ValueError:
        return None


def _decline(code: str, detail: str) -> EligibilityResult:
    return EligibilityResult(False, code, f"Megakernel {detail}; skipping.")


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


def detect_parallelism(runner: RunnerProtocol, tp_size: int) -> list[str]:
    """The parallelism kinds (beyond "tp") the runner uses, in check order.
    Reads config and group sizes only: no collective."""
    p_cfg = runner.vllm_config.parallel_config
    dp = _dp_size(p_cfg)
    ep = p_cfg.enable_expert_parallel
    # Only expert parallelism runs MoE collectives over vLLM's EP group.
    ep_wide = ep and _ep_group_size() > max(1, tp_size)
    pp_multi = not (runner._pp_is_first and runner._pp_is_last)
    dcp = p_cfg.decode_context_parallel_size > 1
    pcp = runner._pcp_enabled or p_cfg.prefill_context_parallel_size > 1
    ep_in_tp = ep and not ep_wide
    flags = (
        ("dp", dp > 1),
        ("ep_wider_than_tp", ep_wide),
        ("dp_lockstep", runner._dp_lockstep_enabled()),
        ("pp", pp_multi),
        ("dcp", dcp),
        ("pcp", pcp),
        ("ep_in_tp", ep_in_tp),
    )
    return [kind for kind, on in flags if on]


def _requires_value(runner: RunnerProtocol, key: str) -> str:
    """The server's value for a `requires` key, as a lowercase string."""
    section, _, attr = REQUIRES_KEYS[key].partition(".")
    value = getattr(getattr(runner.vllm_config, section), attr)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower().removeprefix("torch.")


def check_spec(
    runner: RunnerProtocol, spec: Any, *, view: RunnerView
) -> EligibilityResult | None:
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
    runner: RunnerProtocol, features: frozenset[str]
) -> EligibilityResult | None:
    """The first framework feature that is on but not declared, as a named
    decline (`None` if all are covered). Config only: no collective."""
    v_cfg = runner.vllm_config
    rules: list[tuple[bool, str, str]] = [
        (
            runner.supports_mm_inputs and "multimodal" not in features,
            "multimodal_inputs",
            "requires text-only input_ids (multimodal inputs)",
        ),
        (
            v_cfg.lora_config is not None and "lora" not in features,
            "lora_enabled",
            "does not support LoRA adapters",
        ),
        (
            bool(_aux_hidden_layers(_runner_backbone(runner)))
            and "aux_hidden_states" not in features,
            "aux_hidden_states",
            "does not support aux_hidden_state_layers",
        ),
    ]
    for bad, code, detail in rules:
        if bad:
            return _decline(code, detail)
    spec_cfg = v_cfg.speculative_config
    # Offloading and transfer are separate features: a multi-connector may
    # mix an offloading child with a transfer child.
    kv_cfg = v_cfg.kv_transfer_config
    kv_names = kv_connector_names(kv_cfg)
    offload = any(n in _OFFLOADING_CONNECTORS for n in kv_names)
    kv_bad = kv_transfer_decline(kv_cfg, features) if kv_cfg is not None else None
    kv_code, kv_detail = kv_bad if kv_bad is not None else ("", "")
    rules = [
        (
            spec_cfg is not None
            and spec_cfg.num_speculative_tokens > 0
            and "spec_decode" not in features,
            "spec_decode_unsupported",
            "plugin does not declare speculative decoding",
        ),
        (
            runner.uses_mrope and "mrope" not in features,
            "mrope_unsupported",
            "plugin does not declare M-RoPE",
        ),
        (
            v_cfg.cache_config.enable_prefix_caching
            and "prefix_caching" not in features,
            "prefix_caching_unsupported",
            "plugin does not declare prefix caching",
        ),
        (
            offload and "kv_offload" not in features,
            "kv_offload_unsupported",
            "plugin does not declare KV offloading",
        ),
        (kv_bad is not None, kv_code, kv_detail),
        (
            v_cfg.parallel_config.enable_eplb and "eplb_dynamic" not in features,
            "eplb_dynamic_unsupported",
            "plugin does not declare dynamic EPLB",
        ),
        # vLLM's `ModelConfig.enable_sleep_mode`.
        (
            v_cfg.model_config.enable_sleep_mode and "sleep_mode" not in features,
            "sleep_mode_unsupported",
            "plugin does not declare sleep mode",
        ),
    ]
    for bad, code, detail in rules:
        if bad:
            return _decline(code, detail)
    return None

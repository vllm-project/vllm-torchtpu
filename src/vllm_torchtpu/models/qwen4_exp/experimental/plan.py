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
"""Resolve the experimental Qwen3.8-Flash model plan for one vLLM config.

The only switch is the optional
``--additional-config '{"model_plan": {"profile": "<name>"}}'`` (default
`profiles.DEFAULT_PROFILE`). Each slot takes the first variant of the
profile's list whose ``eligible(vllm_config)`` holds.

The config hook and the model bind the plan of their config (`bind`); code
that has no config at hand (kernels, ops) reads a feature with `value`.
Unbound (unit tests), `UNBOUND_PLAN` applies.
"""

import importlib
from typing import Any

from .profiles import DEFAULT_PROFILE, PROFILES
from .variants import PLAIN, SERVED, VARIANTS

PLAN_KEY = "model_plan"
_PLAN_FIELDS = frozenset({"profile"})
_bound: dict[str, str] | None = None


def profile_name(vllm_config: Any) -> str:
    """The profile ``additional_config["model_plan"]`` names, else the default."""
    additional = vllm_config.additional_config
    plan = additional.get(PLAN_KEY) if isinstance(additional, dict) else None
    if plan is None:
        return DEFAULT_PROFILE
    if not isinstance(plan, dict) or not set(plan) <= _PLAN_FIELDS:
        raise ValueError(
            f'additional_config.{PLAN_KEY} must look like {{"profile": "<name>"}}, '
            f"got {plan!r}"
        )
    name = plan.get("profile", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError(
            f"Unknown Qwen3.8-Flash profile {name!r}; known: {sorted(PROFILES)}"
        )
    return name


def resolve_plan(vllm_config: Any) -> dict[str, str]:
    """{slot: variant name}: per slot, the profile's first eligible variant."""
    name = profile_name(vllm_config)
    plan = {}
    for slot, choices in PROFILES[name].items():
        pick = next(
            (c for c in choices if VARIANTS[slot][c].eligible(vllm_config)), None
        )
        if pick is None:
            raise ValueError(
                f"Qwen3.8-Flash profile {name!r}: no eligible {slot} variant among "
                f"{list(choices)} for this config."
            )
        plan[slot] = pick
    return plan


# Kernel and op modules never import the plan: `bind` pushes their slot
# features into each module's ``configure`` ({option key: feature}).
def _same(*features: str) -> dict[str, str]:
    return {feature: feature for feature in features}


KERNEL_OPTIONS: dict[str, dict[str, str]] = {
    ".kernels.qsa": _same(
        "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT",
    ),
    ".kernels.qsa_rpa": _same(
        "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE",
        "VLLM_TPU_QSA_DECODE_RPA_BKV",
        "VLLM_TPU_QSA_DECODE_RPA_LANES",
        "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES",
        "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK",
        "VLLM_TPU_QSA_DCP_SPLIT_LSE",
        "VLLM_TPU_QSA_PREFIX_SLICE",
    ),
    ".kernels.pooled_qsa": _same(
        "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT",
        "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE",
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY",
        "VLLM_TPU_QSA_FUSED_DCP_BLOCKS",
        "VLLM_TPU_QSA_INDEXER_HOIST_META",
        "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE",
        "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY",
        "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY",
    ),
    ".kernels.block_sparse_gqa_prefill": _same(
        "VLLM_TPU_QSA_PREFILL_BITSET_OWNED",
        "VLLM_TPU_QSA_PREFILL_LEAN",
        "VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED",
    ),
    ".kernels.gdn_v3.wrapper": _same(
        "TPU_GDN_CHUNK_PAIR",
    ),
    ".ops.qsa_op": {
        "QSA_DCP_HEAD_EXCHANGE": "VLLM_TPU_QSA_DCP_HEAD_EXCHANGE",
        "QSA_DCP_GATE_IN_COMBINE": "VLLM_TPU_QSA_DCP_GATE_IN_COMBINE",
        "QSA_DCP_Q_GATHER": "VLLM_TPU_QSA_DCP_Q_GATHER",
        "QSA_DCP_QGATE_PAIR_MAJOR": "VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR",
        "QSA_DCP_SHARD_MERGE": "VLLM_TPU_QSA_DCP_SHARD_MERGE",
    },
    ".ops.gdn_attention_op": {
        "GDN_HOIST_METADATA": "VLLM_TPU_GDN_HOIST_METADATA",
        "GDN_CONV_WEIGHT_KMAJOR": "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR",
    },
}


def configure_kernels(plan: dict[str, str] | None = None) -> None:
    """Hand each kernel / op module its slot features."""
    feats = features(plan)
    for module, keys in KERNEL_OPTIONS.items():
        importlib.import_module(module, __package__).configure(
            **{key: feats[feature] for key, feature in keys.items()}
        )


def bind(vllm_config: Any) -> dict[str, str]:
    """Resolve and keep this process's plan, configure the kernels; returns it."""
    global _bound
    _bound = resolve_plan(vllm_config)
    configure_kernels(_bound)
    return dict(_bound)


def unbind() -> None:
    global _bound
    _bound = None


# Without a bound config (unit tests, tools) every slot takes its plain
# variant: the behavior the code has with no setup-specific feature on.
UNBOUND_PLAN: dict[str, str] = {
    "decode_batch": "small",
    "c48_stack": "off",
    "pool_budget": "auto",
}


def bound_plan() -> dict[str, str]:
    """The bound plan, else `UNBOUND_PLAN`."""
    if _bound is not None:
        return dict(_bound)
    return dict(UNBOUND_PLAN)


def features(
    plan: dict[str, str] | None = None, *, plain: bool | None = None
) -> dict[str, Any]:
    """Feature name -> value for a plan (default: the bound plan).

    The always-served features (`variants.SERVED`) apply to a bound or given
    plan; unbound (``plain``) code sees `variants.PLAIN` instead.
    """
    if plain is None:
        plain = plan is None and _bound is None
    out: dict[str, Any] = dict(PLAIN if plain else SERVED)
    for slot, name in (plan or bound_plan()).items():
        out.update(VARIANTS[slot][name].features)
    return out


def value(name: str) -> Any:
    """One feature of the bound plan (see `variants`)."""
    feats = features()
    if name not in feats:
        raise KeyError(f"no Qwen3.8-Flash plan feature {name!r}")
    return feats[name]


def compile_key_factors() -> dict[str, Any]:
    """The bound plan and its feature values, for the compile-cache key.

    Both change graphs: the features tell the always-served set from the
    plain one, and catch an edit of a variant's values in `variants`, which
    is not one of the hashed source paths.
    """
    return {"model_plan": bound_plan(), "model_plan_features": features()}

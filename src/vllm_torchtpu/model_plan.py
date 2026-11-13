# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model implementation plans: which variant serves each dispatch slot.

A model that opts in owns three tables in its directory (``variants.py``,
``plans.py`` and ``experimental/profiles.py``) and resolves them once per
config into a :class:`ResolvedPlan`. The resolved plan is published under
``additional_config["model_plan"]["resolved"]``, so it rides vLLM's existing
additional-config hash (compile caches key on it) and worker serialization,
and every process activates the same plan from its own copy of the config.

This module is the model-agnostic half: the data types, the resolver and the
process-wide active plan that shared code (runner, layers, the MoE adapter)
reads through :func:`param`. Shared code passes the value that keeps its own
behavior unchanged as ``default``, so a model without a plan is unaffected
(the default-off hook rule for shared-core changes).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ADDITIONAL_CONFIG_KEY = "model_plan"


@dataclass(frozen=True)
class PlanContext:
    """The config facts a variant's eligibility rule may read."""

    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    moe: str | None = None


def _always(_: PlanContext) -> bool:
    return True


@dataclass(frozen=True)
class Variant:
    """One implementation of a slot: the parameters it serves with.

    ``params`` are the values the slot's consumers read; every variant of a
    slot carries the same keys. ``eligible`` decides whether a default plan
    may pick it for a config (first eligible entry wins); a profile that pins
    an ineligible variant is an error.
    """

    params: Mapping[str, Any] = field(default_factory=dict)
    eligible: Callable[[PlanContext], bool] = _always
    description: str = ""


@dataclass(frozen=True)
class ResolvedPlan:
    model: str
    profile: str | None
    slots: Mapping[str, str]
    params: Mapping[str, Mapping[str, Any]]

    def selection(self) -> dict[str, str]:
        """Slot -> variant name, the form the plan snapshots store."""
        return dict(sorted(self.slots.items()))

    def to_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "profile": self.profile,
            "slots": self.selection(),
            "params": {s: dict(p) for s, p in sorted(self.params.items())},
        }

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> ResolvedPlan:
        return cls(
            model=value["model"],
            profile=value.get("profile"),
            slots=MappingProxyType(dict(value["slots"])),
            params=MappingProxyType(
                {s: MappingProxyType(dict(p)) for s, p in value["params"].items()}
            ),
        )


class PlanError(ValueError):
    pass


def resolve(
    model: str,
    context: PlanContext,
    supported_variants: Mapping[str, Mapping[str, Variant]],
    supported_plan: Mapping[str, list[str]],
    *,
    profile: str | None = None,
    profile_overrides: Mapping[str, str] | None = None,
    experimental_variants: Mapping[str, Mapping[str, Variant]] | None = None,
) -> ResolvedPlan:
    """Resolve the default plan, then apply a profile's slot overrides.

    The default plan only ever names SUPPORTED variants. A profile may name a
    variant from either catalog; it pins it, so an ineligible pin raises.
    """
    if supported_plan.keys() != supported_variants.keys():
        raise PlanError(
            f"{model}: plan slots {sorted(supported_plan)} differ from variant "
            f"slots {sorted(supported_variants)}"
        )
    experimental_variants = experimental_variants or {}
    slots: dict[str, str] = {}
    for slot, order in supported_plan.items():
        catalog = supported_variants[slot]
        for name in order:
            if name not in catalog:
                raise PlanError(f"{model}: plan names unknown {slot} variant {name!r}")
            if catalog[name].eligible(context):
                slots[slot] = name
                break
        else:
            raise PlanError(f"{model}: no eligible {slot} variant in {order}")

    for slot, name in (profile_overrides or {}).items():
        if slot not in supported_plan:
            raise PlanError(f"{model}: profile {profile!r} names unknown slot {slot!r}")
        variant = supported_variants[slot].get(name) or experimental_variants.get(
            slot, {}
        ).get(name)
        if variant is None:
            raise PlanError(
                f"{model}: profile {profile!r} names unknown {slot} variant {name!r}"
            )
        if not variant.eligible(context):
            raise PlanError(
                f"{model}: profile {profile!r} pins {slot}={name}, which is not "
                f"eligible for {context}"
            )
        slots[slot] = name

    def variant(slot: str) -> Variant:
        name = slots[slot]
        return supported_variants[slot].get(name) or experimental_variants[slot][name]

    return ResolvedPlan(
        model=model,
        profile=profile,
        slots=MappingProxyType(slots),
        params=MappingProxyType(
            {s: MappingProxyType(dict(variant(s).params)) for s in slots}
        ),
    )


_ACTIVE: ResolvedPlan | None = None


def activate(plan: ResolvedPlan | None) -> None:
    global _ACTIVE
    _ACTIVE = plan


def active() -> ResolvedPlan | None:
    return _ACTIVE


def activate_from_config(vllm_config: VllmConfig) -> ResolvedPlan | None:
    """Activate the plan a model published into this config, if any."""
    additional = getattr(vllm_config, "additional_config", None)
    published = (
        additional.get(ADDITIONAL_CONFIG_KEY) if isinstance(additional, dict) else None
    )
    resolved = published.get("resolved") if isinstance(published, dict) else None
    activate(ResolvedPlan.from_config(resolved) if resolved else None)
    return _ACTIVE


def param(slot: str, key: str, default: Any) -> Any:
    """The active plan's value for ``slot.key``, else ``default``."""
    plan = _ACTIVE
    if plan is None:
        return default
    return plan.params.get(slot, {}).get(key, default)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"

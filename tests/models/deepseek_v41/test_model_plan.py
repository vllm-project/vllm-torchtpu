# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Required CPU checks of the DeepSeek-V4.1 model plan.

1. Plan snapshots: the default plan selects exactly what
   validation/plan_snapshots/*.json records for each standard customer config.
   Regenerate after a deliberate plans.py change by running this file with
   ``DSV41_UPDATE_PLAN_SNAPSHOTS=1``.
2. One-way imports: no customer-path module imports models/deepseek_v41/experimental.
3. Profiles: every profile names known slots and variants, pins every slot, and
   resolves for the config it is written for.
"""

import ast
import json
import os
from pathlib import Path

import pytest

from vllm_torchtpu import model_plan
from vllm_torchtpu.model_plan import PlanContext, PlanError
from vllm_torchtpu.models.deepseek_v41 import plan as dsv41_plan
from vllm_torchtpu.models.deepseek_v41.plans import SUPPORTED_PLAN
from vllm_torchtpu.models.deepseek_v41.variants import SUPPORTED_VARIANTS

SRC = Path(dsv41_plan.__file__).resolve().parents[2]
MODEL_DIR = SRC / "models" / "deepseek_v41"
SNAPSHOTS = MODEL_DIR / "validation" / "plan_snapshots"

# Standard customer configs: snapshot name -> context.
CUSTOMER_CONFIGS = {
    "tp8": PlanContext(tp_size=8, dp_size=1, ep_size=1, moe="gmm"),
    "dp8_ep8_gmm": PlanContext(tp_size=1, dp_size=8, ep_size=8, moe="gmm"),
    "dp8_ep8_fused_ep": PlanContext(
        tp_size=1, dp_size=8, ep_size=8, moe="fused_ep_w4a8"
    ),
    "dp16_ep16_fused_ep": PlanContext(
        tp_size=1, dp_size=16, ep_size=16, moe="fused_ep_w4a8"
    ),
}

# Profile -> the config it is written for.
PROFILE_CONFIGS = {
    "dsv41-dp16-c768": CUSTOMER_CONFIGS["dp16_ep16_fused_ep"],
}


@pytest.mark.parametrize("name", sorted(CUSTOMER_CONFIGS))
def test_default_plan_matches_snapshot(name):
    selection = dsv41_plan.resolve_plan(CUSTOMER_CONFIGS[name]).selection()
    path = SNAPSHOTS / f"{name}.json"
    if os.environ.get("DSV41_UPDATE_PLAN_SNAPSHOTS"):
        path.write_text(model_plan.canonical_json(selection))
    assert path.exists(), f"missing snapshot {path}"
    assert selection == json.loads(path.read_text()), (
        f"the default plan for {name} changed; a deliberate change refreshes "
        "the snapshot and needs Support's approval"
    )


def test_every_snapshot_is_checked():
    assert {p.stem for p in SNAPSHOTS.glob("*.json")} == set(CUSTOMER_CONFIGS)


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_customer_code_never_imports_experimental():
    experimental = MODEL_DIR / "experimental"
    offenders = []
    for path in SRC.rglob("*.py"):
        if experimental in path.parents:
            continue
        hits = [
            name for name in _imports(path) if name.startswith(dsv41_plan._EXPERIMENTAL)
        ]
        if hits:
            offenders.append(f"{path.relative_to(SRC)}: {hits}")
    assert not offenders, "\n".join(offenders)


def test_default_resolution_does_not_load_experimental(monkeypatch):
    def refuse():
        raise AssertionError("the default plan loaded experimental/")

    monkeypatch.setattr(dsv41_plan, "experimental_tables", refuse)
    for context in CUSTOMER_CONFIGS.values():
        dsv41_plan.resolve_plan(context)


def test_variants_of_a_slot_share_their_keys():
    experimental, _ = dsv41_plan.experimental_tables()
    for slot, catalog in SUPPORTED_VARIANTS.items():
        keys = {frozenset(v.params) for v in catalog.values()}
        keys |= {frozenset(v.params) for v in experimental.get(slot, {}).values()}
        # moe_kernel's push variant adds its kernel keywords to `impl`.
        if slot == "moe_kernel":
            assert all("impl" in k for k in keys)
            continue
        assert len(keys) == 1, f"{slot} variants carry different keys: {keys}"
    assert set(experimental) <= set(SUPPORTED_PLAN)


@pytest.mark.parametrize("profile", sorted(PROFILE_CONFIGS))
def test_profile_pins_every_slot_and_resolves(profile):
    _, profiles = dsv41_plan.experimental_tables()
    assert set(profiles) == set(PROFILE_CONFIGS)
    assert set(profiles[profile]) == set(SUPPORTED_PLAN), (
        "active profiles pin every slot, so plans.py edits never move them"
    )
    resolved = dsv41_plan.resolve_plan(PROFILE_CONFIGS[profile], profile)
    assert resolved.selection() == profiles[profile]


def test_dp16_profile_serves_the_landing_push_settings():
    resolved = dsv41_plan.resolve_plan(
        PROFILE_CONFIGS["dsv41-dp16-c768"], "dsv41-dp16-c768"
    )
    assert resolved.params["expert_layout"]["layout"] == "dsv41_flash_dp16_c768"
    moe = resolved.params["moe_kernel"]
    assert moe["impl"] == "push"
    assert (
        moe["dispatch_ahead"],
        moe["light_prepush"],
        moe["early_scale_scatter"],
    ) == (
        25,
        2,
        2,
    )


def test_profile_pinning_an_ineligible_variant_fails():
    with pytest.raises(PlanError, match="not eligible"):
        dsv41_plan.resolve_plan(CUSTOMER_CONFIGS["tp8"], "dsv41-dp16-c768")


def test_unknown_profile_fails():
    with pytest.raises(PlanError, match="Unknown deepseek_v41 model_plan profile"):
        dsv41_plan.resolve_plan(CUSTOMER_CONFIGS["tp8"], "no-such-profile")


def test_removed_env_is_an_error(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_PUSH_DISPATCH_AHEAD", "17")
    with pytest.raises(ValueError, match="moe_kernel"):
        dsv41_plan.reject_removed_env()


def test_resolved_plan_round_trips_through_the_config():
    resolved = dsv41_plan.resolve_plan(
        PROFILE_CONFIGS["dsv41-dp16-c768"], "dsv41-dp16-c768"
    )
    again = model_plan.ResolvedPlan.from_config(
        json.loads(json.dumps(resolved.to_config()))
    )
    assert again.to_config() == resolved.to_config()


def test_param_falls_back_to_the_supported_plan(monkeypatch):
    monkeypatch.setattr(model_plan, "_ACTIVE", None)
    assert dsv41_plan.param("csa2_attention", "deep_gather") is True
    # Shared code without a plan keeps its own (off) default.
    assert model_plan.param("decode_step", "onehot_table_lookup", False) is False

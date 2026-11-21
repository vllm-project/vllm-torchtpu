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
"""The experimental model plan: profiles, variants, default snapshot, no switches."""

import json
import pathlib
import re
from types import SimpleNamespace as NS

import pytest

from vllm_torchtpu.models.qwen38_flash.experimental import plan
from vllm_torchtpu.models.qwen38_flash.experimental.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
)
from vllm_torchtpu.models.qwen38_flash.experimental.variants import VARIANTS

E = pathlib.Path(plan.__file__).parent
SNAPSHOT = E / "validation" / "plan_snapshot.json"


def _config(profile=None, *, max_num_seqs=32, gpu_util=0.92):
    additional = {} if profile is None else {"model_plan": {"profile": profile}}
    return NS(
        additional_config=additional,
        scheduler_config=NS(max_num_seqs=max_num_seqs),
        cache_config=NS(gpu_memory_utilization=gpu_util),
    )


def test_every_profile_pins_every_slot_with_known_variants():
    assert DEFAULT_PROFILE in PROFILES
    for name, slots in PROFILES.items():
        assert set(slots) == set(VARIANTS), name
        for slot, choices in slots.items():
            assert choices and set(choices) <= set(VARIANTS[slot]), (name, slot)


def test_unbound_plan_names_a_variant_of_every_slot():
    assert set(plan.UNBOUND_PLAN) == set(VARIANTS)
    assert plan.features()["QWEN4_EXP_PACK_MAIN_KV"] is False  # plain, unbound
    for slot, name in plan.UNBOUND_PLAN.items():
        assert name in VARIANTS[slot], slot


def test_every_variant_feature_set_is_complete_per_slot():
    for slot, variants in VARIANTS.items():
        keys = {frozenset(v.features) for v in variants.values()}
        assert len(keys) == 1, slot


@pytest.mark.parametrize("module", sorted(plan.KERNEL_OPTIONS))
def test_every_kernel_option_module_imports_and_takes_its_features(module):
    import importlib

    mod = importlib.import_module(module, plan.__package__)
    feats = plan.features()
    mod.configure(**{k: feats[f] for k, f in plan.KERNEL_OPTIONS[module].items()})


def test_default_plan_matches_snapshot():
    snap = json.loads(SNAPSHOT.read_text())
    c = snap["config"]
    got = {
        name: plan.resolve_plan(
            _config(
                name,
                max_num_seqs=c["max_num_seqs"],
                gpu_util=c["gpu_memory_utilization"],
            )
        )
        for name in PROFILES
    }
    assert got == snap["plans"], "update validation/plan_snapshot.json with the PR"
    assert plan.resolve_plan(_config()) == snap["plans"][DEFAULT_PROFILE]


def test_profile_selection_and_eligibility():
    small = plan.resolve_plan(_config(max_num_seqs=16, gpu_util=0.95))
    assert small["decode_batch"] == "small"
    assert small["pool_budget"] == "auto"
    assert plan.resolve_plan(_config("agentx-c48"))["c48_stack"] == "on"
    assert set(PROFILES) == {"agentx-mixed-256k", "agentx-c48"}
    with pytest.raises(ValueError):
        plan.resolve_plan(_config("nope"))
    with pytest.raises(ValueError):
        plan.profile_name(NS(additional_config={"model_plan": {"x": 1}}))


def test_bind_sets_value_and_kernel_options(monkeypatch):
    monkeypatch.setattr(plan, "configure_kernels", lambda p=None: None)
    try:
        plan.bind(_config(max_num_seqs=16))
        assert plan.value("VLLM_TPU_QSA_MAIN_KV_DCP") == 0
        plan.bind(_config())
        assert plan.value("VLLM_TPU_QSA_MAIN_KV_DCP") == 2
        assert plan.value("QWEN4_EXP_PACK_MAIN_KV") is True
        assert "model_plan" in plan.compile_key_factors()
    finally:
        plan.unbind()


# Reads that stay, each a deployment / platform fact, not an optimization switch.
_ALLOWED = {
    "plan.py": "the model_plan profile is the one switch",
    "hooks.py": "migration selector (owner dongruizi, until qwen4_exp retires)",
    "hyperconnection.py": "LIBTPU_INIT_ARGS SparseCore AG offload guard",
    "moe_decode.py": "stands down for the shared fused MoE decode",
    "kernels/streamindex_topk/streamindex_topk.py": "Raiden GLM admission helper",
}
_SWITCH = re.compile(
    r"os\.environ|os\.getenv|\bgetenv\(|import envs\b|\benvs\.[A-Za-z_]"
    r"|vllm\.envs|additional_config"
)


def test_experimental_reads_no_global_switches():
    offenders = []
    for path in sorted(E.rglob("*.py")):
        rel = path.relative_to(E).as_posix()
        if rel in _ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if _SWITCH.search(code):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "\n".join(offenders)

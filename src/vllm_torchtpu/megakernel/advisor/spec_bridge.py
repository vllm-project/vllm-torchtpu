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
"""Bridge declarative ModelSpec + ExecSpec schedules via planner.plan."""

from __future__ import annotations

import ast
import dataclasses
import pkgutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

try:
    import jax.numpy as jnp

    from ..core import exec_spec, planner
    from ..core import modules as core_modules
    from ..core import profile as core_profile
    from ..core import spec as core_spec
    from ..core.collectives import params as collectives_params
    from ..core.dma import vmem as core_vmem
    from ..core.profiles import default as default_profiles
    from ..core.runtime import policy as runtime_policy
except (ImportError, RuntimeError):
    jnp = None
    exec_spec = None
    planner = None
    core_profile = None
    core_spec = None
    collectives_params = None
    core_vmem = None
    default_profiles = None
    runtime_policy = None
    core_modules = None

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..core.exec_spec import ExecSpec, Tiles
    from ..core.spec import ModelSpec
    from .chips import ChipSpec

from ..core.errors import MegakernelConfigError
from .constants import CORE_LIMITS
from .models import ModelProfile, finalize_model_profile
from .specs import CORE_MODULE_FAMILIES, AdvisorReport, MoEMode, WorkloadSpec
from .verdict import advise


class MegakernelPlannerUnavailableError(MegakernelConfigError):
    """Raised when JAX or the planner is unavailable."""

    def __init__(self, message: str = "Megakernel planner requires JAX.") -> None:
        """Initialize the exception with an informative message."""
        super().__init__(message)


def _read_numerics_grid_r(profile_file: str | None) -> tuple[int, ...]:
    """Read `_NUMERICS_GRID_R` literal from live `core/profile.py` AST."""
    if profile_file is None:
        return ()
    tree = ast.parse(Path(profile_file).read_text(encoding="utf-8"))
    for node in tree.body:
        if type(node) is ast.Assign:
            for target in node.targets:
                if type(target) is ast.Name and target.id == "_NUMERICS_GRID_R":
                    val = ast.literal_eval(node.value)
                    if type(val) is tuple:
                        return tuple(int(x) for x in val)
    return ()


def _make_ar_probe_spec(num_slots: int, probe_hidden: int) -> ModelSpec:
    """Build a minimal synthetic ModelSpec with `num_slots` all-reduce slots."""
    if core_spec is None:
        msg = "Megakernel core package requires JAX to verify CORE_LIMITS sync."
        raise MegakernelPlannerUnavailableError(msg)
    slots = []
    for slot_idx in range(num_slots):
        probe_module = core_spec.Module()
        probe_module.name = f"_advisor_ar_probe_{slot_idx}"
        probe_module.reduction = "allreduce"
        slots.append(probe_module)
    return core_spec.ModelSpec(
        name="_advisor_ar_slot_probe",
        config=SimpleNamespace(L=1, H=probe_hidden),
        layer_type=core_spec.LayerType(
            name="_advisor_ar_slot_layer",
            slots=tuple(slots),
        ),
    )


def _probe_max_ar_slots_planner(
    limit: int,
    probe_exec_spec: ExecSpec,
    probe_hidden: int,
) -> list[str]:
    """Verify core planner accepts `limit` AR slots and rejects `limit + 1`."""
    if planner is None:
        msg = "Megakernel core package requires JAX to verify CORE_LIMITS sync."
        raise MegakernelPlannerUnavailableError(msg)
    mismatches: list[str] = []
    try:
        plan_ok = planner.plan(
            _make_ar_probe_spec(limit, probe_hidden), probe_exec_spec
        )
    except MegakernelConfigError as exc:
        mismatches.append(
            f"max_ar_slots_per_layer: core planner rejected {limit} AR slots ({exc})"
        )
    else:
        planned_slots = sum(x is not None for x in plan_ok.slot_ar_index)
        if planned_slots != limit:
            mismatches.append(
                f"max_ar_slots_per_layer: advisor={limit!r} != "
                f"planned_ar_slots={planned_slots!r}"
            )
    try:
        planner.plan(_make_ar_probe_spec(limit + 1, probe_hidden), probe_exec_spec)
    except MegakernelConfigError:
        pass
    else:
        mismatches.append(
            f"max_ar_slots_per_layer: core planner accepted {limit + 1} AR slots "
            f"(exceeds advisor={limit!r})"
        )
    return mismatches


def _verify_max_ar_slots_sync(tiles: Tiles) -> list[str]:
    """Verify CORE_LIMITS.max_ar_slots_per_layer against scratch and planner."""
    if (
        exec_spec is None
        or planner is None
        or core_spec is None
        or collectives_params is None
    ):
        msg = "Megakernel core package requires JAX to verify CORE_LIMITS sync."
        raise MegakernelPlannerUnavailableError(msg)
    limit = CORE_LIMITS.max_ar_slots_per_layer
    if limit < 1:
        return [f"max_ar_slots_per_layer: advisor={limit!r} must be >= 1"]
    mismatches: list[str] = []
    # Minimum 128-lane native vector-width multiple for col_chunk/k_tile.
    probe_hidden = 128
    # B=4 single-token decode across R=8 (smallest hierarchical collective rank
    # in core/collectives/params.py and CORE_LIMITS.verified_ranks).
    probe_exec_spec = exec_spec.ExecSpec(B=4, T=1, R=8)
    ring_scratch = collectives_params.ring_scratch(
        collectives_params.comm_params(probe_exec_spec, tiles, H=probe_hidden)
    )
    for sem_key in ("ar_ssem", "ar_rsem"):
        sem_slots = int(ring_scratch[sem_key].shape[1])
        if sem_slots != limit:
            mismatches.append(
                f"max_ar_slots_per_layer: advisor={limit!r} != "
                f"core {sem_key}.shape[1]={sem_slots!r}"
            )
    mismatches.extend(_probe_max_ar_slots_planner(limit, probe_exec_spec, probe_hidden))
    return mismatches


def verify_core_limits_sync() -> list[str]:
    """Compare advisor CORE_LIMITS against live `megakernel/core` definitions."""
    if (
        exec_spec is None
        or planner is None
        or core_profile is None
        or core_spec is None
        or collectives_params is None
        or core_vmem is None
        or default_profiles is None
        or runtime_policy is None
        or core_modules is None
    ):
        msg = "Megakernel core package requires JAX to verify CORE_LIMITS sync."
        raise MegakernelPlannerUnavailableError(msg)
    tiles = exec_spec.Tiles()
    checks = (
        (
            "max_batch_rows",
            CORE_LIMITS.max_batch_rows,
            exec_spec.MAX_BATCH_ROWS,
        ),
        (
            "verified_ranks",
            CORE_LIMITS.verified_ranks,
            _read_numerics_grid_r(core_profile.__file__),
        ),
        (
            "chip_kinds",
            CORE_LIMITS.chip_kinds,
            tuple(default_profiles.CHIP_BUDGETS.keys()),
        ),
        ("nbd", CORE_LIMITS.nbd, tiles.nbd),
        ("nbe", CORE_LIMITS.nbe, tiles.nbe),
        ("bank", CORE_LIMITS.bank, tiles.bank),
        ("ebank", CORE_LIMITS.ebank, tiles.ebank),
        ("vmem_mib", CORE_LIMITS.vmem_mib, tiles.vmem_mib),
        (
            "fit_vmem_headroom_mib",
            CORE_LIMITS.fit_vmem_headroom_mib,
            core_vmem.FIT_VMEM_HEADROOM_MIB,
        ),
        (
            "default_decode_bucket_top",
            CORE_LIMITS.default_decode_bucket_top,
            max(runtime_policy.DEFAULT_BUCKETS),
        ),
        (
            "default_verify_bucket_top",
            CORE_LIMITS.default_verify_bucket_top,
            max(runtime_policy.DEFAULT_VERIFY_BUCKETS),
        ),
        (
            "core_module_families",
            tuple(sorted(CORE_MODULE_FAMILIES)),
            tuple(
                sorted(
                    info.name
                    for info in pkgutil.iter_modules(core_modules.__path__)
                    if info.ispkg
                )
            ),
        ),
    )
    mismatches = [
        f"{name}: advisor={adv_val!r} != core={core_val!r}"
        for name, adv_val, core_val in checks
        if adv_val != core_val
    ]
    advisor_top = max(
        CORE_LIMITS.default_decode_bucket_top,
        CORE_LIMITS.default_verify_bucket_top,
    )
    if advisor_top > runtime_policy.MAX_BUCKET_ROWS:
        mismatches.append(
            "default_bucket_tops_within_max_bucket_rows: "
            f"advisor top {advisor_top} > core MAX_BUCKET_ROWS "
            f"{runtime_policy.MAX_BUCKET_ROWS}"
        )
    mismatches.extend(_verify_max_ar_slots_sync(tiles))
    return mismatches


def _try_plan_spec(
    model_spec: ModelSpec, exec_spec_obj: ExecSpec
) -> tuple[str, bool, Mapping[str, int], int, str, str, str]:
    """Isolate planner.plan and return parsed weights, ring slots, and dtypes."""
    if planner is None or jnp is None:
        raise MegakernelPlannerUnavailableError
    plan_err = ""
    ar_slots = 2
    is_periodic = True
    mlc: Mapping[str, int] = {}
    wd, ed, kv_dt = "fp8", "", "fp8"
    try:
        plan_obj = planner.plan(model_spec, exec_spec_obj)
        wds = tuple(plan_obj.weight_decls.values())
        mlc = plan_obj.module_layer_counts
        ar_slots = sum(x is not None for x in plan_obj.slot_ar_index)
        rings = ([w for w in wds if w.stream_ring == r] for r in ("static", "dynamic"))
        kv_states = [
            s
            for k, s in plan_obj.state_decls.items()
            if "dn" not in plan_obj.state_owners[k]
        ]
        wd, ed, kv_dt = [
            ("fp8" if jnp.dtype(x[0].dtype).itemsize == 1 else "bf16") if x else d
            for x, d in zip((*rings, kv_states), ("fp8", "", "fp8"), strict=True)
        ]
    except MegakernelConfigError as exc:
        plan_err, is_periodic = str(exc), False
    return plan_err, is_periodic, mlc, ar_slots, wd, ed, kv_dt


def advise_from_spec(
    model_spec: ModelSpec,
    exec_spec_obj: ExecSpec,
    chip: ChipSpec | str = "tpu7x",
    moe_mode: MoEMode = "masked_on_chip",
    dp: int = 1,
) -> AdvisorReport:
    """Bridge declarative ModelSpec + ExecSpec schedules via planner.plan."""
    if dp < 1:
        msg = f"dp must be >= 1, got {dp}"
        raise MegakernelConfigError(msg)
    plan_err, is_periodic, mlc, ar_slots, wd, ed, kv_dt = _try_plan_spec(
        model_spec, exec_spec_obj
    )

    cfg = dataclasses.asdict(model_spec.config)
    num_layers = int(cfg["L"])
    hidden = int(cfg["H"])
    att_every = int(cfg.get("att_every") or 1)
    num_experts = int(cfg.get("E") or 0)
    num_attn = sum(v for k, v in mlc.items() if "gqa" in k or "att" in k)
    num_attn = min(
        num_layers,
        num_attn or (num_layers // att_every if att_every > 1 else num_layers),
    )
    profile = finalize_model_profile(
        ModelProfile(
            name=model_spec.name,
            num_layers=num_layers,
            hidden_size=hidden,
            intermediate_size=int(cfg.get("I") or 4 * hidden),
            num_attention_heads=int(cfg.get("NH") or exec_spec_obj.R),
            num_kv_heads=int(cfg.get("NKV") or exec_spec_obj.R),
            head_dim=int(cfg.get("HD") or 128),
            num_linear_layers=max(0, num_layers - num_attn),
            linear_key_heads=int(cfg.get("KH") or 0),
            linear_value_heads=int(cfg.get("VH") or 0),
            num_experts=num_experts,
            top_k=int(cfg.get("TOPK") or 0) * (num_experts > 0),
            moe_intermediate_size=int(cfg.get("I") or 0),
            shared_expert_intermediate_size=int(cfg.get("IS") or 0),
            num_moe_layers=num_layers * (num_experts > 0),
            num_attention_layers=num_attn,
            linear_key_dim=int(cfg.get("KD") or 128),
            linear_value_dim=int(cfg.get("VD") or 128),
            is_periodic=is_periodic,
            ar_slots_per_layer=ar_slots,
            weight_dtype=wd,
            expert_dtype=ed or wd,
            kv_dtype=kv_dt,
        )
    )
    eff_ep = (
        exec_spec_obj.R
        if (num_experts > 0 and not bool(cfg.get("moe_tp", True)))
        else 1
    )
    workload = WorkloadSpec(
        phase="verify" if exec_spec_obj.T > 1 else "decode",
        batch=exec_spec_obj.NS if exec_spec_obj.T > 1 else exec_spec_obj.B,
        isl=exec_spec_obj.ctx,
        osl=0,
        context=exec_spec_obj.ctx,
        spec_tokens=max(0, exec_spec_obj.T - 1),
        tp=exec_spec_obj.R,
        dp=dp,
        ep=eff_ep,
        moe_mode=moe_mode,
        tiles=exec_spec_obj.tiles,
    )
    return advise(profile, workload, chip, plan_err)

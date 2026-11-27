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
"""Verification bridge checking advisor limits against megakernel/core."""

from __future__ import annotations

import ast
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from ..core import decls as core_decls
from ..core import exec_spec, planner
from ..core import modules as core_modules
from ..core import ops as core_ops
from ..core import profile as core_profile
from ..core import spec as core_spec
from ..core.collectives import params as collectives_params
from ..core.dma import vmem as core_vmem
from ..core.errors import MegakernelConfigError
from ..core.modules.gdn import decls as gdn_decls
from ..core.pool import layout as pool_layout
from ..core.profiles import default as default_profiles
from ..core.runtime import policy as runtime_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ..core.exec_spec import ExecSpec
    from ..core.planner import Plan
    from ..core.spec import LayerType as CoreLayerType
    from ..core.spec import ModelSpec as CoreModelSpec
    from ..core.spec import Module as CoreModule

from .chips import BUILTIN_CHIP_SPECS
from .constants import (
    CHIP_KIND_MAP,
    CORE_KV_DTYPE_ALIASES,
    CORE_LIMITS,
    CORE_STATE_DTYPE_ALIASES,
    CORE_SUPPORTED_DTYPES,
    DOUBLE_BUFFER_SLOTS,
    DTYPE_BYTES,
)
from .specs import CORE_MODULE_FAMILIES

# Minimal valid core ExecSpec probe geometry (R=8, small dense slice H=128).
AR_PROBE_HIDDEN_SIZE: int = 128
AR_PROBE_BATCH: int = 4
AR_PROBE_SEQ_LEN: int = 1
AR_PROBE_RANKS: int = 8


@dataclass(frozen=True)
class _GdnProbeGeom:
    """Minimal GDN geometry for querying live `gdn_cache_decl().rec_dtype`."""

    KH: int = 1
    VH: int = 1
    KD: int = 128
    VD: int = 128
    conv_kernel: int = 4
    conv_taps: int = 3


def _read_numerics_grid_r(profile_file: str | None) -> tuple[int, ...]:
    """Read `_NUMERICS_GRID_R` tuple literal from `core/profile.py`."""
    if profile_file is None:
        return ()
    tree = ast.parse(Path(profile_file).read_text(encoding="utf-8"))
    for node in tree.body:
        if type(node) is ast.Assign:
            for target in node.targets:
                if type(target) is ast.Name and target.id == "_NUMERICS_GRID_R":
                    val = ast.literal_eval(node.value)
                    if type(val) is tuple:
                        return tuple(int(item) for item in val)
    return ()


def _make_ar_probe_spec(
    module_cls: Callable[[], CoreModule],
    layer_type_cls: Callable[..., CoreLayerType],
    model_spec_cls: Callable[..., CoreModelSpec],
    num_slots: int,
    probe_hidden: int,
) -> CoreModelSpec:
    """Build a minimal synthetic ModelSpec with `num_slots` all-reduce slots."""
    slots = []
    for slot_idx in range(num_slots):
        probe_module = module_cls()
        probe_module.name = f"_advisor_ar_probe_{slot_idx}"
        probe_module.reduction = "allreduce"
        slots.append(probe_module)
    return model_spec_cls(
        name="_advisor_ar_slot_probe",
        config=SimpleNamespace(L=1, H=probe_hidden),
        layer_type=layer_type_cls(
            name="_advisor_ar_slot_layer",
            slots=tuple(slots),
        ),
    )


def _probe_max_ar_slots_planner(
    plan_fn: Callable[[CoreModelSpec, ExecSpec], Plan],
    make_spec: Callable[[int], CoreModelSpec],
    limit: int,
    probe_exec_spec: ExecSpec,
) -> list[str]:
    """Verify core planner accepts `limit` AR slots and rejects `limit + 1`."""
    mismatches: list[str] = []
    try:
        planned = plan_fn(make_spec(limit), probe_exec_spec)
    except MegakernelConfigError as exc:
        mismatches.append(
            f"max_ar_slots_per_layer: core planner rejected {limit} AR slots ({exc})"
        )
    else:
        planned_slots = sum(item is not None for item in planned.slot_ar_index)
        if planned_slots != limit:
            mismatches.append(
                f"max_ar_slots_per_layer: advisor={limit!r} != "
                f"planned_ar_slots={planned_slots!r}"
            )
    try:
        plan_fn(make_spec(limit + 1), probe_exec_spec)
    except MegakernelConfigError:
        pass
    else:
        mismatches.append(
            f"max_ar_slots_per_layer: core planner accepted {limit + 1} AR slots "
            f"(exceeds advisor={limit!r})"
        )
    return mismatches


def _verify_max_ar_slots_sync(
    semaphore_slots_by_key: Mapping[str, int],
    plan_fn: Callable[[CoreModelSpec, ExecSpec], Plan],
    make_spec: Callable[[int], CoreModelSpec],
    probe_exec_spec: ExecSpec,
) -> list[str]:
    """Verify `CORE_LIMITS.max_ar_slots_per_layer` against core scratch and planner."""
    limit = CORE_LIMITS.max_ar_slots_per_layer
    if limit < 1:
        return [f"max_ar_slots_per_layer: advisor={limit!r} must be >= 1"]
    mismatches = [
        f"max_ar_slots_per_layer: advisor={limit!r} != "
        f"core {semaphore_key}.shape[1]={semaphore_slots!r}"
        for semaphore_key, semaphore_slots in semaphore_slots_by_key.items()
        if semaphore_slots != limit
    ]
    mismatches.extend(
        _probe_max_ar_slots_planner(plan_fn, make_spec, limit, probe_exec_spec)
    )
    return mismatches


def _probe_core_supported_dtypes(
    normalize_weight_dtypes_fn: Callable[
        [object, Sequence[str], str], Mapping[str, str]
    ],
) -> tuple[str, ...]:
    """Probe core weight-dtype normalizer across known advisor dtype names."""
    accepted: set[str] = set()
    for dtype_name in DTYPE_BYTES:
        try:
            normalized = normalize_weight_dtypes_fn({"w": dtype_name}, ("w",), "probe")
        except ValueError:
            continue
        accepted.add(str(normalized["w"]))
    return tuple(sorted(accepted))


def _canonical_cache_dtype(aliases: Mapping[str, str]) -> str:
    """Derive canonical cache dtype string from `aliases` target names."""
    return "|".join(sorted(set(aliases.values())))


def verify_core_limits_sync() -> list[str]:
    """Compare advisor CORE_LIMITS against cross-source `megakernel/core` invariants.

    Returns:
        List of mismatch messages; empty list means all limits match.
    """
    tiles = exec_spec.Tiles()
    chip_budgets = default_profiles.CHIP_BUDGETS
    core_budget_view = tuple(
        (kind, budget.vmem_mib, budget.vmem_mib, budget.smem_mib)
        for kind, budget in sorted(chip_budgets.items())
    )
    advisor_budget_view = tuple(
        sorted(
            (
                core_kind,
                BUILTIN_CHIP_SPECS[adv_name].vmem_usable_bytes >> 20
                if adv_name in BUILTIN_CHIP_SPECS
                else -1,
                CORE_LIMITS.vmem_mib,
                CORE_LIMITS.smem_mib,
            )
            for adv_name, core_kind in CHIP_KIND_MAP.items()
        )
    )
    advisor_kv_dtype = _canonical_cache_dtype(CORE_KV_DTYPE_ALIASES)
    advisor_state_dtype = _canonical_cache_dtype(CORE_STATE_DTYPE_ALIASES)
    pool_cfg = pool_layout.PoolLayout()
    pool_page_dtype = str(jnp.dtype(pool_cfg.page_dtype))
    core_fp8_dtype = str(jnp.dtype(core_ops.FP8))
    core_kv_dtype = (
        pool_page_dtype
        if pool_page_dtype == core_fp8_dtype
        else f"mismatch(PoolLayout={pool_page_dtype}, FP8={core_fp8_dtype})"
    )
    pool_rec_dtype = str(jnp.dtype(pool_cfg.rec_dtype))
    gdn_rec_dtype = str(jnp.dtype(gdn_decls.gdn_cache_decl(_GdnProbeGeom()).rec_dtype))
    core_rec_dtype = (
        pool_rec_dtype
        if pool_rec_dtype == gdn_rec_dtype
        else f"mismatch(PoolLayout={pool_rec_dtype}, GDN={gdn_rec_dtype})"
    )
    checks = (
        (
            "verified_ranks",
            CORE_LIMITS.verified_ranks,
            _read_numerics_grid_r(core_profile.__file__),
        ),
        ("chip_budgets", advisor_budget_view, core_budget_view),
        ("min_ring_depth", DOUBLE_BUFFER_SLOTS, core_vmem.MIN_RING_DEPTH),
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
        (
            "core_supported_dtypes",
            tuple(sorted(CORE_SUPPORTED_DTYPES)),
            _probe_core_supported_dtypes(core_decls.normalize_weight_dtypes),
        ),
        ("kv_page_dtype", advisor_kv_dtype, core_kv_dtype),
        ("recurrent_state_dtype", advisor_state_dtype, core_rec_dtype),
    )
    mismatches = [
        f"{name}: advisor={adv_val!r} != core={core_val!r}"
        for name, adv_val, core_val in checks
        if adv_val != core_val
    ]
    if tiles.nbd < core_vmem.MIN_RING_DEPTH or tiles.nbe < core_vmem.MIN_RING_DEPTH:
        mismatches.append(
            f"min_ring_depth: Tiles(nbd={tiles.nbd}, nbe={tiles.nbe}) "
            f"< core MIN_RING_DEPTH={core_vmem.MIN_RING_DEPTH}"
        )
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
    probe_exec_spec = exec_spec.ExecSpec(
        B=AR_PROBE_BATCH, T=AR_PROBE_SEQ_LEN, R=AR_PROBE_RANKS
    )
    ring_scratch = collectives_params.ring_scratch(
        collectives_params.comm_params(probe_exec_spec, tiles, H=AR_PROBE_HIDDEN_SIZE)
    )
    semaphore_slots_by_key = {
        key: int(ring_scratch[key].shape[1]) for key in ("ar_ssem", "ar_rsem")
    }
    mismatches.extend(
        _verify_max_ar_slots_sync(
            semaphore_slots_by_key,
            planner.plan,
            lambda slots: _make_ar_probe_spec(
                core_spec.Module,
                core_spec.LayerType,
                core_spec.ModelSpec,
                slots,
                AR_PROBE_HIDDEN_SIZE,
            ),
            probe_exec_spec,
        )
    )
    return mismatches

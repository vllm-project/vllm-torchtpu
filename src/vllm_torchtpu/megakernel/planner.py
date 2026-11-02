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
"""Generic build-time planner deriving weight tiling and schedules from `ModelSpec`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp

from . import pool_alias
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .ops import BF16, F32, rup
from .spec import ExtraInput, ModelSpec, Module, Spec, StateDecl, SwitchSlot, WeightDecl


def k_tile(K: int, N: int, dtype: Any, bank: int) -> int:
    """Largest 128-multiple divisor of K whose `[tk, N]` tile fits `bank` bytes."""
    dt = jnp.dtype(dtype)
    if K <= 0 or K % 128 != 0:
        raise MegakernelConfigError(
            f"Weight K dimension must be a positive multiple of 128, got K={K} "
            f"(N={N}, dtype={dt})"
        )
    best = None
    for tk in range(128, K + 1, 128):
        if K % tk == 0 and pool_alias.storage_size((tk, N), dt) <= bank:
            best = tk
    if best is None:
        min_bytes = pool_alias.storage_size((128, N), dt)
        raise MegakernelConfigError(
            f"No 128-row tile of shape (K={K}, N={N}, dtype={dt}) fits bank={bank} "
            f"bytes (minimum 128x{N} tile requires {min_bytes} bytes)"
        )
    return best


@dataclass(frozen=True)
class Plan:
    """Declarative build-time plan derived from `ModelSpec` and `ExecSpec`."""

    model_spec: ModelSpec
    ex: ExecSpec
    kinds: dict[str, tuple[int, int, Any]]
    static_kinds: tuple[str, ...]
    dynamic_kinds: tuple[str, ...]
    hbm_weights: dict[str, WeightDecl]
    weight_decls: dict[str, WeightDecl]
    weight_owners: dict[str, str]
    state_decls: dict[str, StateDecl]
    state_owners: dict[str, str]
    output_decls: dict[str, StateDecl]
    output_owners: dict[str, str]
    tk: dict[str, int]
    nt: dict[str, int]
    period: int
    branch_counts: tuple[int, ...]
    branch_tiles: tuple[int, ...]
    branch_segs: tuple[list[tuple[str, int, int, int]], ...]
    t_cycle: int
    T_TOTAL: int
    module_layer_counts: dict[str, int]
    module_branch_idx: dict[str, int | None]
    slot_ar_index: tuple[int | None, ...]
    smalls_layout: dict[str, tuple[int, int]]
    smalls_width: int
    shared_widths: dict[str, int]
    w_names: tuple[str, ...]
    s_names: tuple[str, ...]
    o_names: tuple[str, ...]
    module_params: dict[str, Any]


def _discover_schedule(
    model_spec: ModelSpec,
) -> tuple[
    int,
    tuple[int, ...],
    dict[str, int],
    dict[str, int | None],
    tuple[int | None, ...],
]:
    """Derive repeating cycle `(period, branch_counts)`, layer counts, and AR slots.

    Note: `SwitchSlot` currently models repeating periodic schedules
    (`L % period == 0`) with at most 2 active branches per slot. Non-repeating
    prefix layouts (e.g. GLM's 3 dense + 75 MoE layers) will be represented by a
    per-layer schedule table when GLM lands.
    """
    g = model_spec.config
    L = g.L
    period = 1
    branch_counts: tuple[int, ...] = (1,)

    for s in model_spec.layer_type.slots:
        if isinstance(s, SwitchSlot):
            if len(s.active_branches) > 2:
                raise MegakernelConfigError(
                    f"SwitchSlot {s.name!r} supports at most 2 active branches, "
                    f"got {len(s.active_branches)}"
                )
            if len(s.active_branches) > 1:
                if period == 1:
                    period = s.period
                    branch_counts = s.active_counts
                elif s.period != period or s.active_counts != branch_counts:
                    raise MegakernelConfigError(
                        "Multiple SwitchSlots in a LayerType must share "
                        f"period and branch_counts: expected ({period}, "
                        f"{branch_counts}), got ({s.period}, {s.active_counts}) "
                        f"on {s.name!r}"
                    )

    if L <= 0 or L % period != 0:
        raise MegakernelConfigError(
            f"Layer count L={L} must be a positive multiple of SwitchSlot "
            f"period={period}"
        )
    n_cycles = L // period

    module_layer_counts: dict[str, int] = {}
    module_branch_idx: dict[str, int | None] = {}
    slot_ar_index: list[int | None] = []
    next_ar_slot = 0

    for s in model_spec.layer_type.slots:
        if isinstance(s, SwitchSlot):
            active = s.active_branches
            counts = s.active_counts
            reds = {m.reduction for m in active}
            if len(reds) != 1:
                raise MegakernelConfigError(
                    f"SwitchSlot {s.name!r} branches must share reduction policy: "
                    f"{reds}"
                )
            norms = {m.input_norm for m in active}
            if len(norms) != 1:
                raise MegakernelConfigError(
                    f"SwitchSlot {s.name!r} branches must share input_norm: {norms}"
                )
            red = active[0].reduction
            if len(active) == 1:
                m = active[0]
                module_layer_counts[m.name] = L
                module_branch_idx[m.name] = None
            else:
                for b_idx, (m, c_b) in enumerate(zip(active, counts)):
                    module_layer_counts[m.name] = n_cycles * c_b
                    module_branch_idx[m.name] = b_idx
        else:
            red = s.reduction
            module_layer_counts[s.name] = L
            module_branch_idx[s.name] = None

        if red == "allreduce":
            if next_ar_slot >= 2:
                raise MegakernelConfigError(
                    "At most 2 'allreduce' slots per layer are supported by "
                    "double-buffered AllReduceRing"
                )
            slot_ar_index.append(next_ar_slot)
            next_ar_slot += 1
        else:
            slot_ar_index.append(None)

    return (
        period,
        branch_counts,
        module_layer_counts,
        module_branch_idx,
        tuple(slot_ar_index),
    )


def plan(
    model_spec: ModelSpec,
    ex: ExecSpec,
) -> Plan:
    """Derive weight tile sizes, stream schedules, smalls layout, and metadata.

    Pure function of `model_spec` and `ex` (`ex.resolve_tiles(g)`): never fits
    tiles or queries device budgets.
    """
    active_modules = model_spec.all_modules()
    active_names = {m.name for m in active_modules}
    unknown_debug = sorted(set(ex.debug.modules) - active_names)
    if unknown_debug:
        raise MegakernelConfigError(
            f"DebugOptions.modules has unknown or inactive module(s) {unknown_debug}; "
            f"active modules: {sorted(active_names)}"
        )
    g = model_spec.config
    tiles = ex.resolve_tiles(g)
    (
        period,
        branch_counts,
        module_layer_counts,
        module_branch_idx,
        slot_ar_index,
    ) = _discover_schedule(model_spec)

    kinds: dict[str, tuple[int, int, Any]] = {}
    static_kinds: list[str] = []
    dynamic_kinds: list[str] = []
    hbm_weights: dict[str, WeightDecl] = {}
    weight_decls: dict[str, WeightDecl] = {}
    weight_owners: dict[str, str] = {}
    state_decls: dict[str, StateDecl] = {}
    state_owners: dict[str, str] = {}
    output_decls: dict[str, StateDecl] = {}
    output_owners: dict[str, str] = {}

    small_fields_order: list[str] = []
    small_widths: dict[str, int] = {}
    small_slot_owners: dict[str, int] = {}
    shared_widths: dict[str, int] = {
        "acc": g.H,
        "ybuf": 128,
        "xb2": 128,
        "mix": 128,
    }

    module_slot_idx: dict[str, int] = {}
    for slot_i, s in enumerate(model_spec.layer_type.slots):
        if isinstance(s, SwitchSlot):
            for m in s.active_branches:
                module_slot_idx[m.name] = slot_i
        else:
            module_slot_idx[s.name] = slot_i

    for m in model_spec.all_modules():
        slot_i = module_slot_idx[m.name]
        for w in m.weights(g, ex):
            if w.name in weight_decls:
                raise MegakernelConfigError(f"Duplicate weight declaration: {w.name!r}")
            weight_decls[w.name] = w
            weight_owners[w.name] = m.name
            if w.stream_ring == "static":
                kinds[w.name] = (w.K, w.N, w.dtype)
                static_kinds.append(w.name)
            elif w.stream_ring == "dynamic":
                kinds[w.name] = (w.K, w.N, w.dtype)
                dynamic_kinds.append(w.name)
            elif w.stream_ring == "hbm":
                hbm_weights[w.name] = w
            else:
                raise MegakernelConfigError(
                    f"Unsupported stream_ring {w.stream_ring!r} on {w.name}"
                )

        m_small_names: set[str] = set()
        for sf in m.smalls(g, ex):
            m_small_names.add(sf.name)
            if sf.name not in small_widths:
                small_fields_order.append(sf.name)
                small_widths[sf.name] = sf.width
                small_slot_owners[sf.name] = slot_i
            else:
                if small_slot_owners[sf.name] != slot_i:
                    raise MegakernelConfigError(
                        f"Smalls field {sf.name!r} declared by modules in different "
                        f"slots (slot {small_slot_owners[sf.name]} vs slot {slot_i}); "
                        "only branches of the same SwitchSlot may share a smalls "
                        "field name."
                    )
                small_widths[sf.name] = max(small_widths[sf.name], sf.width)
        if m.input_norm is not None and m.input_norm not in m_small_names:
            raise MegakernelConfigError(
                f"Module {m.name!r} declares input_norm={m.input_norm!r} "
                f"which is not in its smalls(): {sorted(m_small_names)}"
            )

        for st in m.states(g, ex):
            if st.name in state_decls:
                prev = state_decls[st.name]
                if not (st == prev and not st.per_layer and st.access == "r"):
                    raise MegakernelConfigError(
                        f"Duplicate state declaration {st.name!r}: {st} vs {prev}"
                    )
                continue
            state_decls[st.name] = st
            state_owners[st.name] = m.name

        for out in m.outputs(g, ex):
            if out.name in output_decls:
                raise MegakernelConfigError(
                    f"Duplicate output declaration: {out.name!r}"
                )
            output_decls[out.name] = out
            output_owners[out.name] = m.name

        for buf_name, req_w in m.shared_widths(g, ex).items():
            shared_widths[buf_name] = max(shared_widths.get(buf_name, 0), req_w)

    tk = {
        k: k_tile(K, N, dt, tiles.ebank if k in dynamic_kinds else tiles.bank)
        for k, (K, N, dt) in kinds.items()
    }
    nt = {k: kinds[k][0] // tk[k] for k in kinds}

    num_branches = len(branch_counts)
    branch_segs_list: list[list[tuple[str, int, int, int]]] = []
    branch_tiles_list: list[int] = []

    for b_idx in range(num_branches):
        cur = 0
        segs: list[tuple[str, int, int, int]] = []
        for s in model_spec.layer_type.slots:
            if isinstance(s, SwitchSlot):
                active = s.active_branches
                mod: Module = active[0] if len(active) == 1 else active[b_idx]
                ordi = 0 if len(active) > 1 else 1
            else:
                mod = s
                ordi = 1
            for w in mod.weights(g, ex):
                if w.stream_ring == "static":
                    nxt = cur + nt[w.name]
                    segs.append((w.name, cur, nxt, ordi))
                    cur = nxt
        branch_segs_list.append(segs)
        branch_tiles_list.append(cur)

    t_cycle = sum(c_b * t_b for c_b, t_b in zip(branch_counts, branch_tiles_list))
    T_TOTAL = (g.L // period) * t_cycle

    lay: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in small_fields_order:
        w_len = small_widths[name]
        lay[name] = (offset, w_len)
        offset += rup(w_len, 128)
    smalls_width = max(128, offset)

    w_names = tuple(
        list(static_kinds)
        + list(dynamic_kinds)
        + list(hbm_weights.keys())
        + ["smalls", "final_norm"]
    )
    s_names = tuple(state_decls.keys())
    o_names = ("hidden",) + tuple(output_decls.keys())

    p_base = Plan(
        model_spec=model_spec,
        ex=ex,
        kinds=kinds,
        static_kinds=tuple(static_kinds),
        dynamic_kinds=tuple(dynamic_kinds),
        hbm_weights=hbm_weights,
        weight_decls=weight_decls,
        weight_owners=weight_owners,
        state_decls=state_decls,
        state_owners=state_owners,
        output_decls=output_decls,
        output_owners=output_owners,
        tk=tk,
        nt=nt,
        period=period,
        branch_counts=branch_counts,
        branch_tiles=tuple(branch_tiles_list),
        branch_segs=tuple(branch_segs_list),
        t_cycle=t_cycle,
        T_TOTAL=T_TOTAL,
        module_layer_counts=module_layer_counts,
        module_branch_idx=module_branch_idx,
        slot_ar_index=slot_ar_index,
        smalls_layout=lay,
        smalls_width=smalls_width,
        shared_widths=shared_widths,
        w_names=w_names,
        s_names=s_names,
        o_names=o_names,
        module_params={},
    )
    return replace(
        p_base,
        module_params={
            m.name: m.plan_params(g, ex, p_base) for m in model_spec.all_modules()
        },
    )


def weight_shapes(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> dict[str, Any]:
    """Per-rank HBM weight operand shapes derived from `Module.weights`."""
    if p is None:
        p = plan(model_spec, ex)
    g = model_spec.config
    out: dict[str, Any] = {}
    for name, w in p.weight_decls.items():
        owner = p.weight_owners[name]
        l_cnt = max(1, p.module_layer_counts[owner])
        if w.stream_ring in ("static", "dynamic"):
            t_shape = (p.nt[name], p.tk[name], w.N)
            if w.layout == "per_layer":
                out[name] = tuple(
                    Spec((*w.lead_dims, *t_shape), w.dtype) for _ in range(l_cnt)
                )
            else:
                out[name] = ((l_cnt, *w.lead_dims, *t_shape), w.dtype)
        else:
            if w.layout == "per_layer":
                out[name] = tuple(
                    Spec((*w.lead_dims, w.K, w.N), w.dtype) for _ in range(l_cnt)
                )
            else:
                out[name] = ((l_cnt, *w.lead_dims, w.K, w.N), w.dtype)
    out["smalls"] = ((g.L, 1, p.smalls_width), F32)
    out["final_norm"] = ((1, g.H), F32)
    return out


def state_shapes(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> dict[str, Any]:
    """Per-rank input state shapes derived from `Module.states`."""
    if p is None:
        p = plan(model_spec, ex)
    out: dict[str, Any] = {}
    for name, st in p.state_decls.items():
        owner = p.state_owners[name]
        l_cnt = max(1, p.module_layer_counts[owner])
        if st.tupled:
            out[name] = tuple(Spec(st.shape, st.dtype) for _ in range(l_cnt))
        elif st.per_layer:
            out[name] = ((l_cnt, *st.shape), st.dtype)
        else:
            out[name] = (st.shape, st.dtype)
    return out


def kernel_out_shapes(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> dict[str, Any]:
    """Per-rank output tensor shapes derived from `Module.outputs` plus `hidden`."""
    if p is None:
        p = plan(model_spec, ex)
    g = model_spec.config
    out: dict[str, Any] = {"hidden": ((ex.M, g.H), BF16)}
    for name, od in p.output_decls.items():
        owner = p.output_owners[name]
        l_cnt = max(1, p.module_layer_counts[owner])
        if od.tupled:
            out[name] = tuple(Spec(od.shape, od.dtype) for _ in range(l_cnt))
        elif od.per_layer:
            out[name] = ((l_cnt, *od.shape), od.dtype)
        else:
            out[name] = (od.shape, od.dtype)
    return out


def encode_ecopy(E: int, ER: int, EW: int, moe_tp: bool, cores: Any) -> list[float]:
    """Pack per-layer redundant expert copy descriptors into an `[E]` f32 row."""
    extra = EW - ER
    out = [0.0] * E
    if moe_tp or extra <= 0:
        return out
    vid = {
        int(g): r0 * ER + s0
        for r0, lst in enumerate(cores)
        for s0, g in enumerate(lst[:ER])
    }
    for r1, lst in enumerate(cores):
        for k, g in enumerate(lst[ER:EW]):
            g_int = int(g)
            if g_int < 0:
                continue
            v = vid.get(g_int)
            if not (0 <= g_int < E) or v is None or not (0 <= v < E):
                warnings.warn(
                    f"encode_ecopy: expert {g_int} at redundant slot "
                    f"(rank={r1}, slot={k}) is out of range [0, {E}) or missing "
                    "from primary slots; ignoring.",
                    stacklevel=2,
                )
                continue
            if v // ER == r1:
                warnings.warn(
                    f"encode_ecopy: expert {g_int} (vid={v}) has a redundant slot "
                    f"(rank={r1}, slot={k}) on the same rank as its primary slot; "
                    "ignoring same-rank replica.",
                    stacklevel=2,
                )
                continue
            if out[v] != 0.0:
                warnings.warn(
                    f"encode_ecopy: expert {g_int} (vid={v}) has multiple "
                    "secondary replicas; overwriting previous replica with "
                    f"(rank={r1}, slot={k}).",
                    stacklevel=2,
                )
            out[v] = float(1 + r1 * extra + k)
    return out


def extra_inputs(model_spec: ModelSpec, ex: ExecSpec) -> list[ExtraInput]:
    """Every module's `extra_inputs`, in module declaration order (unique names)."""
    g = model_spec.config
    out: list[ExtraInput] = []
    for m in model_spec.all_modules():
        out.extend(m.extra_inputs(g, ex))
    names = [x.name for x in out]
    if len(set(names)) != len(names):
        raise MegakernelConfigError(f"duplicate module extra inputs {names}")
    return out

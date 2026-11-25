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
from .spec import Block, ExtraInput, ModelSpec, Spec, StateDecl, SwitchSlot, WeightDecl


def k_tile(K: int, N: int, dtype: Any, bank: int) -> int:
    """Largest 128-multiple divisor of K whose `[tk, N]` tile fits `bank` bytes."""
    best = None
    for tk in range(128, K + 1, 128):
        if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= bank:
            best = tk
    assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, bank)
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
    block_layer_counts: dict[str, int]
    block_branch_idx: dict[str, int | None]
    slot_ar_index: tuple[int | None, ...]
    smalls_layout: dict[str, tuple[int, int]]
    smalls_width: int
    shared_widths: dict[str, int]
    w_names: tuple[str, ...]
    s_names: tuple[str, ...]
    o_names: tuple[str, ...]
    block_params: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return self.__dataclass_fields__.keys()


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
            assert len(s.active_branches) <= 2, (
                f"SwitchSlot {s.name!r} supports at most 2 active branches, "
                f"got {len(s.active_branches)}"
            )
            if len(s.active_branches) > 1:
                if period == 1:
                    period = s.period
                    branch_counts = s.active_counts
                else:
                    assert s.period == period and s.active_counts == branch_counts, (
                        "Multiple SwitchSlots in a LayerType must share "
                        "period and branch_counts"
                    )

    assert L % period == 0, (L, period)
    n_cycles = L // period

    block_layer_counts: dict[str, int] = {}
    block_branch_idx: dict[str, int | None] = {}
    slot_ar_index: list[int | None] = []
    next_ar_slot = 0

    for s in model_spec.layer_type.slots:
        if isinstance(s, SwitchSlot):
            active = s.active_branches
            counts = s.active_counts
            reds = {b.reduction for b in active}
            assert len(reds) == 1, (
                f"SwitchSlot {s.name} branches must share reduction policy: {reds}"
            )
            norms = {b.input_norm for b in active}
            assert len(norms) == 1, (
                f"SwitchSlot {s.name} branches must share input_norm: {norms}"
            )
            red = active[0].reduction
            if len(active) == 1:
                b = active[0]
                block_layer_counts[b.name] = L
                block_branch_idx[b.name] = None
            else:
                for b_idx, (b, c_b) in enumerate(zip(active, counts)):
                    block_layer_counts[b.name] = n_cycles * c_b
                    block_branch_idx[b.name] = b_idx
        else:
            red = s.reduction
            block_layer_counts[s.name] = L
            block_branch_idx[s.name] = None

        if red == "allreduce":
            assert next_ar_slot < 2, (
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
        block_layer_counts,
        block_branch_idx,
        tuple(slot_ar_index),
    )


def plan(
    model_spec: ModelSpec,
    ex: ExecSpec,
    *,
    fit_vmem: bool = True,
) -> Plan:
    """Derive weight tile sizes, stream schedules, smalls layout, and metadata.

    When `fit_vmem=True` (default), `ex.tiles=None` plans the default tiles
    fitted to the model's VMEM budget, as `kernel.build` does
    (`vmem.fit_default_tiles`). Callers that only need weight tile shapes and
    smalls layout (e.g. `pack_weights`) pass `fit_vmem=False` to avoid querying
    device budgets when no JAX backend is initialized.
    """
    if fit_vmem and ex.tiles is None:
        # Local import: `vmem` builds on this module.
        from .vmem import fit_default_tiles

        ex = fit_default_tiles(model_spec, ex)
    active_blocks = model_spec.all_blocks()
    active_names = {b.name for b in active_blocks}
    unknown_debug = sorted(set(ex.debug.blocks) - active_names)
    if unknown_debug:
        raise MegakernelConfigError(
            f"DebugOptions.blocks has unknown or inactive block(s) {unknown_debug}; "
            f"active blocks: {sorted(active_names)}"
        )
    g = model_spec.config
    tiles = ex.resolve_tiles(g)
    (
        period,
        branch_counts,
        block_layer_counts,
        block_branch_idx,
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

    block_slot_idx: dict[str, int] = {}
    for slot_i, s in enumerate(model_spec.layer_type.slots):
        if isinstance(s, SwitchSlot):
            for b in s.active_branches:
                block_slot_idx[b.name] = slot_i
        else:
            block_slot_idx[s.name] = slot_i

    for b in model_spec.all_blocks():
        slot_i = block_slot_idx[b.name]
        for w in b.weights(g, ex):
            assert w.name not in weight_decls, f"Duplicate weight declaration: {w.name}"
            weight_decls[w.name] = w
            weight_owners[w.name] = b.name
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

        b_small_names: set[str] = set()
        for sf in b.smalls(g, ex):
            b_small_names.add(sf.name)
            if sf.name not in small_widths:
                small_fields_order.append(sf.name)
                small_widths[sf.name] = sf.width
                small_slot_owners[sf.name] = slot_i
            else:
                assert small_slot_owners[sf.name] == slot_i, (
                    f"Smalls field {sf.name!r} declared by blocks in different slots "
                    f"(slot {small_slot_owners[sf.name]} vs slot {slot_i}); only "
                    f"branches of the same SwitchSlot may share a smalls field name."
                )
                small_widths[sf.name] = max(small_widths[sf.name], sf.width)
        if b.input_norm is not None:
            assert b.input_norm in b_small_names, (
                f"Block {b.name!r} declares input_norm={b.input_norm!r} "
                f"which is not in its smalls(): {sorted(b_small_names)}"
            )

        for st in b.states(g, ex):
            if st.name in state_decls:
                prev = state_decls[st.name]
                assert st == prev and not st.per_layer and st.access == "r", (
                    f"Duplicate state declaration {st.name!r}: {st} vs {prev}"
                )
                continue
            state_decls[st.name] = st
            state_owners[st.name] = b.name

        for out in b.outputs(g, ex):
            assert out.name not in output_decls, (
                f"Duplicate output declaration: {out.name}"
            )
            output_decls[out.name] = out
            output_owners[out.name] = b.name

        for buf_name, req_w in b.shared_widths(g, ex).items():
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
                blk: Block = active[0] if len(active) == 1 else active[b_idx]
                ordi = 0 if len(active) > 1 else 1
            else:
                blk = s
                ordi = 1
            for w in blk.weights(g, ex):
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
        block_layer_counts=block_layer_counts,
        block_branch_idx=block_branch_idx,
        slot_ar_index=slot_ar_index,
        smalls_layout=lay,
        smalls_width=smalls_width,
        shared_widths=shared_widths,
        w_names=w_names,
        s_names=s_names,
        o_names=o_names,
        block_params={},
    )
    return replace(
        p_base,
        block_params={
            b.name: b.plan_params(g, ex, p_base) for b in model_spec.all_blocks()
        },
    )


def weight_shapes(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> dict[str, Any]:
    """Per-rank HBM weight operand shapes derived from `Block.weights`."""
    if p is None:
        p = plan(model_spec, ex, fit_vmem=False)
    g = model_spec.config
    out: dict[str, Any] = {}
    for name, w in p.weight_decls.items():
        owner = p.weight_owners[name]
        l_cnt = max(1, p.block_layer_counts[owner])
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
    """Per-rank input state shapes derived from `Block.states`."""
    if p is None:
        p = plan(model_spec, ex)
    out: dict[str, Any] = {}
    for name, st in p.state_decls.items():
        owner = p.state_owners[name]
        l_cnt = max(1, p.block_layer_counts[owner])
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
    """Per-rank output tensor shapes derived from `Block.outputs` plus `hidden`."""
    if p is None:
        p = plan(model_spec, ex)
    g = model_spec.config
    out: dict[str, Any] = {"hidden": ((ex.M, g.H), BF16)}
    for name, od in p.output_decls.items():
        owner = p.output_owners[name]
        l_cnt = max(1, p.block_layer_counts[owner])
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
            if v is None:
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
    """Every block's `extra_inputs`, in block declaration order (unique names)."""
    g = model_spec.config
    out: list[ExtraInput] = []
    for b in model_spec.all_blocks():
        out.extend(b.extra_inputs(g, ex))
    names = [x.name for x in out]
    if len(set(names)) != len(names):
        raise MegakernelConfigError(f"duplicate block extra inputs {names}")
    return out

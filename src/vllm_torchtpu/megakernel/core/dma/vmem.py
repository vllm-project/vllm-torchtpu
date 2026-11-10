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
"""VMEM/SMEM/Semaphore scratch allocator, `StreamParams` builder, and tile fitter."""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .. import exec_spec
from ..collectives.params import CommParams, comm_params, ring_scratch, stage_scratch
from ..errors import MegakernelConfigError
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, FP8, round_up
from ..planner import Plan, extra_inputs, plan
from ..profiles.default import chip_budget
from ..spec import ModelSpec, ScratchReq
from . import pool_alias

logger = logging.getLogger(__name__)
U8 = jnp.uint8


# VMEM scratch buffers whose 3D+ FP8 shape `(..., rows_u32, 4, cols)` packs 4
# fp8 bytes per 32-bit sublane word along `shape[-3]` (paged KV / pool pages).
_PACKED_FP8_PAGE_BUFFERS: frozenset[str] = frozenset(
    {
        "epool",
        "kvbuf",
        "gpr_rec",
        "gpr_conv",
        "gdn_in_page",
        "gdn_out_page",
        "kv_scat_page",
    }
)


def _vmem_storage_bytes(
    shape: tuple[int, ...], dtype: Any, *, packed: bool = False
) -> int:
    """Return `(8 * packing, 128)`-tiled VMEM storage bytes for `shape`/`dtype`.

    When `packed=True` (set for `_PACKED_FP8_PAGE_BUFFERS`), sublane-packed
    32-bit word layouts `(..., rows_u32, packing, cols)` (such as paged KV
    `(..., HD // 4, 4, BLK)` fp8) already pack `packing` elements per 32-bit
    sublane word along `shape[-3]`, so their physical row extent is
    `rows_u32 * packing`. Unmarked buffers are never guessed packed from their
    shape.
    """
    dt = jnp.dtype(dtype)
    packing = 32 // (dt.itemsize * 8)
    if not shape:
        sh2: tuple[int, ...] = (1, 1)
    elif len(shape) == 1:
        sh2 = (1, shape[0])
    elif packed and len(shape) >= 3 and packing > 1 and shape[-2] == packing:
        sh2 = (*shape[:-3], shape[-3] * shape[-2], shape[-1])
    else:
        sh2 = shape
    return pool_alias.storage_size(sh2, dt)


def _vmem_buffer_bytes(name: str, shape: tuple[int, ...], dtype: Any) -> int:
    """Return VMEM storage bytes for named scratch buffer `name`."""
    return _vmem_storage_bytes(shape, dtype, packed=(name in _PACKED_FP8_PAGE_BUFFERS))


@dataclass(frozen=True)
class StreamParams:
    """Core weight-streaming ring parameters derived from `Plan` and `ExecSpec`."""

    L: int
    MB: int
    NBD: int
    NBE: int
    BROWS: int
    EBROWS: int
    tk: dict[str, int]
    nt: dict[str, int]
    kinds: dict[str, tuple[int, int, Any]]
    static_kinds: tuple[str, ...]
    dynamic_kinds: tuple[str, ...]
    vshape: dict[str, tuple[int, int]]
    vdtype: dict[str, Any]
    period: int
    branch_counts: tuple[int, ...]
    branch_tiles: tuple[int, ...]
    branch_segs: tuple[list[tuple[str, int, int, int]], ...]
    t_cycle: int
    T_TOTAL: int
    module_branch_idx: dict[str, int | None]
    lay: dict[str, tuple[int, int]]
    S: int
    typed: bool
    dmm_unroll: int = 0
    dense_acc_pair: bool = False
    rel_fetch: bool = False

    def layer_start_tile(self, lyr: Any) -> Any:
        """Return global static stream tile index at the start of layer `lyr`."""
        if len(self.branch_counts) == 1:
            return lyr * self.branch_tiles[0]
        c0 = self.branch_counts[0]
        t0, t1 = self.branch_tiles[0], self.branch_tiles[1]
        rem = lyr % self.period
        return (
            (lyr // self.period) * self.t_cycle
            + jnp.minimum(rem, c0) * t0
            + jnp.maximum(0, rem - c0) * t1
        )

    def module_ord(self, lyr: Any, module_name: str) -> Any:
        """Return layer index into `module_name` stacked weights at layer `lyr`."""
        b_idx = self.module_branch_idx.get(module_name)
        if b_idx is None or len(self.branch_counts) == 1:
            return lyr
        c_b = self.branch_counts[b_idx]
        start_b = sum(self.branch_counts[:b_idx])
        return (lyr // self.period) * c_b + jnp.clip(
            (lyr % self.period) - start_b, 0, c_b - 1
        )

    def lt_to_branch(self, lt: int) -> tuple[int, int]:
        """Map cycle-relative layer `lt` to `(branch_idx, lt_in_branch)`."""
        lo = 0
        for b_idx, c_b in enumerate(self.branch_counts):
            if lt < lo + c_b:
                return b_idx, lt - lo
            lo += c_b
        raise AssertionError(f"Layer type {lt} out of range for period {self.period}")

    def rel_decode(self, lt: int, r: int) -> tuple[int, int, int, int, str, int, int]:
        """Trace-time decode of tile `r` relative to cycle-relative layer `lt`."""
        b_off, lt_, base = 0, lt, 0
        while True:
            b_idx, lt_in_b = self.lt_to_branch(lt_)
            t_len = self.branch_tiles[b_idx]
            if r - base < t_len:
                break
            base += t_len
            lt_ += 1
            if lt_ == self.period:
                lt_, b_off = 0, b_off + 1
        r_in_layer = r - base
        for kind, s_lo, s_hi, ordi in self.branch_segs[b_idx]:
            if r_in_layer < s_hi:
                return b_off, lt_, b_idx, lt_in_b, kind, ordi, base + s_lo
        raise AssertionError(f"Tile {r} out of range for layer type {lt}")


@dataclass(frozen=True)
class CorePlan:
    """Complete build-time plan combining `Plan`, `StreamParams`, and `CommParams`."""

    plan: Plan
    stream: StreamParams
    comm: CommParams
    modules: dict[str, Any]
    phase_map: dict[str, str]
    vmem_bytes: int
    vmem_shapes: dict[str, tuple[tuple[int, ...], Any]]
    NACC: int
    NY: int
    KX2: int
    NMIX: int
    smem_bytes: int = 0
    smem_shapes: dict[str, tuple[tuple[int, ...], Any]] | None = None

    def module(self, name: str) -> Any:
        return self.modules[name]


def _normalize_scratch_reqs(
    raw: dict[str, Any] | list[ScratchReq],
) -> list[ScratchReq]:
    if isinstance(raw, dict):
        return [ScratchReq(name=k, spec=v, phase="persistent") for k, v in raw.items()]
    return list(raw)


def _merge_scratch(
    scratch: dict[str, Any],
    phase_map: dict[str, str],
    reqs: list[ScratchReq],
) -> None:
    """Merge module `ScratchReq`s into `scratch`, overlaying compatible phases."""
    for r in reqs:
        ms = getattr(r.spec, "memory_space", None)
        if ms not in (
            pltpu.MemorySpace.VMEM,
            pltpu.MemorySpace.SMEM,
            pltpu.MemorySpace.SEMAPHORE,
        ):
            raise MegakernelConfigError(
                f"Unsupported scratch spec for {r.name!r}: {r.spec!r}"
            )
        if r.name not in scratch:
            scratch[r.name] = r.spec
            phase_map[r.name] = r.phase
            continue

        cur_spec = scratch[r.name]
        cur_phase = phase_map[r.name]
        cur_ms = cur_spec.memory_space
        if (
            r.phase != "persistent"
            and cur_phase != "persistent"
            and r.phase != cur_phase
            and ms in (pltpu.MemorySpace.VMEM, pltpu.MemorySpace.SMEM)
            and ms == cur_ms
            and r.spec.dtype == cur_spec.dtype
            and len(r.spec.shape) == len(cur_spec.shape)
        ):
            max_shape = tuple(max(a, b) for a, b in zip(cur_spec.shape, r.spec.shape))
            scratch[r.name] = cur_ms(max_shape, cur_spec.dtype)
            phase_map[r.name] = f"{cur_phase}|{r.phase}"
        elif r.spec == cur_spec and r.phase == cur_phase:
            continue
        else:
            raise MegakernelConfigError(
                f"Conflicting scratch buffer {r.name!r}: "
                f"phase {r.phase!r} vs {cur_phase!r}"
            )


def fitted_plan(model_spec: ModelSpec, ex: ExecSpec) -> Plan:
    """Fit default tiles to the device VMEM budget (`fit_default_tiles`) and plan."""
    return plan(model_spec, fit_default_tiles(model_spec, ex))


def build_scratch(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> tuple[CorePlan, dict[str, Any]]:
    """Build `CorePlan` and allocate all core and module VMEM scratch buffers.

    `ex.tiles=None` uses the default tiles fitted to the model's VMEM budget
    (`fit_default_tiles`).
    """
    if p is not None and ex.tiles is None and p.ex.tiles is not None:
        ex = p.ex
    else:
        ex = fit_default_tiles(model_spec, ex)
    if p is None or p.ex != ex:
        p = plan(model_spec, ex)
    g = model_spec.config
    tiles = ex.resolve_tiles(g)

    B, MB = ex.B, ex.MB
    H, L = g.H, g.L

    tk, nt, kinds = p.tk, p.nt, p.kinds
    lay, S = p.smalls_layout, p.smalls_width
    NBD, NBE = tiles.nbd, tiles.nbe
    BROWS, EBROWS = tiles.bank // 128, tiles.ebank // 128

    NACC = p.shared_widths["acc"]
    NY = p.shared_widths["ybuf"]
    KX2 = p.shared_widths["xb2"]
    NMIX = p.shared_widths["mix"]
    typed = bool(ex.interpret)

    cp = comm_params(ex, tiles, H)

    vshape = {k: (tk[k], kinds[k][1]) for k in kinds}
    vdtype = {k: kinds[k][2] for k in kinds}
    for k in kinds:
        assert pool_alias.storage_size(vshape[k], jnp.dtype(vdtype[k])) <= (
            tiles.ebank if k in p.dynamic_kinds else tiles.bank
        ), k

    scratch: dict[str, Any] = {
        "dsem": pltpu.SemaphoreType.DMA((NBD,)),
        "sm": pltpu.VMEM((2, 1, S), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        "res": pltpu.VMEM((MB, H), F32),
        "xb": pltpu.VMEM((MB, H), BF16),
        "xb2": pltpu.VMEM((MB, KX2), BF16),
        "acc": pltpu.VMEM((MB, NACC), F32),
        "ybuf": pltpu.VMEM((MB, NY), F32),
        "part": pltpu.VMEM((MB, H), F32),
        "mix": pltpu.VMEM((MB, NMIX), F32),
        **ring_scratch(cp),
        "fn": pltpu.VMEM((1, H), F32),
        "fn_sem": pltpu.SemaphoreType.DMA(()),
    }
    if p.dynamic_kinds:
        scratch["esem"] = pltpu.SemaphoreType.DMA((NBE,))

    scratch.update(stage_scratch(cp))

    phase_map: dict[str, str] = {k: "persistent" for k in scratch}

    module_plans: dict[str, Any] = dict(p.module_params)
    for m in model_spec.all_modules():
        reqs = _normalize_scratch_reqs(m.scratch(g, ex, p))
        _merge_scratch(scratch, phase_map, reqs)

    if typed:
        for k in p.static_kinds:
            scratch["dring_" + k] = pltpu.VMEM((NBD,) + vshape[k], vdtype[k])
            phase_map["dring_" + k] = "persistent"
        for k in p.dynamic_kinds:
            scratch["ering_" + k] = pltpu.VMEM((NBE,) + vshape[k], vdtype[k])
            phase_map["ering_" + k] = "persistent"
        # A module that stages its KV ring in the expert pool (`KVE`) needs the
        # typed pool to hold `(2, HD // 4, 4, BLK)` fp8 KV items.
        kve = [mp for mp in module_plans.values() if getattr(mp, "KVE", False)]
        if p.dynamic_kinds and kve:
            scratch["epool"] = pltpu.VMEM((NBE, 2, g.HD // 4, 4, kve[0].BLK), FP8)
            phase_map["epool"] = "persistent"
    else:
        scratch["dpool"] = pltpu.VMEM((NBD * BROWS, 128), U8)
        phase_map["dpool"] = "persistent"
        if p.dynamic_kinds:
            scratch["epool"] = pltpu.VMEM((NBE * EBROWS, 128), U8)
            phase_map["epool"] = "persistent"

    vmem_bytes = 0
    vmem_shapes: dict[str, tuple[tuple[int, ...], Any]] = {}
    smem_bytes = B * 4  # pos_ref [B] int32
    smem_shapes: dict[str, tuple[tuple[int, ...], Any]] = {"pos": ((B,), jnp.int32)}
    for x in extra_inputs(model_spec, ex):
        if x.memory_space == "smem":
            smem_bytes += math.prod(x.shape) * jnp.dtype(x.dtype).itemsize
            smem_shapes[x.name] = (tuple(x.shape), x.dtype)
    for k, st in p.state_decls.items():
        if st.memory_space == "smem":
            l_cnt = max(1, p.module_layer_counts[p.state_owners[k]])
            sh = (l_cnt, *st.shape) if st.per_layer else st.shape
            smem_bytes += math.prod(sh) * jnp.dtype(st.dtype).itemsize
            smem_shapes[k] = (tuple(sh), st.dtype)
    for k, v in scratch.items():
        ms = getattr(v, "memory_space", None)
        if ms == pltpu.MemorySpace.VMEM:
            sh = tuple(v.shape)
            vmem_bytes += _vmem_buffer_bytes(k, sh, v.dtype)
            vmem_shapes[k] = (sh, v.dtype)
        elif ms == pltpu.MemorySpace.SMEM:
            sh = tuple(v.shape)
            smem_bytes += math.prod(sh) * jnp.dtype(v.dtype).itemsize
            smem_shapes[k] = (sh, v.dtype)
        elif ms == pltpu.MemorySpace.SEMAPHORE:
            continue
        else:
            raise MegakernelConfigError(f"Unsupported scratch spec for {k!r}: {v!r}")

    vmem_limit_bytes = tiles.vmem_mib * 1024 * 1024
    if vmem_bytes > vmem_limit_bytes:
        breakdown = sorted(
            (
                (
                    _vmem_buffer_bytes(k, sh, dt),
                    k,
                    sh,
                    dt,
                )
                for k, (sh, dt) in vmem_shapes.items()
            ),
            reverse=True,
        )
        top_str = ", ".join(
            f"{k}={sz / (1024 * 1024):.2f}MiB {sh} {jnp.dtype(dt).name}"
            for sz, k, sh, dt in breakdown[:10]
        )
        vmem_mib = vmem_bytes / (1024 * 1024)
        raise MegakernelConfigError(
            f"VMEM scratch allocation ({vmem_mib:.2f} MiB = {vmem_bytes:,} B) "
            f"exceeds tiles.vmem_mib budget "
            f"({tiles.vmem_mib} MiB = {vmem_limit_bytes:,} B). "
            f"Top buffers: {top_str}"
        )

    smem_limit_bytes = tiles.smem_mib * 1024 * 1024
    if smem_bytes > smem_limit_bytes:
        smem_kib = smem_bytes / 1024
        raise MegakernelConfigError(
            f"SMEM allocation ({smem_kib:.2f} KiB = {smem_bytes:,} B) "
            f"exceeds tiles.smem_mib budget "
            f"({tiles.smem_mib} MiB = {smem_limit_bytes:,} B)."
        )

    sp = StreamParams(
        L=L,
        MB=MB,
        NBD=NBD,
        NBE=NBE,
        BROWS=BROWS,
        EBROWS=EBROWS,
        tk=tk,
        nt=nt,
        kinds=kinds,
        static_kinds=p.static_kinds,
        dynamic_kinds=p.dynamic_kinds,
        vshape=vshape,
        vdtype=vdtype,
        period=p.period,
        branch_counts=p.branch_counts,
        branch_tiles=p.branch_tiles,
        branch_segs=p.branch_segs,
        t_cycle=p.t_cycle,
        T_TOTAL=p.T_TOTAL,
        module_branch_idx=p.module_branch_idx,
        lay=lay,
        S=S,
        typed=typed,
        dmm_unroll=tiles.dmm_unroll,
        dense_acc_pair=tiles.dense_acc_pair,
        rel_fetch=tiles.rel_fetch,
    )
    core_plan = CorePlan(
        plan=p,
        stream=sp,
        comm=cp,
        modules=module_plans,
        phase_map=phase_map,
        vmem_bytes=vmem_bytes,
        vmem_shapes=vmem_shapes,
        NACC=NACC,
        NY=NY,
        KX2=KX2,
        NMIX=NMIX,
        smem_bytes=smem_bytes,
        smem_shapes=smem_shapes,
    )
    return core_plan, scratch


# Smallest dense / expert ring depth `fit_default_tiles` shrinks to (one slot
# in use, one being refilled).
MIN_RING_DEPTH = 2
# Non-scratch VMEM headroom (MiB) reserved by `fit_default_tiles` below
# `tiles.vmem_mib` (the Mosaic `vmem_limit_bytes`). `core_plan.vmem_bytes`
# counts only Pallas `scratch_shapes`; Mosaic also places Pallas I/O window
# refs and compiler-lowered VMEM temporaries (~0.85-4.96 MiB across B=48..64)
# inside `vmem_limit_bytes`, and XLA:TPU's LLO register allocator places spill
# slots (~3.70 MiB on 60-layer hybrid MoE at B >= 40) in hardware VMEM (63.94 MiB
# on TPU v7x). Reserving 6 MiB below the 62 MiB Mosaic limit caps default-tile
# static scratch at 56 MiB so both Mosaic scoped VMEM (<= 62 MiB) and XLA:TPU
# total VMEM (<= 63.94 MiB) fit on every default bucket.
FIT_VMEM_HEADROOM_MIB = 6
# VMEM budget used while probing a candidate (the real budget is checked after).
_PROBE_VMEM_MIB = 1 << 20


def fit_vmem_budget_bytes(tiles: exec_spec.Tiles) -> int:
    """Maximum static `scratch_shapes` VMEM bytes allowed by `fit_default_tiles`."""
    eff_mib = max(0, tiles.vmem_mib - FIT_VMEM_HEADROOM_MIB)
    return eff_mib * 1024 * 1024


def _ring_bytes(core_plan: CorePlan) -> tuple[int, int]:
    """Allocated `(dense, expert)` weight-ring VMEM bytes of `core_plan`."""
    dense = expert = 0
    for k, (sh, dt) in core_plan.vmem_shapes.items():
        nbytes = _vmem_buffer_bytes(k, sh, dt)
        if k == "dpool" or k.startswith("dring_"):
            dense += nbytes
        elif k == "epool" or k.startswith("ering_"):
            expert += nbytes
    return dense, expert


_LOGGED_FIT_KEYS: set[tuple[str, int, int, int, int, int, int]] = set()
# Bucket row ceiling and token-per-sequence grid across which `fit_default_tiles`
# pins one model-wide `nbe` (the minimum over the model's buckets, since `nbe`
# is `approx` in `TILE_NUMERICS` and gates module pipelining paths).
DEFAULT_FIT_MAX_ROWS = exec_spec.MAX_BATCH_ROWS
DEFAULT_FIT_T: tuple[int, ...] = (1, 2, 4, 8)


def _fit_ring_depths_uncached(
    model_spec: ModelSpec,
    probe_ex: ExecSpec,
    t: exec_spec.Tiles,
    budget: int,
    shrink_nbe: bool,
) -> tuple[exec_spec.Tiles | None, int, exec_spec.Tiles]:
    while True:
        probe = dataclasses.replace(
            probe_ex, tiles=dataclasses.replace(t, vmem_mib=_PROBE_VMEM_MIB)
        )
        core_plan, _ = build_scratch(model_spec, probe)
        if core_plan.vmem_bytes <= budget:
            return t, core_plan.vmem_bytes, t
        dense, expert = _ring_bytes(core_plan)
        can_e = shrink_nbe and expert > 0 and t.nbe > MIN_RING_DEPTH
        can_d = dense > 0 and t.nbd > MIN_RING_DEPTH
        if can_e and (expert >= dense or not can_d):
            t = dataclasses.replace(t, nbe=t.nbe - 1)
        elif can_d:
            t = dataclasses.replace(t, nbd=t.nbd - 1)
        else:
            return None, core_plan.vmem_bytes, t


@functools.lru_cache(maxsize=256)
def _fit_ring_depths_cached(
    model_spec: ModelSpec,
    probe_ex: ExecSpec,
    t: exec_spec.Tiles,
    budget: int,
    shrink_nbe: bool,
) -> tuple[exec_spec.Tiles | None, int, exec_spec.Tiles]:
    return _fit_ring_depths_uncached(model_spec, probe_ex, t, budget, shrink_nbe)


def _fit_ring_depths(
    model_spec: ModelSpec,
    probe_ex: ExecSpec,
    t: exec_spec.Tiles,
    budget: int,
    *,
    shrink_nbe: bool = True,
) -> tuple[exec_spec.Tiles | None, int, exec_spec.Tiles]:
    try:
        hash((model_spec, probe_ex, t, budget, shrink_nbe))
    except TypeError:
        return _fit_ring_depths_uncached(model_spec, probe_ex, t, budget, shrink_nbe)
    return _fit_ring_depths_cached(model_spec, probe_ex, t, budget, shrink_nbe)


def fit_default_tiles(
    model_spec: ModelSpec,
    ex: ExecSpec,
    *,
    max_rows: int = DEFAULT_FIT_MAX_ROWS,
) -> ExecSpec:
    """Return `ex` with explicit tiles: the default rule fitted to its VMEM budget.

    No-op when `ex.tiles` is set (explicit and profile tiles are never changed;
    an over-budget plan raises in `build_scratch`). Otherwise starts from
    `default_tiles(B, R)` sized for the device's chip (`chip_budget()`), pins
    one model-wide `nbe` (the minimum fitted `nbe` at `max_rows`
    (`MAX_BATCH_ROWS`) across every `T` in `DEFAULT_FIT_T` the model serves, so
    `nbe` never varies across buckets of a model and cannot flip module
    pipelining paths), and then shrinks only `nbd` (down to `MIN_RING_DEPTH`,
    with `shrink_nbe=False`) until the bucket's planned VMEM scratch fits
    `fit_vmem_budget_bytes(t)` (`vmem_mib - FIT_VMEM_HEADROOM_MIB`). Raises
    `MegakernelConfigError` if the plan does not fit with that `nbe`.
    """
    if ex.tiles is not None:
        return ex
    t0 = exec_spec.default_tiles(ex.B, R=ex.R, chip=chip_budget())
    budget = fit_vmem_budget_bytes(t0)
    model_nbe = t0.nbe
    for tw in DEFAULT_FIT_T:
        bw = round_up(max_rows, tw)
        w_ex = dataclasses.replace(ex, B=bw, T=tw)
        try:
            t_w, _, t_w_last = _fit_ring_depths(
                model_spec, w_ex, t0, budget, shrink_nbe=True
            )
        except MegakernelConfigError:
            continue
        model_nbe = min(model_nbe, (t_w or t_w_last).nbe)
    t_fit, vmem_bytes, t_last = _fit_ring_depths(
        model_spec,
        ex,
        dataclasses.replace(t0, nbe=model_nbe),
        budget,
        shrink_nbe=False,
    )
    if t_fit is None:
        need = vmem_bytes / (1024 * 1024)
        budget_mib = budget / (1024 * 1024)
        raise MegakernelConfigError(
            f"default tiles do not fit the {budget_mib:.2f} MiB VMEM scratch budget "
            f"({t_last.vmem_mib} MiB limit - {FIT_VMEM_HEADROOM_MIB} MiB headroom) for "
            f"ModelSpec {model_spec.name!r} (B={ex.B}, T={ex.T}, R={ex.R}): "
            f"{need:.2f} MiB with ring depths nbd={t_last.nbd}, nbe={t_last.nbe}; "
            "pass explicit tiles or a named profile"
        )
    if (t_fit.nbd, t_fit.nbe) != (t0.nbd, t0.nbe):
        log_key = (
            model_spec.name,
            ex.B,
            ex.R,
            t0.nbd,
            t0.nbe,
            t_fit.nbd,
            t_fit.nbe,
        )
        if log_key not in _LOGGED_FIT_KEYS:
            _LOGGED_FIT_KEYS.add(log_key)
            logger.info(
                "fit_default_tiles shrunk ring depths for ModelSpec %r "
                "(B=%d, R=%d): nbd %d->%d, nbe %d->%d "
                "(%.2f MiB / %.2f MiB budget)",
                model_spec.name,
                ex.B,
                ex.R,
                t0.nbd,
                t_fit.nbd,
                t0.nbe,
                t_fit.nbe,
                vmem_bytes / (1024 * 1024),
                budget / (1024 * 1024),
            )
    return dataclasses.replace(ex, tiles=t_fit)

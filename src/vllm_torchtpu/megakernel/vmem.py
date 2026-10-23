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
"""Core VMEM/SMEM/Semaphore allocator and `StreamParams` / `CommParams` builders."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from . import exec_spec, pool_alias
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .ops import BF16, F32, FP8, rup
from .planner import Plan, extra_inputs, plan
from .profiles.default import chip_budget
from .spec import ModelSpec, ScratchReq

U8 = jnp.uint8


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
    block_branch_idx: dict[str, int | None]
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

    def block_ord(self, lyr: Any, block_name: str) -> Any:
        """Return layer index into `block_name` stacked weights at layer `lyr`."""
        b_idx = self.block_branch_idx.get(block_name)
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
class CommParams:
    """Core all-reduce and inter-rank collective parameters."""

    R: int
    B: int
    M: int
    MR: int
    MB: int
    H: int
    PR: int
    PC: int
    HPC: int
    HIER: int
    RSAG: bool
    WAR: bool
    AR_R: int
    AR_C: int
    CW: int
    HC2: int
    AR_ROUNDS: tuple[int, ...]
    AR_PEERS: int
    N_HIER3: int = 3


@dataclass(frozen=True)
class CorePlan:
    """Complete build-time plan combining `Plan`, `StreamParams`, and `CommParams`."""

    plan: Plan
    stream: StreamParams
    comm: CommParams
    blocks: dict[str, Any]
    phase_map: dict[str, str]
    vmem_bytes: int
    vmem_shapes: dict[str, tuple[tuple[int, ...], Any]]
    NACC: int
    NY: int
    KX2: int
    NMIX: int
    smem_bytes: int = 0
    smem_shapes: dict[str, tuple[tuple[int, ...], Any]] | None = None

    def block(self, name: str) -> Any:
        return self.blocks[name]


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
    """Merge block `ScratchReq`s into `scratch`, overlaying compatible phases."""
    for r in reqs:
        if r.name not in scratch:
            scratch[r.name] = r.spec
            phase_map[r.name] = r.phase
            continue

        cur_spec = scratch[r.name]
        cur_phase = phase_map[r.name]
        if (
            r.phase != "persistent"
            and cur_phase != "persistent"
            and r.phase != cur_phase
            and hasattr(r.spec, "shape")
            and hasattr(cur_spec, "shape")
            and getattr(r.spec, "dtype", None) == getattr(cur_spec, "dtype", None)
            and getattr(r.spec, "memory_space", None)
            == getattr(cur_spec, "memory_space", None)
            and len(r.spec.shape) == len(cur_spec.shape)
        ):
            max_shape = tuple(max(a, b) for a, b in zip(cur_spec.shape, r.spec.shape))
            ms = getattr(cur_spec, "memory_space", pltpu.VMEM)
            scratch[r.name] = ms(max_shape, cur_spec.dtype)
            phase_map[r.name] = f"{cur_phase}|{r.phase}"
        elif r.spec == cur_spec and r.phase == cur_phase:
            continue
        else:
            raise MegakernelConfigError(
                f"Conflicting scratch buffer {r.name!r}: "
                f"phase {r.phase!r} vs {cur_phase!r}"
            )


def build_scratch(
    model_spec: ModelSpec,
    ex: ExecSpec,
    p: Plan | None = None,
) -> tuple[CorePlan, dict[str, Any]]:
    """Build `CorePlan` and allocate all core and block VMEM scratch buffers.

    `ex.tiles=None` uses the default tiles fitted to the model's VMEM budget,
    as `kernel.build` does (`fit_default_tiles`).
    """
    ex = fit_default_tiles(model_spec, ex)
    if p is None:
        p = plan(model_spec, ex)
    g = model_spec.config
    tiles = ex.resolve_tiles(g)

    R, B, M, MR, MB = ex.R, ex.B, ex.M, ex.MR, ex.MB
    H, L = g.H, g.L
    PC = min(1024, H)
    assert H % PC == 0, (H, PC)
    HPC = H // PC
    PR = rup(B * HPC, 16)

    tk, nt, kinds = p.tk, p.nt, p.kinds
    lay, S = p.smalls_layout, p.smalls_width
    NBD, NBE = tiles.nbd, tiles.nbe
    BROWS, EBROWS = tiles.bank // 128, tiles.ebank // 128

    NACC = p.shared_widths["acc"]
    NY = p.shared_widths["ybuf"]
    KX2 = p.shared_widths["xb2"]
    NMIX = p.shared_widths["mix"]
    typed = bool(ex.interpret)

    HIER = tiles.hier if R > 1 else 0
    RSAG = tiles.rsag and R > 1 and HIER == 0
    WAR = (
        (ex.WIDE or MR * H <= PR * PC or (RSAG and (PC // R) % 128 != 0))
        if tiles.war < 0
        else bool(tiles.war)
    )
    AR_R = MR if WAR else PR
    AR_C = H if WAR else PC
    # No slot reduces through the core all-reduce (every block reduces itself,
    # `slot_ar_index` all None): its buffers are never touched, so size them at
    # one tile instead of [MR, H] (Qwen3.8's HC wrappers own h-wide rings).
    core_ar = any(i is not None for i in p.slot_ar_index)
    if not core_ar:
        AR_R, AR_C = min(AR_R, 8), 128 * max(R, 2)
    CW = AR_C // R
    HC2 = AR_C // 2
    N_HIER3 = max(1, R.bit_length() - 1)
    assert not RSAG or CW % 128 == 0
    assert HIER in (0, 1, 2, 3) and (
        HIER == 0 or R == 8 or (HIER == 3 and R >= 2 and (R & (R - 1)) == 0)
    )
    assert HIER != 2 or HC2 % 128 == 0
    AR_ROUNDS = (
        tuple(range(N_HIER3))
        if HIER == 3
        else ((0, 1, 2) if HIER == 2 else ((0, 1) if (RSAG or HIER == 1) else (0,)))
    )
    AR_PEERS = 1 if HIER == 3 else (3 if HIER in (1, 2) else max(R - 1, 1))

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
        # Blocks that remap `part` to their own buffer (`owns_part`) never read
        # the core one; when every slot does, keep it at one tile.
        "part": pltpu.VMEM((8, 128) if _all_own_part(model_spec) else (MB, H), F32),
        "mix": pltpu.VMEM((MB, NMIX), F32),
        "send": pltpu.VMEM(
            (N_HIER3, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, R, AR_R, CW)
            if RSAG
            else (2, AR_R, AR_C),
            BF16,
        ),
        "recv": pltpu.VMEM(
            (N_HIER3, 2, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, 2, AR_R, AR_C)
            if HIER == 1
            else (2, R, AR_R, CW)
            if RSAG
            else (2, R, AR_R, AR_C),
            BF16,
        ),
        "ar_ssem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        "ar_rsem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        "fn": pltpu.VMEM((1, H), F32),
        "fn_sem": pltpu.SemaphoreType.DMA(()),
    }
    if p.dynamic_kinds:
        scratch["esem"] = pltpu.SemaphoreType.DMA((NBE,))

    if RSAG:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, CW), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, R, AR_R, CW), BF16)
    elif HIER == 1:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, AR_C), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, AR_C), BF16)
    elif HIER == 2:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, HC2), BF16)
        scratch["s2_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["s2_recv"] = pltpu.VMEM((2, 2, AR_R, HC2), BF16)

    phase_map: dict[str, str] = {k: "persistent" for k in scratch}

    block_plans: dict[str, Any] = dict(p.block_params)
    for b in model_spec.all_blocks():
        reqs = _normalize_scratch_reqs(b.scratch(g, ex, p))
        _merge_scratch(scratch, phase_map, reqs)

    if typed:
        for k in p.static_kinds:
            scratch["dring_" + k] = pltpu.VMEM((NBD,) + vshape[k], vdtype[k])
            phase_map["dring_" + k] = "persistent"
        for k in p.dynamic_kinds:
            scratch["ering_" + k] = pltpu.VMEM((NBE,) + vshape[k], vdtype[k])
            phase_map["ering_" + k] = "persistent"
        # A block that stages its KV ring in the expert pool (`KVE`) needs the
        # typed pool to hold `(2, HD // 4, 4, BLK)` fp8 KV items.
        kve = [bp for bp in block_plans.values() if getattr(bp, "KVE", False)]
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
            l_cnt = max(1, p.block_layer_counts[p.state_owners[k]])
            sh = (l_cnt, *st.shape) if st.per_layer else st.shape
            smem_bytes += math.prod(sh) * jnp.dtype(st.dtype).itemsize
            smem_shapes[k] = (tuple(sh), st.dtype)
    for k, v in scratch.items():
        if hasattr(v, "shape"):
            ms_str = str(getattr(v, "memory_space", "")).upper()
            if "VMEM" in ms_str:
                vmem_bytes += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
                vmem_shapes[k] = (tuple(v.shape), v.dtype)
            elif "SMEM" in ms_str:
                smem_bytes += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
                smem_shapes[k] = (tuple(v.shape), v.dtype)

    vmem_limit_bytes = tiles.vmem_mib * 1024 * 1024
    if vmem_bytes > vmem_limit_bytes:
        breakdown = sorted(
            (
                (math.prod(sh) * jnp.dtype(dt).itemsize, k, sh, dt)
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
        block_branch_idx=p.block_branch_idx,
        lay=lay,
        S=S,
        typed=typed,
        dmm_unroll=tiles.dmm_unroll,
        dense_acc_pair=tiles.dense_acc_pair,
        rel_fetch=tiles.rel_fetch,
    )
    cp = CommParams(
        R=R,
        B=B,
        M=M,
        MR=MR,
        MB=MB,
        H=H,
        PR=PR,
        PC=PC,
        HPC=HPC,
        HIER=HIER,
        RSAG=RSAG,
        WAR=WAR,
        AR_R=AR_R,
        AR_C=AR_C,
        CW=CW,
        HC2=HC2,
        AR_ROUNDS=AR_ROUNDS,
        AR_PEERS=AR_PEERS,
        N_HIER3=N_HIER3,
    )
    core_plan = CorePlan(
        plan=p,
        stream=sp,
        comm=cp,
        blocks=block_plans,
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
# VMEM budget used while probing a candidate (the real budget is checked after).
_PROBE_VMEM_MIB = 1 << 20


def _ring_bytes(core_plan: CorePlan) -> tuple[int, int]:
    """Allocated `(dense, expert)` weight-ring VMEM bytes of `core_plan`."""
    dense = expert = 0
    for k, (sh, dt) in core_plan.vmem_shapes.items():
        nbytes = math.prod(sh) * jnp.dtype(dt).itemsize
        if k == "dpool" or k.startswith("dring_"):
            dense += nbytes
        elif k == "epool" or k.startswith("ering_"):
            expert += nbytes
    return dense, expert


def fit_default_tiles(model_spec: ModelSpec, ex: ExecSpec) -> ExecSpec:
    """Return `ex` with explicit tiles: the default rule fitted to its VMEM budget.

    No-op when `ex.tiles` is set (explicit and profile tiles are never changed;
    an over-budget plan raises in `build_scratch`). Otherwise starts from
    `default_tiles(B, R)` sized for the device's chip (`chip_budget()`) and,
    while the planned VMEM scratch exceeds
    `vmem_mib`, drops one slot from the larger allocated weight ring (dense
    `nbd` or expert `nbe`), down to `MIN_RING_DEPTH`. Raises
    `MegakernelConfigError` if the plan does not fit even then.
    """
    if ex.tiles is not None:
        return ex
    t = exec_spec.default_tiles(ex.B, R=ex.R, chip=chip_budget())
    budget = t.vmem_mib * 1024 * 1024
    while True:
        probe = dataclasses.replace(
            ex, tiles=dataclasses.replace(t, vmem_mib=_PROBE_VMEM_MIB)
        )
        core_plan, _ = build_scratch(model_spec, probe)
        if core_plan.vmem_bytes <= budget:
            return dataclasses.replace(ex, tiles=t)
        dense, expert = _ring_bytes(core_plan)
        can_e = expert > 0 and t.nbe > MIN_RING_DEPTH
        can_d = dense > 0 and t.nbd > MIN_RING_DEPTH
        if can_e and (expert >= dense or not can_d):
            t = dataclasses.replace(t, nbe=t.nbe - 1)
        elif can_d:
            t = dataclasses.replace(t, nbd=t.nbd - 1)
        else:
            need = core_plan.vmem_bytes / (1024 * 1024)
            raise MegakernelConfigError(
                f"default tiles do not fit the {t.vmem_mib} MiB VMEM budget for "
                f"ModelSpec {model_spec.name!r} (B={ex.B}, T={ex.T}, R={ex.R}): "
                f"{need:.2f} MiB with ring depths nbd={t.nbd}, nbe={t.nbe}; "
                "pass explicit tiles or a named profile"
            )


def _all_own_part(model_spec: Any) -> bool:
    """Every slot block (each SwitchSlot branch too) sets `owns_part = True`."""
    blocks = []
    for slot in model_spec.layer_type.slots:
        blocks.extend(getattr(slot, "branches", None) or (slot,))
    return bool(blocks) and all(getattr(b, "owns_part", False) for b in blocks)

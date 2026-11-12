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
"""Scratchpad layout, stage-slice packing, and Pallas BlockSpecs for Qwen3.8-2.4T."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, cast

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.megakernel.core.dma import pool_alias
from vllm_torchtpu.megakernel.core.profiles.default import chip_budget

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax._src.pallas.mosaic.core import MemoryRef
    from jax._src.state.types import TransformedRef

    from .shapes import Qwen38Config

from .constants import (
    BF16_BYTES,
    F32_BYTES,
    H_TILED_4D_KEYS,
    H_TILED_4D_SCALE_KEYS,
    H_TILED_5D_KEYS,
    H_TILED_5D_SCALE_KEYS,
    H_TILED_ALL_KEYS,
    I32_BYTES,
    IN_IDX_POOL,
    MIB_BYTES,
    MIN_MATRIX_NDIM,
    MOE_WEIGHT_NDIM,
    OUT_IDX_POOL,
    PAD2_SIZE,
    RESIDENT_WEIGHT_KEYS,
    STAGE_ROW_TILE,
    TPU_V7X_DEVICE_KIND,
    VMEM_ALIGN_BYTES,
)


class Qwen38PlanError(TypeError):
    """Raised when a Pallas scratch specification has an invalid non-semaphore dtype."""


def vmem_limit_bytes(device_kind: str = TPU_V7X_DEVICE_KIND) -> int:
    """Return usable VMEM budget in bytes from the static megakernel chip profile.

    Queries `chip_budget(device_kind)` directly without touching `jax.devices()`
    or `pltpu.get_tpu_info()` so offline planning never claims a TPU (`R-4.4`).
    """
    return chip_budget(device_kind).vmem_mib * MIB_BYTES


def smem_limit_bytes(device_kind: str = TPU_V7X_DEVICE_KIND) -> int:
    """Return usable SMEM budget in bytes from the static megakernel chip profile."""
    return chip_budget(device_kind).smem_mib * MIB_BYTES


@dataclasses.dataclass(frozen=True, slots=True)
class StageSlice:
    """Byte-aligned scratchpad slice descriptor for a stage-local weight buffer."""

    name: str
    shape: tuple[int, ...]
    tiled_shape: tuple[int, ...]
    dtype: jnp.dtype
    offset_bytes: int
    size_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class StageScratchPlan:
    """Shared stage-disjoint scratchpad plan backed by `pool_alias.view`."""

    peak_bytes: int
    slices: Mapping[str, StageSlice]

    def scratch_spec(
        self, *, interpret: bool = False
    ) -> MemoryRef | dict[str, MemoryRef]:
        """Return Pallas VMEM scratch spec (per-slice dict in interpret mode)."""
        if interpret:
            return {
                s.name: pltpu.VMEM(s.tiled_shape, s.dtype) for s in self.slices.values()
            }
        return pltpu.VMEM(
            (self.peak_bytes // VMEM_ALIGN_BYTES, VMEM_ALIGN_BYTES),
            jnp.uint8,
        )

    def view(
        self,
        stage_ref: dict[str, jax.Array] | jax.Array | TransformedRef,
        slc: StageSlice | str,
    ) -> jax.Array:
        """Return a typed 2D view into the shared stage scratchpad buffer."""
        target = self.slices[slc] if isinstance(slc, str) else slc
        if isinstance(stage_ref, dict):
            v = stage_ref[target.name]
        else:
            rows = target.size_bytes // VMEM_ALIGN_BYTES
            v = pool_alias.view(
                stage_ref.at[pl.ds(0, rows)], target.tiled_shape, target.dtype
            )
        if target.shape != target.tiled_shape:
            return cast(
                "jax.Array",
                v.at[..., : target.shape[-2], : target.shape[-1]],
            )
        return cast("jax.Array", v)


def build_stage_scratch_plan(
    cfg: Qwen38Config,
) -> tuple[StageScratchPlan, dict[str, StageSlice]]:
    """Pack stage-disjoint down-projection weight buffers onto shared scratchpad.

    All three stage-disjoint packed FP4 down-projection buffers share the
    trailing row width `(cfg.H // 2,)` in `uint8`:
    - Attention stage: `wo_buf` `(cfg.Q_R * cfg.D, cfg.H // 2)`
    - GDN stage: `wout_buf` `(cfg.Z_R, cfg.H // 2)`
    - MoE stage: `wse2_buf` `(cfg.SE_R, cfg.H // 2)`
    """
    shapes = {
        "wo_buf": (cfg.Q_R * cfg.D, cfg.H // 2),
        "wout_buf": (cfg.Z_R, cfg.H // 2),
        "wse2_buf": (cfg.SE_R, cfg.H // 2),
    }
    dt = jnp.dtype(jnp.uint8)
    slices: dict[str, StageSlice] = {}
    for name, (rows, cols) in shapes.items():
        r_pad = ((rows + STAGE_ROW_TILE - 1) // STAGE_ROW_TILE) * STAGE_ROW_TILE
        c_pad = ((cols + VMEM_ALIGN_BYTES - 1) // VMEM_ALIGN_BYTES) * VMEM_ALIGN_BYTES
        slices[name] = StageSlice(
            name=name,
            shape=(rows, cols),
            tiled_shape=(r_pad, c_pad),
            dtype=dt,
            offset_bytes=0,
            size_bytes=r_pad * c_pad * int(dt.itemsize),
        )
    peak_bytes = max(s.size_bytes for s in slices.values())
    return StageScratchPlan(peak_bytes=peak_bytes, slices=slices), slices


def _collective_scratch_specs(cfg: Qwen38Config) -> dict[str, MemoryRef]:
    """Build hierarchical all-reduce and persistent activation VMEM specs."""
    hier_g = min(cfg.hier_group, cfg.R)
    num_groups = max(1, cfg.R // hier_g)
    return {
        "ar_send": pltpu.VMEM((max(1, hier_g - 1), cfg.B, cfg.H), jnp.bfloat16),
        "ar_recv": pltpu.VMEM((2, max(1, hier_g - 1), cfg.B, cfg.H), jnp.bfloat16),
        "ar_stage_send": pltpu.VMEM(
            (max(1, num_groups - 1), cfg.B, cfg.H), jnp.float32
        ),
        "ar_stage_recv": pltpu.VMEM(
            (2, max(1, num_groups - 1), cfg.B, cfg.H), jnp.float32
        ),
        "x_cur": pltpu.VMEM((cfg.B, cfg.H), jnp.bfloat16),
        "x_norm": pltpu.VMEM((cfg.B, cfg.H), jnp.bfloat16),
    }


def _attention_scratch_specs(cfg: Qwen38Config) -> dict[str, MemoryRef]:
    """Build full-attention weight, KV page, and DCP8 exchange VMEM specs."""
    g = cfg.fp4_group
    dcp = cfg.DCP
    q_r = cfg.Q_R
    d = cfg.D
    return {
        "wqkv_buf": pltpu.VMEM((2, cfg.h_tile, cfg.ATT_QKV_R // 2), jnp.uint8),
        "wqkv_s_buf": pltpu.VMEM((2, cfg.h_tile // g, cfg.ATT_QKV_R), jnp.bfloat16),
        "wo_s_buf": pltpu.VMEM(((q_r * d) // g, cfg.H), jnp.bfloat16),
        "kv_page_buf": pltpu.VMEM((2, 2, cfg.K_R, d, cfg.page_size), jnp.float8_e4m3fn),
        "kv_write_buf": pltpu.VMEM((2, cfg.K_R, d, cfg.page_size), jnp.float8_e4m3fn),
        "q_local_send": pltpu.VMEM((max(1, dcp - 1), cfg.B, q_r, d), jnp.bfloat16),
        "q_dcp_recv": pltpu.VMEM((dcp, cfg.B, q_r, d), jnp.bfloat16),
        "lse_part_send": pltpu.VMEM(
            (max(1, dcp - 1), cfg.B, dcp * q_r, d + 2), jnp.float32
        ),
        "lse_part_recv": pltpu.VMEM((dcp, cfg.B, dcp * q_r, d + 2), jnp.float32),
    }


def _gdn_scratch_specs(cfg: Qwen38Config) -> dict[str, MemoryRef]:
    """Build GDN projection and recurrent/conv state ping-pong VMEM specs."""
    g = cfg.fp4_group
    qkv = cfg.QKV
    z_r = cfg.Z_R
    return {
        "win_buf": pltpu.VMEM((2, cfg.h_tile, (qkv + z_r) // 2), jnp.uint8),
        "win_s_buf": pltpu.VMEM((2, cfg.h_tile // g, qkv + z_r), jnp.bfloat16),
        "wba_buf": pltpu.VMEM((cfg.H, 2 * cfg.VH_R), jnp.bfloat16),
        "wconv_buf": pltpu.VMEM((cfg.conv_kernel, qkv), jnp.bfloat16),
        "wout_s_buf": pltpu.VMEM((z_r // g, cfg.H), jnp.bfloat16),
        "rec_buf": pltpu.VMEM((2, cfg.B, cfg.VH_R, cfg.KD, cfg.VD), jnp.float32),
        "conv_buf": pltpu.VMEM((2, cfg.B, cfg.conv_kernel - 1, qkv), jnp.bfloat16),
    }


def _moe_scratch_specs(cfg: Qwen38Config) -> dict[str, MemoryRef]:
    """Build MoE router, shared-expert, routed-expert, and FP4 output VMEM specs."""
    g = cfg.fp4_group
    i_r = cfg.I_R
    se_r = cfg.SE_R
    max_out_cols = max(cfg.H, cfg.ATT_QKV_R, cfg.QKV + cfg.Z_R, 2 * se_r, 2 * i_r)
    return {
        "wgate_buf": pltpu.VMEM((cfg.H, cfg.E), jnp.bfloat16),
        "wsegate_buf": pltpu.VMEM((cfg.H, 1), jnp.bfloat16),
        "wse13_buf": pltpu.VMEM((2, cfg.h_tile, se_r), jnp.uint8),
        "wse13_s_buf": pltpu.VMEM((2, cfg.h_tile // g, 2 * se_r), jnp.bfloat16),
        "wse2_s_buf": pltpu.VMEM((se_r // g, cfg.H), jnp.bfloat16),
        "we13_buf": pltpu.VMEM((2, cfg.h_tile, i_r), jnp.uint8),
        "we13_s_buf": pltpu.VMEM((2, cfg.h_tile // g, 2 * i_r), jnp.bfloat16),
        "we2_buf": pltpu.VMEM((i_r, cfg.H // 2), jnp.uint8),
        "we2_s_buf": pltpu.VMEM((i_r // g, cfg.H), jnp.bfloat16),
        "fp4_out_buf": pltpu.VMEM((cfg.B, max_out_cols), jnp.float32),
    }


def make_vmem_scratch_specs(
    cfg: Qwen38Config, *, interpret: bool = False
) -> dict[str, MemoryRef | dict[str, MemoryRef]]:
    """Build the VMEM scratchpad specifications for `pl.pallas_call`."""
    stage_plan, _ = build_stage_scratch_plan(cfg)
    return {
        "stage_scratch_u8": stage_plan.scratch_spec(interpret=bool(interpret)),
        **_collective_scratch_specs(cfg),
        **_attention_scratch_specs(cfg),
        **_gdn_scratch_specs(cfg),
        **_moe_scratch_specs(cfg),
    }


def make_semaphore_specs(cfg: Qwen38Config) -> dict[str, MemoryRef]:
    """Build the DMA and regular semaphore specifications for `pl.pallas_call`."""
    dcp = cfg.DCP
    hier_g = min(cfg.hier_group, cfg.R)
    num_groups = max(1, cfg.R // hier_g)
    return {
        "ar_ssem1": pltpu.SemaphoreType.DMA((2, max(1, hier_g - 1))),
        "ar_rsem1": pltpu.SemaphoreType.DMA((2, max(1, hier_g - 1))),
        "ar_ssem2": pltpu.SemaphoreType.DMA((2, max(1, num_groups - 1))),
        "ar_rsem2": pltpu.SemaphoreType.DMA((2, max(1, num_groups - 1))),
        "b_sem": pltpu.SemaphoreType.REGULAR((4,)),
        "w_sem": pltpu.SemaphoreType.DMA((4,)),
        "state_sem": pltpu.SemaphoreType.DMA((4,)),
        "kv_sem": pltpu.SemaphoreType.DMA((4,)),
        "dcp_q_ssem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
        "dcp_q_rsem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
        "dcp_lse_ssem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
        "dcp_lse_rsem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
    }


def make_scratch_specs(
    cfg: Qwen38Config, *, interpret: bool = False
) -> dict[str, MemoryRef | dict[str, MemoryRef]]:
    """Build the scratchpad dictionary for `pl.pallas_call`."""
    return {
        **make_vmem_scratch_specs(cfg, interpret=interpret),
        **make_semaphore_specs(cfg),
    }


def _padded_vmem_bytes(shape: tuple[int, ...], itemsize: int) -> int:
    """Compute aligned scratchpad bytes with trailing (8/32, 128) tile padding."""
    if len(shape) >= MIN_MATRIX_NDIM:
        r_tile = STAGE_ROW_TILE if itemsize == 1 else 8
        r_pad = ((int(shape[-2]) + r_tile - 1) // r_tile) * r_tile
        c_pad = (
            (int(shape[-1]) + VMEM_ALIGN_BYTES - 1) // VMEM_ALIGN_BYTES
        ) * VMEM_ALIGN_BYTES
        elems = int(math.prod(shape[:-2])) * r_pad * c_pad
    elif len(shape) == 1:
        elems = (
            (int(shape[0]) + VMEM_ALIGN_BYTES - 1) // VMEM_ALIGN_BYTES
        ) * VMEM_ALIGN_BYTES
    else:
        elems = 1
    return elems * int(itemsize)


def resident_vmem_bytes(cfg: Qwen38Config) -> int:
    """Compute aligned VMEM bytes of resident inputs, outputs, and norm weights."""
    resident_blocks: tuple[tuple[tuple[int, ...], int], ...] = (
        ((cfg.B, cfg.H), BF16_BYTES),
        ((cfg.B, cfg.H), BF16_BYTES),
        ((cfg.L, cfg.H), BF16_BYTES),
        ((cfg.L, cfg.H), BF16_BYTES),
        ((cfg.H,), BF16_BYTES),
        ((cfg.NATT, cfg.D), BF16_BYTES),
        ((cfg.NATT, cfg.D), BF16_BYTES),
        ((cfg.NGDN, cfg.VH_R), F32_BYTES),
        ((cfg.NGDN, cfg.VH_R), F32_BYTES),
        ((cfg.NGDN, cfg.VD), BF16_BYTES),
    )
    return sum(_padded_vmem_bytes(sh, itemsize) for sh, itemsize in resident_blocks)


def scratch_footprint(cfg: Qwen38Config) -> dict[str, int | float | bool]:
    """Compute aligned byte footprint of scratch/resident/scalar buffers and limits."""
    specs = make_vmem_scratch_specs(cfg)
    breakdown: dict[str, int] = {}
    total = 0
    raw_total = 0
    for name, spec in specs.items():
        if isinstance(spec, dict) or spec.memory_space == pltpu.SEMAPHORE:
            continue
        try:
            dt = jnp.dtype(spec.dtype)
        except TypeError as exc:
            msg = f"Invalid scratch spec dtype for {name!r}: {spec.dtype!r}"
            raise Qwen38PlanError(msg) from exc
        sh = tuple(int(d) for d in spec.shape)
        raw_nb = int(math.prod(sh)) * int(dt.itemsize)
        nb = _padded_vmem_bytes(sh, int(dt.itemsize))
        breakdown[name] = nb
        total += nb
        raw_total += raw_nb
    res_bytes = resident_vmem_bytes(cfg)
    total += res_bytes
    stage_plan, slices = build_stage_scratch_plan(cfg)
    unpacked_stage_bytes = sum(s.size_bytes for s in slices.values())
    smem_bytes = total_scalar_bytes(cfg)
    v_limit = vmem_limit_bytes()
    s_limit = smem_limit_bytes()
    return {
        **breakdown,
        "resident_vmem_bytes": res_bytes,
        "stage_scratch_high_water_bytes": stage_plan.peak_bytes,
        "stage_scratch_saved_bytes": unpacked_stage_bytes - stage_plan.peak_bytes,
        "raw_total_bytes": raw_total,
        "total_bytes": total,
        "total_mib": round(total / MIB_BYTES, 3),
        "limit_bytes": v_limit,
        "scalar_bytes": smem_bytes,
        "scalar_limit_bytes": s_limit,
        "fits_in_scalar_limit": smem_bytes < s_limit,
        "fits_in_limit": (total < v_limit) and (smem_bytes < s_limit),
    }


def total_scalar_bytes(cfg: Qwen38Config) -> int:
    """Compute total scalar-memory input bytes (`pos` + `page_table`)."""
    pos_bytes = cfg.B * I32_BYTES
    pt_bytes = cfg.NATT * cfg.B * cfg.max_pages_per_seq * I32_BYTES
    return pos_bytes + pt_bytes


def normalize_weights_for_pallas(
    weights: dict[str, jax.Array], cfg: Qwen38Config
) -> dict[str, jax.Array]:
    """Reshape H-tiled weights into explicit `(n_htiles, h_tile, ...)` axes."""
    n_ht = cfg.H // cfg.h_tile
    g_ht = cfg.h_tile // cfg.fp4_group
    out: dict[str, jax.Array] = {}
    for k, arr in weights.items():
        v = arr
        if k in H_TILED_4D_KEYS:
            v = v.reshape(v.shape[0], v.shape[1], n_ht, cfg.h_tile, v.shape[3])
        elif k in H_TILED_4D_SCALE_KEYS:
            v = v.reshape(v.shape[0], v.shape[1], n_ht, g_ht, v.shape[3])
        elif k in H_TILED_5D_KEYS:
            v = v.reshape(
                v.shape[0], v.shape[1], v.shape[2], n_ht, cfg.h_tile, v.shape[4]
            )
        elif k in H_TILED_5D_SCALE_KEYS:
            v = v.reshape(v.shape[0], v.shape[1], v.shape[2], n_ht, g_ht, v.shape[4])

        if k in H_TILED_ALL_KEYS:
            ax = 3 if v.ndim == MOE_WEIGHT_NDIM else 2
            if v.shape[ax] == 1:
                v = jnp.repeat(v, PAD2_SIZE, axis=ax)
        out[k] = v
    return out


def make_io_aliases(n_att: int = 1) -> dict[int, int]:
    """Return flat leaf `input_output_aliases` for `n_att` region pools and GDN."""
    aliases = {IN_IDX_POOL + idx: OUT_IDX_POOL + idx for idx in range(n_att)}
    aliases[IN_IDX_POOL + n_att] = OUT_IDX_POOL + n_att
    aliases[IN_IDX_POOL + n_att + 1] = OUT_IDX_POOL + n_att + 1
    return aliases


def make_io_specs(
    weights: dict[str, jax.Array],
    *,
    n_att: int = 1,
    has_forced_routing: bool = False,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Construct `in_specs` and `out_specs` for the single-step `pl.pallas_call`.

    Small routing metadata (`pos`, `page_table`, and optional `forced_top_ids`)
    reside in scalar memory; small per-layer norm/decay vectors
    (`RESIDENT_WEIGHT_KEYS`) reside in vector scratchpad; the `n_att` runner-owned
    region pools, GDN state pools (`conv`, `rec`), and all projection/MoE weights
    stay in `pl.ANY` (HBM) and are streamed via explicit DMA.
    """
    w_specs = {
        k: pl.BlockSpec(
            memory_space=pltpu.VMEM if k in RESIDENT_WEIGHT_KEYS else pl.ANY
        )
        for k in weights
    }
    pool_specs = tuple(pl.BlockSpec(memory_space=pl.ANY) for _ in range(n_att))
    base_in: list[object] = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # x_in: (B, H) bf16
        pl.BlockSpec(memory_space=pltpu.SMEM),  # pos: (B,) int32
        pl.BlockSpec(
            memory_space=pltpu.SMEM
        ),  # page_table: (NATT, B, max_pages_per_seq)
        pool_specs,  # pools: tuple of NATT x (num_pages, 2, K_R, D, page_size)
        pl.BlockSpec(memory_space=pl.ANY),  # conv: (NGDN, B, conv_kernel-1, QKV)
        pl.BlockSpec(memory_space=pl.ANY),  # rec: (NGDN, B, VH_R, KD, VD)
        w_specs,
    ]
    if has_forced_routing:
        base_in.append(pl.BlockSpec(memory_space=pltpu.SMEM))  # forced_top_ids
    in_specs = tuple(base_in)
    out_specs: tuple[object, ...] = (
        pl.BlockSpec(memory_space=pltpu.VMEM),  # x_out: (B, H) bf16
        pool_specs,  # pool_out: tuple of NATT region pools aliased in place
        pl.BlockSpec(memory_space=pl.ANY),  # conv_out
        pl.BlockSpec(memory_space=pl.ANY),  # rec_out
    )
    return in_specs, out_specs


def assert_no_all_gather(hlo_text: str, *, context: str = "") -> None:
    """Fail if the lowered HLO/StableHLO text contains any `all-gather` instruction."""
    for line in hlo_text.splitlines():
        s = line.strip()
        if not s or s.startswith(("//", "#")):
            continue
        if "all-gather" in s or "all_gather" in s:
            prefix = f"[{context}] " if context else ""
            msg = f"{prefix}Forbidden all-gather collective found in HLO: {s}"
            raise AssertionError(msg)

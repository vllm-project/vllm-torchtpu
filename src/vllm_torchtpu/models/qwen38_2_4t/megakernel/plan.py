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
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.megakernel.dma import pool_alias
from vllm_torchtpu.megakernel.profiles.default import chip_budget

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax._src.pallas.mosaic.core import MemoryRef
    from jax._src.state.types import TransformedRef

    from .shapes import Qwen38Config

_PAD2_SIZE: int = 2
_MIN_2D_NDIM: int = 2
_MOE_WEIGHT_NDIM: int = 6


def vmem_limit_bytes() -> int:
    """Return usable VMEM budget in bytes from `pltpu.get_tpu_info()` or chip budget."""
    if jax.devices()[0].platform == "tpu":
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    return chip_budget().vmem_mib * 1024 * 1024


def smem_limit_bytes() -> int:
    """Return SMEM budget in bytes from `pltpu.get_tpu_info()` or chip budget."""
    if jax.devices()[0].platform == "tpu":
        return int(pltpu.get_tpu_info().smem_capacity_bytes)
    return chip_budget().smem_mib * 1024 * 1024


def _nbytes(shape: tuple[int, ...], dtype: jnp.dtype | str | type) -> int:
    """Return total bytes for an array of the given shape and element type."""
    return int(math.prod(shape)) * int(jnp.dtype(dtype).itemsize)


@dataclasses.dataclass(frozen=True)
class StageSlice:
    """Byte-aligned scratchpad slice descriptor for a stage-local weight buffer."""

    name: str
    shape: tuple[int, ...]
    tiled_shape: tuple[int, ...]
    dtype: jnp.dtype
    offset_bytes: int
    size_bytes: int


@dataclasses.dataclass(frozen=True)
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
        return pltpu.VMEM((self.peak_bytes // 128, 128), jnp.uint8)

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
            rows = target.size_bytes // 128
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
    row_tile = 32
    dt = jnp.dtype(jnp.uint8)
    slices: dict[str, StageSlice] = {}
    for name, (r, c) in shapes.items():
        r_pad = ((r + row_tile - 1) // row_tile) * row_tile
        c_pad = ((c + 127) // 128) * 128
        slices[name] = StageSlice(
            name=name,
            shape=(r, c),
            tiled_shape=(r_pad, c_pad),
            dtype=dt,
            offset_bytes=0,
            size_bytes=r_pad * c_pad * int(dt.itemsize),
        )
    peak_bytes = max(s.size_bytes for s in slices.values())
    return StageScratchPlan(peak_bytes=peak_bytes, slices=slices), slices


def make_scratch_specs(
    cfg: Qwen38Config, *, interpret: bool = False
) -> dict[str, MemoryRef | dict[str, MemoryRef]]:
    """Build the scratchpad dictionary for `pl.pallas_call`.

    Composes `StageScratchPlan` for stage-disjoint down-projection weight buffers and
    allocates explicit ping-pong scratch for the local 2-level `hier=4` all-reduce.
    """
    b = cfg.B
    h = cfg.H
    d = cfg.D
    dcp = cfg.DCP
    q_r = cfg.Q_R
    k_r = cfg.K_R
    qkv = cfg.QKV
    z_r = cfg.Z_R
    vh_r = cfg.VH_R
    kd = cfg.KD
    vd = cfg.VD
    i_r = cfg.I_R
    se_r = cfg.SE_R
    g = cfg.fp4_group
    h_tile = cfg.h_tile
    hier_g = min(cfg.hier_group, cfg.R)
    num_groups = max(1, cfg.R // hier_g)

    stage_plan, _ = build_stage_scratch_plan(cfg)

    return {
        # Stage-disjoint down-projection weight scratchpad (wo_buf, wout_buf, wse2_buf)
        "stage_scratch_u8": stage_plan.scratch_spec(interpret=bool(interpret)),
        # Local 2-level hier=4 all-reduce buffers: send buffers use 1 slot (local
        # outbound DMAs complete before batch.wait() returns), while receive buffers
        # and DMA semaphores use 2 ping-pong slots (ar_step & 1) so a fast peer rank
        # entering the next all-reduce cannot overwrite a peer's receive buffer before
        # it finishes summing the prior slot without an inter-collective barrier.
        "ar_send": pltpu.VMEM((max(1, hier_g - 1), b, h), jnp.bfloat16),
        "ar_recv": pltpu.VMEM((2, max(1, hier_g - 1), b, h), jnp.bfloat16),
        "ar_stage_send": pltpu.VMEM((max(1, num_groups - 1), b, h), jnp.float32),
        "ar_stage_recv": pltpu.VMEM((2, max(1, num_groups - 1), b, h), jnp.float32),
        "ar_ssem1": pltpu.SemaphoreType.DMA((2, max(1, hier_g - 1))),
        "ar_rsem1": pltpu.SemaphoreType.DMA((2, max(1, hier_g - 1))),
        "ar_ssem2": pltpu.SemaphoreType.DMA((2, max(1, num_groups - 1))),
        "ar_rsem2": pltpu.SemaphoreType.DMA((2, max(1, num_groups - 1))),
        "b_sem": pltpu.SemaphoreType.REGULAR((4,)),
        # Persistent residual & normalized hidden buffers
        "x_cur": pltpu.VMEM((b, h), jnp.bfloat16),
        "x_norm": pltpu.VMEM((b, h), jnp.bfloat16),
        # Streaming H-tile FP4 weight + scale ping-pong buffers
        "wqkv_buf": pltpu.VMEM((2, h_tile, cfg.ATT_QKV_R // 2), jnp.uint8),
        "wqkv_s_buf": pltpu.VMEM((2, h_tile // g, cfg.ATT_QKV_R), jnp.bfloat16),
        "wo_s_buf": pltpu.VMEM(((q_r * d) // g, h), jnp.bfloat16),
        "win_buf": pltpu.VMEM((2, h_tile, (qkv + z_r) // 2), jnp.uint8),
        "win_s_buf": pltpu.VMEM((2, h_tile // g, qkv + z_r), jnp.bfloat16),
        "wba_buf": pltpu.VMEM((h, 2 * vh_r), jnp.bfloat16),
        "wconv_buf": pltpu.VMEM((cfg.conv_kernel, qkv), jnp.bfloat16),
        "wout_s_buf": pltpu.VMEM((z_r // g, h), jnp.bfloat16),
        "wgate_buf": pltpu.VMEM((h, cfg.E), jnp.bfloat16),
        "wsegate_buf": pltpu.VMEM((h, 1), jnp.bfloat16),
        "wse13_buf": pltpu.VMEM((2, h_tile, se_r), jnp.uint8),
        "wse13_s_buf": pltpu.VMEM((2, h_tile // g, 2 * se_r), jnp.bfloat16),
        "wse2_s_buf": pltpu.VMEM((se_r // g, h), jnp.bfloat16),
        # Routed expert weights & scales (H-tile ping-pong for e13, single-slot for e2)
        "we13_buf": pltpu.VMEM((2, h_tile, i_r), jnp.uint8),
        "we13_s_buf": pltpu.VMEM((2, h_tile // g, 2 * i_r), jnp.bfloat16),
        "we2_buf": pltpu.VMEM((i_r, h // 2), jnp.uint8),
        "we2_s_buf": pltpu.VMEM((i_r // g, h), jnp.bfloat16),
        # GDN recurrent + conv state ping-pong buffers (2 slots)
        "rec_buf": pltpu.VMEM((2, b, vh_r, kd, vd), jnp.float32),
        "conv_buf": pltpu.VMEM((2, b, cfg.conv_kernel - 1, qkv), jnp.bfloat16),
        # DCP8 attention KV page ping-pong + query/LSE cross-rank exchange buffers
        "kv_page_buf": pltpu.VMEM((2, 2, k_r, d, cfg.page_size), jnp.float8_e4m3fn),
        "kv_write_buf": pltpu.VMEM((2, k_r, d, cfg.page_size), jnp.float8_e4m3fn),
        "q_local_send": pltpu.VMEM((max(1, dcp - 1), b, q_r, d), jnp.bfloat16),
        "q_dcp_recv": pltpu.VMEM((dcp, b, q_r, d), jnp.bfloat16),
        "lse_part_send": pltpu.VMEM(
            (max(1, dcp - 1), b, dcp * q_r, d + 2), jnp.float32
        ),
        "lse_part_recv": pltpu.VMEM((dcp, b, dcp * q_r, d + 2), jnp.float32),
        # Matmul output accumulator scratch
        "fp4_out_buf": pltpu.VMEM(
            (b, max(h, cfg.ATT_QKV_R, qkv + z_r, 2 * se_r, 2 * i_r)), jnp.float32
        ),
        # DMA semaphores
        "w_sem": pltpu.SemaphoreType.DMA((4,)),
        "state_sem": pltpu.SemaphoreType.DMA((4,)),
        "kv_sem": pltpu.SemaphoreType.DMA((4,)),
        "dcp_q_ssem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
        "dcp_q_rsem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
        "dcp_lse_ssem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
        "dcp_lse_rsem": pltpu.SemaphoreType.DMA((max(1, dcp - 1),)),
    }


def _padded_vmem_bytes(shape: tuple[int, ...], itemsize: int) -> int:
    """Compute aligned scratchpad bytes with trailing (8/32, 128) tile padding."""
    if len(shape) >= _MIN_2D_NDIM:
        r_tile = 32 if itemsize == 1 else 8
        r_pad = ((int(shape[-2]) + r_tile - 1) // r_tile) * r_tile
        c_pad = ((int(shape[-1]) + 127) // 128) * 128
        elems = int(math.prod(shape[:-2])) * r_pad * c_pad
    elif len(shape) == 1:
        elems = ((int(shape[0]) + 127) // 128) * 128
    else:
        elems = 1
    return elems * int(itemsize)


def resident_vmem_bytes(cfg: Qwen38Config) -> int:
    """Compute aligned VMEM bytes of resident inputs, outputs, and norm weights."""
    resident_blocks: tuple[tuple[tuple[int, ...], int], ...] = (
        ((cfg.B, cfg.H), 2),  # x_in
        ((cfg.B, cfg.H), 2),  # x_out
        ((cfg.L, cfg.H), 2),  # in_norm
        ((cfg.L, cfg.H), 2),  # post_norm
        ((cfg.H,), 2),  # final_norm
        ((cfg.NATT, cfg.D), 2),  # q_norm
        ((cfg.NATT, cfg.D), 2),  # k_norm
        ((cfg.NGDN, cfg.VH_R), 4),  # a_log
        ((cfg.NGDN, cfg.VH_R), 4),  # dt_bias
        ((cfg.NGDN, cfg.VD), 2),  # gdn_norm
    )
    return sum(_padded_vmem_bytes(sh, itemsize) for sh, itemsize in resident_blocks)


def scratch_footprint(cfg: Qwen38Config) -> dict[str, int | float | bool]:
    """Compute aligned byte footprint of scratch/resident/scalar buffers and limits."""
    specs = make_scratch_specs(cfg)
    breakdown: dict[str, int] = {}
    total = 0
    raw_total = 0
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            try:
                dt = jnp.dtype(spec.dtype)
            except TypeError:
                continue
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
        "total_mib": round(total / (1024 * 1024), 3),
        "limit_bytes": v_limit,
        "scalar_bytes": smem_bytes,
        "scalar_limit_bytes": s_limit,
        "fits_in_scalar_limit": smem_bytes < s_limit,
        "fits_in_limit": (total < v_limit) and (smem_bytes < s_limit),
    }


def total_scalar_bytes(cfg: Qwen38Config) -> int:
    """Compute total scalar-memory input bytes (`pos` + `page_table`)."""
    pos_bytes = cfg.B * 4
    pt_bytes = cfg.NATT * cfg.B * cfg.max_pages_per_seq * 4
    return pos_bytes + pt_bytes


# Weight tensors whose H dimension is tiled into (H // h_tile, h_tile) for 4D/5D DMA
_H_TILED_4D_KEYS = frozenset({"wqkv", "w_inproj", "w_se13"})
_H_TILED_4D_SCALE_KEYS = frozenset({"wqkv_s", "w_inproj_s", "w_se13_s"})
_H_TILED_5D_KEYS = frozenset({"w_e13"})
_H_TILED_5D_SCALE_KEYS = frozenset({"w_e13_s"})
_PAD2_AXIS2_KEYS = (
    _H_TILED_4D_KEYS
    | _H_TILED_4D_SCALE_KEYS
    | _H_TILED_5D_KEYS
    | _H_TILED_5D_SCALE_KEYS
)


def normalize_weights_for_pallas(
    weights: dict[str, jax.Array], cfg: Qwen38Config
) -> dict[str, jax.Array]:
    """Reshape H-tiled weights into explicit `(n_htiles, h_tile, ...)` axes."""
    n_ht = cfg.H // cfg.h_tile
    g_ht = cfg.h_tile // cfg.fp4_group
    out: dict[str, jax.Array] = {}
    for k, arr in weights.items():
        v = arr
        if k in _H_TILED_4D_KEYS:
            v = v.reshape(v.shape[0], v.shape[1], n_ht, cfg.h_tile, v.shape[3])
        elif k in _H_TILED_4D_SCALE_KEYS:
            v = v.reshape(v.shape[0], v.shape[1], n_ht, g_ht, v.shape[3])
        elif k in _H_TILED_5D_KEYS:
            v = v.reshape(
                v.shape[0], v.shape[1], v.shape[2], n_ht, cfg.h_tile, v.shape[4]
            )
        elif k in _H_TILED_5D_SCALE_KEYS:
            v = v.reshape(v.shape[0], v.shape[1], v.shape[2], n_ht, g_ht, v.shape[4])

        if k in _PAD2_AXIS2_KEYS:
            ax = 3 if v.ndim == _MOE_WEIGHT_NDIM else 2
            if v.shape[ax] == 1:
                v = jnp.repeat(v, _PAD2_SIZE, axis=ax)
        out[k] = v
    return out


def _unpad_weight(k: str, v: jax.Array, n_ht: int) -> jax.Array:
    """Remove axis padding added to satisfy Pallas tiling requirements."""
    if k not in _PAD2_AXIS2_KEYS:
        return v
    if k in {"wqkv", "wqkv_s", "w_inproj", "w_inproj_s", "w_se13", "w_se13_s"}:
        if n_ht == 1 and v.shape[2] == _PAD2_SIZE:
            return v[:, :, :1, ...]
    elif k in {"w_e13", "w_e13_s"} and n_ht == 1 and v.shape[3] == _PAD2_SIZE:
        return v[:, :, :, :1, ...]
    return v


def _untile_weight(k: str, v: jax.Array) -> jax.Array:
    """Collapse tiled H dimensions back to canonical rank."""
    if k in _H_TILED_4D_KEYS or k in _H_TILED_4D_SCALE_KEYS:
        return v.reshape(v.shape[0], v.shape[1], v.shape[2] * v.shape[3], v.shape[4])
    if k in _H_TILED_5D_KEYS or k in _H_TILED_5D_SCALE_KEYS:
        return v.reshape(
            v.shape[0],
            v.shape[1],
            v.shape[2],
            v.shape[3] * v.shape[4],
            v.shape[5],
        )
    return v


def denormalize_weights_from_pallas(
    norm_weights: dict[str, jax.Array], cfg: Qwen38Config
) -> dict[str, jax.Array]:
    """Restore normalized Pallas weight dict back to canonical shapes."""
    n_ht = cfg.H // cfg.h_tile
    out: dict[str, jax.Array] = {}
    for k, arr in norm_weights.items():
        v = _unpad_weight(k, arr, n_ht)
        out[k] = _untile_weight(k, v)
    return out


_RESIDENT_WEIGHT_KEYS = frozenset(
    {
        "in_norm",
        "post_norm",
        "final_norm",
        "q_norm",
        "k_norm",
        "a_log",
        "dt_bias",
        "gdn_norm",
    }
)


def make_io_specs(
    weights: dict[str, jax.Array],
    *,
    has_forced_routing: bool = False,
) -> tuple[Any, Any]:
    """Construct `in_specs` and `out_specs` for the single-step `pl.pallas_call`.

    Small routing metadata (`pos`, `page_table`, and optional `forced_top_ids`)
    reside in scalar memory; small per-layer norm/decay vectors
    (`_RESIDENT_WEIGHT_KEYS`) reside in vector scratchpad; large state pools (`pool`,
    `conv`, `rec`) and all projection/MoE weights stay in `pl.ANY` (HBM) and are
    streamed via explicit DMA.
    """
    w_specs = {
        k: pl.BlockSpec(
            memory_space=pltpu.VMEM if k in _RESIDENT_WEIGHT_KEYS else pl.ANY
        )
        for k in weights
    }
    base_in = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # x_in: (B, H) bf16
        pl.BlockSpec(memory_space=pltpu.SMEM),  # pos: (B,) int32
        pl.BlockSpec(
            memory_space=pltpu.SMEM
        ),  # page_table: (NATT, B, max_pages_per_seq)
        pl.BlockSpec(
            memory_space=pl.ANY
        ),  # pool: (NATT, num_pages, 2, K_R, D, page_size)
        pl.BlockSpec(memory_space=pl.ANY),  # conv: (NGDN, B, conv_kernel-1, QKV)
        pl.BlockSpec(memory_space=pl.ANY),  # rec: (NGDN, B, VH_R, KD, VD)
        w_specs,
    ]
    if has_forced_routing:
        base_in.append(pl.BlockSpec(memory_space=pltpu.SMEM))  # forced_top_ids
    in_specs = tuple(base_in)
    out_specs = (
        pl.BlockSpec(memory_space=pltpu.VMEM),  # x_out: (B, H) bf16
        pl.BlockSpec(memory_space=pl.ANY),  # pool_out
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

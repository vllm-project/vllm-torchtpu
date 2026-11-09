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
"""Diffusion attention backends for TPU execution.

``TpuFlashAttentionBackend`` (the default) runs unmasked, non-causal
bf16 attention through a Pallas flash kernel tuned for diffusion
sequence lengths. Masked, causal, or non-bf16 calls fall back with a
one-time warning to ``TpuSDPAImpl`` (PyTorch SDPA).
"""

import functools
import math
from collections.abc import Mapping, Sequence
from functools import cache as run_once
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.forward_context import get_ulysses_mode

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.diffusion_attention.kernel import (
    flash_attention_bshd,
    ring_ulysses_flash_attention_bshd,
)
from vllm_torchtpu.kernels.diffusion_attention.ring_kernel import (
    ring_attention_sharded,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class RingCfgLaneMismatchError(ValueError, RuntimeError):
    """Raised when ring attention partners do not belong to the same CFG lane."""


_RUNTIME_RING_CFG_STATE: dict[str, Any] | None = None


def set_ring_cfg_validation_state(state: dict[str, Any] | None) -> None:
    """Records runtime CFG/ring group layout for same-CFG-lane validation."""
    global _RUNTIME_RING_CFG_STATE
    _RUNTIME_RING_CFG_STATE = state


def get_ring_cfg_validation_state() -> dict[str, Any] | None:
    """Returns the active runtime CFG/ring group layout state, if any."""
    return _RUNTIME_RING_CFG_STATE


def clear_ring_cfg_validation_state() -> None:
    """Clears any cached runtime CFG/ring group layout state."""
    global _RUNTIME_RING_CFG_STATE
    _RUNTIME_RING_CFG_STATE = None


def ranks_ordered_by_device_id(device_ids_by_rank: Sequence[int]) -> list[int]:
    """Returns global ranks ordered by ascending PJRT device ID 0..N-1."""
    device_ids = [int(d) for d in device_ids_by_rank]
    n = len(device_ids)
    if sorted(device_ids) != list(range(n)):
        raise ValueError(
            f"device_ids_by_rank {device_ids} is not a permutation of 0..{n - 1}"
        )
    return [r for _, r in sorted((dev_id, r) for r, dev_id in enumerate(device_ids))]


def _orthogonal_position_groups(
    sizes: tuple[int, int, int, int, int], axis_index: int
) -> list[list[int]]:
    """Builds orthogonal position groups for (tp, sp, pp, cfg, dp)."""
    grid = np.arange(int(np.prod(sizes)), dtype=np.int64).reshape(sizes, order="F")
    moved = np.moveaxis(grid, axis_index, -1)
    return [list(map(int, row)) for row in moved.reshape(-1, sizes[axis_index])]


def compute_device_id_group_layout(
    device_ids_by_rank: Sequence[int],
    *,
    cfg_size: int = 1,
    ring_size: int = 1,
    uly_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int | None = None,
    use_ulysses_low: bool = True,
) -> dict[str, Any]:
    """Computes vLLM-Omni parallel group rank lists laid out by PJRT device ID."""
    world_size = len(device_ids_by_rank)
    sp_size = ring_size * uly_size
    non_dp = tp_size * sp_size * pp_size * cfg_size
    if world_size % non_dp != 0:
        raise ValueError(
            f"world_size ({world_size}) is not divisible by non_dp ({non_dp})"
        )
    resolved_dp = (world_size // non_dp) if dp_size is None else dp_size
    if resolved_dp * non_dp != world_size:
        raise ValueError(
            f"dp_size ({resolved_dp}) * non_dp ({non_dp}) != world_size ({world_size})"
        )
    ranks_by_dev = ranks_ordered_by_device_id(device_ids_by_rank)
    sizes = (tp_size, sp_size, pp_size, cfg_size, resolved_dp)

    def _map_axis(axis_idx: int) -> list[list[int]]:
        return [
            [ranks_by_dev[pos] for pos in group]
            for group in _orthogonal_position_groups(sizes, axis_idx)
        ]

    tp_groups = _map_axis(0)
    sp_groups = _map_axis(1)
    pp_groups = _map_axis(2)
    cfg_groups = _map_axis(3)
    dp_groups = _map_axis(4)

    ulysses_groups: list[list[int]] = []
    ring_groups: list[list[int]] = []
    for group_ranks in sp_groups:
        if use_ulysses_low:
            for i in range(ring_size):
                ulysses_groups.append(
                    list(group_ranks[i * uly_size : (i + 1) * uly_size])
                )
            for i in range(uly_size):
                ring_groups.append(list(group_ranks[i::uly_size]))
        else:
            for i in range(uly_size):
                ring_groups.append(
                    list(group_ranks[i * ring_size : (i + 1) * ring_size])
                )
            for i in range(ring_size):
                ulysses_groups.append(list(group_ranks[i::ring_size]))

    cfg_lane_by_rank: dict[int, int] = {}
    for cfg_group in cfg_groups:
        for lane_idx, r in enumerate(cfg_group):
            cfg_lane_by_rank[int(r)] = lane_idx

    return {
        "device_ids_by_rank": [int(d) for d in device_ids_by_rank],
        "ranks_by_device_id": ranks_by_dev,
        "cfg_groups": cfg_groups,
        "sp_groups": sp_groups,
        "ulysses_groups": ulysses_groups,
        "ring_groups": ring_groups,
        "dp_groups": dp_groups,
        "pp_groups": pp_groups,
        "tp_groups": tp_groups,
        "cfg_lane_by_rank": cfg_lane_by_rank,
    }


def validate_ring_cfg_lanes(
    *,
    cfg_groups: Sequence[Sequence[int]] | None = None,
    ring_groups: Sequence[Sequence[int]] | None = None,
    cfg_lane_by_rank: Mapping[int, int] | Sequence[int] | None = None,
    device_ids_by_rank: Sequence[int] | None = None,
    mesh_device_ids: Any | None = None,
    outer_size: int | None = None,
    ring_size: int | None = None,
    uly_size: int | None = None,
) -> None:
    """Verifies that every ring partner belongs to the same CFG lane."""
    lane_map: dict[int, int] = {}
    if cfg_lane_by_rank is not None:
        if isinstance(cfg_lane_by_rank, Mapping):
            lane_map = {int(r): int(lane) for r, lane in cfg_lane_by_rank.items()}
        else:
            lane_map = {r: int(lane) for r, lane in enumerate(cfg_lane_by_rank)}
    elif cfg_groups is not None:
        for cfg_group in cfg_groups:
            for lane_idx, r in enumerate(cfg_group):
                lane_map[int(r)] = lane_idx

    if ring_groups is not None and lane_map:
        for ring_group in ring_groups:
            group_list = [int(r) for r in ring_group]
            lanes = [lane_map[r] for r in group_list]
            if len(set(lanes)) > 1:
                raise RingCfgLaneMismatchError(
                    f"Ring partners {group_list} span multiple CFG lanes {lanes}; "
                    "every ring partner must be in the same CFG lane."
                )

    if mesh_device_ids is not None and device_ids_by_rank is not None and lane_map:
        dev_ids = [int(d) for d in device_ids_by_rank]
        rank_for_dev = {dev_id: r for r, dev_id in enumerate(dev_ids)}
        if hasattr(mesh_device_ids, "devices"):
            grid_objs = mesh_device_ids.devices
            mesh_shape = grid_objs.shape
            flat_mesh_devs = [int(d.id) for d in grid_objs.flat]
        else:
            arr = np.asarray(mesh_device_ids)
            flat_mesh_devs = [
                int(x.id) if hasattr(x, "id") else int(x) for x in arr.flat
            ]
            mesh_shape = arr.shape if arr.ndim == 3 else None

        if mesh_shape is None:
            if outer_size is None or ring_size is None or uly_size is None:
                raise ValueError(
                    "outer_size, ring_size, and uly_size are required when "
                    "mesh_device_ids is flat"
                )
            mesh_shape = (outer_size, ring_size, uly_size)

        grid = np.asarray(flat_mesh_devs, dtype=np.int64).reshape(mesh_shape)
        b_dim, r_dim, u_dim = mesh_shape
        ring_lookup: dict[int, list[int]] = {}
        if ring_groups is not None:
            for rg in ring_groups:
                rg_list = [int(r) for r in rg]
                for r in rg_list:
                    ring_lookup[r] = rg_list

        for b_idx in range(b_dim):
            for u_idx in range(u_dim):
                fiber_devs = [int(grid[b_idx, r_idx, u_idx]) for r_idx in range(r_dim)]
                fiber_ranks = [rank_for_dev[d] for d in fiber_devs]
                fiber_lanes = [lane_map[r] for r in fiber_ranks]
                if len(set(fiber_lanes)) > 1:
                    raise RingCfgLaneMismatchError(
                        f"Mesh ring fiber (batch={b_idx}, uly={u_idx}) with device IDs "
                        f"{fiber_devs} maps to ranks {fiber_ranks} across mismatched "
                        f"CFG lanes {fiber_lanes}; every ring partner must be in the "
                        "same CFG lane."
                    )
                if ring_lookup and fiber_ranks[0] in ring_lookup:
                    expected = ring_lookup[fiber_ranks[0]]
                    if fiber_ranks != expected:
                        raise RingCfgLaneMismatchError(
                            f"Mesh ring fiber (batch={b_idx}, uly={u_idx}) with device "
                            f"IDs {fiber_devs} maps to ranks {fiber_ranks}, which does "
                            f"not match distributed ring group {expected}."
                        )


def _get_device_ordered_pallas_mesh(
    axis_names: tuple[str, ...],
    mesh_shape: tuple[int, ...],
) -> jax.sharding.Mesh:
    """Builds the Pallas JAX mesh with devices sorted by ascending device ID."""
    raw_mesh = pallas.get_pallas_mesh(axis_names=axis_names, mesh_shape=mesh_shape)
    flat_devices = list(raw_mesh.devices.flat)
    sorted_devices = sorted(flat_devices, key=lambda d: int(d.id))
    device_ids = [int(d.id) for d in sorted_devices]
    if device_ids != list(range(len(sorted_devices))):
        raise ValueError(
            f"Pallas mesh device IDs {device_ids} are not 0..{len(sorted_devices) - 1}"
        )
    devices_grid = np.array(sorted_devices, dtype=object).reshape(mesh_shape)
    axis_types = getattr(raw_mesh, "_axis_types", None)
    if axis_types is not None:
        return jax.sharding.Mesh(devices_grid, axis_names, axis_types=axis_types)
    return jax.sharding.Mesh(devices_grid, axis_names)


class TpuSDPAImpl(SDPAImpl):
    """TPU implementation of SDPA attention for diffusion models."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            query, key, value, attn_metadata, mask_mode="broadcast_k"
        )


class TpuSDPABackend(SDPABackend):
    """TPU attention backend for diffusion models using PyTorch SDPA."""

    @staticmethod
    def get_name() -> str:
        return "TPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TpuSDPAImpl]:
        return TpuSDPAImpl


@run_once
def _flash_attention_op():
    """Builds the torch custom op wrapping the Pallas kernel (once, eagerly)."""

    def diffusion_flash_attention(
        q: jax.Array, k: jax.Array, v: jax.Array, scale: float
    ) -> jax.Array:
        return flash_attention_bshd(q, k, v, scale=scale)

    op = pallas.jax_op("pallas::diffusion_flash_attention", diffusion_flash_attention)
    op.register_fake(lambda q, k, v, scale: q.new_empty((*q.shape[:3], v.shape[-1])))
    return op


class TpuFlashAttentionImpl(TpuSDPAImpl):
    """Pallas flash attention for diffusion transformers on TPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Build the op outside any compiled region.
        self._op = _flash_attention_op()

    def _needs_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: Any | None,
    ) -> bool:
        attn_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        # TODO: Revisit this fallback once torch_tpu SDPA lets callers select its
        # flash kernel and supports masks; until then masked, causal and non-bf16
        # calls take PyTorch SDPA.
        if self.causal:
            reason = "causal=True"
        elif attn_mask is not None:
            reason = "attn_mask is not None"
        elif query.dtype != torch.bfloat16:
            reason = f"query.dtype={query.dtype} (requires torch.bfloat16)"
        else:
            return False
        logger.warning_once(
            "TpuFlashAttentionImpl falling back to PyTorch SDPA (%s).",
            reason,
        )
        return True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        if query.shape[2] % key.shape[2] != 0:
            # Neither path supports it: vLLM-Omni's SDPA raises the same way.
            raise ValueError(
                f"num_heads ({query.shape[2]}) must be a multiple of "
                f"num_kv_heads ({key.shape[2]})"
            )
        if self._needs_sdpa_fallback(query, key, attn_metadata):
            return super().forward(query, key, value, attn_metadata)
        return self._op(query, key, value, self.softmax_scale)


class TpuFlashAttentionBackend(SDPABackend):
    """TPU diffusion attention backend using a Pallas flash kernel."""

    @staticmethod
    def get_name() -> str:
        return "TPU_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[TpuFlashAttentionImpl]:
        return TpuFlashAttentionImpl


_NON_RING_FLASH_PREFS = frozenset(
    {
        "sdpa",
        "torch",
        "torch_sdpa",
        "tpu_sdpa",
        "cudnn_attn",
        "flashinfer_attn",
        "trtllm_attn",
        "sage_attn",
        "sage_attn_3",
        "flash_attn_hub",
        "flash_attn_3_hub",
    }
)


@functools.cache
def _ring_ulysses_attention_op(
    outer_size: int,
    ring_size: int,
    uly_size: int,
    fuse_ulysses: bool,
):
    """Builds the torch custom op wrapping the 2D Ring-Ulysses Pallas kernel."""
    mesh = _get_device_ordered_pallas_mesh(
        axis_names=("batch", "ring", "uly"),
        mesh_shape=(outer_size, ring_size, uly_size),
    )
    if _RUNTIME_RING_CFG_STATE is not None and ring_size > 1:
        validate_ring_cfg_lanes(
            cfg_groups=_RUNTIME_RING_CFG_STATE.get("cfg_groups"),
            ring_groups=_RUNTIME_RING_CFG_STATE.get("ring_groups"),
            cfg_lane_by_rank=_RUNTIME_RING_CFG_STATE.get("cfg_lane_by_rank"),
            device_ids_by_rank=_RUNTIME_RING_CFG_STATE.get("device_ids_by_rank"),
            mesh_device_ids=mesh,
            outer_size=outer_size,
            ring_size=ring_size,
            uly_size=uly_size,
        )
    spec = PartitionSpec("batch", "ring", "uly", None, None, None, None)
    uly_axis = "uly" if fuse_ulysses else None

    def _body(q: jax.Array, k: jax.Array, v: jax.Array, scale: float) -> jax.Array:
        out = ring_ulysses_flash_attention_bshd(
            q[0, 0, 0],
            k[0, 0, 0],
            v[0, 0, 0],
            ring_axis="ring",
            uly_axis=uly_axis,
            scale=scale,
        )
        return out[None, None, None]

    def _fn(q: jax.Array, k: jax.Array, v: jax.Array, scale: float) -> jax.Array:
        return jax.shard_map(
            functools.partial(_body, scale=scale),
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(q, k, v)

    mode = "ring_ulysses" if fuse_ulysses else "ring"
    op_name = f"pallas::diffusion_{mode}_attn_{outer_size}_{ring_size}_{uly_size}"
    # jax_op's stub types partition specs as tuple[tuple[str, ...], ...].
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        op_name,
        _fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec, None),
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda q, k, v, scale: q.new_empty((*q.shape[:6], v.shape[-1])))
    return op


# Below 4096 rows per rank, pure-ring per-head ICI hops cost more than
# intra-chip D2D all_to_all + bulk ppermute.
_SHORT_SHARD_2D_RING_MAX = 4096


@functools.cache
def _pallas_ring_attention_op(
    outer_size: int,
    ring_size: int,
    uly_size: int,
    fuse_ulysses: bool,
    scale: float,
    use_fixed_m: bool,
):
    """Builds the torch custom op for ICI-DMA Pallas ring attention."""
    axis_names = ("batch", "ring", "uly")
    mesh = _get_device_ordered_pallas_mesh(
        axis_names=axis_names,
        mesh_shape=(outer_size, ring_size, uly_size),
    )
    if _RUNTIME_RING_CFG_STATE is not None and ring_size > 1:
        validate_ring_cfg_lanes(
            cfg_groups=_RUNTIME_RING_CFG_STATE.get("cfg_groups"),
            ring_groups=_RUNTIME_RING_CFG_STATE.get("ring_groups"),
            cfg_lane_by_rank=_RUNTIME_RING_CFG_STATE.get("cfg_lane_by_rank"),
            device_ids_by_rank=_RUNTIME_RING_CFG_STATE.get("device_ids_by_rank"),
            mesh_device_ids=mesh,
            outer_size=outer_size,
            ring_size=ring_size,
            uly_size=uly_size,
        )
    spec = PartitionSpec("batch", "ring", "uly", None, None, None, None)
    do_fuse = bool(fuse_ulysses and uly_size > 1)

    def _body(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        if do_fuse:
            lead = (0, 1, 2)
            q_u = jax.lax.all_to_all(
                jnp.squeeze(q, axis=lead),
                "uly",
                split_axis=2,
                concat_axis=1,
                tiled=True,
            )
            k_u = jax.lax.all_to_all(
                jnp.squeeze(k, axis=lead),
                "uly",
                split_axis=2,
                concat_axis=1,
                tiled=True,
            )
            v_u = jax.lax.all_to_all(
                jnp.squeeze(v, axis=lead),
                "uly",
                split_axis=2,
                concat_axis=1,
                tiled=True,
            )
            out_u = ring_attention_sharded(
                jnp.expand_dims(q_u, axis=lead),
                jnp.expand_dims(k_u, axis=lead),
                jnp.expand_dims(v_u, axis=lead),
                ring_axis="ring",
                axis_names=axis_names,
                scale=scale,
                use_fixed_m=use_fixed_m,
            )
            out_s = jax.lax.all_to_all(
                jnp.squeeze(out_u, axis=lead),
                "uly",
                split_axis=1,
                concat_axis=2,
                tiled=True,
            )
            return jnp.expand_dims(out_s, axis=lead)
        return ring_attention_sharded(
            q,
            k,
            v,
            ring_axis="ring",
            axis_names=axis_names,
            scale=scale,
            use_fixed_m=use_fixed_m,
        )

    def _fn(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return jax.shard_map(
            _body,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(q, k, v)

    scale_hex = f"{int(np.float64(scale).view(np.uint64)):016x}"
    fm_tag = "fm1" if use_fixed_m else "fm0"
    mode = "ring_ulysses_ici" if do_fuse else "ring_ici"
    op_name = (
        f"pallas::diffusion_{mode}_attn_"
        f"{outer_size}_{ring_size}_{uly_size}_{fm_tag}_{scale_hex}"
    )
    # jax_op's stub types partition specs as tuple[tuple[str, ...], ...].
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        op_name,
        _fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec),
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda q, k, v: q.new_empty(q.shape))
    return op


@functools.cache
def _short_2d_ring_attention_op(
    outer_size: int,
    ring_size: int,
    scale: float,
):
    """Builds intra-chip Ulysses=2 + ppermute ring op for short shards."""
    axis_names = ("batch", "ring", "uly")
    mesh = _get_device_ordered_pallas_mesh(
        axis_names=axis_names,
        mesh_shape=(outer_size, ring_size // 2, 2),
    )
    if _RUNTIME_RING_CFG_STATE is not None and ring_size > 1:
        validate_ring_cfg_lanes(
            cfg_groups=_RUNTIME_RING_CFG_STATE.get("cfg_groups"),
            ring_groups=_RUNTIME_RING_CFG_STATE.get("ring_groups"),
            cfg_lane_by_rank=_RUNTIME_RING_CFG_STATE.get("cfg_lane_by_rank"),
            device_ids_by_rank=_RUNTIME_RING_CFG_STATE.get("device_ids_by_rank"),
            mesh_device_ids=[int(d.id) for d in mesh.devices.flat],
            outer_size=outer_size,
            ring_size=ring_size,
            uly_size=1,
        )
    spec = PartitionSpec("batch", "ring", "uly", None, None, None, None)

    def _body(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        out = ring_ulysses_flash_attention_bshd(
            q[0, 0, 0],
            k[0, 0, 0],
            v[0, 0, 0],
            ring_axis="ring",
            uly_axis="uly",
            scale=scale,
        )
        return out[None, None, None]

    def _fn(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return jax.shard_map(
            _body,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(q, k, v)

    scale_hex = f"{int(np.float64(scale).view(np.uint64)):016x}"
    op_name = (
        f"pallas::diffusion_short_2d_ring_attn_{outer_size}_{ring_size}_{scale_hex}"
    )
    # jax_op's stub types partition specs as tuple[tuple[str, ...], ...].
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        op_name,
        _fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec),
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda q, k, v: q.new_empty(q.shape))
    return op


def _can_use_ici_ring_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    ring_size: int,
    uly_size: int,
    fuse_ulysses: bool,
) -> bool:
    """Whether a ring attention call fits the ICI-DMA Pallas ring kernel."""
    if not envs.VLLM_TORCHTPU_OMNI_RING_KERNEL or ring_size < 2:
        return False
    if query.device.type != "tpu":
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    batch, q_seq, num_q_heads, head_dim = query.shape
    kv_batch, kv_seq, num_kv_heads, k_dim = key.shape
    if batch != kv_batch or q_seq != kv_seq:
        return False
    if head_dim != 128 or k_dim != 128 or value.shape != key.shape:
        return False
    if not (query.dtype == key.dtype == value.dtype == torch.bfloat16):
        return False
    if fuse_ulysses and uly_size > 1:
        if num_q_heads % uly_size != 0 or num_kv_heads % uly_size != 0:
            return False
        q_seq_eff = q_seq * uly_size
        q_heads_eff = num_q_heads // uly_size
        kv_heads_eff = num_kv_heads // uly_size
    else:
        q_seq_eff = q_seq
        q_heads_eff = num_q_heads
        kv_heads_eff = num_kv_heads
    return q_seq_eff >= 256 and kv_heads_eff > 0 and q_heads_eff % kv_heads_eff == 0


def tpu_ring_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    *,
    ring_size: int,
    uly_size: int = 1,
    outer_size: int | None = None,
    fuse_ulysses: bool = False,
) -> torch.Tensor:
    """Runs 1D ring or fused 2D Ring-Ulysses flash attention on TPU."""
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError(
            f"num_heads ({query.shape[2]}) must be a multiple of "
            f"num_kv_heads ({key.shape[2]})"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if outer_size is None:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else ring_size * uly_size
        )
        outer_size = max(1, world_size // (ring_size * uly_size))
    if _RUNTIME_RING_CFG_STATE is not None and ring_size > 1:
        validate_ring_cfg_lanes(
            cfg_groups=_RUNTIME_RING_CFG_STATE.get("cfg_groups"),
            ring_groups=_RUNTIME_RING_CFG_STATE.get("ring_groups"),
            cfg_lane_by_rank=_RUNTIME_RING_CFG_STATE.get("cfg_lane_by_rank"),
            device_ids_by_rank=_RUNTIME_RING_CFG_STATE.get("device_ids_by_rank"),
            mesh_device_ids=list(range(outer_size * ring_size * uly_size)),
            outer_size=outer_size,
            ring_size=ring_size,
            uly_size=uly_size,
        )
    fuse = bool(fuse_ulysses and uly_size > 1)
    q = query.contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    k = key.contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    v = value.contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    if _can_use_ici_ring_kernel(
        query,
        key,
        value,
        ring_size=ring_size,
        uly_size=uly_size,
        fuse_ulysses=fuse,
    ):
        if (
            not fuse
            and uly_size == 1
            and ring_size >= 4
            and ring_size % 2 == 0
            and query.shape[1] <= _SHORT_SHARD_2D_RING_MAX
            and query.shape[2] % 2 == 0
            and key.shape[2] % 2 == 0
        ):
            short_op = _short_2d_ring_attention_op(outer_size, ring_size, float(scale))
            return short_op(q, k, v).reshape(*query.shape[:3], value.shape[-1])
        ring_op = _pallas_ring_attention_op(
            outer_size,
            ring_size,
            uly_size,
            fuse,
            float(scale),
            bool(envs.VLLM_TORCHTPU_OMNI_RING_FIXED_M),
        )
        return ring_op(q, k, v).reshape(*query.shape[:3], value.shape[-1])

    op = _ring_ulysses_attention_op(outer_size, ring_size, uly_size, fuse)
    out = op(q, k, v, float(scale))
    return out.reshape(*query.shape[:3], value.shape[-1])


def can_run_tpu_ring_attention(
    ring_runner: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any | None,
    *,
    causal: bool = False,
) -> bool:
    """Whether a RingParallelAttention call fits the TPU Pallas ring kernel."""
    pref = getattr(ring_runner, "attn_backend_pref", None)
    if pref is not None and str(pref).lower() in _NON_RING_FLASH_PREFS:
        return False
    if (
        causal
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
    ):
        return False
    if (
        key.shape[2] <= 0
        or query.shape[2] % key.shape[2] != 0
        or key.shape[2] != value.shape[2]
        or query.shape[-1] != key.shape[-1]
        or value.shape[-1] % 8 != 0
    ):
        return False
    if attn_metadata is not None:
        if (
            getattr(attn_metadata, "attn_mask", None) is not None
            or getattr(attn_metadata, "joint_query", None) is not None
            or getattr(attn_metadata, "joint_key", None) is not None
            or getattr(attn_metadata, "joint_value", None) is not None
        ):
            return False
        extra = getattr(attn_metadata, "extra", None) or {}
        if (
            extra.get("valid_kv_length") is not None
            or extra.get("gate_compress") is not None
            or extra.get("kv_cache_dtype") not in (None, "float")
        ):
            return False
    sp_group = getattr(ring_runner, "_sp_group", None)
    return int(getattr(sp_group, "ring_world_size", 0)) >= 1


def can_fuse_tpu_ring_ulysses(
    attn_layer: Any,
    strategy: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any | None,
) -> bool:
    """Whether an Attention forward call can fuse Ulysses all-to-all with Ring."""
    ring_runner = getattr(attn_layer, "ring_runner", None)
    if not getattr(attn_layer, "use_ring", False) or ring_runner is None:
        return False
    if get_ulysses_mode(default="strict") != "strict":
        return False
    if (
        strategy is getattr(attn_layer, "_no_parallel_strategy", None)
        or getattr(strategy, "name", None) != "ulysses"
        or getattr(attn_layer, "_explicit_cross_contract", False)
        or attn_layer.is_paged_kv_active()
    ):
        return False
    skip = getattr(getattr(attn_layer, "attention", None), "skip", None)
    if skip is not None and getattr(skip, "configured", False):
        return False
    if (
        getattr(strategy, "_scatter_idx", 2) != 2
        or getattr(strategy, "_gather_idx", 1) != 1
    ):
        return False
    sp_group = getattr(ring_runner, "_sp_group", None)
    uly_size = int(getattr(sp_group, "ulysses_world_size", 1))
    if (
        uly_size <= 1
        or query.shape[2] % uly_size != 0
        or key.shape[2] % uly_size != 0
        or value.shape[2] % uly_size != 0
        or (query.shape[2] // uly_size) % (key.shape[2] // uly_size) != 0
    ):
        return False
    return can_run_tpu_ring_attention(
        ring_runner,
        query,
        key,
        value,
        attn_metadata,
        causal=bool(getattr(attn_layer, "causal", False)),
    )

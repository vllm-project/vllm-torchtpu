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
"""Row-tiled fused QKV+gate split, per-head Q/K RMSNorm, and partial NeoX RoPE
for Qwen3-Next and Qwen3.5 attention as a single Pallas call per layer.

Inputs:

  qkv      [rows, (2*H + 2*Hk) * D]  bf16/f32  [q_gate (2*H*D) | k (Hk*D) | v (Hk*D)]
  q_w, k_w [D] f32                             norm weights, already including + 1
  cos_sin  [rows, R]                           rope cache rows: [cos (R/2) | sin (R/2)]

Outputs: q [rows, H*D] and k [rows, Hk*D] (normed, rotated on the first R
dims), v [rows, Hk*D], and gate [rows, H*D] (pre-sigmoid), all in ``qkv.dtype``.
Passing ``qkv`` whole avoids XLA materializing strided 2D column-slice HBM copies
for ``q_gate``, ``k``, and ``v`` before the Pallas call.

Row counts divisible by ``num_sublanes`` execute over a parallel 1D row grid
``(rows // tile,)`` with stacked-head vectorized RMSNorm and RoPE plus a
shape-proportional ``vmem_limit_bytes`` Scoped VMEM cap, scaling to prefill and
large mixed batches without VMEM growth or full-budget Scoped VMEM eviction.
Small unaligned row counts up to ``max(MAX_UNALIGNED_ROWS, 4 * num_sublanes)``
(such as 1- or 4-row decode buckets) run in a single whole-tensor VMEM step when
they fit in the Scoped VMEM budget.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MAX_TILE = 64
MAX_UNALIGNED_ROWS = 32
_MIB = 1024 * 1024
# Below 1 MiB of qkv (for example <= 24 rows of [rows, 17408] bfloat16 = 835 KiB
# at H=32, Hk=2, D=256), XLA keeps qkv in Scoped VMEM out of qkv_proj and fuses
# column slicing, RMSNorm, and NeoX RoPE inline without a custom-call barrier.
MIN_PALLAS_QKV_BYTES = _MIB
_FLOAT32_BYTES = 4
_SCOPED_VMEM_FRACTION = 0.5
_VMEM_HEADROOM_BYTES = 2 * _MIB
_INTERPRET_VMEM_CAPACITY_BYTES = 16 * _MIB
_MIN_SCOPED_VMEM_LIMIT_BYTES = 4 * _MIB
_16MIB_SCOPED_VMEM_LIMIT_BYTES = 16 * _MIB
_DEFAULT_SCOPED_VMEM_LIMIT_BYTES = 32 * _MIB
_64MIB_SCOPED_VMEM_LIMIT_BYTES = 64 * _MIB
_SCOPED_VMEM_HEADROOM_FACTOR = 1.25
_16MIB_SCOPED_VMEM_CHIPS = frozenset(
    {
        pltpu.ChipVersion.TPU_V2,
        pltpu.ChipVersion.TPU_V3,
        pltpu.ChipVersion.TPU_V4I,
        pltpu.ChipVersion.TPU_V4,
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V5P,
    }
)
_64MIB_SCOPED_VMEM_CHIPS = frozenset(
    {
        pltpu.ChipVersion.TPU_8I,
        pltpu.ChipVersion.TPU_8T,
    }
)


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _max_unaligned_rows(num_sublanes: int) -> int:
    return max(MAX_UNALIGNED_ROWS, 4 * num_sublanes)


def _default_scoped_vmem_bytes(
    chip_version: pltpu.ChipVersion | None = None,
) -> int:
    if chip_version in _16MIB_SCOPED_VMEM_CHIPS:
        return _16MIB_SCOPED_VMEM_LIMIT_BYTES
    if chip_version in _64MIB_SCOPED_VMEM_CHIPS:
        return _64MIB_SCOPED_VMEM_LIMIT_BYTES
    return _DEFAULT_SCOPED_VMEM_LIMIT_BYTES


def _usable_vmem_budget(
    vmem_capacity_bytes: int,
    *,
    chip_version: pltpu.ChipVersion | None = None,
) -> int:
    scoped_limit = min(
        _default_scoped_vmem_bytes(chip_version),
        int(_SCOPED_VMEM_FRACTION * vmem_capacity_bytes),
    )
    headroom = min(_VMEM_HEADROOM_BYTES, scoped_limit // 5)
    return max(0, scoped_limit - headroom)


def _vmem_tile_bytes(
    rows: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype_itemsize: int,
    *,
    buffer_count: int,
    num_sublanes: int = 8,
    stacked: bool = False,
) -> int:
    packing = _FLOAT32_BYTES // dtype_itemsize
    padded_rows_qkv = _align_up(rows, num_sublanes * packing)
    padded_rows_f32 = _align_up(rows, num_sublanes)
    qkv_cols = (2 * num_heads + 2 * num_kv_heads) * head_dim
    padded_rotary_cols = _align_up(rotary_dim, 128)
    row_buffers = buffer_count * (
        2 * padded_rows_qkv * qkv_cols * dtype_itemsize
        + padded_rows_f32 * padded_rotary_cols * _FLOAT32_BYTES
    )
    # [1, head_dim] float32 weights pad sublanes to [num_sublanes, head_dim].
    weight_buffers = buffer_count * 2 * num_sublanes * head_dim * _FLOAT32_BYTES
    stacked_scratch = (
        2 * max(num_heads, num_kv_heads) * padded_rows_f32 * head_dim * _FLOAT32_BYTES
        if stacked
        else 0
    )
    return row_buffers + weight_buffers + stacked_scratch


def scoped_vmem_limit_bytes(
    tile_vmem_bytes: int,
    vmem_capacity_bytes: int,
    *,
    chip_version: pltpu.ChipVersion | None = None,
) -> int | None:
    """Shape-proportional Scoped VMEM limit for ``qk_norm_rope_gate``.

    Without an explicit ``vmem_limit_bytes``, Pallas emits no Scoped VMEM cap
    and XLA reserves the global Scoped VMEM budget (32 MiB on TPU v7x and v6e,
    16 MiB on TPU v5p and v5e, 64 MiB on TPU Gen 8), evicting live residual and
    activation tensors to HBM across layers. Returns ``None`` when the estimated
    footprint plus headroom meets or exceeds the chip cap so Pallas keeps the
    target default.
    """
    if tile_vmem_bytes <= 0 or vmem_capacity_bytes <= 0:
        raise ValueError(
            "scoped_vmem_limit_bytes inputs must be positive, got "
            f"{tile_vmem_bytes=}, {vmem_capacity_bytes=}."
        )
    default_scoped = _default_scoped_vmem_bytes(chip_version)
    upper = min(default_scoped, int(0.8 * vmem_capacity_bytes))
    need = int(tile_vmem_bytes * _SCOPED_VMEM_HEADROOM_FACTOR) + _MIB
    if need >= upper:
        return None
    return max(need, min(_MIN_SCOPED_VMEM_LIMIT_BYTES, upper))


def legal_row_tile(rows: int, max_tile: int, *, num_sublanes: int = 8) -> int:
    """Largest multiple of ``num_sublanes`` <= ``max_tile`` dividing ``rows``."""
    if num_sublanes < 8 or num_sublanes % 8 != 0:
        raise ValueError(
            f"num_sublanes must be a positive multiple of 8, got {num_sublanes}."
        )
    if rows < num_sublanes or rows % num_sublanes != 0:
        raise ValueError(
            f"rows must be a positive multiple of {num_sublanes}, got {rows}."
        )
    if max_tile < num_sublanes or max_tile % num_sublanes != 0:
        raise ValueError(
            f"max_tile must be a positive multiple of {num_sublanes}, got {max_tile}."
        )
    for tile in range(min(rows, max_tile), num_sublanes - 1, -num_sublanes):
        if rows % tile == 0:
            return tile
    raise AssertionError(
        "Unreachable: rows and max_tile are positive multiples of num_sublanes."
    )


def can_run_qk_norm_rope_gate(
    rows: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    *,
    dtype_itemsize: int = 2,
    vmem_capacity_bytes: int,
    num_sublanes: int = 8,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = 0,
) -> bool:
    """Return True when ``qk_norm_rope_gate_rows`` meets alignment and VMEM bounds."""
    if min_pallas_qkv_bytes < 0:
        return False
    if num_sublanes < 8 or num_sublanes % 8 != 0:
        return False
    if rows <= 0 or num_heads <= 0 or num_kv_heads <= 0:
        return False
    if head_dim <= 0 or head_dim % 128 != 0:
        return False
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        return False
    if dtype_itemsize not in (2, 4):
        return False

    qkv_cols = (2 * num_heads + 2 * num_kv_heads) * head_dim
    if (
        min_pallas_qkv_bytes > 0
        and rows * qkv_cols * dtype_itemsize < min_pallas_qkv_bytes
    ):
        return True

    vmem_budget = _usable_vmem_budget(vmem_capacity_bytes, chip_version=chip_version)
    if rows >= num_sublanes and rows % num_sublanes == 0:
        double_buffered_min_bytes = _vmem_tile_bytes(
            num_sublanes,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            dtype_itemsize,
            buffer_count=2,
            num_sublanes=num_sublanes,
            stacked=True,
        )
        if double_buffered_min_bytes <= vmem_budget:
            return True
        if rows != num_sublanes:
            return False
        single_buffered_min_bytes = _vmem_tile_bytes(
            num_sublanes,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            dtype_itemsize,
            buffer_count=1,
            num_sublanes=num_sublanes,
            stacked=True,
        )
        return single_buffered_min_bytes <= vmem_budget

    if rows > _max_unaligned_rows(num_sublanes):
        return False
    whole_bytes = _vmem_tile_bytes(
        rows,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        dtype_itemsize,
        buffer_count=1,
        num_sublanes=num_sublanes,
        stacked=False,
    )
    return whole_bytes <= vmem_budget


def _rms_norm(x, weight, eps):
    variance = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * weight


def _prepare_rope_tables(cos_sin, rows: int, head_dim: int, rotary_dim: int):
    """Build full-width ``[rows, head_dim]`` cos and signed-sin tables once per tile."""
    half = rotary_dim // 2
    cos = cos_sin[:, :half]
    sin = cos_sin[:, half:rotary_dim]
    cos_rot = jnp.concatenate([cos, cos], axis=1)
    sin_rot = jnp.concatenate([-sin, sin], axis=1)
    if rotary_dim == head_dim:
        return cos_rot, sin_rot
    pass_cols = head_dim - rotary_dim
    cos_full = jnp.concatenate(
        [cos_rot, jnp.ones((rows, pass_cols), dtype=jnp.float32)], axis=1
    )
    sin_full = jnp.concatenate(
        [sin_rot, jnp.zeros((rows, pass_cols), dtype=jnp.float32)], axis=1
    )
    return cos_full, sin_full


def _apply_rope_tables(x, cos_full, sin_full, rotary_dim: int):
    half = rotary_dim // 2
    swapped_parts = [x[:, half:rotary_dim], x[:, :half]]
    if rotary_dim < x.shape[1]:
        swapped_parts.append(x[:, rotary_dim:])
    x_swapped = jnp.concatenate(swapped_parts, axis=1)
    return x * cos_full + x_swapped * sin_full


def _stack_norm_rope(
    head_slices: list[jax.Array],
    weight: jax.Array,
    cos_full: jax.Array,
    sin_full: jax.Array,
    *,
    rotary_dim: int,
    eps: float,
    out_dtype,
) -> jax.Array:
    """Normalize and rotate sublane-aligned head slices in one vectorized pass."""
    head_count = len(head_slices)
    if head_count == 1:
        normed = _rms_norm(head_slices[0].astype(jnp.float32), weight, eps)
        return _apply_rope_tables(normed, cos_full, sin_full, rotary_dim).astype(
            out_dtype
        )
    rows = head_slices[0].shape[0]
    stacked = jnp.concatenate(head_slices, axis=0).astype(jnp.float32)
    cos_tiled = jnp.tile(cos_full, (head_count, 1))
    sin_tiled = jnp.tile(sin_full, (head_count, 1))
    rotated = _apply_rope_tables(
        _rms_norm(stacked, weight, eps), cos_tiled, sin_tiled, rotary_dim
    ).astype(out_dtype)
    return jnp.concatenate(
        [rotated[h * rows : (h + 1) * rows, :] for h in range(head_count)],
        axis=1,
    )


def _kernel(
    qkv_ref,
    q_weight_ref,
    k_weight_ref,
    cos_sin_ref,
    q_out_ref,
    k_out_ref,
    v_out_ref,
    gate_out_ref,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    num_sublanes: int = 8,
):
    rows = qkv_ref.shape[0]
    cos_full, sin_full = _prepare_rope_tables(
        cos_sin_ref[...], rows, head_dim, rotary_dim
    )
    q_weight = q_weight_ref[...]
    k_weight = k_weight_ref[...]
    k_offset = num_heads * 2 * head_dim

    if rows % num_sublanes == 0:
        gate_slices = [
            qkv_ref[:, (2 * h + 1) * head_dim : (2 * h + 2) * head_dim]
            for h in range(num_heads)
        ]
        gate_out_ref[...] = (
            gate_slices[0] if num_heads == 1 else jnp.concatenate(gate_slices, axis=1)
        )
        q_slices = [
            qkv_ref[:, (2 * h) * head_dim : (2 * h + 1) * head_dim]
            for h in range(num_heads)
        ]
        q_out_ref[...] = _stack_norm_rope(
            q_slices,
            q_weight,
            cos_full,
            sin_full,
            rotary_dim=rotary_dim,
            eps=eps,
            out_dtype=q_out_ref.dtype,
        )
        k_slices = [
            qkv_ref[:, k_offset + h * head_dim : k_offset + (h + 1) * head_dim]
            for h in range(num_kv_heads)
        ]
        k_out_ref[...] = _stack_norm_rope(
            k_slices,
            k_weight,
            cos_full,
            sin_full,
            rotary_dim=rotary_dim,
            eps=eps,
            out_dtype=k_out_ref.dtype,
        )
    else:
        for head in range(num_heads):
            q_start = head * 2 * head_dim
            q = qkv_ref[:, q_start : q_start + head_dim].astype(jnp.float32)
            gate = qkv_ref[:, q_start + head_dim : q_start + 2 * head_dim]
            q = _apply_rope_tables(
                _rms_norm(q, q_weight, eps), cos_full, sin_full, rotary_dim
            )
            out_cols = slice(head * head_dim, (head + 1) * head_dim)
            q_out_ref[:, out_cols] = q.astype(q_out_ref.dtype)
            gate_out_ref[:, out_cols] = gate

        for head in range(num_kv_heads):
            in_cols = slice(
                k_offset + head * head_dim, k_offset + (head + 1) * head_dim
            )
            out_cols = slice(head * head_dim, (head + 1) * head_dim)
            k = qkv_ref[:, in_cols].astype(jnp.float32)
            k = _apply_rope_tables(
                _rms_norm(k, k_weight, eps), cos_full, sin_full, rotary_dim
            )
            k_out_ref[:, out_cols] = k.astype(k_out_ref.dtype)

    v_offset = k_offset + num_kv_heads * head_dim
    v_out_ref[...] = qkv_ref[:, v_offset:]


_STATIC_ARGNAMES = (
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "rotary_dim",
    "eps",
    "max_tile",
    "vmem_capacity_bytes",
    "num_sublanes",
    "chip_version",
    "min_pallas_qkv_bytes",
    "interpret",
)


def unfused_qk_norm_rope_gate_rows(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
):
    """Plain-JAX QKV+gate split, per-head Q/K RMSNorm, and partial NeoX RoPE.

    Emits no ``pallas_call`` so ``jax.export`` lowers small Scoped VMEM-resident
    buckets inline into surrounding XLA fusions without a custom-call barrier.
    """
    if qkv.ndim != 2:
        raise ValueError(f"qkv must be 2D, got shape {qkv.shape}.")
    if qkv.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"qkv dtype must be bfloat16 or float32, got {qkv.dtype}.")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and positive, got {eps}.")
    if num_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "num_heads and num_kv_heads must be positive, "
            f"got {num_heads}, {num_kv_heads}."
        )
    if head_dim <= 0 or head_dim % 128 != 0:
        raise ValueError(
            f"head_dim must be a positive multiple of 128, got {head_dim}."
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            "rotary_dim must be a positive even integer <= head_dim, "
            f"got {rotary_dim} / {head_dim}."
        )
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(
            f"q_weight and k_weight must have shape ({head_dim},), "
            f"got {q_weight.shape} and {k_weight.shape}."
        )
    rows, total_cols = qkv.shape
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}.")
    q_cols = num_heads * head_dim
    kv_cols = num_kv_heads * head_dim
    expected_cols = 2 * q_cols + 2 * kv_cols
    if total_cols != expected_cols:
        raise ValueError(
            f"qkv width {total_cols} != expected {expected_cols} for "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        )
    if cos_sin.shape != (rows, rotary_dim):
        raise ValueError(
            f"cos_sin shape {cos_sin.shape} != expected {(rows, rotary_dim)}."
        )

    q_gate_cols = 2 * q_cols
    q_and_gate = qkv[:, :q_gate_cols].reshape(rows, num_heads, 2 * head_dim)
    q = q_and_gate[:, :, :head_dim]
    gate = q_and_gate[:, :, head_dim:]
    k_heads = qkv[:, q_gate_cols : q_gate_cols + kv_cols].reshape(
        rows, num_kv_heads, head_dim
    )
    v = qkv[:, q_gate_cols + kv_cols :]
    q_weight_f32 = q_weight.astype(jnp.float32)
    k_weight_f32 = k_weight.astype(jnp.float32)
    half = rotary_dim // 2
    cos = cos_sin[:, :half].astype(jnp.float32)[:, None, :]
    sin = cos_sin[:, half:rotary_dim].astype(jnp.float32)[:, None, :]

    def _norm_and_rope(x, weight_f32):
        normed = _rms_norm(x.astype(jnp.float32), weight_f32, eps)
        x1 = normed[..., :half]
        x2 = normed[..., half:rotary_dim]
        parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        if rotary_dim < head_dim:
            parts.append(normed[..., rotary_dim:])
        return jnp.concatenate(parts, axis=-1).astype(qkv.dtype)

    return (
        _norm_and_rope(q, q_weight_f32).reshape(rows, q_cols),
        _norm_and_rope(k_heads, k_weight_f32).reshape(rows, kv_cols),
        v,
        gate.reshape(rows, q_cols),
    )


@functools.partial(jax.jit, static_argnames=_STATIC_ARGNAMES)
def qk_norm_rope_gate(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    max_tile: int = DEFAULT_MAX_TILE,
    vmem_capacity_bytes: int | None = None,
    num_sublanes: int | None = None,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = 0,
    interpret: bool = False,
):
    """Gather RoPE rows by position, then run ``qk_norm_rope_gate_rows``."""
    cos_sin = jnp.take(cos_sin_cache, positions.astype(jnp.int32), axis=0)
    return qk_norm_rope_gate_rows(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        max_tile=max_tile,
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_sublanes=num_sublanes,
        chip_version=chip_version,
        min_pallas_qkv_bytes=min_pallas_qkv_bytes,
        interpret=interpret,
    )


@functools.partial(jax.jit, static_argnames=_STATIC_ARGNAMES)
def qk_norm_rope_gate_rows(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    max_tile: int = DEFAULT_MAX_TILE,
    vmem_capacity_bytes: int | None = None,
    num_sublanes: int | None = None,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = 0,
    interpret: bool = False,
):
    """Run fused split, Q/K RMSNorm, and NeoX RoPE on pre-gathered RoPE rows.

    ``qkv`` has shape ``[rows, (2 * num_heads + 2 * num_kv_heads) * head_dim]``,
    ``q_weight`` and ``k_weight`` ``[head_dim]`` (already including ``+ 1``), and
    ``cos_sin`` ``[rows, rotary_dim]``. Returns ``(q, k, v, gate)`` in ``qkv.dtype``.
    """
    if qkv.ndim != 2:
        raise ValueError(f"qkv must be 2D, got shape {qkv.shape}.")
    if qkv.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(f"qkv dtype must be bfloat16 or float32, got {qkv.dtype}.")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and positive, got {eps}.")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(
            f"q_weight and k_weight must have shape ({head_dim},), "
            f"got {q_weight.shape} and {k_weight.shape}."
        )

    dtype_itemsize = jnp.dtype(qkv.dtype).itemsize
    on_tpu_device = not interpret and pltpu.is_tpu_device()
    tpu_info = pltpu.get_tpu_info() if on_tpu_device else None
    has_tpu_vmem = vmem_capacity_bytes is not None or tpu_info is not None
    if vmem_capacity_bytes is not None:
        vmem_cap = vmem_capacity_bytes
    elif tpu_info is not None:
        vmem_cap = tpu_info.vmem_capacity_bytes
    else:
        vmem_cap = _INTERPRET_VMEM_CAPACITY_BYTES
    resolved_sublanes = (
        num_sublanes
        if num_sublanes is not None
        else (tpu_info.num_sublanes if tpu_info is not None else 8)
    )
    if resolved_sublanes < 8 or resolved_sublanes % 8 != 0:
        raise ValueError(
            f"num_sublanes must be a positive multiple of 8, got {resolved_sublanes}."
        )
    if max_tile < resolved_sublanes or max_tile % resolved_sublanes != 0:
        raise ValueError(
            f"max_tile must be a positive multiple of {resolved_sublanes}, "
            f"got {max_tile}."
        )
    resolved_chip = (
        chip_version
        if chip_version is not None
        else (tpu_info.chip_version if tpu_info is not None else None)
    )

    rows, total_cols = qkv.shape
    q_cols = num_heads * head_dim
    kv_cols = num_kv_heads * head_dim
    expected_cols = 2 * q_cols + 2 * kv_cols
    if total_cols != expected_cols:
        raise ValueError(
            f"qkv width {total_cols} != expected {expected_cols} for "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        )
    if cos_sin.shape != (rows, rotary_dim):
        raise ValueError(
            f"cos_sin shape {cos_sin.shape} != expected {(rows, rotary_dim)}."
        )

    if min_pallas_qkv_bytes < 0:
        raise ValueError(
            f"min_pallas_qkv_bytes must be non-negative, got {min_pallas_qkv_bytes}."
        )
    if not can_run_qk_norm_rope_gate(
        rows,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        dtype_itemsize=dtype_itemsize,
        vmem_capacity_bytes=vmem_cap,
        num_sublanes=resolved_sublanes,
        chip_version=resolved_chip,
        min_pallas_qkv_bytes=min_pallas_qkv_bytes,
    ):
        raise ValueError(
            "qk_norm_rope_gate alignment or VMEM precondition not met: "
            f"rows={rows}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, rotary_dim={rotary_dim}."
        )

    if (
        min_pallas_qkv_bytes > 0
        and rows * total_cols * dtype_itemsize < min_pallas_qkv_bytes
    ):
        return unfused_qk_norm_rope_gate_rows(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
        )

    kernel = functools.partial(
        _kernel,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        num_sublanes=resolved_sublanes,
    )
    out_shape = (
        jax.ShapeDtypeStruct((rows, q_cols), qkv.dtype),
        jax.ShapeDtypeStruct((rows, kv_cols), qkv.dtype),
        jax.ShapeDtypeStruct((rows, kv_cols), qkv.dtype),
        jax.ShapeDtypeStruct((rows, q_cols), qkv.dtype),
    )
    q_w = q_weight.reshape(1, head_dim).astype(jnp.float32)
    k_w = k_weight.reshape(1, head_dim).astype(jnp.float32)
    cs = cos_sin.astype(jnp.float32)

    if rows >= resolved_sublanes and rows % resolved_sublanes == 0:
        vmem_budget = _usable_vmem_budget(vmem_cap, chip_version=resolved_chip)
        effective_max_tile = resolved_sublanes
        for candidate in range(
            min(rows, max_tile), resolved_sublanes - 1, -resolved_sublanes
        ):
            if (
                _vmem_tile_bytes(
                    candidate,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rotary_dim,
                    dtype_itemsize,
                    buffer_count=1 if candidate == rows else 2,
                    num_sublanes=resolved_sublanes,
                    stacked=True,
                )
                <= vmem_budget
            ):
                effective_max_tile = candidate
                break
        tile_rows = legal_row_tile(
            rows, effective_max_tile, num_sublanes=resolved_sublanes
        )
        tile_vmem_bytes = _vmem_tile_bytes(
            tile_rows,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            dtype_itemsize,
            buffer_count=1 if tile_rows == rows else 2,
            num_sublanes=resolved_sublanes,
            stacked=True,
        )
        vmem_limit = (
            scoped_vmem_limit_bytes(
                tile_vmem_bytes, vmem_cap, chip_version=resolved_chip
            )
            if not interpret and has_tpu_vmem
            else None
        )
        return pl.pallas_call(
            kernel,
            grid=(rows // tile_rows,),
            in_specs=[
                pl.BlockSpec((tile_rows, total_cols), lambda i: (i, 0)),
                pl.BlockSpec((1, head_dim), lambda i: (0, 0)),
                pl.BlockSpec((1, head_dim), lambda i: (0, 0)),
                pl.BlockSpec((tile_rows, rotary_dim), lambda i: (i, 0)),
            ],
            out_specs=(
                pl.BlockSpec((tile_rows, q_cols), lambda i: (i, 0)),
                pl.BlockSpec((tile_rows, kv_cols), lambda i: (i, 0)),
                pl.BlockSpec((tile_rows, kv_cols), lambda i: (i, 0)),
                pl.BlockSpec((tile_rows, q_cols), lambda i: (i, 0)),
            ),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit,
            ),
            name=f"qk_norm_rope_gate_t{tile_rows}",
            interpret=pltpu.InterpretParams() if interpret else False,
        )(qkv, q_w, k_w, cs)

    whole_vmem_bytes = _vmem_tile_bytes(
        rows,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        dtype_itemsize,
        buffer_count=1,
        num_sublanes=resolved_sublanes,
        stacked=False,
    )
    whole_vmem_limit = (
        scoped_vmem_limit_bytes(whole_vmem_bytes, vmem_cap, chip_version=resolved_chip)
        if not interpret and has_tpu_vmem
        else None
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[vmem] * 4,
        out_specs=(vmem, vmem, vmem, vmem),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=whole_vmem_limit),
        name="qk_norm_rope_gate_whole",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(qkv, q_w, k_w, cs)


def reference(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
):
    """vLLM's eager path in plain JAX: cos/sin gather, qkv/gate split, per-head
    norm in f32, NeoX rope on the first rotary_dim dims."""
    cos_sin = jnp.take(cos_sin_cache, positions.astype(jnp.int32), axis=0)
    rows = qkv.shape[0]
    q_gate_cols = num_heads * 2 * head_dim
    kv_cols = num_kv_heads * head_dim
    q_gate = qkv[:, :q_gate_cols]
    k = qkv[:, q_gate_cols : q_gate_cols + kv_cols]
    v = qkv[:, q_gate_cols + kv_cols :]
    q_and_gate = q_gate.reshape(rows, num_heads, 2 * head_dim)
    q, gate = q_and_gate[:, :, :head_dim], q_and_gate[:, :, head_dim:]
    k_heads = k.reshape(rows, num_kv_heads, head_dim)

    def norm(x, weight):
        x = x.astype(jnp.float32)
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)

    half = rotary_dim // 2
    cos = cos_sin[:, :half].astype(jnp.float32)[:, None, :]
    sin = cos_sin[:, half:rotary_dim].astype(jnp.float32)[:, None, :]

    def rope(x):
        x1, x2 = x[..., :half], x[..., half:rotary_dim]
        parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        if rotary_dim < head_dim:
            parts.append(x[..., rotary_dim:])
        return jnp.concatenate(parts, -1)

    q = rope(norm(q, q_weight)).astype(qkv.dtype)
    k_heads = rope(norm(k_heads, k_weight)).astype(qkv.dtype)
    return (
        q.reshape(rows, num_heads * head_dim),
        k_heads.reshape(rows, kv_cols),
        v,
        gate.reshape(rows, num_heads * head_dim),
    )

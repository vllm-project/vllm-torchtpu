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
"""Fused QKV+gate split, Q/K RMSNorm, and NeoX RoPE for Qwen3-Next/Qwen3.5 on TPU.

``TPU_FUSED_QK_NORM_ROPE=1`` runs ``qkv_proj`` output splitting, per-head Q/K
RMSNorm, partial NeoX RoPE, and gate extraction as one ``pallas.jax_op`` per
attention layer (``kernels/qk_norm_rope_gate``): row-tiled Pallas at or above
``MIN_PALLAS_QKV_BYTES`` (``1 MiB`` of ``qkv``, passing ``qkv`` whole to avoid
strided 2D column-slice HBM copies) and inline XLA ops below ``1 MiB``.
"""

import functools
import hashlib
import math
import threading
from typing import Any

import jax
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qk_norm_rope_gate import (
    MIN_PALLAS_QKV_BYTES,
    can_run_qk_norm_rope_gate,
    qk_norm_rope_gate_rows,
)

_ops_lock = threading.Lock()
_ops_by_geometry: dict[
    tuple[
        int,
        int,
        int,
        int,
        float,
        int | None,
        int | None,
        pltpu.ChipVersion | None,
        int,
    ],
    Any,
] = {}


def _eps_op_tag(eps_f: float) -> str:
    return hashlib.sha256(eps_f.hex().encode()).hexdigest()[:8]


def _run_kernel(
    qkv: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    cos_sin: jax.Array,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    vmem_capacity_bytes: int | None,
    num_sublanes: int | None,
    chip_version: pltpu.ChipVersion | None,
    min_pallas_qkv_bytes: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # cos_sin holds the rope rows for these tokens, gathered by position in
    # torch before the call.
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
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_sublanes=num_sublanes,
        chip_version=chip_version,
        min_pallas_qkv_bytes=min_pallas_qkv_bytes,
    )


def _validate_geometry(
    num_heads: int, num_kv_heads: int, head_dim: int, rotary_dim: int, eps: float
) -> float:
    eps_f = float(eps)
    if not (math.isfinite(eps_f) and eps_f > 0):
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
    return eps_f


def _get_op(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    vmem_capacity_bytes: int | None = None,
    num_sublanes: int | None = None,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = MIN_PALLAS_QKV_BYTES,
):
    """Return the torch op for one geometry. ``prebuild()`` builds it at model
    construction; the traced forward only looks it up, because Dynamo cannot
    register an op while tracing."""
    eps_f = _validate_geometry(num_heads, num_kv_heads, head_dim, rotary_dim, eps)
    vmem_key = None if vmem_capacity_bytes is None else int(vmem_capacity_bytes)
    sublanes_key = None if num_sublanes is None else int(num_sublanes)
    min_bytes_key = int(min_pallas_qkv_bytes)
    if min_bytes_key < 0:
        raise ValueError(
            f"min_pallas_qkv_bytes must be non-negative, got {min_pallas_qkv_bytes}."
        )
    if sublanes_key is not None and (sublanes_key < 8 or sublanes_key % 8 != 0):
        raise ValueError(
            f"num_sublanes must be a positive multiple of 8, got {num_sublanes}."
        )
    key = (
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps_f,
        vmem_key,
        sublanes_key,
        chip_version,
        min_bytes_key,
    )
    op = _ops_by_geometry.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _ops_by_geometry.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        vmem_tag = "auto" if vmem_key is None else str(vmem_key)
        sublanes_tag = "auto" if sublanes_key is None else str(sublanes_key)
        chip_tag = "auto" if chip_version is None else chip_version.value
        kernel_fn = functools.partial(
            _run_kernel,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps_f,
            vmem_capacity_bytes=vmem_key,
            num_sublanes=sublanes_key,
            chip_version=chip_version,
            min_pallas_qkv_bytes=min_bytes_key,
        )
        op = pallas.jax_op(
            f"pallas::qk_norm_rope_gate_h{num_heads}_kv{num_kv_heads}"
            f"_d{head_dim}_r{rotary_dim}_e{_eps_op_tag(eps_f)}"
            f"_v{vmem_tag}_s{sublanes_tag}_c{chip_tag}_m{min_bytes_key}",
            kernel_fn,
        )

        def _fake(qkv, q_weight, k_weight, cos_sin):
            del q_weight, k_weight, cos_sin
            rows = qkv.shape[0]
            return (
                qkv.new_empty(rows, num_heads * head_dim),
                qkv.new_empty(rows, num_kv_heads * head_dim),
                qkv.new_empty(rows, num_kv_heads * head_dim),
                qkv.new_empty(rows, num_heads * head_dim),
            )

        op.register_fake(_fake)
        _ops_by_geometry[key] = op
        return op


def prebuild(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    vmem_capacity_bytes: int | None = None,
    num_sublanes: int | None = None,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = MIN_PALLAS_QKV_BYTES,
) -> None:
    """Register the op for this geometry outside any traced region."""
    _get_op(
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_sublanes=num_sublanes,
        chip_version=chip_version,
        min_pallas_qkv_bytes=min_pallas_qkv_bytes,
    )


def _is_tpu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "tpu"


def can_dispatch_fused_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor | None,
    cos_sin_cache: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    vmem_capacity_bytes: int,
    *,
    num_sublanes: int = 8,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = MIN_PALLAS_QKV_BYTES,
    check_rows: bool = True,
) -> bool:
    """Return True when ``qkv`` can run through ``fused_qk_rmsnorm_rope_gate``."""
    if min_pallas_qkv_bytes < 0:
        return False
    if not _is_tpu_tensor(qkv):
        return False
    if num_sublanes < 8 or num_sublanes % 8 != 0:
        return False
    if qkv.ndim != 2 or not qkv.is_contiguous():
        return False
    if qkv.dtype not in (torch.bfloat16, torch.float32):
        return False
    expected_cols = (2 * num_q_heads + 2 * num_kv_heads) * head_dim
    if qkv.shape[1] != expected_cols:
        return False
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32)
        or cos_sin_cache.device != qkv.device
    ):
        return False
    if (
        positions is None
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != qkv.device
    ):
        return False
    if positions.ndim == 1:
        if check_rows and positions.shape[0] != qkv.shape[0]:
            return False
    elif positions.ndim == 2:
        if positions.shape[0] != 3 or (
            check_rows and positions.shape[1] != qkv.shape[0]
        ):
            return False
    else:
        return False
    if not check_rows:
        return True
    return can_run_qk_norm_rope_gate(
        rows=qkv.shape[0],
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        dtype_itemsize=qkv.element_size(),
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_sublanes=num_sublanes,
        chip_version=chip_version,
        min_pallas_qkv_bytes=min_pallas_qkv_bytes,
    )


def fused_qk_rmsnorm_rope_gate(
    qkv: torch.Tensor,
    q_weight_plus_one: torch.Tensor,
    k_weight_plus_one: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    vmem_capacity_bytes: int | None = None,
    num_sublanes: int | None = None,
    chip_version: pltpu.ChipVersion | None = None,
    min_pallas_qkv_bytes: int = MIN_PALLAS_QKV_BYTES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return post-norm, post-RoPE ``(q, k)``, ``v``, and the pre-sigmoid ``gate``.

    ``positions`` is ``[rows]`` or, for text-only M-RoPE, ``[3, rows]`` with equal rows.
    """
    if not _is_tpu_tensor(qkv):
        raise ValueError(f"qkv must be on a TPU device, got device={qkv.device}.")
    if qkv.ndim != 2 or not qkv.is_contiguous():
        raise ValueError(
            f"qkv must be a contiguous 2D tensor, got shape={tuple(qkv.shape)} "
            f"contiguous={qkv.is_contiguous()}."
        )
    if qkv.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"qkv dtype must be bfloat16 or float32, got dtype={qkv.dtype}."
        )
    if tuple(q_weight_plus_one.shape) != (head_dim,) or tuple(
        k_weight_plus_one.shape
    ) != (head_dim,):
        raise ValueError(
            "q_weight_plus_one and k_weight_plus_one must have shape "
            f"({head_dim},), got {tuple(q_weight_plus_one.shape)} and "
            f"{tuple(k_weight_plus_one.shape)}."
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError(
            f"cos_sin_cache must be 2D with width {rotary_dim}, "
            f"got shape={tuple(cos_sin_cache.shape)}."
        )
    if positions.ndim == 1:
        token_positions = positions
    elif positions.ndim == 2 and positions.shape[0] == 3:
        token_positions = positions[0]
    else:
        raise ValueError(
            "positions must have shape [rows] or [3, rows], "
            f"got shape={tuple(positions.shape)}."
        )
    op = _get_op(
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_sublanes=num_sublanes,
        chip_version=chip_version,
        min_pallas_qkv_bytes=min_pallas_qkv_bytes,
    )
    cos_sin = cos_sin_cache.index_select(0, token_positions)
    return op(qkv, q_weight_plus_one, k_weight_plus_one, cos_sin)

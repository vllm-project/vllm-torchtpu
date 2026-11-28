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
"""Tensor layout contracts between Qwen4Exp attention stages."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Literal

import torch

# ── 1. Pure data types ─────────────────────────────────────────────────────


class Axis(enum.StrEnum):
    T = "T"
    H_KV = "H_kv"
    G = "G"
    D = "D"
    KV = "KV"


H_Q = (Axis.H_KV, Axis.G)
LaneAlign = int | Literal["page_size"]


@dataclass(frozen=True)
class TensorLayout:
    """Major-to-minor axes + TPU sublane packing and lane alignment."""

    axes: tuple[Axis | tuple[Axis, ...], ...]
    dtype: torch.dtype
    pack_sublane: bool = False  # splits axes[-2] -> (ceil(d/p), p), p = 4 // itemsize
    lane_align: LaneAlign = 1  # pads axes[-1] to multiple (128 or max(128, page_size))
    fold_lanes: bool = False  # folds axes[-1] > 128 into axes[-2] for 8-bit KV
    quantized: bool = False  # static KV scales already applied


@dataclass(frozen=True)
class QKVLayout:
    """Boundary contract between Step 1 (QKVProj) and Step 2 (CoreAttn)."""

    q: TensorLayout
    kv: TensorLayout | tuple[TensorLayout, TensorLayout]


# ── 2. Concrete attention layouts ──────────────────────────────────────────


def unfused_o(dtype: torch.dtype = torch.bfloat16) -> TensorLayout:
    return TensorLayout((Axis.T, H_Q, Axis.D), dtype)


def unfused_qkv(dtype: torch.dtype = torch.bfloat16) -> QKVLayout:
    k = TensorLayout((Axis.T, Axis.H_KV, Axis.D), dtype)
    return QKVLayout(q=unfused_o(dtype), kv=(k, k))


def packed_q(dtype: torch.dtype = torch.bfloat16) -> TensorLayout:
    return TensorLayout(
        (Axis.H_KV, Axis.T, Axis.G, Axis.D), dtype, pack_sublane=True, lane_align=128
    )


def kv_nhd(
    dtype: torch.dtype = torch.float8_e4m3fn,
    *,
    fold_lanes: bool = True,
    quantized: bool = True,
) -> TensorLayout:
    return TensorLayout(
        (Axis.T, (Axis.H_KV, Axis.KV), Axis.D),
        dtype,
        pack_sublane=True,
        lane_align=128,
        fold_lanes=fold_lanes,
        quantized=quantized,
    )


def kv_hnd(
    dtype: torch.dtype = torch.float8_e4m3fn, *, quantized: bool = True
) -> TensorLayout:
    return TensorLayout(
        ((Axis.H_KV, Axis.KV), Axis.D, Axis.T),
        dtype,
        pack_sublane=True,
        lane_align="page_size",
        quantized=quantized,
    )


# ── 3. Verification mechanics ──────────────────────────────────────────────


def expected_shape(
    layout: TensorLayout, *, T: int, H_kv: int, G: int, D: int, page_size: int = 128
) -> tuple[int, ...]:
    vals = {Axis.T: T, Axis.H_KV: H_kv, Axis.G: G, Axis.D: D, Axis.KV: 2}
    sizes = [
        math.prod(vals[a] for a in (g if isinstance(g, tuple) else (g,)))
        for g in layout.axes
    ]
    p = 4 // layout.dtype.itemsize
    if layout.fold_lanes and p >= 4 and sizes[-1] > 128:
        fold = math.ceil(sizes[-1] / 128)
        while fold > 1 and sizes[-1] % (fold * 128) != 0:
            fold //= 2
        sizes[-2] *= fold
        sizes[-1] //= fold
    align = (
        max(128, page_size) if layout.lane_align == "page_size" else layout.lane_align
    )
    sizes[-1] = math.ceil(sizes[-1] / align) * align
    if not layout.pack_sublane:
        return tuple(sizes)
    return (*sizes[:-2], math.ceil(sizes[-2] / p), p, sizes[-1])


def verify_tensor(
    layout: TensorLayout,
    tensor: torch.Tensor,
    *,
    T: int,
    H_kv: int,
    G: int,
    D: int,
    page_size: int = 128,
) -> None:
    shape = expected_shape(layout, T=T, H_kv=H_kv, G=G, D=D, page_size=page_size)
    if tuple(tensor.shape) != shape or tensor.dtype != layout.dtype:
        raise ValueError(
            f"Expected {shape} {layout.dtype}, "
            f"got {tuple(tensor.shape)} {tensor.dtype}."
        )


def verify_qkv(
    layout: QKVLayout,
    q: torch.Tensor,
    kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    T: int,
    H_kv: int,
    G: int,
    D: int,
    page_size: int = 128,
) -> None:
    verify_tensor(layout.q, q, T=T, H_kv=H_kv, G=G, D=D, page_size=page_size)
    if isinstance(layout.kv, tuple):
        if not isinstance(kv, tuple) or len(kv) != 2:
            raise ValueError("Expected separate (k, v) tensors.")
        verify_tensor(
            layout.kv[0], kv[0], T=T, H_kv=H_kv, G=G, D=D, page_size=page_size
        )
        verify_tensor(
            layout.kv[1], kv[1], T=T, H_kv=H_kv, G=G, D=D, page_size=page_size
        )
    else:
        if not isinstance(kv, torch.Tensor):
            raise ValueError("Expected packed kv tensor.")
        verify_tensor(layout.kv, kv, T=T, H_kv=H_kv, G=G, D=D, page_size=page_size)

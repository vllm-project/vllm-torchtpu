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
"""Pure-Python layout math for quantized GDN recurrent state in HBM/VMEM."""

from __future__ import annotations

GDN_SSM_STATE_QUANT_CHOICES = ("", "int8_block16")
INT8_BLOCK16_GROUP_SIZE = 16
INT8_PER_U32 = 4


def gdn_ssm_quant_q_and_scale_u32_rows(
    head_k_dim: int, quant_mode: str = ""
) -> tuple[int, int] | None:
    """Return ``(q_u32_rows, scale_u32_rows)`` per ``(head, v_lane)`` for SSM state.

    Requires ``head_k_dim % 128 == 0`` so both the packed INT8 value tile
    (``head_k_dim // 4`` uint32 rows) and the FP32 block-16 scale tile
    (``head_k_dim // 16`` uint32 rows) are multiples of the 8-sublane VREG
    height.
    """
    if not quant_mode:
        return None
    if quant_mode not in GDN_SSM_STATE_QUANT_CHOICES:
        raise ValueError(
            f"Unsupported GDN SSM state quant mode {quant_mode!r}; "
            f"expected one of {GDN_SSM_STATE_QUANT_CHOICES}."
        )
    if head_k_dim <= 0 or head_k_dim % 128 != 0:
        raise ValueError(
            f"GDN SSM state quant {quant_mode!r} requires head_k_dim "
            f"divisible by 128; got {head_k_dim}."
        )
    return head_k_dim // INT8_PER_U32, head_k_dim // INT8_BLOCK16_GROUP_SIZE


def gdn_ssm_quant_u32_rows(head_k_dim: int, quant_mode: str = "") -> int | None:
    """Return total uint32 rows per ``(head, v_lane)`` for quantized SSM state."""
    pair = gdn_ssm_quant_q_and_scale_u32_rows(head_k_dim, quant_mode)
    if pair is None:
        return None
    return pair[0] + pair[1]


def gdn_ssm_state_elem_bytes(
    unquantized_itemsize: int, head_k_dim: int = 128, quant_mode: str = ""
) -> float:
    """Effective bytes per logical recurrent element (1.25 for int8_block16)."""
    u32_rows = gdn_ssm_quant_u32_rows(head_k_dim, quant_mode)
    if u32_rows is None:
        return float(unquantized_itemsize)
    return (u32_rows * 4.0) / float(head_k_dim)

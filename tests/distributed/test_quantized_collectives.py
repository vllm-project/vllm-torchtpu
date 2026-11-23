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
"""Unit tests for distributed quantized collective helpers."""

from typing import TYPE_CHECKING, cast

import pytest
import torch

from vllm_torchtpu.distributed.quantized_collectives import (
    _MIN_QUANTIZED_ALL_GATHER_NDIM as MIN_QUANTIZED_ALL_GATHER_NDIM,
)
from vllm_torchtpu.distributed.quantized_collectives import quantized_all_gather
from vllm_torchtpu.layers.core.quantization import dequantize_tensor, quantize_tensor

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

_FP8_E4M3_FINFO = torch.finfo(torch.float8_e4m3fn)
# FP8 e4m3fn unit roundoff (half of one ULP: eps / 2).
FP8_E4M3_UNIT_ROUNDOFF = float(_FP8_E4M3_FINFO.eps) / 2.0
# Absolute error floor per unit of row max: half the FP8 subnormal spacing
# divided by the FP8 max value.
FP8_E4M3_SUBNORMAL_ERROR_PER_ROW_MAX = (
    float(_FP8_E4M3_FINFO.smallest_normal)
    * float(_FP8_E4M3_FINFO.eps)
    / (2.0 * float(_FP8_E4M3_FINFO.max))
)


def fp8_round_trip_tol(x: torch.Tensor) -> tuple[float, float]:
    """Derive analytical (rtol, atol) for per-row FP8 e4m3fn round-trip.

    Accounts for FP8 quantization unit roundoff plus three input-dtype rounding
    sites (1.0 / scale, x * scale_inv, and the final cast in dequantize_tensor),
    and both FP8 and output-dtype subnormal floors.
    """
    out_info = torch.finfo(x.dtype)
    out_eps = float(out_info.eps)
    rtol = (1.0 + FP8_E4M3_UNIT_ROUNDOFF) * (1.0 + out_eps / 2.0) ** 3 - 1.0
    max_abs = float(x.abs().max().item()) if x.numel() else 0.0
    atol = (
        FP8_E4M3_SUBNORMAL_ERROR_PER_ROW_MAX * max_abs * (1.0 + out_eps)
        + float(out_info.smallest_normal) * out_eps / 2.0
    )
    return rtol, atol


class _RecordingGroup:
    """Stand-in collective group recording all_gather shapes, dtypes, and dims."""

    def __init__(self, world_size: int) -> None:
        """Store simulated world size and initialize call log."""
        self.world_size = world_size
        self.calls: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Record call metadata and repeat tensor across simulated ranks."""
        self.calls.append((tuple(tensor.shape), tensor.dtype, dim))
        return torch.cat([tensor] * self.world_size, dim=dim)


class _DropScaleGroup:
    """Stand-in collective group that replaces gathered float32 scales with ones."""

    def __init__(self, world_size: int) -> None:
        """Store simulated world size."""
        self.world_size = world_size

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Concatenate across ranks and overwrite float32 scales with ones."""
        gathered = torch.cat([tensor] * self.world_size, dim=dim)
        if gathered.dtype == torch.float32:
            return torch.ones_like(gathered)
        return gathered


def _as_group(group: _RecordingGroup | _DropScaleGroup) -> "GroupCoordinator":
    """Cast stand-in group to GroupCoordinator for strict typing."""
    return cast("GroupCoordinator", group)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantized_all_gather_matches_local_round_trip(dtype: torch.dtype) -> None:
    """Gathered rows match local FP8 quantize/dequantize repeated across ranks."""
    torch.manual_seed(0)
    rows_per_rank = 8
    hidden = 1024
    world_size = 2
    x = torch.randn(rows_per_rank, hidden, dtype=dtype)
    group = _RecordingGroup(world_size=world_size)
    out = quantized_all_gather(x, _as_group(group))
    quantized, scale = quantize_tensor(x, quant_dtype=torch.float8_e4m3fn, axis=-1)
    expected_local = dequantize_tensor(quantized, scale, axis=-1, out_dtype=dtype)
    expected = expected_local.repeat(group.world_size, 1)
    assert out.dtype == dtype
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantized_all_gather_round_trip_analytic_bounds(dtype: torch.dtype) -> None:
    """Round-trip stays within analytical FP8 bounds and keeps zero rows finite."""
    torch.manual_seed(7)
    rows_per_rank = 8
    hidden = 64
    world_size = 2
    zero_row = 3
    # Trim inside rank 1 so padding removal exercises cross-rank narrowing.
    num_rows = world_size * rows_per_rank - zero_row
    x = (torch.randn(rows_per_rank, hidden) * 3.0).to(dtype)
    x[zero_row].zero_()
    group = _RecordingGroup(world_size=world_size)
    out = quantized_all_gather(x, _as_group(group), num_rows=num_rows)

    assert out.dtype == dtype
    assert out.shape == (num_rows, hidden)
    assert not out.isnan().any()
    assert torch.equal(out[zero_row], torch.zeros_like(out[zero_row]))
    assert torch.equal(
        out[zero_row + rows_per_rank], torch.zeros_like(out[zero_row + rows_per_rank])
    )
    ref = x.repeat(world_size, 1).narrow(0, 0, num_rows)
    rtol, atol = fp8_round_trip_tol(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_quantized_all_gather_bfloat16_exceeding_float16_max_stays_finite() -> None:
    """Bfloat16 rows above float16 max stay finite without float16 intermediates."""
    torch.manual_seed(7)
    rows_per_rank = 8
    hidden = 64
    world_size = 2
    x = (torch.randn(rows_per_rank, hidden) * 3.0).to(torch.bfloat16)
    # float16 max (65504.0) rounds to 65536.0 in bfloat16, which overflows if
    # any intermediate narrows to float16.
    x[0].fill_(float(torch.finfo(torch.float16).max))
    out = quantized_all_gather(x, _as_group(_RecordingGroup(world_size=world_size)))
    assert torch.isfinite(out[0]).all()
    rtol, _ = fp8_round_trip_tol(x[0])
    torch.testing.assert_close(out[0], x[0], rtol=rtol, atol=0.0)


def test_quantized_all_gather_detects_dropped_scale() -> None:
    """Replacing gathered float32 scales with ones violates analytical bounds."""
    torch.manual_seed(7)
    rows_per_rank = 8
    hidden = 64
    world_size = 2
    x = (torch.randn(rows_per_rank, hidden) * 3.0).to(torch.bfloat16)
    ref = x.repeat(world_size, 1)
    rtol, atol = fp8_round_trip_tol(x)
    corrupted = quantized_all_gather(
        x, _as_group(_DropScaleGroup(world_size=world_size))
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(corrupted, ref, rtol=rtol, atol=atol)


def test_quantized_all_gather_wire_format() -> None:
    """FP8 payload travels as int8 rows followed by float32 per-row scales."""
    rows_per_rank = 8
    hidden = 1024
    world_size = 2
    x = torch.randn(rows_per_rank, hidden, dtype=torch.bfloat16)
    group = _RecordingGroup(world_size=world_size)
    quantized_all_gather(x, _as_group(group))
    assert group.calls == [
        (tuple(x.shape), torch.int8, 0),
        ((x.shape[0], 1), torch.float32, 0),
    ]


def test_quantized_all_gather_trims_to_num_rows() -> None:
    """Passing num_rows trims gathered rows across rank boundaries."""
    torch.manual_seed(1)
    rows_per_rank = 8
    hidden = 1024
    world_size = 2
    # Cut through rank 1 rows: rows_per_rank < num_rows < world_size * rows_per_rank.
    num_rows = rows_per_rank + rows_per_rank // 2
    x = torch.randn(rows_per_rank, hidden, dtype=torch.bfloat16)
    full = quantized_all_gather(x, _as_group(_RecordingGroup(world_size=world_size)))
    trimmed = quantized_all_gather(
        x, _as_group(_RecordingGroup(world_size=world_size)), num_rows=num_rows
    )
    assert trimmed.shape == (num_rows, hidden)
    assert torch.equal(trimmed, full[:num_rows])


def test_quantized_all_gather_rejects_low_rank_input() -> None:
    """Reject 1-D inputs before issuing any collective call."""
    group = _RecordingGroup(world_size=2)
    low_rank = torch.randn(16, dtype=torch.bfloat16)
    expected_match = f"ndim >= {MIN_QUANTIZED_ALL_GATHER_NDIM}"
    with pytest.raises(ValueError, match=expected_match):
        quantized_all_gather(low_rank, _as_group(group))
    assert group.calls == []

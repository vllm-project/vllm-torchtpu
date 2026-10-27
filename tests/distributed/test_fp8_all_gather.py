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
"""Unit tests for the shared FP8 all-gather helper."""

import pytest
import torch

import vllm_torchtpu.distributed.fp8_all_gather as fp8_mod
import vllm_torchtpu.layers.core.quantization as quant_mod
from vllm_torchtpu.distributed.fp8_all_gather import fp8_all_gather

# Exact IEEE-754 / OFP8 quantization + dual out_dtype rounding error terms:
# float8_e4m3fn has 3 mantissa bits (normal half-ULP = 2**-4, subnormal half-ULP
# = 2**-10 in [-448, 448]), followed by scale.to(dtype) and q_deq * scale.
_FP8_E4M3_REL_HALF_ULP = 2.0**-4
_FP8_E4M3_SUBNORM_REL_FLOOR = 2.0**-10 / float(torch.finfo(torch.float8_e4m3fn).max)


def _fp8_round_trip_tol(x: torch.Tensor) -> tuple[float, float]:
    u_out = float(torch.finfo(x.dtype).eps) / 2.0
    two_roundings = (1.0 + u_out) ** 2
    rtol = (1.0 + _FP8_E4M3_REL_HALF_ULP) * two_roundings - 1.0
    amax = float(x.detach().float().abs().max())
    atol = amax * _FP8_E4M3_SUBNORM_REL_FLOOR * two_roundings
    return rtol, atol


@pytest.mark.parametrize("tp", [2, 8])
def test_fp8_all_gather_round_trip(tp: int) -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 64
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    out = fp8_all_gather(x, lambda t: torch.cat([t] * tp, dim=0))
    assert out.dtype == torch.bfloat16
    assert out.shape == (tp * rows, hidden)
    ref = torch.cat([x] * tp, dim=0)
    rtol, atol = _fp8_round_trip_tol(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_fp8_all_gather_subchannel_scales_on_wire() -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 1024
    block_size = 512
    tp = 2
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    log: list[tuple[tuple[int, ...], torch.dtype]] = []

    def logging_gather(t: torch.Tensor) -> torch.Tensor:
        log.append((tuple(t.shape), t.dtype))
        return torch.cat([t] * tp, dim=0)

    out = fp8_all_gather(x, logging_gather, block_size=block_size)
    assert log == [((8, 1024), torch.int8), ((8, 2), torch.bfloat16)]
    ref = torch.cat([x] * tp, dim=0)
    rtol, atol = _fp8_round_trip_tol(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_fp8_all_gather_zero_row_stays_zero() -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 64
    tp = 2
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    x[3].zero_()
    out = fp8_all_gather(x, lambda t: torch.cat([t] * tp, dim=0))
    assert not out.isnan().any()
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    assert torch.equal(out[3 + rows], torch.zeros_like(out[3 + rows]))


def test_fp8_all_gather_bf16_max_row_round_trips() -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 64
    tp = 2
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    x[0].fill_(65504.0)
    out = fp8_all_gather(x, lambda t: torch.cat([t] * tp, dim=0))
    rtol, _ = _fp8_round_trip_tol(x[0])
    torch.testing.assert_close(out[0], x[0], rtol=rtol, atol=0.0)


def test_fp8_all_gather_non_dividing_block_raises() -> None:
    torch.manual_seed(7)
    x = (torch.randn(8, 96) * 3.0).to(torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by block_size"):
        fp8_all_gather(x, lambda t: torch.cat([t, t], dim=0), block_size=64)


def test_fp8_all_gather_none_block_uses_row_scale() -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 64
    tp = 2
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    log: list[tuple[tuple[int, ...], torch.dtype]] = []

    def logging_gather(t: torch.Tensor) -> torch.Tensor:
        log.append((tuple(t.shape), t.dtype))
        return torch.cat([t] * tp, dim=0)

    out = fp8_all_gather(x, logging_gather, block_size=None)
    assert log[1] == ((8, 1), torch.bfloat16)
    rtol, atol = _fp8_round_trip_tol(x)
    torch.testing.assert_close(out, torch.cat([x] * tp, dim=0), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_all_gather_dequantizes_in_input_dtype_without_fp32_upcast(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_dequant(*args: object, **kwargs: object) -> None:
        raise AssertionError("no fp32 upcast")

    monkeypatch.setattr(quant_mod, "dequantize_tensor", fail_dequant, raising=False)
    monkeypatch.setattr(fp8_mod, "dequantize_tensor", fail_dequant, raising=False)

    torch.manual_seed(7)
    x = (torch.randn(8, 64) * 3.0).to(dtype)
    wire_log: list[tuple[tuple[int, ...], torch.dtype]] = []

    def logging_gather(t: torch.Tensor) -> torch.Tensor:
        wire_log.append((tuple(t.shape), t.dtype))
        return torch.cat([t, t], dim=0)

    out_row = fp8_all_gather(x, logging_gather, block_size=None)
    assert wire_log == [((8, 64), torch.int8), ((8, 1), dtype)]
    assert out_row.dtype == dtype
    ref = torch.cat([x, x], dim=0)
    rtol, atol = _fp8_round_trip_tol(x)
    torch.testing.assert_close(out_row, ref, rtol=rtol, atol=atol)

    wire_log.clear()
    out_blk = fp8_all_gather(x, logging_gather, block_size=32)
    assert wire_log == [((8, 64), torch.int8), ((8, 2), dtype)]
    assert out_blk.dtype == dtype
    torch.testing.assert_close(out_blk, ref, rtol=rtol, atol=atol)


def test_fp8_all_gather_negative_control_unscaled_exceeds_tolerance() -> None:
    torch.manual_seed(7)
    x = (torch.randn(8, 64) * 3.0).to(torch.bfloat16)
    ref = torch.cat([x, x], dim=0)
    rtol, atol = _fp8_round_trip_tol(x)

    def drop_scale_gather(t: torch.Tensor) -> torch.Tensor:
        gathered = torch.cat([t, t], dim=0)
        if gathered.dtype == torch.bfloat16:
            return torch.ones_like(gathered)
        return gathered

    corrupted = fp8_all_gather(x, drop_scale_gather, block_size=None)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(corrupted, ref, rtol=rtol, atol=atol)

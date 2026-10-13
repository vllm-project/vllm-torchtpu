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

from vllm_torchtpu.distributed.fp8_all_gather import fp8_all_gather


@pytest.mark.parametrize("tp", [2, 8])
def test_fp8_all_gather_round_trip(tp: int) -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 64
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    out = fp8_all_gather(x, lambda t: torch.cat([t] * tp, dim=0), block_size=512)
    assert out.dtype == torch.bfloat16
    assert out.shape == (tp * rows, hidden)
    ref = torch.cat([x] * tp, dim=0)
    torch.testing.assert_close(out, ref, rtol=1.0 / 32, atol=float(x.abs().max()) / 32)


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
    assert log == [((8, 1024), torch.int8), ((8, 2), torch.float32)]
    ref = torch.cat([x] * tp, dim=0)
    torch.testing.assert_close(out, ref, rtol=1.0 / 32, atol=float(x.abs().max()) / 32)


def test_fp8_all_gather_zero_row_stays_zero() -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 64
    tp = 2
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    x[3].zero_()
    out = fp8_all_gather(x, lambda t: torch.cat([t] * tp, dim=0), block_size=512)
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
    out = fp8_all_gather(x, lambda t: torch.cat([t] * tp, dim=0), block_size=512)
    torch.testing.assert_close(out[0], x[0], rtol=1.0 / 32, atol=0.0)


def test_fp8_all_gather_non_dividing_block_uses_row_scale() -> None:
    torch.manual_seed(7)
    rows = 8
    hidden = 96
    block_size = 64
    tp = 2
    x = (torch.randn(rows, hidden) * 3.0).to(torch.bfloat16)
    log: list[tuple[tuple[int, ...], torch.dtype]] = []

    def logging_gather(t: torch.Tensor) -> torch.Tensor:
        log.append((tuple(t.shape), t.dtype))
        return torch.cat([t] * tp, dim=0)

    out = fp8_all_gather(x, logging_gather, block_size=block_size)
    assert log[1] == ((8, 1), torch.float32)
    ref = torch.cat([x] * tp, dim=0)
    torch.testing.assert_close(out, ref, rtol=1.0 / 32, atol=float(x.abs().max()) / 32)


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
    assert log[1] == ((8, 1), torch.float32)
    torch.testing.assert_close(
        out, torch.cat([x] * tp, dim=0), rtol=1.0 / 32, atol=float(x.abs().max()) / 32
    )

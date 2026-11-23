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
"""fresh_buffer on TPU: a copy in a buffer of its own, exact for non-NaN float8."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_torchtpu.utils import fresh_buffer

FLOAT8_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e8m0fnu,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", FLOAT8_DTYPES, ids=str)
def test_fresh_buffer_keeps_every_float8_code(dtype):
    """The bfloat16 round trip keeps every non-NaN byte pattern; NaNs stay NaN
    (their sign or payload may change)."""
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)
    x = codes.to("tpu")
    out = fresh_buffer(x)
    assert out.dtype == dtype
    assert out.is_contiguous()
    got = out.cpu()
    nan = torch.isnan(codes.float())
    assert torch.equal(got.view(torch.uint8)[~nan], codes.view(torch.uint8)[~nan])
    assert torch.isnan(got.float()[nan]).all()


class _RecordOps(TorchDispatchMode):
    """Records the name of every aten op dispatched while active."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket))
        return func(*args, **(kwargs or {}))


@pytest.mark.tpu_test
def test_fresh_buffer_float8_never_adds():
    """Float8 never reaches `aten.add`, which torch_tpu rejects for float8 after
    0.2.0, so a revert to `+ 0` fails here even on builds that still have it.
    The input is a transposed view, like the FP8 linear weight in `fp8.py`."""
    w = torch.randn(8, 4).to(torch.float8_e4m3fn)
    x = w.to("tpu").t()
    with _RecordOps() as rec:
        out = fresh_buffer(x)
    assert "aten.add" not in rec.ops, rec.ops
    assert out.shape == (4, 8)
    assert out.is_contiguous()
    want = w.t().contiguous().view(torch.uint8)
    assert torch.equal(out.cpu().view(torch.uint8), want)


@pytest.mark.tpu_test
def test_fresh_buffer_bfloat16_on_tpu():
    x = torch.randn(4, 8, dtype=torch.bfloat16).to("tpu")
    out = fresh_buffer(x[:, :4])
    assert out.shape == (4, 4)
    assert out.is_contiguous()
    assert torch.equal(out.cpu(), x[:, :4].cpu())

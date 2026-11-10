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
"""The static per-tensor KV quantizer in both modes of
quantized_matmul.util.RECIPROCAL_SCALE: by default main's arithmetic (divide
by the scale), with the flag the reciprocal multiply and the scale-1.0 clamp
in the tensor's own dtype. Each is checked bit for bit against its formula
on CPU, on values where the two formulas round to different FP8 values."""

import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.layers.core import quantization

pytestmark = pytest.mark.cpu_test

FP8 = torch.float8_e4m3fn
INFO = torch.finfo(FP8)


@pytest.fixture
def mode(monkeypatch):
    def set_mode(reciprocal):
        monkeypatch.setattr(util, "RECIPROCAL_SCALE", reciprocal)

    return set_mode


def _tensor(scale):
    """Random values, values past and near the FP8 maximum, and FP8 rounding
    midpoints times the scale: dividing such a value by the scale and
    multiplying it by the scale's reciprocal can land on either side of the
    midpoint."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((8, 512), generator=generator) * 40.0
    x[0, :4] = torch.tensor([600.0, -600.0, 447.0, 1e-4])
    values = torch.arange(1, 127, dtype=torch.uint8).view(FP8).to(torch.float32)
    midpoints = (values[:-1] + values[1:]) / 2 * torch.tensor(scale)
    row = torch.zeros(512)
    row[: midpoints.numel()] = midpoints
    row[256 : 256 + midpoints.numel()] = -midpoints
    return torch.cat([x, row[None]])


def _main(x, scale):
    q = torch.clamp(x.to(torch.float32) / scale, INFO.min, INFO.max)
    return q.to(FP8)


def _reciprocal(x, scale):
    if scale == 1.0:
        return torch.clamp(x, INFO.min, INFO.max).to(FP8)
    q = torch.clamp(x.to(torch.float32) * (1.0 / scale), INFO.min, INFO.max)
    return q.to(FP8)


def _bits(q):
    return q.view(torch.uint8)


SCALES = [0.37, 1.7, 1.0]


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("reciprocal", [False, True], ids=["main", "reciprocal"])
def test_quantize_kv_matches_its_mode(mode, reciprocal, scale):
    mode(reciprocal)
    x = _tensor(scale)
    want = (_reciprocal if reciprocal else _main)(x, scale)
    key, value = quantization.quantize_kv(FP8, x, x.clone(), scale, scale)
    assert value is not None
    assert torch.equal(_bits(key), _bits(want))
    assert torch.equal(_bits(value), _bits(want))


@pytest.mark.parametrize("scale", SCALES[:2])
def test_the_two_formulas_differ_on_this_tensor(scale):
    """Without values where they disagree, the test of each mode could not
    tell which branch ran."""
    x = _tensor(scale)
    assert not torch.equal(_bits(_main(x, scale)), _bits(_reciprocal(x, scale)))


def test_the_default_is_mains_arithmetic():
    assert util.RECIPROCAL_SCALE is False

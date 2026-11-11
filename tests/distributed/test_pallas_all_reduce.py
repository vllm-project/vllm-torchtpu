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
"""pallas_all_reduce: the config switch and the unarmed fallback."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.kernels.collectives import tp_all_reduce


def _config(additional):
    return SimpleNamespace(additional_config=additional)


def test_off_by_default():
    assert pallas_all_reduce.max_bytes_from_config(_config({})) == 0
    assert pallas_all_reduce.max_bytes_from_config(_config(None)) == 0
    assert not pallas_all_reduce.arm_from_config(_config({}), torch.device("cpu"))


def test_max_bytes_capped_at_kernel_limit():
    key = pallas_all_reduce.MAX_BYTES_KEY
    assert pallas_all_reduce.max_bytes_from_config(_config({key: 524288})) == 524288
    assert (
        pallas_all_reduce.max_bytes_from_config(_config({key: 1 << 30}))
        == tp_all_reduce.MAX_VIEW_BYTES
    )


@pytest.mark.parametrize("bad", [-1, "512", 1.5, True])
def test_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        pallas_all_reduce.max_bytes_from_config(
            _config({pallas_all_reduce.MAX_BYTES_KEY: bad})
        )


def test_unarmed_all_reduce_defers_to_caller(monkeypatch):
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", None)
    x = torch.zeros(8, 1024, dtype=torch.bfloat16)
    assert pallas_all_reduce.all_reduce((0, 1), x) is None


def test_armed_all_reduce_routes_only_its_group_and_bf16_2d(monkeypatch):
    calls = []

    def op(x, rank):
        calls.append(tuple(x.shape))
        return x + 1

    armed = pallas_all_reduce._Armed(ranks=(0, 1), op=op, rank=torch.zeros(1, 1))
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", armed)
    x = torch.zeros(8, 1024, dtype=torch.bfloat16)
    assert pallas_all_reduce.all_reduce((0, 2), x) is None  # another group
    assert pallas_all_reduce.all_reduce((0, 1), x.float()) is None  # not bf16
    assert pallas_all_reduce.all_reduce((0, 1), x[0]) is None  # not 2-D
    out = pallas_all_reduce.all_reduce((0, 1), x)
    assert out is not None and calls == [(8, 1024)]

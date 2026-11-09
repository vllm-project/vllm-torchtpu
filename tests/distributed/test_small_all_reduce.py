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
"""tp_small_all_reduce routing (CPU): which tensors take the kernel, and
that the GroupCoordinator patch sends exactly those to it."""

import types

import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import small_all_reduce as sar


@pytest.fixture(autouse=True)
def _reset():
    yield
    sar._ARMED = None
    sar._MAX_BYTES = sar.DEFAULT_MAX_BYTES


def _x(rows, hidden=6144, dtype=torch.bfloat16):
    return torch.zeros(rows, hidden, dtype=dtype)


@pytest.mark.parametrize(
    "rows,want",
    [
        (0, False),
        (1, True),
        (8, True),
        (40, True),
        (42, True),  # 516,096 B
        (43, False),
        (8192, False),  # prefill
    ],
)
def test_use_kernel_rows(rows, want):
    # 512 KiB / (2 B x 6144) = 42.67 rows.
    assert sar.use_kernel(rows, 6144) is want


def test_eligible_is_static():
    assert sar.eligible(_x(16), 8) and sar.eligible(_x(8192), 8)
    assert not sar.eligible(_x(16, dtype=torch.float32), 8)
    assert not sar.eligible(torch.zeros(2, 16, 6144, dtype=torch.bfloat16), 8)
    assert not sar.eligible(_x(16, hidden=6144 + 128), 8)  # h / n not lane aligned


def test_patch_routes_only_eligible(monkeypatch):
    calls = []

    def fake_op(x):
        calls.append(tuple(x.shape))
        return x + 1

    stock_calls = []
    cls = parallel_state.GroupCoordinator
    monkeypatch.setattr(
        cls, "all_reduce", lambda self, x: stock_calls.append(tuple(x.shape)) or x
    )
    sar._patch_group_all_reduce()
    sar._ARMED = (fake_op, 8, "tp:0")
    tp = types.SimpleNamespace(world_size=8, unique_name="tp:0")
    other = types.SimpleNamespace(world_size=8, unique_name="dcp:0")

    out = cls.all_reduce(tp, _x(16))
    assert calls == [(16, 6144)] and stock_calls == []
    assert torch.equal(out, _x(16) + 1)
    cls.all_reduce(tp, _x(256))  # the op decides kernel vs psum by rows
    cls.all_reduce(tp, _x(16, dtype=torch.float32))
    cls.all_reduce(other, _x(16))
    assert calls == [(16, 6144), (256, 6144)]
    assert stock_calls == [(16, 6144), (16, 6144)]
    # Patching twice wraps once.
    patched = cls.all_reduce
    sar._patch_group_all_reduce()
    assert cls.all_reduce is patched


def test_unarmed_is_stock():
    assert sar.maybe_all_reduce(_x(16), "tp:0") is None


def test_selftest_rows_cover_decode_and_top():
    assert sar.selftest_rows(6144) == [1, 4, 5, 8, 15, 16, 17, 32, 40, 42]
    sar._MAX_BYTES = 16 * 6144 * 2
    assert sar.selftest_rows(6144) == [1, 4, 5, 8, 15, 16]


def _config(additional):
    return types.SimpleNamespace(additional_config=additional)


def test_config_off_by_default(monkeypatch):
    armed = []
    monkeypatch.setattr(sar, "arm", lambda *a: armed.append(a) or True)
    assert sar.arm_from_config(_config({}), "cpu") is False
    assert sar.arm_from_config(_config(None), "cpu") is False
    assert sar.arm_from_config(_config({"tp_small_all_reduce": False}), "cpu") is False
    assert armed == []


def test_config_threshold(monkeypatch):
    monkeypatch.setattr(sar, "arm", lambda *a: True)
    cfg = _config(
        {"tp_small_all_reduce": True, "tp_small_all_reduce_max_bytes": 32 * 6144 * 2}
    )
    cfg.model_config = types.SimpleNamespace(get_hidden_size=lambda: 6144)
    assert sar.arm_from_config(cfg, "cpu") is True
    assert sar.use_kernel(1, 6144) and sar.use_kernel(32, 6144)
    assert not sar.use_kernel(33, 6144)
    with pytest.raises(ValueError):
        sar.arm_from_config(_config({"tp_small_all_reduce": "yes"}), "cpu")
    bad = _config({"tp_small_all_reduce": True, "tp_small_all_reduce_max_bytes": 0})
    with pytest.raises(ValueError):
        sar.arm_from_config(bad, "cpu")

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
"""CPU unit tests for shared Pallas helper functions in `kernels/pallas_util.py`."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pallas_util


@pytest.mark.parametrize("target_cols", [128, 192, 256, 384])
def test_broadcast_minor(target_cols: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=128))
    src = jnp.arange(256, dtype=jnp.float32).reshape(2, 128)
    tiles = pallas_util.cdiv(target_cols, 128)
    expected = jnp.tile(src, (1, tiles))[:, :target_cols]
    np.testing.assert_array_equal(
        pallas_util.broadcast_minor(src, (2, target_cols)), expected
    )


@pytest.mark.parametrize(
    ("stride", "expected"),
    [(1, False), (3, False), (5, False), (2, True), (4, True), (8, True)],
)
def test_has_bank_conflicts(stride: int, expected: bool) -> None:
    assert pallas_util.has_bank_conflicts(stride) is expected


@pytest.mark.parametrize(
    ("device_kind", "expected"),
    [
        ("TPU v5e", 5),
        ("TPU v5litepod", 5),
        ("TPU v5 lite", 5),
        ("TPU v5i", 5),
        ("TPU v6e", 6),
        ("TPU7x", 7),
        ("TPU8x", 8),
        ("TPU v10", 10),
        ("cpu", -1),
        ("NVIDIA H100", -1),
    ],
)
def test_get_tpu_version(
    device_kind: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jax, "devices", lambda: [SimpleNamespace(device_kind=device_kind)]
    )
    assert pallas_util.get_tpu_version() == expected


def test_alignment_and_packing_helpers() -> None:
    assert pallas_util.cdiv(17, 8) == 3
    assert pallas_util.align_to(17, 8) == 24
    assert int(pallas_util.pl_align_to(jnp.int32(17), 8)) == 24
    assert pallas_util.next_power_of_2(1) == 1
    assert pallas_util.next_power_of_2(5) == 8
    assert pallas_util.get_dtype_bitwidth(jnp.bfloat16) == 16
    assert pallas_util.get_dtype_packing(jnp.float8_e4m3fn) == 4

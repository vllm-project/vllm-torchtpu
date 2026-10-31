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
"""Which TPUs the SparseCore kernels target, and their TensorCore fallback."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core import ragged_gather_v2 as ragged_gather_v2_mod
from vllm_torchtpu.kernels.sparse_core import support


@pytest.mark.parametrize(
    ("chip", "expected"),
    [
        (pltpu.ChipVersion.TPU_V5P, True),
        (pltpu.ChipVersion.TPU_V6E, True),
        (pltpu.ChipVersion.TPU_7, True),
        (pltpu.ChipVersion.TPU_7X, True),
        (pltpu.ChipVersion.TPU_V5E, False),  # No SparseCore.
        (pltpu.ChipVersion.TPU_8I, False),  # Not validated.
        (pltpu.ChipVersion.TPU_8T, False),  # Not validated.
    ],
)
def test_get_sparse_core_info(chip, expected):
    info = pltpu.get_tpu_info_for_chip(chip, 1)
    sc_info = support.get_sparse_core_info(info)
    assert (sc_info is not None) is expected
    if expected:
        assert sc_info is info.sparse_core


def test_get_sparse_core_info_reads_current_tpu(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(support.pltpu, "get_tpu_info", lambda: info)
    assert support.get_sparse_core_info() is info.sparse_core


def test_get_sparse_core_info_on_non_tpu_host(monkeypatch):
    def _raise():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr(support.pltpu, "get_tpu_info", _raise)
    assert support.get_sparse_core_info() is None


def test_ragged_gather_v2_falls_back_on_unvalidated_chip(monkeypatch):
    # TPU8i reports a SparseCore, but these kernels are not validated on it.
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_8I, 1)
    assert info.sparse_core is not None
    monkeypatch.setattr(support.pltpu, "get_tpu_info", lambda: info)
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    indices = jnp.array([3, 1, 7, 0], dtype=jnp.int32)

    out = ragged_gather_v2_mod.ragged_gather_v2(
        x, indices, jnp.array([0], jnp.int32), jnp.array([4], jnp.int32)
    )

    np.testing.assert_array_equal(np.asarray(out), np.asarray(x)[[3, 1, 7, 0]])

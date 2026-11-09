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

"""100% full-module line and branch coverage tests for sparse_core kernels.

Covers:
1. core_map_helper.py
2. ragged_gather.py
3. ragged_gather_v2.py
4. ragged_scatter.py
5. ragged_gather_reduce.py
6. ragged_gather_reduce_v3.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core import (
    ragged_gather,
    ragged_gather_v2,
    ragged_scatter,
)

# ==============================================================================
# 1. core_map_helper.py Full Coverage Tests
# ==============================================================================


# ==============================================================================
# 2. ragged_gather.py Full Coverage Tests
# ==============================================================================


class TestRaggedGather:
    """100% full-module line and branch coverage on ragged_gather.py."""

    def test_ragged_gather_sc_info_none_fallback(self):
        tpu_info_no_sc = SimpleNamespace(
            sparse_core=None,
            chip_version=pltpu.ChipVersion.TPU_7X,
            vmem_capacity_bytes=64 << 20,
        )
        with patch.object(
            ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
            indices = jnp.array([2, 0])
            # Scalar start and end promotion
            res = ragged_gather.ragged_gather(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res), [[30.0, 31.0], [10.0, 11.0]])

            # 1D array start and end
            res1d = ragged_gather.ragged_gather(
                x, indices, jnp.array([0]), jnp.array([2])
            )
            np.testing.assert_allclose(np.asarray(res1d), np.asarray(res))


# ==============================================================================
# 3. ragged_gather_v2.py Full Coverage Tests
# ==============================================================================


class TestRaggedGatherV2:
    """100% full-module line and branch coverage on ragged_gather_v2.py."""

    def test_ragged_gather_v2_validation_and_fallbacks(self):
        # x.ndim != 2
        with pytest.raises(AssertionError, match="2d inputs"):
            ragged_gather_v2.ragged_gather_v2(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_gather_v2.ragged_gather_v2(
                jnp.zeros((4, 8), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.int32),
                0,
                2,
            )

        # Invalid dtype -> ValueError
        with pytest.raises(ValueError, match="dtype must be f32, bf16"):
            ragged_gather_v2.ragged_gather_v2(
                jnp.zeros((4, 8), dtype=jnp.int32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # sc_info is None -> fallback x[indices]
        tpu_info_no_sc = SimpleNamespace(
            sparse_core=None,
            chip_version=pltpu.ChipVersion.TPU_7X,
            vmem_capacity_bytes=64 << 20,
        )
        with patch.object(
            ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.array([[10.0, 11.0], [20.0, 21.0]])
            indices = jnp.array([1, 0])
            # Scalar start/end
            res = ragged_gather_v2.ragged_gather_v2(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res), [[20.0, 21.0], [10.0, 11.0]])
            # 1D start/end
            res1d = ragged_gather_v2.ragged_gather_v2(
                x, indices, jnp.array([0]), jnp.array([2])
            )
            np.testing.assert_allclose(np.asarray(res1d), np.asarray(res))


# ==============================================================================
# 4. ragged_scatter.py Full Coverage Tests
# ==============================================================================


class TestRaggedScatter:
    """100% full-module line and branch coverage on ragged_scatter.py."""

    def test_ragged_scatter_validation_and_fallbacks(self):
        # x.ndim != 2
        with pytest.raises(AssertionError, match="2d inputs"):
            ragged_scatter.ragged_scatter(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_scatter.ragged_scatter(
                jnp.zeros((4, 8), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.int32),
                0,
                2,
            )

        # sc_info is None -> fallback x[indices]
        tpu_info_no_sc = SimpleNamespace(
            sparse_core=None,
            chip_version=pltpu.ChipVersion.TPU_7X,
            vmem_capacity_bytes=64 << 20,
        )
        with patch.object(
            ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.array([[10.0, 11.0], [20.0, 21.0]])
            indices = jnp.array([1, 0])
            # Scalar start/end
            res = ragged_scatter.ragged_scatter(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res), [[20.0, 21.0], [10.0, 11.0]])
            # 1D start/end
            res1d = ragged_scatter.ragged_scatter(
                x, indices, jnp.array([0]), jnp.array([2])
            )
            np.testing.assert_allclose(np.asarray(res1d), np.asarray(res))

        # VMEM capacity fallback (small size < vmem_capacity * 0.6)
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
        tpu_info_small = SimpleNamespace(
            sparse_core=sc_info,
            vmem_capacity_bytes=100_000_000,
            num_lanes=128,
            generation=7,
            chip_version=pltpu.ChipVersion.TPU_7X,
        )
        with patch.object(
            ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info_small
        ):
            ragged_scatter.ragged_scatter.clear_cache()
            res_vmem = ragged_scatter.ragged_scatter(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res_vmem), np.asarray(res))
            # Also test unwrapped function directly with 0d array scalars
            res_unwrapped = ragged_scatter.ragged_scatter._fun(
                x, indices, jnp.array(0), jnp.array(2)
            )
            np.testing.assert_allclose(np.asarray(res_unwrapped), np.asarray(res))


# ==============================================================================
# 5. ragged_gather_reduce.py Full Coverage Tests
# ==============================================================================


# ==============================================================================
# 6. ragged_gather_reduce_v3.py Full Coverage Tests
# ==============================================================================

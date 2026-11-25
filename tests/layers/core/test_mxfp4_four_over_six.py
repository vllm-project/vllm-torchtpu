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
"""Block scale selection for Four Over Six MoE requantization."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor


def _skip_without_float4():
    try:
        jnp.asarray([0.0, 1.0], jnp.float32).astype(jnp.float4_e2m1fn)
    except Exception as exc:  # pragma: no cover - device dependent
        pytest.skip(f"jax float4_e2m1fn unavailable: {exc}")


@pytest.mark.parametrize("pack", [False, True])
def test_four_over_six_selects_per_block_and_column(pack):
    _skip_without_float4()
    rows = np.tile([[4.0, 6.0, 0.0], [1.5, 1.0, 0.0]], (256, 1))
    w = np.concatenate([rows, rows[:, [1, 0, 2]]]).astype(np.float32)
    codes, scales = quantize_to_native_fp4_kmajor(w, block=512, pack=pack)
    np.testing.assert_array_equal(scales, [[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]])
    expected = jnp.asarray(w).astype(jnp.float4_e2m1fn)
    if pack:
        from jax import lax

        pairs = expected.reshape(512, 2, 3).swapaxes(-1, -2)
        expected = lax.bitcast_convert_type(pairs, jnp.uint8)
    np.testing.assert_array_equal(codes, expected)

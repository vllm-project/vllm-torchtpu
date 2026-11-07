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
"""gmm_fused_moe against the 2 gmm_v2 calls it replaces in moe_gmm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import gmm_fused_moe
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2

GROUPS, HIDDEN, INTER = 16, 1024, 512


def _weights():
    keys = jax.random.split(jax.random.key(7), 4)

    def weight(key, shape):
        return jax.random.normal(key, shape, jnp.bfloat16).astype(jnp.float8_e4m3fn)

    def scale(key, shape):
        return jax.random.uniform(key, shape, jnp.float32, 0.5, 1.5) * 0.02

    return (
        weight(keys[0], (GROUPS, HIDDEN, 2 * INTER)),
        scale(keys[1], (GROUPS, 1, 1, 2 * INTER)),
        weight(keys[2], (GROUPS, INTER, HIDDEN)),
        scale(keys[3], (GROUPS, 1, 1, HIDDEN)),
    )


def _two_calls(x, w1, s1, w2, s2, group_sizes):
    group_offset = jnp.array(0, dtype=jnp.int32)
    h = gmm_v2(
        x,
        w1,
        group_sizes,
        rhs_scale=s1,
        group_offset=group_offset,
        zero_initialize=False,
        fuse_act="silu",
    )
    return gmm_v2(
        h[:, : w2.shape[1]],
        w2,
        group_sizes,
        rhs_scale=s2,
        group_offset=group_offset,
        zero_initialize=False,
        preferred_element_type=x.dtype,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "group_sizes",
    [
        # Experts of 0, 1 to 128, 129 to 256 and over 1,024 rows (2 segments).
        [0, 40, 77, 19, 130, 9, 256, 3, 0, 61, 1, 1100, 70, 2, 51, 248],
        [128] * GROUPS,
        [0] * GROUPS,
    ],
)
def test_gmm_fused_moe_matches_two_gmm_v2_calls(group_sizes):
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")
    w1, s1, w2, s2 = _weights()
    rows = int(sum(group_sizes))
    x = jax.random.normal(jax.random.key(1), (max(rows, 256), HIDDEN), jnp.bfloat16)
    sizes = jnp.array(group_sizes, dtype=jnp.int32)

    expected = _two_calls(x, w1, s1, w2, s2, sizes)
    actual = gmm_fused_moe.gmm_fused_moe(
        x, w1, s1, w2, s2, sizes, jnp.array(0, dtype=jnp.int32), fuse_act="silu"
    )

    assert gmm_fused_moe.supports(w1, s1, None, w2, s2, None, "silu", None)
    np.testing.assert_array_equal(
        np.asarray(actual[:rows]).view(np.uint16),
        np.asarray(expected[:rows]).view(np.uint16),
    )

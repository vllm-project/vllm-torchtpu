# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The output projection with the residual add (``megakernel.qwen38.projection``)
in TPU interpret mode on the host."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.megakernel.qwen38.projection import fp8_projection  # noqa: E402


def test_fp8_projection_matches_the_rounded_residual_sum():
    rng = np.random.default_rng(0)
    rows, contraction, out = 1024, 512, 1024
    x = jnp.asarray(rng.standard_normal((rows, contraction)), jnp.float32)
    weight = jnp.asarray(rng.standard_normal((out, contraction)) * 0.5, jnp.float32)
    x, weight = x.astype(jnp.float8_e4m3fn), weight.astype(jnp.float8_e4m3fn)
    residual = jnp.asarray(rng.standard_normal((rows, out)), jnp.bfloat16)
    scale = jnp.float32(0.01)

    got = fp8_projection(x, weight, scale, residual, interpret=True)

    product = np.asarray(x, np.float64) @ np.asarray(weight, np.float64).T
    projected = jnp.asarray(product * 0.01, jnp.float32).astype(jnp.bfloat16)
    want = (residual.astype(jnp.float32) + projected.astype(jnp.float32)).astype(
        jnp.bfloat16
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

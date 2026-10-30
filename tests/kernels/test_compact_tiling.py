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
"""TPU tests for the compact-tiling pin.

The pin aliases its operand, so the case that matters is an operand that is
still read after the pin: the result and that read must both see the original
values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.compact_tiling import pin_compact_tiling


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((16, 4096), jnp.bfloat16),
        ((1024, 4096), jnp.bfloat16),
        ((24, 256), jnp.float32),
    ],
)
def test_pin_keeps_values_for_every_reader(shape, dtype):
    x = jax.random.normal(jax.random.key(0), shape, dtype)

    @jax.jit
    def run(x):
        y = x * 2
        pinned = pin_compact_tiling(y)
        return pinned, y @ jnp.ones((shape[1], 128), dtype)

    pinned, product = run(x)
    expected = x * 2
    np.testing.assert_array_equal(np.asarray(pinned), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(product),
        np.asarray(jax.jit(lambda y: y @ jnp.ones((shape[1], 128), dtype))(expected)),
    )

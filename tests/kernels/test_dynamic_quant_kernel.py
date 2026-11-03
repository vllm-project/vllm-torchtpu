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
"""In-kernel activation quantization vs the plain-JAX quantized matmul.

The interpret-mode cases gate the row-block pipeline, the slot alternation and
the chunked output without a TPU; the compiled cases check Mosaic's result
bit-exact against XLA's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul import dynamic_quant_kernel, util


def _reference(x, w_q, w_scale):
    x_q, x_scale = util.quantize_tensor(x, w_q.dtype)
    out = jax.lax.dot_general(
        x_q, w_q, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    return (out * x_scale * w_scale[None]).astype(x.dtype)


def _operands(rows, k, n):
    rng = np.random.default_rng(rows + k + n)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    x[0] = 0.0
    x[1] *= 64.0
    w = (rng.standard_normal((k, n)) * 0.02).astype(np.float32)
    w_q, w_scale = util.quantize_tensor(jnp.asarray(w), jnp.float8_e4m3fn, dim=0)
    return jnp.asarray(x, jnp.bfloat16), w_q, w_scale.reshape(-1)


@pytest.mark.parametrize(
    "rows,k,n",
    [
        (512, 4096, 2560),  # one row block
        (1536, 1024, 4096),  # odd number of row blocks
        (1024, 4096, 256),  # one output chunk
        (1024, 128, 4096),  # more output chunks than quantization steps
        (1024, 256, 384),  # output chunk narrower than OUT_BLOCK
    ],
)
def test_matches_plain_jax(rows, k, n):
    x, w_q, w_scale = _operands(rows, k, n)
    assert dynamic_quant_kernel.supports(x, w_q, w_scale)
    with pltpu.force_tpu_interpret_mode():
        got = dynamic_quant_kernel.dynamic_quant_matmul(x, w_q, w_scale)
    want = jax.jit(_reference)(x, w_q, w_scale)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert not np.asarray(got[0]).any()


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows,k,n", [(16384, 4096, 2560), (1024, 256, 384)])
def test_compiled_matches_xla(rows, k, n):
    x, w_q, w_scale = _operands(rows, k, n)
    got = dynamic_quant_kernel.dynamic_quant_matmul(x, w_q, w_scale)
    want = jax.jit(_reference)(x, w_q, w_scale)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize(
    "x_shape,x_dtype,w_dtype",
    [
        ((768, 512), jnp.bfloat16, jnp.float8_e4m3fn),  # rows not a block multiple
        ((0, 512), jnp.bfloat16, jnp.float8_e4m3fn),
        ((512, 512), jnp.float32, jnp.float8_e4m3fn),
        ((512, 512), jnp.bfloat16, jnp.int8),
    ],
)
def test_unsupported_operands_stay_on_plain_jax(x_shape, x_dtype, w_dtype):
    x = jnp.zeros(x_shape, x_dtype)
    w_q = jnp.zeros((x_shape[1], 512), w_dtype)
    assert not dynamic_quant_kernel.supports(x, w_q, jnp.ones((512,), jnp.float32))

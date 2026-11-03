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


def _split_cols(split, n):
    # "tail" splits the output like GDN's in_proj_qkvz: q/k/v, then z (the last
    # 1024 columns at TP 8); "heads" also writes z as heads of 128.
    return 0 if split == "none" else 128 if n <= 512 else 1024


def _split_head_dim(split):
    return 128 if split == "heads" else 0


def _matmul(x, w_q, w_scale, split, weight_nk):
    if weight_nk:
        # The caller's weight is the transpose of a stored [n, k] array.
        w_q = jnp.asarray(np.ascontiguousarray(np.asarray(w_q).T)).T
    return dynamic_quant_kernel.dynamic_quant_matmul(
        x,
        w_q,
        w_scale,
        split_cols=_split_cols(split, w_q.shape[1]),
        weight_nk=weight_nk,
        split_head_dim=_split_head_dim(split),
    )


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
@pytest.mark.parametrize("split", ["none", "tail", "heads"])
@pytest.mark.parametrize("weight_nk", [False, True])
def test_matches_plain_jax(rows, k, n, split, weight_nk):
    x, w_q, w_scale = _operands(rows, k, n)
    assert dynamic_quant_kernel.supports(
        x, w_q, w_scale, _split_cols(split, n), _split_head_dim(split)
    )
    with pltpu.force_tpu_interpret_mode():
        got = _matmul(x, w_q, w_scale, split, weight_nk)
    want = jax.jit(_reference)(x, w_q, w_scale)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert not np.asarray(got[0]).any()


def _norm_weight(k):
    rng = np.random.default_rng(k)
    return jnp.asarray(1.0 + 0.1 * rng.standard_normal(k), jnp.float32)


@pytest.mark.parametrize(
    "rows,k,n",
    [
        (512, 512, 1024),  # one row block
        (1536, 384, 512),  # odd row blocks; k not a multiple of NORM_LANES
    ],
)
@pytest.mark.parametrize("split", ["none", "tail", "heads"])
def test_norm_matches_normalizing_first(rows, k, n, split):
    x, w_q, w_scale = _operands(rows, k, n)
    norm_weight = _norm_weight(k)
    cols, head_dim = _split_cols(split, n), _split_head_dim(split)
    assert dynamic_quant_kernel.supports(x, w_q, w_scale, cols, head_dim, norm_weight)
    with pltpu.force_tpu_interpret_mode():
        got = dynamic_quant_kernel.dynamic_quant_matmul(
            x,
            w_q,
            w_scale,
            split_cols=cols,
            split_head_dim=head_dim,
            norm_weight=norm_weight,
            norm_eps=1e-6,
        )
    want = jax.jit(_reference)(util.rms_norm(x, norm_weight, 1e-6), w_q, w_scale)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert not np.asarray(got[0]).any()


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows,k,n", [(16384, 4096, 2560), (1024, 256, 384)])
@pytest.mark.parametrize("split", ["none", "tail", "heads"])
@pytest.mark.parametrize("weight_nk", [False, True])
def test_compiled_matches_xla(rows, k, n, split, weight_nk):
    x, w_q, w_scale = _operands(rows, k, n)
    got = _matmul(x, w_q, w_scale, split, weight_nk)
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


@pytest.mark.parametrize("split_cols", [64, -128, 512, 640])
def test_unsupported_split_stays_on_plain_jax(split_cols):
    # No 128-multiple chunk divides 64; the others leave no first part.
    x = jnp.zeros((512, 512), jnp.bfloat16)
    w_q = jnp.zeros((512, 512), jnp.float8_e4m3fn)
    w_scale = jnp.ones((512,), jnp.float32)
    assert not dynamic_quant_kernel.supports(x, w_q, w_scale, split_cols)


@pytest.mark.parametrize("split_cols,split_head_dim", [(0, 128), (256, 64), (256, 384)])
def test_unsupported_split_heads_stay_on_plain_jax(split_cols, split_head_dim):
    # Heads need a split, a 128-multiple size, and to tile the output chunks.
    x = jnp.zeros((512, 512), jnp.bfloat16)
    w_q = jnp.zeros((512, 1024), jnp.float8_e4m3fn)
    w_scale = jnp.ones((1024,), jnp.float32)
    assert not dynamic_quant_kernel.supports(
        x, w_q, w_scale, split_cols, split_head_dim
    )

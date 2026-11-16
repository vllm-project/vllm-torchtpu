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
"""One pass over a sequence-parallel token shard: residual add, Gemma RMSNorm
and per-row e4m3 quantization of the normed rows for the next all-gather.

``shard [S, H]`` bf16 plus a bf16 update ``[S, H]`` (a reduce-scatter
output). Outputs the new residual (bf16), the normed rows as e4m3 and their
per-row scale ``amax / 448``. Rounding points follow model.py's unfused
chain: the residual sum rounds to bf16, the norm runs in f32 (the TPU
``rms_norm`` provider) and rounds to bf16 before the amax and quantization;
the quantization multiplies by the scale's reciprocal, so a rare element can
land one e4m3 step from the unfused ``x / scale``.
As XLA fusions the chain re-read the shard three times (add + stats, amax,
quantize: 31.8 us per 1024-row shard against ~12 us for one pass, tpu7x).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
BLOCK_ROWS = 128


def _kernel(eps, shard_ref, upd_ref, w_ref, res_ref, normed_ref, scale_ref):
    r = shard_ref[...].astype(jnp.float32) + upd_ref[...].astype(jnp.float32)
    r = r.astype(jnp.bfloat16)
    res_ref[...] = r
    rf = r.astype(jnp.float32)
    var = jnp.mean(rf * rf, axis=-1, keepdims=True)
    normed = (rf * jax.lax.rsqrt(var + eps) * w_ref[...]).astype(jnp.bfloat16)
    nf = normed.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(nf), axis=-1, keepdims=True), 1e-12) / FP8_MAX
    normed_ref[...] = (nf * (1.0 / scale)).astype(FP8)
    scale_ref[...] = scale


@functools.partial(jax.jit, static_argnames=("eps", "block_rows", "interpret"))
def add_norm_quant(
    shard: jax.Array,
    update: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    block_rows: int = BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(shard + update, e4m3 rows of rms_norm(shard + update) * weight, their
    f32 scales [S, 1]). ``weight`` is the f32 Gemma weight (1 + w), ``[H]``."""
    rows, hidden = shard.shape
    tm = min(block_rows, rows)
    assert rows % tm == 0, (rows, tm)
    row = pl.BlockSpec((tm, hidden), lambda i: (i, 0))
    col = pl.BlockSpec((tm, 1), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(_kernel, eps),
        grid=(rows // tm,),
        in_specs=[row, row, w_spec],
        out_specs=[row, row, col],
        out_shape=[
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, hidden), FP8),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 << 20
        ),
        interpret=interpret,
    )(shard, update, weight.reshape(1, hidden).astype(jnp.float32))

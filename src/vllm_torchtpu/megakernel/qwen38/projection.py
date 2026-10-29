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
"""A layer's output projection and residual add as one Pallas call per core:
``residual + (x @ weight.T) * scale`` with FP8 ``x`` and ``weight`` (``[out,
in]``, the checkpoint's layout), one dot over the whole contraction per
output tile."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32

ROW_TILE = 1024  # x rows per tile: one weight pass per row tile
COLUMN_TILE = 512  # output columns per tile


def fp8_projection(x, weight, scale, residual, *, interpret=False):
    """``residual + x @ weight.T * scale`` as BF16 ``[rows, out]``: ``x``
    ``[rows, in]`` and ``weight`` ``[out, in]`` FP8, ``scale`` ``[1]`` F32
    (both static scales' product), ``residual`` ``[rows, out]`` BF16; the
    projection rounded to BF16, then summed in F32."""
    rows, contraction = x.shape
    out = weight.shape[0]
    row_tile, column_tile = min(ROW_TILE, rows), min(COLUMN_TILE, out)
    assert rows % row_tile == 0 and out % column_tile == 0

    def kernel(scale_ref, x_ref, w_ref, residual_ref, out_ref):
        product = jax.lax.dot_general(
            x_ref[...], w_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=F32
        )
        # The projection rounds to BF16 before the residual add (the
        # reference's rounding points).
        projected = (product * scale_ref[0]).astype(BF16)
        out_ref[...] = (residual_ref[...].astype(F32) + projected.astype(F32)).astype(
            BF16
        )

    return pl.pallas_call(
        kernel,
        grid=(rows // row_tile, out // column_tile),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec((row_tile, contraction), lambda i, j: (i, 0)),
            pl.BlockSpec((column_tile, contraction), lambda i, j: (j, 0)),
            pl.BlockSpec((row_tile, column_tile), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((row_tile, column_tile), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((rows, out), BF16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=2
            * (row_tile + column_tile)
            * contraction
            * x.dtype.itemsize
            + 2**24,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="qwen38_fp8_projection",
    )(jnp.reshape(scale, (1,)).astype(F32), x, weight, residual)

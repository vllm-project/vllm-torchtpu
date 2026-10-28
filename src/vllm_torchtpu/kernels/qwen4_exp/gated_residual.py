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
"""Merge BF16 residual branches with FP32 per-token scalar gates."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    RESIDUAL_PIPELINE,
    HcHardware,
    supports_residual,
)


def _kernel(residual_ref, block_ref, gates_ref, output_ref, *, branches, hidden):
    block = block_ref[...].astype(jnp.float32)
    gates = gates_ref[...]
    for branch in range(branches):
        start = branch * hidden
        residual = residual_ref[:, start : start + hidden].astype(jnp.float32)
        output_ref[:, start : start + hidden] = (
            residual + block * gates[:, branch : branch + 1]
        ).astype(output_ref.dtype)


@functools.cache
def gated_residual_fn(
    branches: int, hidden: int, token_tile: int, hardware: HcHardware | None = None
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Build a tiled residual merge with BF16 inputs/output and FP32 gates.

    Args:
        branches: Number of contiguous residual branches.
        hidden: Width of each branch, aligned to 128 TPU vector lanes.
        token_tile: Full token rows per tile, aligned to dtype sublane packing.
        hardware: Optional immutable layout facts; resolved at TPU tracing otherwise.

    Returns:
        A JAX function taking residual, block output, and scalar gates.

    Raises:
        ValueError: Geometry, operand shapes, or dtypes are unsupported.
    """
    if branches <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError("Residual branches must be positive and width 128-aligned")
    if token_tile <= 0:
        raise ValueError("Residual token tile must be positive")

    def core(residual: jax.Array, block: jax.Array, gates: jax.Array) -> jax.Array:
        tokens = residual.shape[0]
        if (
            tokens == 0
            or tokens % token_tile
            or residual.shape != (tokens, branches * hidden)
            or block.shape != (tokens, hidden)
            or gates.shape != (tokens, branches)
            or residual.dtype != jnp.bfloat16
            or block.dtype != jnp.bfloat16
            or gates.dtype != jnp.float32
        ):
            raise ValueError("Residual merge requires full BF16 tiles and FP32 gates")
        layout = hardware or HcHardware.from_tpu_info(pltpu.get_tpu_info())
        if not supports_residual(branches, hidden, token_tile, layout):
            raise ValueError(
                "Residual merge requires supported packing and VMEM budget"
            )
        return pl.pallas_call(
            functools.partial(_kernel, branches=branches, hidden=hidden),
            grid=(tokens // token_tile,),
            in_specs=(
                pl.BlockSpec(
                    (token_tile, branches * hidden),
                    lambda i: (i, 0),
                    pipeline_mode=RESIDUAL_PIPELINE,
                ),
                pl.BlockSpec(
                    (token_tile, hidden),
                    lambda i: (i, 0),
                    pipeline_mode=RESIDUAL_PIPELINE,
                ),
                pl.BlockSpec(
                    (token_tile, branches),
                    lambda i: (i, 0),
                    pipeline_mode=RESIDUAL_PIPELINE,
                ),
            ),
            out_specs=pl.BlockSpec(
                (token_tile, branches * hidden),
                lambda i: (i, 0),
                pipeline_mode=RESIDUAL_PIPELINE,
            ),
            out_shape=jax.ShapeDtypeStruct(residual.shape, residual.dtype),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=layout.vmem_bytes),
            # Logical unpadded operand/output bytes, not exact DMA bus traffic.
            cost_estimate=pl.CostEstimate(
                flops=2 * tokens * branches * hidden,
                transcendentals=0,
                bytes_accessed=sum(
                    a.size * a.dtype.itemsize for a in (residual, block, gates)
                )
                + residual.size * residual.dtype.itemsize,
            ),
            name=f"gated_residual_b{branches}_h{hidden}_m{token_tile}",
        )(residual, block, gates)

    return core

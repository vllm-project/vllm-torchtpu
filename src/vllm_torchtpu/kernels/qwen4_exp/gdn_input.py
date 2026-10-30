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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3.config import GDNConfig
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput

_COMPACT_TILE_TOKENS = 128


def _compact_qkv_kernel(qkv_ref: jax.Array, out_ref: jax.Array) -> None:
    out_ref[...] = qkv_ref[...].astype(jnp.float32)[:, None, :]


def supports_compact_qkv(
    shape: tuple[int, int],
    dtype: jnp.dtype,
    *,
    pipeline_mode: pl.Buffered | None = None,
) -> bool:
    """Whether aligned tile buffers fit the device VMEM capacity."""
    tokens, width = shape
    if (
        dtype != jnp.bfloat16
        or tokens < _COMPACT_TILE_TOKENS
        or tokens % _COMPACT_TILE_TOKENS
        or width <= 0
    ):
        return False
    info = pltpu.get_tpu_info()
    if pipeline_mode is None:
        pipeline_mode = pl.Buffered(buffer_count=GDNConfig.num_buffers)
    window_bytes = (
        pipeline_mode.buffer_count
        * _COMPACT_TILE_TOKENS
        * width
        * (jnp.dtype(dtype).itemsize + jnp.dtype(jnp.float32).itemsize)
    )
    return width % info.num_lanes == 0 and window_bytes <= info.vmem_capacity_bytes


def compact_qkv_input(
    qkv: jax.Array, *, pipeline_mode: pl.Buffered | None = None
) -> jax.Array:
    """Cast directly into FP32 Conv1D input tiles."""
    tokens, width = qkv.shape
    if pipeline_mode is None:
        pipeline_mode = pl.Buffered(buffer_count=GDNConfig.num_buffers)
    if not supports_compact_qkv(
        (tokens, width), qkv.dtype, pipeline_mode=pipeline_mode
    ):
        return qkv.astype(jnp.float32).reshape(tokens, 1, width)

    return pl.pallas_call(
        _compact_qkv_kernel,
        grid=(tokens // _COMPACT_TILE_TOKENS,),
        in_specs=(
            pl.BlockSpec(
                (_COMPACT_TILE_TOKENS, width),
                lambda tile: (tile, 0),
                pipeline_mode=pipeline_mode,
            ),
        ),
        out_specs=pl.BlockSpec(
            (_COMPACT_TILE_TOKENS, 1, width),
            lambda tile: (tile, 0, 0),
            pipeline_mode=pipeline_mode,
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, 1, width), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="gdn_compact_qkv_input",
    )(qkv)


def prepare_gdn_input(qkv: jax.Array) -> jax.Array | Conv1DInput:
    """Prepare a shard-local projection when its conversion tiles fit."""
    tokens, width = qkv.shape
    if not supports_compact_qkv((tokens, width), qkv.dtype):
        return qkv
    return Conv1DInput(compact_qkv_input(qkv), qkv.dtype)

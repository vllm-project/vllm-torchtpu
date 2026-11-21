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
"""Few-row linear on an e4m3 ``[K, N]`` weight with one scale per column.

``y = bf16(bf16(x @ w) * s)`` (s bf16 or f32), rounded as the XLA matmul on
``bf16(w)`` rounds it. The bf16 rows meet the e4m3 weight on the MXU (a mixed
bf16 x e4m3 matmul): XLA's dot instead widens every weight element on the
vector unit inside its fusion, which bounds it at decode row counts.

The weight is an operand in VMEM, so XLA prefetches it ahead of the call as
it does for its own dots, and the kernel is the matmul alone.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_VMEM_MARGIN = 2 << 20


def max_rows(k: int, n: int) -> int:
    """The most rows for which `fp8_linear` serves an e4m3 ``[k, n]`` weight:
    while reading the weight outlasts the MXU work at its bf16 rate. 0 on
    chips without a native bf16 x e4m3 matmul, when `k` is not a whole number
    of lanes or `n` of MXU columns, or when the weight takes more than half of
    VMEM (XLA holds it there beside the step's other buffers)."""
    info = pltpu.get_tpu_info()
    if not info.is_matmul_supported(jnp.bfloat16, jnp.float8_e4m3fn):
        return 0
    if k % info.num_lanes or n % info.mxu_column_size:
        return 0
    if 2 * k * n > info.vmem_capacity_bytes:
        return 0
    return info.bf16_ops_per_second // (2 * info.mem_bw_bytes_per_second)


def _kernel(x_ref, s_ref, w_ref, o_ref):
    acc = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    rounded = acc.astype(o_ref.dtype).astype(jnp.float32)
    o_ref[...] = (rounded * s_ref[...].astype(jnp.float32)).astype(o_ref.dtype)


def fp8_linear(
    x: jax.Array,
    w: jax.Array,
    s: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """bf16 ``[rows, N]`` of bf16 ``x [rows, K]`` against the e4m3
    ``w [K, N]`` and its per-column ``s [1, N]``."""
    rows, k = x.shape
    n = w.shape[1]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((rows, n), x.dtype),
        in_specs=[vmem, vmem, vmem],
        out_specs=vmem,
        compiler_params=pltpu.CompilerParams(
            # The operands are XLA's VMEM buffers; the kernel's own are the
            # f32 product and its bf16-rounded copy.
            vmem_limit_bytes=2 * rows * n * 4 + _VMEM_MARGIN
        ),
        name=f"fp8_linear-k_{k}-n_{n}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x, s, pltpu.with_memory_space_constraint(w, pltpu.VMEM))

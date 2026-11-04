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
"""The push layer's combine with the arrival buffer resident in VMEM.

At decode sizes the whole arrival buffer of a shard fits in VMEM (2128 rows
of 5 KB at DP16 C768), so one Pallas call can land it with a single copy and
read every token's `topk` rows with dynamic VMEM loads, instead of XLA's
epilogue, whose row gather is offloaded to the SparseCore and waited for by
the TensorCore. The arithmetic is the epilogue's, in its order: per token,
term k = f32(row) * coef[k] summed over k = 0..topk-1 in f32, then cast to the
output dtype, so the result is the same bit for bit.
"""

import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2 import host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import COMBINE_MIN_TOKENS

# Arrival buffers above this many bytes keep the XLA epilogue: the buffer,
# the output and their staging have to fit the kernel's VMEM budget.
MAX_ARRIVAL_BYTES = 24 * 2**20
# Tokens one loop iteration combines; a static unroll inside the loop.
TOKENS_PER_STEP = 8


def combine_vmem_applies(t_local, arrivals_shape, wire_dtype):
    """Whether this shard's arrival buffer fits the VMEM-resident combine.

    Below COMBINE_MIN_TOKENS XLA does not offload the epilogue's gather and
    the epilogue stays the cheaper of the two (+3.3 us a layer for this
    kernel at the drafter's 240 tokens, isolated).
    """
    rows, blocks, lanes = arrivals_shape
    nbytes = rows * blocks * lanes * jnp.dtype(wire_dtype).itemsize
    return (
        t_local >= COMBINE_MIN_TOKENS
        and nbytes <= MAX_ARRIVAL_BYTES
        and t_local % TOKENS_PER_STEP == 0
    )


def _build(*, t_local, topk, recv_rows, lane_blocks, lanes, wire_dtype, out_dtype):
    steps = t_local // TOKENS_PER_STEP

    def kernel(rows_sm, coef_sm, arr_hbm, out_vm, arr_vm, sem):
        copy = pltpu.make_async_copy(arr_hbm, arr_vm, sem)
        copy.start()
        copy.wait()

        def step(s, carry):
            for j in range(TOKENS_PER_STEP):
                i = s * TOKENS_PER_STEP + j
                acc = None
                for k in range(topk):
                    p = i * topk + k
                    term = arr_vm[rows_sm[p]].astype(jnp.float32) * coef_sm[p]
                    acc = term if acc is None else acc + term
                out_vm[i] = acc.astype(out_dtype)
            return carry

        lax.fori_loop(0, steps, step, jnp.int32(0))

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
            scratch_shapes=(
                pltpu.VMEM((recv_rows, lane_blocks, lanes), wire_dtype),
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, lane_blocks, lanes), out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=host.vmem_limit(), disable_bounds_checks=True
        ),
        name=f"moe_push_combine_vmem_t{t_local}_k{topk}_r{recv_rows}",
    )

    def fn(arrivals, coef, rows):
        """arrivals [R, blocks, lanes], coef f32 [t, K], rows i32 [t, K]
        (already clamped to R - 1) -> [t, blocks * lanes]."""
        out = call(
            rows.astype(jnp.int32).reshape(-1),
            coef.astype(jnp.float32).reshape(-1),
            arrivals,
        )
        return out.reshape(t_local, lane_blocks * lanes)

    return fn


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def build_combine_vmem_kernel(**kwargs):
    """Memoizing front door for the VMEM-resident combine."""
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    fn = _CACHE.get(key)
    if fn is None:
        with _CACHE_LOCK:
            fn = _CACHE.get(key)
            if fn is None:
                fn = _build(**kwargs)
                _CACHE[key] = fn
    return fn

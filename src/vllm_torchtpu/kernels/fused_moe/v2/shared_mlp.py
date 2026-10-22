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
"""Experimental local-token shared FP8 MLP with down-weight prefetch.

Uses the JAX 0.11.2 explicit MXU accumulator API,
with explicit DMA and full-row quantization between the two projections.
This candidate is intentionally limited to the reviewed decode shape.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def fused_shared_mlp(
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    *,
    swiglu_limit: float | None,
    diagnostic: bool = False,
) -> jax.Array:
    """Compute a replicated-weight shared MLP for 288 local decode tokens."""
    rows, hidden = x.shape
    intermediate = w2.shape[0]
    if (rows, hidden, intermediate) != (288, 5120, 2304):
        raise ValueError("This candidate requires local shape (288, 5120, 2304)")
    if w1.shape != (hidden, 2 * intermediate) or w2.shape[1] != hidden:
        raise ValueError("Shared weights must use canonical K,N layout")
    if w1.dtype != jnp.float8_e4m3fn or w2.dtype != w1.dtype:
        raise ValueError("Shared weights must use FP8 E4M3FN")
    if x.dtype != jnp.bfloat16:
        raise ValueError("Shared activations must use BF16")
    if s1.shape != (2 * intermediate,) or s2.shape != (hidden,):
        raise ValueError("Shared scales must have one value per output channel")
    if s1.dtype != jnp.float32 or s2.dtype != jnp.float32:
        raise ValueError("Shared per-channel scales must use FP32")
    # Match baseline input quantization outside the fused shared kernel.
    xq, xs = quantize_tensor(x, w1.dtype)
    bm, bn, bk = 96, 256, 256
    mxu_rows = (128, 160)
    mxu_offsets = (0, 128)
    down_n = 1024

    def kernel(
        x_ref,
        xs_ref,
        w1_ref,
        s1_ref,
        w2_ref,
        s2_ref,
        *outputs_and_scratch,
    ):
        output_count = 5 if diagnostic else 1
        y_ref = outputs_and_scratch[0]
        (
            gu_vm,
            down_vm,
            gate_vm,
            midq_vm,
            mids_vm,
            out_vm,
            gu_sem,
            down_sem,
            out_sem,
        ) = outputs_and_scratch[output_count:]

        def snapshot(src, output_index):
            if diagnostic:
                copy = pltpu.make_async_copy(
                    src, outputs_and_scratch[output_index], out_sem.at[0]
                )
                copy.start()
                copy.wait()

        def load_gu(half):
            return pltpu.make_async_copy(
                w1_ref.at[:, pl.ds(half * intermediate, intermediate)],
                gu_vm.at[half],
                gu_sem.at[half],
            )

        def load_down(tile, slot):
            return pltpu.make_async_copy(
                w2_ref.at[:, pl.ds(tile * down_n, down_n)],
                down_vm.at[slot],
                down_sem.at[slot],
            )

        load_gu(0).start()
        load_gu(1).start()

        # Adapted from JAX jax-v0.11.2 tests/pallas/tpu_pallas_test.py,
        # ExplicitMXUTest: split local rows over two MXUs and retain the
        # complete K reduction in hardware accumulators before popping.
        accs = tuple(
            jax.empty_ref(
                jax.ShapeDtypeStruct((mxu_rows[mxu], bn), jnp.float32),
                memory_space=pltpu.ACC(mxu),
            )
            for mxu in range(2)
        )

        def accumulate(lhs_ref, rhs_ref, n, k_steps):
            def dot_body(k, unused):
                rhs = rhs_ref[pl.ds(k * bk, bk), pl.ds(n, bn)]
                for mxu in range(2):
                    pltpu.matmul_push_rhs(rhs, staging_register=0, mxu_index=mxu)
                for mxu in range(2):
                    pltpu.matmul_acc_lhs(
                        accs[mxu],
                        lhs_ref[
                            pl.ds(mxu_offsets[mxu], mxu_rows[mxu]), pl.ds(k * bk, bk)
                        ],
                        load_staged_rhs=0,
                    )
                return unused

            lax.fori_loop(0, k_steps, dot_body, 0)

        def project_half(half):
            load_gu(half).wait()

            def tile_body(index, unused):
                n = index * bn
                accumulate(x_ref, gu_vm.at[half], n, hidden // bk)
                for mxu in range(2):
                    m = mxu_offsets[mxu]
                    value = pltpu.matmul_pop(accs[mxu])
                    value = value * xs_ref[pl.ds(m, mxu_rows[mxu]), :]
                    value = value * s1_ref[pl.ds(half * intermediate + n, bn)][None, :]
                    value = value.astype(jnp.bfloat16)
                    if half == 0:
                        gate_vm[pl.ds(m, mxu_rows[mxu]), pl.ds(n, bn)] = value
                    else:
                        gate = gate_vm[pl.ds(m, mxu_rows[mxu]), pl.ds(n, bn)]
                        if swiglu_limit is not None:
                            gate = jnp.minimum(gate, swiglu_limit)
                            value = jnp.clip(value, -swiglu_limit, swiglu_limit)
                        activated = jax.nn.silu(gate).astype(jnp.bfloat16)
                        # Gate is dead after this tile; reuse it for SwiGLU.
                        gate_vm[pl.ds(m, mxu_rows[mxu]), pl.ds(n, bn)] = (
                            activated * value
                        ).astype(jnp.bfloat16)
                return unused

            lax.fori_loop(0, intermediate // bn, tile_body, 0)

        with jax.named_scope("shared_gate"):
            project_half(0)
        snapshot(gate_vm, 1)
        # Gate slab has been consumed. Prefetch down while up/SwiGLU execute.
        load_down(0, 0).start()
        with jax.named_scope("shared_up_swiglu"):
            project_half(1)

        snapshot(gate_vm, 2)

        def quant_body(m, unused):
            mid = gate_vm[pl.ds(m * bm, bm), :]
            q, scale = quantize_tensor(mid, w1.dtype)
            midq_vm[pl.ds(m * bm, bm), :] = q
            mids_vm[pl.ds(m * bm, bm), :] = scale
            return unused

        with jax.named_scope("shared_full_row_quant"):
            lax.fori_loop(0, rows // bm, quant_body, 0)

        snapshot(midq_vm, 3)
        snapshot(mids_vm, 4)

        def down_tile(tile, unused):
            slot = tile % 2
            load_down(tile, slot).wait()

            @pl.when(tile + 1 < hidden // down_n)
            def prefetch_next():
                load_down(tile + 1, (slot + 1) % 2).start()

            def compute(index, carry):
                n = index * bn
                accumulate(midq_vm, down_vm.at[slot], n, intermediate // bk)
                for mxu in range(2):
                    m = mxu_offsets[mxu]
                    value = pltpu.matmul_pop(accs[mxu])
                    value = value * mids_vm[pl.ds(m, mxu_rows[mxu]), :]
                    value = value * s2_ref[pl.ds(tile * down_n + n, bn)][None, :]
                    out_vm[slot, pl.ds(m, mxu_rows[mxu]), pl.ds(n, bn)] = value.astype(
                        jnp.bfloat16
                    )
                return carry

            @pl.when(tile >= 2)
            def wait_output_reuse():
                pltpu.make_async_copy(
                    out_vm.at[slot], out_vm.at[slot], out_sem.at[slot]
                ).wait()

            lax.fori_loop(0, down_n // bn, compute, 0)
            pltpu.make_async_copy(
                out_vm.at[slot],
                y_ref.at[:, pl.ds(tile * down_n, down_n)],
                out_sem.at[slot],
            ).start()
            return unused

        with jax.named_scope("shared_down"):
            lax.fori_loop(0, hidden // down_n, down_tile, 0)
        for slot in range(2):
            pltpu.make_async_copy(
                out_vm.at[slot], out_vm.at[slot], out_sem.at[slot]
            ).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    output_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    if diagnostic:
        output_shape = (
            output_shape,
            jax.ShapeDtypeStruct((rows, intermediate), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, intermediate), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, intermediate), w1.dtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        )
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(vmem, vmem, hbm, vmem, hbm, vmem),
            out_specs=(hbm,) * 5 if diagnostic else hbm,
            scratch_shapes=(
                pltpu.VMEM((2, hidden, intermediate), w1.dtype),
                pltpu.VMEM((2, intermediate, down_n), w1.dtype),
                pltpu.VMEM((rows, intermediate), jnp.bfloat16),
                pltpu.VMEM((rows, intermediate), w1.dtype),
                pltpu.VMEM((rows, 1), jnp.float32),
                pltpu.VMEM((2, rows, down_n), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            grid=(),
        ),
        name="fused_shared_gate_up_down_t288_acc",
    )(xq, xs, w1, s1, w2, s2)

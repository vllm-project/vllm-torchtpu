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
"""A Gemma-style post-attention block for decode buckets, as one weight stream.

Between the attention output's all-reduce and the MLP output's all-reduce a
Gemma4 decoder layer runs a post-attention RMSNorm, the residual add, a
pre-feedforward RMSNorm, the W8A8 gate/up projection, GeGLU (tanh GELU) and
the W8A8 down projection. For a decode bucket that is two weight streams of
tens of MB each with five small XLA passes around them. This kernel does all
of it in one call: the norms and the rows' FP8 quantization run while the
first gate/up tiles land, the down projection's first tiles are issued
during the last gate/up tiles, and the GeGLU and its quantization run while
they land.

Arithmetic follows the unfused layer: `rms_norm` (f32, weight in f32, cast to
the activation dtype), the residual add rounded to the activation dtype,
`quantize_tensor`'s per-row FP8 quantization, the FP8 dots accumulated in f32
and scaled per row and column, the gate/up output rounded to the activation
dtype, and `GeluAndMul("tanh")` in f32 rounded once. Mosaic's accumulation
order and the single GELU rounding can move low bits against XLA's.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul import decode_gemv

_SQRT_2_OVER_PI = math.sqrt(2 / math.pi)
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _rms_norm(x: jax.Array, w: jax.Array, eps: float, out_dtype) -> jax.Array:
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * lax.rsqrt(var + eps) * w.astype(jnp.float32)).astype(out_dtype)


def _gelu_tanh(x: jax.Array) -> jax.Array:
    return 0.5 * x * (1.0 + jnp.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def supports(rows: int, hidden: int, inter: int) -> bool:
    """Whether a `[rows, hidden]` bucket with `[hidden, 2 * inter]` and
    `[inter, hidden]` FP8 weights fits: whole lanes, whole FP8 row tiles, and
    both weight rings, the gate/up rows and the f32 GeGLU in VMEM. At
    Gemma4-31B TP2 it admits up to 32 rows; the deviceless tpu7x compile of
    32 and 48 rows succeeds (jax 0.11.2, libtpu 0.0.50)."""
    if not (
        rows % 16 == 0
        and hidden % decode_gemv._ROW_TILE == 0
        and hidden % decode_gemv._LANES == 0
        and inter % decode_gemv._LANES == 0
    ):
        return False
    tn1, d1, _ = decode_gemv.tile_config(hidden, 2 * inter)
    tn2, d2, _ = decode_gemv.tile_config(inter, hidden)
    mp = -(-rows // decode_gemv._ROW_TILE) * decode_gemv._ROW_TILE
    rings = d1 * hidden * tn1 + d2 * inter * tn2
    # bf16 inputs, outputs and residual, the f32 norm rows and the FP8 rows;
    # the bf16 gate/up rows.
    rows_bytes = mp * hidden * (4 * 2 + 4 + 1) + mp * 2 * inter * 2
    geglu = mp * inter * (3 * 4 + 2 + 1)  # f32 temporaries, bf16 y, FP8 y
    small = (2 * inter + 3 * hidden) * 4 + mp * (tn1 + tn2) * 4
    total = rings + rows_bytes + geglu + small
    return total <= decode_gemv._VMEM_BUDGET * _VMEM_LIMIT_BYTES


@functools.partial(jax.jit, static_argnames=("eps", "tile_bytes", "interpret"))
def decode_geglu_mlp(
    attn: jax.Array,
    residual: jax.Array,
    post_attn_norm_w: jax.Array,
    pre_ff_norm_w: jax.Array,
    w_gate_up: jax.Array,
    s_gate_up: jax.Array,
    w_down: jax.Array,
    s_down: jax.Array,
    *,
    eps: float,
    tile_bytes: int = decode_gemv._TILE_BYTES,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`(down(GeGLU(gate_up(x))), r)` with `r = post_attn_norm(attn) + residual`
    and `x = pre_ff_norm(r)`, before the down projection's all-reduce.

    `attn`, `residual` `[m, hidden]` (m a multiple of 16); FP8 `w_gate_up`
    `[hidden, 2 * inter]` ([gate | up]) and `w_down` `[inter, hidden]` in the
    canonical `(k, n)` layout with f32 per-column scales.
    """
    m, hidden = attn.shape
    inter = w_down.shape[0]
    act = attn.dtype
    q_dtype = w_gate_up.dtype
    tn1, d1, sp1 = decode_gemv.tile_config(hidden, 2 * inter, tile_bytes=tile_bytes)
    tn2, d2, sp2 = decode_gemv.tile_config(inter, hidden, tile_bytes=tile_bytes)
    nt1, nt2 = 2 * inter // tn1, hidden // tn2
    mp = -(-m // decode_gemv._ROW_TILE) * decode_gemv._ROW_TILE

    def kernel(
        attn_ref,
        res_ref,
        wpa_hbm,
        wpf_hbm,
        w1_hbm,
        s1_hbm,
        w2_hbm,
        s2_hbm,
        out_ref,
        res_out_ref,
        ring1,
        sem1,
        ring2,
        sem2,
        wpa_vmem,
        wpf_vmem,
        s1_vmem,
        s2_vmem,
        small_sem,
        xq_ref,
        gu_ref,
        yq_ref,
    ):
        kc1, kc2 = hidden // sp1, inter // sp2

        def copy1(t, slot, s):
            return pltpu.make_async_copy(
                w1_hbm.at[pl.ds(s * kc1, kc1), pl.ds(t * tn1, tn1)],
                ring1.at[slot, pl.ds(s * kc1, kc1)],
                sem1.at[slot, s],
            )

        def copy2(t, slot, s):
            return pltpu.make_async_copy(
                w2_hbm.at[pl.ds(s * kc2, kc2), pl.ds(t * tn2, tn2)],
                ring2.at[slot, pl.ds(s * kc2, kc2)],
                sem2.at[slot, s],
            )

        norm1 = pltpu.make_async_copy(wpa_hbm, wpa_vmem, small_sem.at[0])
        norm2 = pltpu.make_async_copy(wpf_hbm, wpf_vmem, small_sem.at[1])
        scales1 = pltpu.make_async_copy(s1_hbm, s1_vmem, small_sem.at[2])
        scales2 = pltpu.make_async_copy(s2_hbm, s2_vmem, small_sem.at[3])
        for c in (norm1, norm2, scales1, scales2):
            c.start()
        for t in range(d1):
            for s in range(sp1):
                copy1(t, t, s).start()

        norm1.wait()
        norm2.wait()
        h = _rms_norm(attn_ref[...], wpa_vmem[...].reshape(1, hidden), eps, act)
        r = (h.astype(jnp.float32) + res_ref[...].astype(jnp.float32)).astype(act)
        res_out_ref[...] = r
        x = _rms_norm(r, wpf_vmem[...].reshape(1, hidden), eps, act)
        xq, xs = decode_gemv._quantize_rows(decode_gemv._pad_rows(x, mp), q_dtype)
        xq_ref[...] = xq

        def gate_up_body(t, carry):
            slot = lax.rem(t, d1)
            for s in range(sp1):
                copy1(t, slot, s).wait()
            # Issued ahead of the tiles, the scales have landed with the first.
            pl.when(t == 0)(scales1.wait)
            acc = jnp.dot(xq_ref[...], ring1[slot], preferred_element_type=jnp.float32)
            col = pl.multiple_of(t * tn1, decode_gemv._LANES)
            out = acc * xs * s1_vmem[pl.ds(col, tn1)].reshape(1, tn1)
            gu_ref[:, pl.ds(col, tn1)] = out.astype(act)
            nxt = t + d1

            @pl.when(nxt < nt1)
            def _():
                for s in range(sp1):
                    copy1(nxt, slot, s).start()

            # The down projection's first tiles start, into their own ring,
            # in the iterations where the gate/up ring stops refilling.
            for i in range(min(d1, d2)):

                @pl.when(nxt - nt1 == i)
                def _():
                    for s in range(sp2):
                        copy2(i, i, s).start()

            return carry

        lax.fori_loop(0, nt1, gate_up_body, 0)
        for i in range(d1, d2):  # first down tiles past the gate/up loop's end
            for s in range(sp2):
                copy2(i, i, s).start()

        gu = gu_ref[...].astype(jnp.float32)
        y = (_gelu_tanh(gu[:, :inter]) * gu[:, inter:]).astype(act)
        yq, ys = decode_gemv._quantize_rows(y, q_dtype)
        yq_ref[...] = yq
        scales2.wait()

        def down_body(t, carry):
            slot = lax.rem(t, d2)
            for s in range(sp2):
                copy2(t, slot, s).wait()
            acc = jnp.dot(yq_ref[...], ring2[slot], preferred_element_type=jnp.float32)
            col = pl.multiple_of(t * tn2, decode_gemv._LANES)
            out = acc * ys * s2_vmem[pl.ds(col, tn2)].reshape(1, tn2)
            out_ref[:, pl.ds(col, tn2)] = out[:m].astype(out_ref.dtype)

            @pl.when(t + d2 < nt2)
            def _():
                for s in range(sp2):
                    copy2(t + d2, slot, s).start()

            return carry

        lax.fori_loop(0, nt2, down_body, 0)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, hidden), act),
            jax.ShapeDtypeStruct((m, hidden), act),
        ),
        in_specs=[vmem, vmem, hbm, hbm, hbm, hbm, hbm, hbm],
        out_specs=(vmem, vmem),
        scratch_shapes=[
            pltpu.VMEM((d1, hidden, tn1), q_dtype),
            pltpu.SemaphoreType.DMA((d1, sp1)),
            pltpu.VMEM((d2, inter, tn2), q_dtype),
            pltpu.SemaphoreType.DMA((d2, sp2)),
            pltpu.VMEM((hidden,), post_attn_norm_w.dtype),
            pltpu.VMEM((hidden,), pre_ff_norm_w.dtype),
            pltpu.VMEM((2 * inter,), jnp.float32),
            pltpu.VMEM((hidden,), jnp.float32),
            pltpu.SemaphoreType.DMA((4,)),
            pltpu.VMEM((mp, hidden), q_dtype),
            pltpu.VMEM((mp, 2 * inter), act),
            pltpu.VMEM((mp, inter), q_dtype),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        name=f"decode_geglu_mlp_tn{tn1}x{tn2}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(
        attn,
        residual,
        decode_gemv._hbm(post_attn_norm_w),
        decode_gemv._hbm(pre_ff_norm_w),
        decode_gemv._hbm(w_gate_up),
        decode_gemv._hbm(s_gate_up.astype(jnp.float32)),
        decode_gemv._hbm(w_down),
        decode_gemv._hbm(s_down.astype(jnp.float32)),
    )

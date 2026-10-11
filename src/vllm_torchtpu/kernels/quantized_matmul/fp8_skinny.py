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
"""FP8 dense matmul for skinny activations (decode, M <= 16), weight streamed from HBM.

At decode a per-channel FP8 linear is a weight stream: 16 rows against a 6-34 MB weight.
XLA stages the whole weight into VMEM before the MXU starts on several of these sites
(or streams it in a few windows that each pay a fixed cost), so the call takes about
twice the weight's read time. This kernel keeps the weight in HBM and walks it through a
small VMEM ring with several DMAs in flight, in one grid step, computing each tile as it
lands.

Numerics: the activation is quantized in-kernel with `util.quantize_block`'s formula as
XLA compiles it -- scale = f32(amax|x|) * f32(1/fp8_max), scale == 0 -> 1,
x_q = fp8(clip(f32(x) * (1/scale))) -- bit-identical x_q and scale. x_q is then held as
bf16 (every e4m3 value is exact in bf16) and fed to a bf16 x fp8 MXU dot, whose f32
accumulation is exact. XLA's fp8 x fp8 dot accumulates below f32 precision (0.4-0.6%
from f64 at these shapes), so outputs differ from `xla_quantized_matmul` and sit closer
to the f64 product of the same quantized operands.

Tilings, both NN on a K-major `[K, N]` weight with N % 128 == 0 (a weight whose N is
not gets a zero-padded copy at load, `padded_n`):
  'n': tiles W[:, j*t:(j+1)*t] (strided windows); each tile's output columns are
       written once.
  'k': tiles W[j*t:(j+1)*t, :] (contiguous rows); f32 accumulator in VMEM.
An N-major weight would need an NT dot, whose transposed tile is relaid through the XLU
before the push and costs about as much as the stream itself (fp8-gemm probe 04).
The ring follows tokamax `fused_moe_rs`: the prologue starts `nbuf` copies; the steady
state waits slot j, computes, and starts tile j + nbuf into the same slot. The tile loop
is unrolled, so slots and lane offsets are static.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 16
LANE = 128

# Measured per site on tpu7x (GLM-5.2-FP8, TP16 q-replicate decode, cold weight)
# in write-tpu-kernels fp8-gemm probes 03-04: fastest config, ties to the smaller ring.
TUNED = {  # (k, n) -> (split, tile, nbuf)
    (6144, 2688): ("k", 1024, 3),  # q_a + kv_a (padded copy of 2624)
    (2048, 16384): ("n", 1024, 4),  # q_b
    (1024, 6144): ("n", 2048, 3),  # o_proj
    (6144, 1536): ("k", 1536, 3),  # dense gate_up
}


def pick_config(k: int, n: int) -> tuple[str, int, int]:
    if (k, n) in TUNED:
        return TUNED[(k, n)]
    for t in (2048, 1024, 512, 256, 128):
        if n % t == 0 and n // t >= 3:
            return ("n", t, 3)
    return ("n", n, 1)


def supports(m: int, w_q: jax.Array, w_s: jax.Array) -> bool:
    """True when this kernel serves the call: M <= 16, per-channel fp8, and a (K, N)
    with a measured config. Other shapes stay on XLA: an untuned shape's ring could
    exceed VMEM, and nothing says it would win."""
    return (
        m <= MAX_M
        and w_q.ndim == 2
        and w_s.ndim == 1
        and w_q.dtype == jnp.float8_e4m3fn
        and tuple(w_q.shape) in TUNED
        and w_s.shape[0] == w_q.shape[1]
    )


def _kernel(
    idx_ref,
    x_ref,
    ws_ref,
    w_hbm,
    o_ref,
    xq_ref,
    xs_ref,
    wbuf,
    sem,
    *acc,
    split,
    t,
    n_tiles,
    nbuf,
    q_dtype,
):
    # Weight slice: 0 in production (a [1, ...] view); a stack index in the benchmark.
    s = idx_ref[0]

    def copy(j):
        if split == "n":
            src = w_hbm.at[s, :, pl.ds(j * t, t)]
        else:
            src = w_hbm.at[s, pl.ds(j * t, t), :]
        return pltpu.make_async_copy(src, wbuf.at[j % nbuf], sem.at[j % nbuf])

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()

    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        copy(j).wait()
        w = wbuf[j % nbuf]
        if split == "n":
            cols = slice(j * t, (j + 1) * t)
            y = jax.lax.dot_general(
                xq_ref[...], w, dn, preferred_element_type=jnp.float32
            )
            o_ref[:, cols] = (y * xs_ref[...] * ws_ref[:, cols]).astype(o_ref.dtype)
        else:
            y = jax.lax.dot_general(
                xq_ref[:, j * t : (j + 1) * t],
                w,
                dn,
                preferred_element_type=jnp.float32,
            )
            if j == 0:
                acc[0][...] = y
            else:
                acc[0][...] += y
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
    if split == "k":
        o_ref[...] = (acc[0][...] * xs_ref[...] * ws_ref[...]).astype(o_ref.dtype)


@functools.cache
def _build(m, k, n, n_stack, w_dtype, out_dtype, split, t, nbuf):
    full = k if split == "k" else n
    if full % t or t % LANE:
        raise ValueError(
            f"tile {t} does not divide {split}-extent {full} in lane multiples"
        )
    n_tiles = full // t
    nbuf = max(1, min(nbuf, n_tiles))
    tile = (k, t) if split == "n" else (t, n)
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),
        pltpu.VMEM((m, 1), jnp.float32),
        pltpu.VMEM((nbuf, *tile), w_dtype),
        pltpu.SemaphoreType.DMA((nbuf,)),
    ]
    if split == "k":
        scratch.append(pltpu.VMEM((m, n), jnp.float32))
    kern = functools.partial(
        _kernel, split=split, t=t, n_tiles=n_tiles, nbuf=nbuf, q_dtype=w_dtype
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g, idx: (0, 0)),
            pl.BlockSpec((1, n), lambda g, idx: (0, 0)),
            # W [s, k, n], pinned to HBM (MSA cannot move it into VMEM).
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, n), lambda g, idx: (0, 0)),
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"fp8_skinny_{split}{t}_b{nbuf}_m{m}_k{k}_n{n}",
    )


def fp8_skinny_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    *,
    n_logical: int | None = None,
    config: tuple[str, int, int] | None = None,
) -> jax.Array:
    """x [m<=16, k] bf16, w_q fp8 [k, n] with n % 128 == 0, w_s f32 [n]
    -> [m, n_logical or n] in x.dtype."""
    return fp8_skinny_matmul_stacked(
        x,
        w_q[None],
        jnp.zeros((1,), jnp.int32),
        w_s,
        n_logical=n_logical,
        config=config,
    )


def fp8_skinny_matmul_stacked(x, w_stack, idx, w_s, *, n_logical=None, config=None):
    """As `fp8_skinny_matmul`; the kernel reads the weight from `w_stack[idx[0]]`."""
    m, k = x.shape
    n_stack, _, n = w_stack.shape
    split, t, nbuf = config or pick_config(k, n)
    call = _build(m, k, n, n_stack, w_stack.dtype, x.dtype, split, t, nbuf)
    out = call(idx, x, w_s.reshape(1, n).astype(jnp.float32), w_stack)
    if n_logical is not None and n_logical != n:
        out = out[:, :n_logical]
    return out


def padded_n(n: int, multiple: int = LANE) -> int:
    """The output width a weight is padded to at load (zero outputs, unit scale) so
    the kernel tiles it exactly."""
    return -(-n // multiple) * multiple

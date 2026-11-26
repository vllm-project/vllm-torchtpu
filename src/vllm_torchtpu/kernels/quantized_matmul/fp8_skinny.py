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
"""FP8 dense matmul for skinny activations (M <= 16), weight streamed from HBM."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 16
LANE = 128
DN = (((1,), (0,)), ((), ()))
VMEM = pl.BlockSpec(memory_space=pltpu.VMEM)
HBM = pl.BlockSpec(memory_space=pltpu.HBM)

TUNED: dict[tuple[int, int], tuple[str, int, int]] = {
    (6144, 2688): ("k", 1024, 3),
    (2048, 16384): ("n", 1024, 4),
    (1024, 6144): ("n", 2048, 3),
    (6144, 1536): ("k", 1536, 3),
}


def pick_config(k: int, n: int) -> tuple[str, int, int]:
    return TUNED[(k, n)]


def supports(m: int, w_q: jax.Array, w_s: jax.Array) -> bool:
    return (
        0 < m <= MAX_M
        and w_q.ndim == 2
        and w_s.ndim == 1
        and w_q.dtype == jnp.float8_e4m3fn
        and tuple(w_q.shape) in TUNED
        and w_s.shape[0] == w_q.shape[1]
    )


def quantize_into(x_ref, xq_ref, xs_ref, q_dtype) -> None:
    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    s = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(jnp.float32) * jnp.float32(
        1.0 / q_max
    )
    s = jnp.where(s == 0, jnp.float32(1.0), s)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / s), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = s


def _kernel(
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
    scale_dma,
):
    ws_copy = None
    if scale_dma:
        ws_vmem, ws_sem, acc = acc[-2], acc[-1], acc[:-2]
        ws_copy = pltpu.make_async_copy(ws_ref, ws_vmem, ws_sem)
        ws_copy.start()
        ws_ref = ws_vmem

    def copy(j):
        src = (
            w_hbm.at[:, pl.ds(j * t, t)]
            if split == "n"
            else w_hbm.at[pl.ds(j * t, t), :]
        )
        return pltpu.make_async_copy(src, wbuf.at[j % nbuf], sem.at[j % nbuf])

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()
    quantize_into(x_ref, xq_ref, xs_ref, q_dtype)
    for j in range(n_tiles):
        copy(j).wait()
        lhs = xq_ref[...] if split == "n" else xq_ref[:, j * t : (j + 1) * t]
        y = jax.lax.dot_general(
            lhs, wbuf[j % nbuf], DN, preferred_element_type=jnp.float32
        )
        if split == "n":
            if ws_copy is not None and j == 0:
                ws_copy.wait()
            cols = slice(j * t, (j + 1) * t)
            o_ref[:, cols] = (y * xs_ref[...] * ws_ref[:, cols]).astype(o_ref.dtype)
        elif j == 0:
            acc[0][...] = y
        else:
            acc[0][...] += y
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
    if split == "k":
        if ws_copy is not None:
            ws_copy.wait()
        o_ref[...] = (acc[0][...] * xs_ref[...] * ws_ref[...]).astype(o_ref.dtype)


@functools.cache
def _build(
    m, k, n, w_dtype, out_dtype, split, t, nbuf, scale_dma=False, interpret=False
):
    full = k if split == "k" else n
    if full % t or t % LANE:
        raise ValueError(
            f"tile {t} must divide {split}={full} in {LANE}-lane multiples"
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
    if scale_dma:
        scratch += [pltpu.VMEM((1, n), jnp.float32), pltpu.SemaphoreType.DMA]
    kern = functools.partial(
        _kernel,
        split=split,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        scale_dma=scale_dma,
    )
    sd = "_sd" if scale_dma else ""
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        in_specs=[VMEM, HBM if scale_dma else VMEM, HBM],
        out_specs=VMEM,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(dimension_semantics=()),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fp8_skinny_{split}{t}_b{nbuf}{sd}_m{m}_k{k}_n{n}",
    )


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def _call_constrained(call, x, w_s, w_q, scale_hbm, weight_hbm):
    if scale_hbm:
        w_s = pltpu.with_memory_space_constraint(w_s, pltpu.HBM)
    if weight_hbm:
        w_q = pltpu.with_memory_space_constraint(w_q, pltpu.HBM)
    return call(x, w_s, w_q)


def fp8_skinny_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    *,
    n_logical: int | None = None,
    config: tuple[str, int, int] | None = None,
    scale_dma: bool = False,
    hbm_pin: bool = False,
    interpret: bool = False,
) -> jax.Array:
    m, k = x.shape
    _, n = w_q.shape
    split, t, nbuf = config or pick_config(k, n)
    call = _build(m, k, n, w_q.dtype, x.dtype, split, t, nbuf, scale_dma, interpret)
    w_s2 = w_s.reshape(1, n).astype(jnp.float32)
    out = (
        _call_constrained(call, x, w_s2, w_q, scale_dma, hbm_pin)
        if (scale_dma or hbm_pin) and not interpret
        else call(x, w_s2, w_q)
    )
    return out[:, :n_logical] if n_logical is not None and n_logical != n else out


def window_supports(k: int, n: int, width: int) -> bool:
    config = TUNED.get((k, n))
    return config is not None and config[0] == "n" and width % config[1] == 0


SCALE_DMA_SITES = {(1024, 6144)}


def scale_dma_applies(k: int, n: int) -> bool:
    return (k, n) in SCALE_DMA_SITES


def padded_n(n: int, multiple: int = LANE) -> int:
    return -(-n // multiple) * multiple

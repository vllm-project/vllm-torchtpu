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
"""W8A16 matmul for MiniMax M3 MXFP8 dense linears at decode row counts.

``x [m, k] bf16 @ (w [k, n] e4m3 * scale [k // 32, n])``. The kernel starts the
DMAs of all K chunks of the weight at once, then for each chunk waits, converts
it to bf16, multiplies by its power-of-two block scale (exact) and accumulates
the chunk's matmul in f32. The MXU sees the same bf16 weight values that
load-time dequantization produces, while HBM streams 1 + 1/16 B per weight
instead of 2 B.
"""

import functools
from collections.abc import Callable
from threading import Lock

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MXFP8_BLOCK = 32
# K -> number of K chunks, measured on v7x at the per-rank TP8 shapes (qkv and
# shared gate_up K=6144, o_proj K=1024, shared down K=384). Larger weights (the
# dense layers' MLP) measured slower than bf16 and keep the bf16 path.
_CHUNKS = {6144: 3, 1024: 8, 384: 3}
_MAX_ELEMS = 6144 * 1536

_ANY = getattr(pl, "ANY", None) or getattr(pltpu, "ANY", None)


def supports_shape(k: int, n: int) -> bool:
    return k in _CHUNKS and k * n <= _MAX_ELEMS and n % 128 == 0


def _kernel(x_ref, s_ref, w_hbm, o_ref, wbuf, sem, *, chunks, kc, n):
    copies = [
        pltpu.make_async_copy(w_hbm.at[pl.ds(c * kc, kc), :], wbuf.at[c], sem.at[c])
        for c in range(chunks)
    ]
    for cp in copies:
        cp.start()
    acc = None
    for c in range(chunks):
        copies[c].wait()
        w = wbuf[c].astype(jnp.bfloat16)
        w = (w.reshape(kc // MXFP8_BLOCK, MXFP8_BLOCK, n) * s_ref[c][:, None, :]).reshape(
            kc, n
        )
        part = jnp.dot(
            x_ref[:, c * kc : (c + 1) * kc], w, preferred_element_type=jnp.float32
        )
        acc = part if acc is None else acc + part
    o_ref[...] = acc.astype(o_ref.dtype)


@jax.jit
def mxfp8_w8a16_matmul(x: jax.Array, w: jax.Array, scale: jax.Array) -> jax.Array:
    """x [m, k] bf16, w [k, n] float8_e4m3fn, scale [k // 32, n] bf16 -> [m, n]."""
    m, k = x.shape
    n = w.shape[1]
    chunks = _CHUNKS[k]
    kc = k // chunks
    s3 = scale.reshape(chunks, kc // MXFP8_BLOCK, n)
    return pl.pallas_call(
        functools.partial(_kernel, chunks=chunks, kc=kc, n=n),
        in_specs=[
            pl.BlockSpec((m, k), lambda: (0, 0)),
            pl.BlockSpec((chunks, kc // MXFP8_BLOCK, n), lambda: (0, 0, 0)),
            pl.BlockSpec(memory_space=_ANY),
        ],
        out_specs=pl.BlockSpec((m, n), lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=[
            pltpu.VMEM((chunks, kc, n), w.dtype),
            pltpu.SemaphoreType.DMA((chunks,)),
        ],
        name="mxfp8_w8a16_matmul",
    )(x, s3, w)


_op: Callable | None = None
_op_lock = Lock()


def mxfp8_w8a16_op() -> Callable:
    """The cached torch op; call once eagerly before tracing a forward."""
    global _op
    if _op is not None:
        return _op
    with _op_lock:
        if _op is None:
            from torch_tpu._internal import pallas

            op = pallas.jax_op("pallas::minimax_mxfp8_w8a16", mxfp8_w8a16_matmul)

            def _fake(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor):
                del scale
                return torch.empty(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)

            op.register_fake(_fake)
            _op = op
    return _op


def mxfp8_w8a16_linear(
    x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Torch entry: ``x [..., k] @ w8a16(w [k, n], scale [k // 32, n])``."""
    out_shape = (*x.shape[:-1], w.shape[1])
    return mxfp8_w8a16_op()(x.reshape(-1, x.shape[-1]), w, scale).reshape(out_shape)

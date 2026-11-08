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
"""mHC gating / softmax / Sinkhorn as one Pallas kernel.

The gates act on `(T, 2M + M*M)` mix logits (M = hc_mult = 4, so 24 values
per token). Written as jnp, the Sinkhorn loop unrolls into ~100 tiny XLA ops
per call -- every row/column normalization is its own reduce fusion -- and a
45-layer model calls it 90 times per step, so at decode the step pays launch
overhead for ~10K ops that each move a few bytes (~1 ms of an 8.6 ms
GLM-5.3-Flash decode step).

Here the token axis goes on lanes and the 24 values on sublanes, so a row or
column sum of the M x M matrix is a handful of adds over fixed sublane rows,
and all iterations stay in vregs.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

LANE = 128
DEFAULT_TOKEN_BLOCK = 512


def _gates_kernel(
    mix_ref,  # [2M + M*M, tb] f32
    sqrsum_ref,  # [1, tb] f32
    scale_ref,  # [3, 1] f32
    base_ref,  # [2M + M*M, 1] f32
    out_ref,  # [2M + M*M, tb] f32
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    m = hc_mult
    norm = jax.lax.rsqrt(sqrsum_ref[...] / (m * hidden_size) + rms_eps)
    mix = mix_ref[...] * norm
    base = base_ref[...]
    scale_pre = scale_ref[0:1, :]
    scale_post = scale_ref[1:2, :]
    scale_comb = scale_ref[2:3, :]

    def row(k):
        return mix[k : k + 1, :]

    def bias(k):
        return base[k : k + 1, :]

    out = [jax.nn.sigmoid(row(k) * scale_pre + bias(k)) + hc_pre_eps for k in range(m)]
    out += [
        jax.nn.sigmoid(row(k) * scale_post + bias(k)) * hc_post_mult_value
        for k in range(m, 2 * m)
    ]

    comb = [
        [
            row(2 * m + i * m + j) * scale_comb + bias(2 * m + i * m + j)
            for j in range(m)
        ]
        for i in range(m)
    ]
    # softmax over the last axis, then + eps: matches `utils.mhc_pre_gates`.
    for i in range(m):
        peak = functools.reduce(jnp.maximum, comb[i])
        e = [jnp.exp(c - peak) for c in comb[i]]
        inv = 1.0 / functools.reduce(jnp.add, e)
        comb[i] = [x * inv + hc_sinkhorn_eps for x in e]

    def normalize_cols(c):
        for j in range(m):
            inv = 1.0 / (
                functools.reduce(jnp.add, [c[i][j] for i in range(m)]) + hc_sinkhorn_eps
            )
            for i in range(m):
                c[i][j] = c[i][j] * inv
        return c

    def normalize_rows(c):
        for i in range(m):
            inv = 1.0 / (functools.reduce(jnp.add, c[i]) + hc_sinkhorn_eps)
            c[i] = [x * inv for x in c[i]]
        return c

    comb = normalize_cols(comb)
    for _ in range(sinkhorn_repeat - 1):
        comb = normalize_cols(normalize_rows(comb))

    out += [comb[i][j] for i in range(m) for j in range(m)]
    out_ref[...] = jnp.concatenate(out, axis=0)


def mhc_pre_gates_pallas(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = DEFAULT_TOKEN_BLOCK,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Drop-in for `utils.mhc_pre_gates`; same arguments and results."""
    num_tokens, width = mixes.shape
    padded = -(-num_tokens // LANE) * LANE
    block = min(token_block_size, padded)
    padded = -(-padded // block) * block

    mixes_t = jnp.pad(mixes.astype(jnp.float32).T, ((0, 0), (0, padded - num_tokens)))
    # Pad rows get sqrsum 1 so the rsqrt stays finite; they are sliced off.
    sqrsum_t = jnp.pad(
        sqrsum.astype(jnp.float32).reshape(1, num_tokens),
        ((0, 0), (0, padded - num_tokens)),
        constant_values=1.0,
    )
    out = pl.pallas_call(
        functools.partial(
            _gates_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        ),
        grid=(padded // block,),
        in_specs=[
            pl.BlockSpec((width, block), lambda i: (0, i)),
            pl.BlockSpec((1, block), lambda i: (0, i)),
            pl.BlockSpec((3, 1), lambda i: (0, 0)),
            pl.BlockSpec((width, 1), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((width, block), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((width, padded), jnp.float32),
    )(
        mixes_t,
        sqrsum_t,
        hc_scale.astype(jnp.float32).reshape(3, 1),
        hc_base.astype(jnp.float32).reshape(width, 1),
    )
    out = out[:, :num_tokens].T
    pre_mix = out[:, :hc_mult]
    post_mix = out[:, hc_mult : 2 * hc_mult]
    comb_mix = out[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult)
    return pre_mix, post_mix, comb_mix

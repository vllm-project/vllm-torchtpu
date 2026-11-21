# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Engram gate: mix the looked-up value into the hc streams.

One pass over the flat ``[T, M*D]`` residual. For each stream ``m`` it
computes the normalized dot of the stream against its key, the signed-sqrt
sigmoid gate, and ``stream + gate * value``. This is the arithmetic of
``Engram.forward`` (and of upstream's fused CUDA kernel): FP32 math on BF16
inputs, one BF16 rounding of the output. Only the order of the three
per-stream sums over ``D`` differs from the XLA reductions.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
# Columns per step of the per-stream loops: four vregs of FP32 per row group.
COLUMNS = 512
VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def token_tile(tokens: int, max_tile: int = 64) -> int:
    """Tokens per grid step: all of a small batch, else a divisor that keeps
    BF16 rows sublane aligned (16)."""
    if tokens <= max_tile:
        return tokens
    for tile in range(max_tile, 15, -16):
        if tokens % tile == 0:
            return tile
    return 16


def _gate_kernel(
    hidden_ref, kv_ref, q_ref, k_ref, *rest, mult, dim, eps, clamp, masked
):
    if masked:
        mask_ref, out_ref = rest
    else:
        (out_ref,) = rest
    tile = hidden_ref.shape[0]
    columns = COLUMNS if dim % COLUMNS == 0 else dim

    def lane_sum(x):
        return functools.reduce(
            jnp.add, [x[:, i : i + LANES] for i in range(0, x.shape[1], LANES)]
        )

    gates = []
    for m in range(mult):
        dot = jnp.zeros((tile, LANES), jnp.float32)
        hidden_squares = jnp.zeros((tile, LANES), jnp.float32)
        key_squares = jnp.zeros((tile, LANES), jnp.float32)
        for c in range(0, dim, columns):
            hidden = hidden_ref[:, pl.ds(m * dim + c, columns)].astype(jnp.float32)
            key = kv_ref[:, pl.ds(m * dim + c, columns)].astype(jnp.float32)
            q = q_ref[pl.ds(m, 1), pl.ds(c, columns)]
            k = k_ref[pl.ds(m, 1), pl.ds(c, columns)]
            dot = dot + lane_sum(hidden * q * k * key)
            hidden_squares = hidden_squares + lane_sum(hidden * hidden)
            key_squares = key_squares + lane_sum(key * key)
        dot = jnp.sum(dot, axis=1, keepdims=True)
        hidden_mean = jnp.sum(hidden_squares, axis=1, keepdims=True) / dim
        key_mean = jnp.sum(key_squares, axis=1, keepdims=True) / dim
        dot = dot * jax.lax.rsqrt(hidden_mean + eps)
        dot = dot * (jax.lax.rsqrt(key_mean + eps) * dim**-0.5)
        magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp))
        gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude))
        if masked:
            gate = gate * mask_ref[...]
        gates.append(gate)
    for c in range(0, dim, columns):
        value = kv_ref[:, pl.ds(mult * dim + c, columns)].astype(jnp.float32)
        for m in range(mult):
            cols = pl.ds(m * dim + c, columns)
            hidden = hidden_ref[:, cols].astype(jnp.float32)
            out_ref[:, cols] = (hidden + gates[m] * value).astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "clamp", "max_tile", "interpret"))
def engram_gate(
    hidden: jax.Array,
    kv: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    token_mask: jax.Array | None = None,
    *,
    eps: float,
    clamp: float,
    max_tile: int = 64,
    interpret: bool = False,
) -> jax.Array:
    """``hidden + gate * value`` for every hc stream.

    Args:
        hidden: [T, M*D] BF16, the M hc streams side by side.
        kv: [T, (M+1)*D] BF16, the M keys then the shared value.
        q_weight, k_weight: [M, D].
        token_mask: [T], zero (False) closes a token's gate; None keeps all.

    Returns:
        [T, M*D] in ``hidden.dtype``.
    """
    tokens, width = hidden.shape
    mult, dim = q_weight.shape
    assert width == mult * dim and kv.shape == (tokens, (mult + 1) * dim)
    assert k_weight.shape == (mult, dim) and dim % LANES == 0
    tile = token_tile(tokens, max_tile)
    padded = -(-tokens // tile) * tile
    masked = token_mask is not None
    inputs = [hidden, kv]
    if masked:
        inputs.append(token_mask.astype(jnp.float32).reshape(tokens, 1))
    if padded != tokens:
        inputs = [jnp.pad(a, ((0, padded - tokens), (0, 0))) for a in inputs]
    hidden, kv, *mask = inputs

    def rows(width):
        return pl.BlockSpec((tile, width), lambda i: (i, 0))

    whole = pl.BlockSpec((mult, dim), lambda i: (0, 0))
    out = pl.pallas_call(
        functools.partial(
            _gate_kernel, mult=mult, dim=dim, eps=eps, clamp=clamp, masked=masked
        ),
        grid=(padded // tile,),
        in_specs=[rows(width), rows(kv.shape[1]), whole, whole]
        + ([rows(1)] if masked else []),
        out_specs=rows(width),
        out_shape=jax.ShapeDtypeStruct((padded, width), hidden.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT_BYTES
        ),
        interpret=interpret,
    )(hidden, kv, q_weight.astype(jnp.float32), k_weight.astype(jnp.float32), *mask)
    return out[:tokens] if padded != tokens else out


def engram_gate_reference(
    hidden, kv, q_weight, k_weight, token_mask=None, *, eps, clamp
):
    """``Engram.forward``'s gate arithmetic in XLA, on the [T, M, D] view."""
    tokens = hidden.shape[0]
    mult, dim = q_weight.shape
    streams = hidden.reshape(tokens, mult, dim).astype(jnp.float32)
    keys = kv[:, : mult * dim].reshape(tokens, mult, dim).astype(jnp.float32)
    value = kv[:, mult * dim :].astype(jnp.float32)
    dot = (streams * q_weight * k_weight * keys).sum(-1)
    dot = dot * jax.lax.rsqrt(jnp.mean(jnp.square(streams), -1) + eps)
    dot = dot * (jax.lax.rsqrt(jnp.mean(jnp.square(keys), -1) + eps) * dim**-0.5)
    magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude))
    if token_mask is not None:
        gate = gate * token_mask[:, None]
    out = streams + gate[:, :, None] * value[:, None, :]
    return out.astype(hidden.dtype).reshape(tokens, mult * dim)

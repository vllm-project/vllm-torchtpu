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
"""DeepSeek-V4.1 Pallas mHC post-mix residual recombination kernel."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16_TILE_ROWS = 16


def _default_vmem_limit_bytes() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes) * 3 // 4


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int],
    vmem_limit_bytes: int | None = None,
) -> tuple[int, int]:
    """Select sublane-aligned ``(token_block, padded_tokens)`` fitting VMEM."""
    limit = (
        _default_vmem_limit_bytes() if vmem_limit_bytes is None else vmem_limit_bytes
    )
    tb = min(token_block_size, _round_up(num_tokens, BF16_TILE_ROWS))
    while tb > BF16_TILE_ROWS and vmem_need(tb) > limit:
        tb //= 2
    return tb, _round_up(num_tokens, tb)


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Zero-pad the leading token axis of each 2D array to ``padded_tokens``."""
    pad = padded_tokens - arrays[0].shape[0]
    if pad == 0:
        return arrays
    return tuple(jnp.pad(arr, ((0, pad), (0, 0))) for arr in arrays)


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Trim leading token rows added by ``pad_to`` back to ``num_tokens``."""
    if arrays[0].shape[0] == num_tokens:
        return arrays
    return tuple(arr[:num_tokens] for arr in arrays)


def _post_kernel(
    x_ref: jax.Ref,
    res_ref: jax.Ref,
    post_ref: jax.Ref,
    comb_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    hc_mult: int,
    hidden_size: int,
) -> None:
    x = x_ref[...].astype(jnp.float32)
    post = post_ref[...]
    comb = comb_ref[...]

    streams = [
        res_ref[:, i * hidden_size : (i + 1) * hidden_size].astype(jnp.float32)
        for i in range(hc_mult)
    ]
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * streams[i]
        out_ref[:, j * hidden_size : (j + 1) * hidden_size] = acc.astype(out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("token_block_size", "vmem_limit_bytes", "interpret"),
)
def mhc_post_2d(
    x: jax.Array,
    res2d: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Recombine ``bf16[T, D]`` output into flat ``bf16[T, M * D]`` residual streams.

    Mixes ``res2d: bf16[T, M * D]`` with ``post_layer_mix: f32[T, M[, 1]]`` and
    ``comb_res_mix: f32[T, M, M]``.
    """
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post_layer_mix.dtype == jnp.float32, post_layer_mix.dtype
    assert comb_res_mix.dtype == jnp.float32, comb_res_mix.dtype

    hc_mult = comb_res_mix.shape[-1]
    hidden_size = res2d.shape[-1] // hc_mult

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    num_tokens = res2d.shape[0]
    limit = (
        _default_vmem_limit_bytes() if vmem_limit_bytes is None else vmem_limit_bytes
    )

    def _vmem_need(tb: int) -> int:
        # Avoid underestimating VMEM with single-buffered bytes: double-buffered
        # bf16 streams plus f32 accumulators need 2x headroom to prevent spills.
        return 2 * (
            tb * hidden_size * (2 + 2 * hc_mult * 2) * 2 + tb * hidden_size * 4 * 2
        )

    tb, padded_tokens = select_token_block(
        num_tokens,
        token_block_size,
        vmem_need=_vmem_need,
        vmem_limit_bytes=limit,
    )
    x2d, res2d, post2d, comb2d = pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=limit,
        disable_bounds_checks=True,
    )

    out = pl.pallas_call(
        functools.partial(_post_kernel, hc_mult=hc_mult, hidden_size=hidden_size),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(
            (padded_tokens, hc_mult * hidden_size), res2d.dtype
        ),
        compiler_params=compiler_params,
        interpret=interpret,
    )(x2d, res2d, post2d, comb2d)

    (out,) = trim_to(num_tokens, out)
    return out


def mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Recombine ``x=(..., D)`` into bfloat16 ``residual=(..., M, D)`` streams."""
    assert residual.dtype == jnp.bfloat16, residual.dtype

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]

    out2d = mhc_post_2d(
        x.reshape(-1, hidden_size),
        residual.reshape(-1, hc_mult * hidden_size),
        post_layer_mix.reshape(-1, hc_mult, 1),
        comb_res_mix.reshape(-1, hc_mult, hc_mult),
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
        interpret=interpret,
    )
    return out2d.reshape(*outer_shape, hc_mult, hidden_size)


__all__ = [
    "BF16_TILE_ROWS",
    "mhc_post",
    "mhc_post_2d",
    "pad_to",
    "select_token_block",
    "trim_to",
]

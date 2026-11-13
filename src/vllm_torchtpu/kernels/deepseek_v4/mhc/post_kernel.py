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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES


def _post_kernel(x_ref, res_ref, post_ref, comb_ref, out_ref, *, hc_mult, hidden_size):
    """Strip-tiled stream recombine avoiding multi-stream f32 register spills."""
    post = post_ref[...]  # (tb, hc_mult) f32
    comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)
    dh = 1280 if hidden_size % 1280 == 0 else hidden_size
    num_strips = hidden_size // dh

    for s in range(num_strips):
        s_lo = s * dh
        s_hi = s_lo + dh
        x_f32 = x_ref[:, s_lo:s_hi].astype(jnp.float32)
        accs = [post[:, j : j + 1] * x_f32 for j in range(hc_mult)]
        for i in range(hc_mult):
            old_i = res_ref[
                :, i * hidden_size + s_lo : i * hidden_size + s_hi
            ].astype(jnp.float32)
            for j in range(hc_mult):
                k = i * hc_mult + j
                accs[j] = accs[j] + comb[:, k : k + 1] * old_i
        for j in range(hc_mult):
            out_ref[:, j * hidden_size + s_lo : j * hidden_size + s_hi] = accs[
                j
            ].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("token_block_size", "vmem_limit_bytes"))
def mhc_post_2d(
    x: jax.Array,
    res2d: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 128,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """2D-native entry: the kernel's own layout as public API."""
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post_layer_mix.dtype == jnp.float32, post_layer_mix.dtype
    assert comb_res_mix.dtype == jnp.float32, comb_res_mix.dtype

    hc_mult = comb_res_mix.shape[-1]
    hidden_size = res2d.shape[-1] // hc_mult

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    num_tokens = res2d.shape[0]

    padded = utils.round_up(num_tokens, utils.SUBLANE)
    if padded <= 64:
        tb, padded_tokens = padded, padded
    else:
        cap = min(token_block_size, max(utils.SUBLANE, (padded // 2 // utils.SUBLANE) * utils.SUBLANE))
        tb = utils.SUBLANE
        for cand in range((cap // utils.SUBLANE) * utils.SUBLANE, utils.SUBLANE - 1, -utils.SUBLANE):
            if padded % cand == 0:
                tb = cand
                break
        padded_tokens = utils.round_up(num_tokens, tb)
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )
    buf_mode = pl.Buffered(buffer_count=2)

    out = pl.pallas_call(
        functools.partial(_post_kernel, hc_mult=hc_mult, hidden_size=hidden_size),
        grid=(padded_tokens // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0), pipeline_mode=buf_mode),
            pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0), pipeline_mode=buf_mode),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0), pipeline_mode=buf_mode),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0), pipeline_mode=buf_mode),
        ],
        out_specs=pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0), pipeline_mode=buf_mode),
        out_shape=jax.ShapeDtypeStruct(
            (padded_tokens, hc_mult * hidden_size), res2d.dtype
        ),
        compiler_params=compiler_params,
    )(x2d, res2d, post2d, comb2d)

    (out,) = utils.trim_to(num_tokens, out)
    return out


def mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 128,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """Pallas replacement for vLLM's ``mhc_post_torch``."""
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
    )
    return out2d.reshape(*outer_shape, hc_mult, hidden_size)


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
"""The model's last mHC post fused with the hc head collapse.

One pass per token block: recombine the streams as ``post_kernel`` does,
round them to bf16 as it stores them, then the head's RMS scale, its 4-wide
projection, sigmoid gates and weighted stream sum. The (T, hc_mult * hidden)
post output is never written to HBM.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES

TOKEN_BLOCK = 64
LANE_CHUNK = 256

# Position of the optional final-RMSNorm weight among the kernel's refs,
# counting the shared-expert slot.
_NORM_REF_INDEX = 8


def _post_head_kernel(
    x_ref,
    shared_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_ref,
    sc_ref,
    hb_ref,
    norm_w_ref,
    out_ref,
    *,
    hc_mult,
    hidden_size,
    lane_chunk,
    rms_eps,
    hc_eps,
    routed_scale,
    norm_eps,
):
    """One token block: post recombine, then the head collapse in VMEM.

    Walks the hidden axis in lane chunks so each chunk's f32 values die
    before the next; only the bf16 streams stay resident between passes.
    With ``shared_ref``, the MoE output is formed first, as the seam kernel
    forms it (``fused_post_pre_kernel``). With ``norm_w_ref``, the collapsed
    rows are written after the model's final RMSNorm.
    """
    m, h = hc_mult, hidden_size
    post = post_ref[...]  # (tb, m) f32
    comb = comb_ref[...]  # (tb, m * m) f32, row-major (i, j)
    chunks = [slice(c, c + lane_chunk) for c in range(0, h, lane_chunk)]

    # Same ops in the same order as post_kernel, so the bf16 streams match
    # the ones it would have stored.
    new_bf = [[] for _ in range(m)]
    sq = None
    for cols in chunks:
        x = x_ref[:, cols].astype(jnp.float32)
        if shared_ref is not None:
            x = (shared_ref[:, cols] + x * routed_scale).astype(jnp.bfloat16)
            x = x.astype(jnp.float32)
        streams = [
            res_ref[:, i * h + cols.start : i * h + cols.stop].astype(jnp.float32)
            for i in range(m)
        ]
        for j in range(m):
            acc = post[:, j : j + 1] * x
            for i in range(m):
                k = i * m + j
                acc = acc + comb[:, k : k + 1] * streams[i]
            nb = acc.astype(jnp.bfloat16)
            new_bf[j].append(nb)
            hf = nb.astype(jnp.float32)
            sq = hf * hf if sq is None else sq + hf * hf
    ms = jnp.sum(sq, axis=-1, keepdims=True) * (1.0 / (m * h))
    rs = jax.lax.rsqrt(ms + rms_eps)

    # The head's f32 linear runs at default precision: one bf16 pass on the
    # normalized rows, so normalize first and round as that pass does.
    dn = (((1,), (1,)), ((), ()))
    logits = None
    for j in range(m):
        for c, cols in enumerate(chunks):
            lhs = (new_bf[j][c].astype(jnp.float32) * rs).astype(jnp.bfloat16)
            w = fn_ref[:, j * h + cols.start : j * h + cols.stop]
            part = jax.lax.dot_general(lhs, w, dn, preferred_element_type=jnp.float32)
            logits = part if logits is None else logits + part
    z = logits * sc_ref[0, 0] + hb_ref[...]
    pre = 1.0 / (1.0 + jnp.exp(-z)) + hc_eps

    for c, cols in enumerate(chunks):
        t = [pre[:, j : j + 1] * new_bf[j][c].astype(jnp.float32) for j in range(m)]
        # Pairing and the 0.0 seed follow XLA's reduce over the stream axis.
        y = 0.0 + ((t[0] + t[2]) + (t[1] + t[3]))
        out_ref[:, cols] = y.astype(out_ref.dtype)
    if norm_w_ref is not None:
        # Re-read what was stored, as the unfused RMSNorm reads it.
        out_ref[...] = utils.rms_norm_rows(out_ref[...], norm_w_ref[...], norm_eps)


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_eps",
        "routed_scale",
        "token_block_size",
        "lane_chunk",
        "vmem_limit_bytes",
        "norm_eps",
    ),
)
def mhc_post_head_2d(
    x: jax.Array,
    res2d: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_eps: float,
    shared: jax.Array | None = None,
    routed_scale: float = 1.0,
    token_block_size: int = TOKEN_BLOCK,
    lane_chunk: int = LANE_CHUNK,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    norm_weight: jax.Array | None = None,
    norm_eps: float = 0.0,
) -> jax.Array:
    """Fused post + head collapse on the flat stream layout.

    Args:
        x: (num_tokens, hidden_size) bfloat16, the last sublayer's output.
        res2d: (num_tokens, hc_mult * hidden_size) bfloat16.
        post_layer_mix: (num_tokens, hc_mult[, 1]) float32.
        comb_res_mix: (num_tokens, hc_mult, hc_mult) float32.
        hc_fn: (hc_mult, hc_mult * hidden_size) float32 head projection.
        hc_scale: (1,) float32. hc_base: (hc_mult,) float32.
        shared: optional (num_tokens, hidden_size) float32 shared-expert
            output; x is then the routed sum before ``routed_scale``.
        norm_weight: optional (hidden_size,) bfloat16 weight of the model's
            final RMSNorm (epsilon ``norm_eps``); the rows are then returned
            normalized, bit for bit that norm (``utils.rms_norm_rows``).

    Returns:
        (num_tokens, hidden_size) in ``res2d.dtype``.
    """
    assert res2d.dtype == jnp.bfloat16, res2d.dtype
    assert post_layer_mix.dtype == jnp.float32, post_layer_mix.dtype
    assert comb_res_mix.dtype == jnp.float32, comb_res_mix.dtype

    hc_mult = comb_res_mix.shape[-1]
    hidden_size = res2d.shape[-1] // hc_mult
    assert hc_mult == 4, f"stream sum order is written for 4 hc streams, got {hc_mult}"
    assert hc_fn.shape == (hc_mult, hc_mult * hidden_size), hc_fn.shape
    assert hidden_size % lane_chunk == 0, (hidden_size, lane_chunk)

    x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
    post2d = post_layer_mix.reshape(-1, hc_mult)
    comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
    num_tokens = res2d.shape[0]

    tb, padded_tokens = utils.select_token_block(num_tokens, token_block_size)
    x2d, res2d, post2d, comb2d = utils.pad_to(padded_tokens, x2d, res2d, post2d, comb2d)
    x_inputs = [x2d]
    if shared is not None:
        assert shared.dtype == jnp.float32, shared.dtype
        x_inputs += utils.pad_to(padded_tokens, shared.reshape(-1, hidden_size))

    # Rounded as XLA rounds an f32 operand for a default-precision dot.
    fn_bf = hc_fn.astype(jnp.bfloat16)
    sc2d = hc_scale.reshape(1, 1).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult).astype(jnp.float32)
    norm_inputs, norm_specs = [], []
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,), norm_weight.shape
        assert norm_weight.dtype == jnp.bfloat16, norm_weight.dtype
        norm_inputs.append(norm_weight)
        norm_specs.append(pl.BlockSpec((hidden_size,), lambda i: (0,)))

    kernel = functools.partial(
        _post_head_kernel,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        lane_chunk=lane_chunk,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        routed_scale=routed_scale,
        norm_eps=norm_eps,
    )
    if norm_weight is None:
        kernel = utils.without_ref(kernel, _NORM_REF_INDEX)
    if shared is None:
        kernel = utils.without_shared(kernel)
    out = pl.pallas_call(
        kernel,
        grid=(padded_tokens // tb,),
        in_specs=[
            *[pl.BlockSpec((tb, hidden_size), lambda i: (i, 0))] * len(x_inputs),
            pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
            pl.BlockSpec((hc_mult, hc_mult * hidden_size), lambda i: (0, 0)),
            pl.BlockSpec((1, 1), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult), lambda i: (0, 0)),
            *norm_specs,
        ],
        out_specs=pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, hidden_size), res2d.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
    )(*x_inputs, res2d, post2d, comb2d, fn_bf, sc2d, hb2d, *norm_inputs)

    (out,) = utils.trim_to(num_tokens, out)
    return out


def mhc_post_head(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_eps: float,
    *,
    shared: jax.Array | None = None,
    routed_scale: float = 1.0,
    norm_weight: jax.Array | None = None,
    norm_eps: float = 0.0,
) -> jax.Array:
    """``mhc_post`` followed by ``mhc_collapse_head``, as one kernel.

    Args:
        x: (..., hidden_size) bfloat16. residual: (..., hc_mult,
            hidden_size) bfloat16. post_layer_mix: (..., hc_mult, 1) and
            comb_res_mix: (..., hc_mult, hc_mult), float32.
        shared, routed_scale, norm_weight, norm_eps: as in
            ``mhc_post_head_2d``.

    Returns:
        (..., hidden_size) in ``residual.dtype``.
    """
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    out2d = mhc_post_head_2d(
        x.reshape(-1, hidden_size),
        residual.reshape(-1, hc_mult * hidden_size),
        post_layer_mix.reshape(-1, hc_mult, 1),
        comb_res_mix.reshape(-1, hc_mult, hc_mult),
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        shared=None if shared is None else shared.reshape(-1, hidden_size),
        routed_scale=routed_scale,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return out2d.reshape(*outer_shape, hidden_size)

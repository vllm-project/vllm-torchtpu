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


def _post_head_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    *refs,
    hc_mult,
    hidden_size,
    lane_chunk,
    rms_eps,
    hc_eps,
    want_head=True,
    want_mean=False,
):
    """One token block: post recombine, then the head collapse in VMEM.

    Walks the hidden axis in lane chunks so each chunk's f32 values die
    before the next; only the bf16 streams stay resident between passes.
    """
    refs = list(refs)
    if want_head:
        fn_ref, sc_ref, hb_ref = refs[:3]
        refs = refs[3:]
    out_ref = refs.pop(0) if want_head else None
    mean_ref = refs.pop(0) if want_mean else None
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
            if want_head:
                hf = nb.astype(jnp.float32)
                sq = hf * hf if sq is None else sq + hf * hf
        if want_mean:
            # torch ``mean(dim=1)`` over the 4 bf16 streams, as XLA reduces
            # them: f32 (0+2)+(1+3), times 1/4, one rounding.
            f = [new_bf[j][-1].astype(jnp.float32) for j in range(m)]
            mean_ref[:, cols] = (((f[0] + f[2]) + (f[1] + f[3])) * 0.25).astype(
                mean_ref.dtype
            )
    if not want_head:
        return
    ms = jnp.sum(sq, axis=-1, keepdims=True) / (m * h)
    rs = jax.lax.rsqrt(ms + rms_eps)

    # The head's f32 linear: rc7 runs torch's F.linear on f32 operands, which
    # torch_tpu does at full f32 precision (a bf16 one-pass projection moved
    # 25% of outputs by several ulp). So form it on the VPU in f32: per output
    # j, accumulate norm * fn[j] lane-chunk by lane-chunk, reduce lanes once.
    accs = [None] * m
    for c, cols in enumerate(chunks):
        for i in range(m):
            nrm = new_bf[i][c].astype(jnp.float32) * rs
            for j in range(m):
                w = fn_ref[j : j + 1, i * h + cols.start : i * h + cols.stop]
                prod = nrm * w
                accs[j] = prod if accs[j] is None else accs[j] + prod
    logits = jnp.concatenate(
        [jnp.sum(a, axis=-1, keepdims=True) for a in accs], axis=1
    )
    z = logits * sc_ref[0, 0] + hb_ref[...]
    pre = 1.0 / (1.0 + jnp.exp(-z)) + hc_eps

    for c, cols in enumerate(chunks):
        t = [pre[:, j : j + 1] * new_bf[j][c].astype(jnp.float32) for j in range(m)]
        # Pairing and the 0.0 seed follow XLA's reduce over the stream axis.
        y = 0.0 + ((t[0] + t[2]) + (t[1] + t[3]))
        out_ref[:, cols] = y.astype(out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_eps",
        "token_block_size",
        "lane_chunk",
        "vmem_limit_bytes",
        "want_head",
        "want_mean",
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
    token_block_size: int = TOKEN_BLOCK,
    lane_chunk: int = LANE_CHUNK,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    want_head: bool = True,
    want_mean: bool = False,
) -> tuple[jax.Array, ...]:
    """Fused post + head collapse (and/or the stream mean) on the flat
    stream layout. Returns (head,) / (mean,) / (head, mean).

    Args:
        x: (num_tokens, hidden_size) bfloat16, the last sublayer's output.
        res2d: (num_tokens, hc_mult * hidden_size) bfloat16.
        post_layer_mix: (num_tokens, hc_mult[, 1]) float32.
        comb_res_mix: (num_tokens, hc_mult, hc_mult) float32.
        hc_fn: (hc_mult, hc_mult * hidden_size) float32 head projection.
        hc_scale: (1,) float32. hc_base: (hc_mult,) float32.

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

    fn_bf = hc_fn.astype(jnp.float32)  # f32: the projection runs on the VPU
    sc2d = hc_scale.reshape(1, 1).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult).astype(jnp.float32)

    in_specs = [
        pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
        pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
        pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
    ]
    args = [x2d, res2d, post2d, comb2d]
    if want_head:
        in_specs += [
            pl.BlockSpec((hc_mult, hc_mult * hidden_size), lambda i: (0, 0)),
            pl.BlockSpec((1, 1), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult), lambda i: (0, 0)),
        ]
        args += [fn_bf, sc2d, hb2d]
    n_out = int(want_head) + int(want_mean)
    outs = pl.pallas_call(
        functools.partial(
            _post_head_kernel,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            lane_chunk=lane_chunk,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            want_head=want_head,
            want_mean=want_mean,
        ),
        grid=(padded_tokens // tb,),
        in_specs=in_specs,
        out_specs=[pl.BlockSpec((tb, hidden_size), lambda i: (i, 0))] * n_out,
        out_shape=[jax.ShapeDtypeStruct((padded_tokens, hidden_size), res2d.dtype)]
        * n_out,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
    )(*args)
    return tuple(utils.trim_to(num_tokens, *outs))


def mhc_post_head(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array | None,
    hc_scale: jax.Array | None,
    hc_base: jax.Array | None,
    rms_eps: float,
    hc_eps: float,
    *,
    want_head: bool = True,
    want_mean: bool = False,
) -> tuple[jax.Array, ...]:
    """``mhc_post`` followed by ``mhc_collapse_head`` and/or the stream mean
    (``.mean(dim=-2)``), without writing the (..., hc_mult, hidden) streams.

    Returns (head,), (mean,) or (head, mean), each (..., hidden_size).
    """
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    if not want_head:
        hc_fn = jnp.zeros((hc_mult, hc_mult * hidden_size), jnp.float32)
        hc_scale = jnp.ones((1,), jnp.float32)
        hc_base = jnp.zeros((hc_mult,), jnp.float32)
    outs = mhc_post_head_2d(
        x.reshape(-1, hidden_size),
        residual.reshape(-1, hc_mult * hidden_size),
        post_layer_mix.reshape(-1, hc_mult, 1),
        comb_res_mix.reshape(-1, hc_mult, hc_mult),
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        want_head=want_head,
        want_mean=want_mean,
    )
    return tuple(o.reshape(*outer_shape, hidden_size) for o in outs)

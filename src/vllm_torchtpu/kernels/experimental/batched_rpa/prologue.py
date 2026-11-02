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
"""Batched-RPA inputs straight from a fused qkv projection's output rows.

`qkv_to_rpa_inputs` does in one pass what a q/k-normed, rotary attention (Gemma4)
does between its qkv projection and the kernel: per-head RMSNorm of q and k
(weighted) and v (unweighted), neox rotary on q and k, the static per-tensor
FP8 quantization of k and v, and `wrapper.prepare_inputs`' layouts. It writes
q as `[kv_heads, tokens, q_per_kv // 2, 2, head_dim]` and the new K/V as
`[tokens, slots, 4, head_dim]`, the layouts the kernel reads, so nothing is
relaid out in between. It covers bf16 queries, an FP8 KV cache in the
HEAD_ALONG_SUBLANE layout and static per-tensor KV scales; `prepare_inputs`
also handles SEQ_ALONG_LANE and per-token scales, which this does not.

The arithmetic is the one XLA runs for those ops today: the norms in f32 with
their outputs fed to rotary unrounded, rotary's output rounded to bf16 (q is
read in bf16; k is quantized from the bf16 value) and v quantized from the f32
norm output.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_KV_PACKING = 4  # FP8 rows per packed K/V slot
_Q_PACKING = 2  # bf16 query heads per packed row pair
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_VMEM_BUDGET_BYTES = 46 * 1024 * 1024
_BLOCK_TOKENS = (512, 256, 128, 64, 32, 16)


def _block_vmem_bytes(bt: int, width: int, num_q_heads: int, slots: int, d: int) -> int:
    io = bt * width * 2 + num_q_heads * bt * d * 2 + bt * slots * _KV_PACKING * d
    return 2 * io + 8 * bt * d * 4  # double-buffered blocks + f32 temporaries


def choose_block_tokens(
    tokens: int, width: int, num_q_heads: int, slots: int, d: int
) -> int:
    """The largest token block that divides `tokens` and fits the VMEM budget."""
    for bt in _BLOCK_TOKENS:
        if tokens % bt == 0 and (
            _block_vmem_bytes(bt, width, num_q_heads, slots, d) <= _VMEM_BUDGET_BYTES
        ):
            return bt
    raise ValueError(f"no token block for {tokens} tokens of width {width}")


def _rms(x: jax.Array, w: jax.Array | None, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return x32 if w is None else x32 * w


def _rotary(y: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Neox rotary on the leading `2 * cos.shape[-1]` lanes of `y`."""
    half = cos.shape[-1]
    y1, y2 = y[..., :half], y[..., half : 2 * half]
    parts = [y1 * cos - y2 * sin, y2 * cos + y1 * sin]
    if 2 * half < y.shape[-1]:
        parts.append(y[..., 2 * half :])
    return jnp.concatenate(parts, axis=-1)


def _quantize(x: jax.Array, scale: float, dtype: jnp.dtype) -> jax.Array:
    if scale != 1.0:  # a division by 1 is folded away on the XLA path
        x = x / scale
    info = jnp.finfo(dtype)
    return jnp.clip(x, float(info.min), float(info.max)).astype(dtype)


def _kernel(
    qkv_ref,
    cos_ref,
    sin_ref,
    wq_ref,
    wk_ref,
    q_ref,
    kv_ref,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    eps,
    k_scale,
    v_scale,
):
    group = num_q_heads // num_kv_heads
    cos = cos_ref[...].astype(jnp.float32)
    sin = sin_ref[...].astype(jnp.float32)
    wq, wk = wq_ref[...], wk_ref[...]

    def head(j):
        return qkv_ref[:, j * head_dim : (j + 1) * head_dim]

    for j in range(num_q_heads):
        q = _rotary(_rms(head(j), wq, eps), cos, sin).astype(q_ref.dtype)
        h, rest = divmod(j, group)
        q_ref[h, :, rest // _Q_PACKING, rest % _Q_PACKING, :] = q
    rows = []
    for h in range(num_kv_heads):
        k = _rotary(_rms(head(num_q_heads + h), wk, eps), cos, sin)
        k = k.astype(jnp.bfloat16).astype(jnp.float32)
        v = _rms(head(num_q_heads + num_kv_heads + h), None, eps)
        rows += [
            _quantize(k, k_scale, kv_ref.dtype),
            _quantize(v, v_scale, kv_ref.dtype),
        ]
    slots = kv_ref.shape[1]
    zeros = jnp.zeros((kv_ref.shape[0], head_dim), kv_ref.dtype)
    rows += [zeros] * (slots * _KV_PACKING - len(rows))  # prepare_inputs' padding
    for r, row in enumerate(rows):
        kv_ref[:, r // _KV_PACKING, r % _KV_PACKING, :] = row


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "eps",
        "k_scale",
        "v_scale",
        "kv_dtype",
        "block_tokens",
        "interpret",
    ),
)
def _qkv_to_rpa_inputs(
    qkv: jax.Array,  # [tokens, (num_q_heads + 2 * num_kv_heads) * head_dim] bf16
    cos: jax.Array,  # [tokens, rotary_dim // 2], the rotary cache rows of the tokens
    sin: jax.Array,  # [tokens, rotary_dim // 2]
    q_norm_weight: jax.Array,  # [head_dim]
    k_norm_weight: jax.Array,  # [head_dim]
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    k_scale: float,
    v_scale: float,
    kv_dtype: jnp.dtype = jnp.float8_e4m3fn,
    block_tokens: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`qkv_to_rpa_inputs` on a token count that is a multiple of 16."""
    tokens, width = qkv.shape
    group = num_q_heads // num_kv_heads
    if width != (num_q_heads + 2 * num_kv_heads) * head_dim:
        raise ValueError(f"qkv width {width} does not match the head counts")
    if qkv.dtype != jnp.bfloat16 or group % _Q_PACKING or head_dim % 128:
        raise ValueError(
            f"needs bf16 qkv, an even number of q heads per kv head and a "
            f"lane-aligned head_dim: {qkv.dtype}, {group}, {head_dim}"
        )
    half = cos.shape[-1]
    if cos.shape != (tokens, half) or sin.shape != cos.shape or 2 * half > head_dim:
        raise ValueError(f"rotary rows {cos.shape} / {sin.shape} for {qkv.shape}")
    slots = -(-2 * num_kv_heads // _KV_PACKING)
    bt = block_tokens or choose_block_tokens(
        tokens, width, num_q_heads, slots, head_dim
    )
    if tokens % bt:
        raise ValueError(f"block of {bt} tokens does not tile {tokens} tokens")
    kernel = functools.partial(
        _kernel,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    row_spec = lambda cols: pl.BlockSpec((bt, cols), lambda i: (i, 0))  # noqa: E731
    weight_spec = pl.BlockSpec((1, head_dim), lambda i: (0, 0))
    return pl.pallas_call(
        kernel,
        grid=(tokens // bt,),
        in_specs=[
            row_spec(width),
            row_spec(half),
            row_spec(half),
            weight_spec,
            weight_spec,
        ],
        out_specs=[
            pl.BlockSpec(
                (num_kv_heads, bt, group // _Q_PACKING, _Q_PACKING, head_dim),
                lambda i: (0, i, 0, 0, 0),
            ),
            pl.BlockSpec((bt, slots, _KV_PACKING, head_dim), lambda i: (i, 0, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(
                (num_kv_heads, tokens, group // _Q_PACKING, _Q_PACKING, head_dim),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct((tokens, slots, _KV_PACKING, head_dim), kv_dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT_BYTES
        ),
        name="qkv_to_rpa_inputs",
        interpret=interpret,
    )(
        qkv,
        cos,
        sin,
        q_norm_weight.astype(jnp.float32).reshape(1, head_dim),
        k_norm_weight.astype(jnp.float32).reshape(1, head_dim),
    )


def qkv_to_rpa_inputs(
    qkv: jax.Array,  # [tokens, (num_q_heads + 2 * num_kv_heads) * head_dim] bf16
    cos: jax.Array,  # [tokens, rotary_dim // 2], the rotary cache rows of the tokens
    sin: jax.Array,  # [tokens, rotary_dim // 2]
    q_norm_weight: jax.Array,  # [head_dim]
    k_norm_weight: jax.Array,  # [head_dim]
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """(q_hbm, new_kv_hbm) as `wrapper.prepare_inputs` returns them.

    A token count that 16-token blocks do not tile (an odd bucket) runs on
    zero rows padded to the next multiple of 16, and the outputs are cut back.
    """
    tokens = qkv.shape[0]
    pad = -tokens % _BLOCK_TOKENS[-1]
    if pad:
        rows = ((0, pad), (0, 0))
        qkv, cos, sin = (jnp.pad(a, rows) for a in (qkv, cos, sin))
    q, kv = _qkv_to_rpa_inputs(qkv, cos, sin, q_norm_weight, k_norm_weight, **kwargs)
    return (q[:, :tokens], kv[:tokens]) if pad else (q, kv)

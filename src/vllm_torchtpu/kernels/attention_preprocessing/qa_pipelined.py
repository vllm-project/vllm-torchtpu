# SPDX-License-Identifier: Apache-2.0
"""Q-A/KV + RMSNorm + SWA RoPE with double-buffered token tiles.

Same arithmetic as ``serial.fused_qa_kv_norm_swa_serial``; only buffering
changes. The weight block index never changes, so it stays single-buffered and
is fetched once. Token tiles and both outputs keep two buffers, so the next
tile's activation fetch and the previous tile's writeback overlap compute.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing.serial import (
    _check_projection,
    _check_rope,
    _qa_kv_norm_swa_kernel,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    QA_TOKEN_BLOCK,
    QA_VMEM_LIMIT_BYTES,
)


def fused_qa_kv_norm_swa_pipelined(
    x: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    gamma_qr: jax.Array,
    gamma_kv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float = 1e-6,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return BF16 (normalized qr, normalized and rotated SWA KV)."""
    _check_projection(x, weight, weight_scale)
    rank, head_dim = gamma_qr.size, gamma_kv.size
    if weight.shape[1] != rank + head_dim:
        raise ValueError("Projection width must equal qr width plus KV width")
    _check_rope(x, cos, sin, head_dim)
    tokens, hidden = x.shape
    if tokens % QA_TOKEN_BLOCK:
        raise ValueError(f"Pad token rows to a multiple of {QA_TOKEN_BLOCK}")
    once = pl.Buffered(buffer_count=1)
    tb = QA_TOKEN_BLOCK
    return pl.pallas_call(
        functools.partial(_qa_kv_norm_swa_kernel, eps=eps),
        grid=(tokens // tb,),
        in_specs=(
            pl.BlockSpec((tb, hidden), lambda t: (t, 0)),
            pl.BlockSpec(weight.shape, lambda t: (0, 0), once),
            pl.BlockSpec((rank + head_dim,), lambda t: (0,), once),
            pl.BlockSpec(gamma_qr.shape, lambda t: (0,), once),
            pl.BlockSpec(gamma_kv.shape, lambda t: (0,), once),
            pl.BlockSpec((tb, cos.shape[1]), lambda t: (t, 0)),
            pl.BlockSpec((tb, sin.shape[1]), lambda t: (t, 0)),
        ),
        out_specs=(
            pl.BlockSpec((tb, rank), lambda t: (t, 0)),
            pl.BlockSpec((tb, head_dim), lambda t: (t, 0)),
        ),
        scratch_shapes=(pltpu.VMEM((tb, rank + head_dim), jnp.float32),),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, rank), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, head_dim), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=QA_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="qa_kv_norm_swa_pipelined",
    )(x, weight, weight_scale.reshape(-1), gamma_qr, gamma_kv, cos, sin)

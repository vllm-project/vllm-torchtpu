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
"""Fused head-major-to-token-major per-token FP8 quantization for batched RPA.

`batched_rpa` writes its attention output into the packed query buffer
`[kv_heads, tokens, q_per_kv // 2, 2, padded_head_dim]` in bf16, where each
32-bit word holds one lane of two query heads for one token. Downstream FP8
`o_proj` expects row-major `[tokens, num_q_heads * head_dim]` quantized per
token alongside `scale [tokens, 1]` in float32.

`quantize_head_major_output` gathers each `(kv_head, head_pair, 128-lane tile)`
via double-buffered strided 32-bit DMAs, unpacks the bf16 head pairs in VMEM,
and emits quantized token-major rows and exact `quantize_tensor` row scales in
one pass.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128
_PACKING = 2  # bf16 query heads per 32-bit word
_MAX_SLOT_WORDS = 1 << 19  # 512 Ki words (2 MiB) per double-buffered DMA slot
_MAX_BLOCK_TOKENS = 512
_DEFAULT_CPU_VMEM_BYTES = 1 << 24  # 16 MiB fallback when TPU info is unavailable


def _vmem_capacity_bytes(default: int = _DEFAULT_CPU_VMEM_BYTES) -> int:
    try:
        info = pltpu.get_tpu_info()
    except Exception:
        return default
    if info is None:
        return default
    return int(info.vmem_capacity_bytes)


def block_tokens(
    tokens: int,
    num_slabs: int,
    vmem_limit_bytes: int | None = None,
) -> int:
    """Return the largest power-of-two token block fitting one DMA slot."""
    limit = (
        _vmem_capacity_bytes() if vmem_limit_bytes is None else int(vmem_limit_bytes)
    )
    slot_words = min(_MAX_SLOT_WORDS, limit // 8 // 4)
    bt = min(tokens & -tokens, _MAX_BLOCK_TOKENS)
    while bt > 8 and num_slabs * bt * _LANES > slot_words:
        bt //= 2
    return bt


def _quantize_output_kernel(
    o_hbm,
    q_ref,
    s_ref,
    buf,
    sem,
    *,
    slabs: tuple[tuple[int, int, int], ...],
    groups: int,
    head_dim: int,
    bt: int,
):
    step, steps = pl.program_id(0), pl.num_programs(0)
    words = o_hbm.bitcast(jnp.uint32)  # [kv_heads, tokens, q_per_kv // 2, 1, d]

    def copies(block, slot):
        t0 = pl.multiple_of(block * bt, bt)
        return [
            pltpu.make_async_copy(
                words.at[h, pl.ds(t0, bt), pair, 0, pl.ds(lane, _LANES)],
                buf.at[slot, j],
                sem.at[slot],
            )
            for j, (h, pair, lane) in enumerate(slabs)
        ]

    slot = step % 2

    @pl.when(step == 0)
    def _():
        for c in copies(0, 0):
            c.start()

    @pl.when(step + 1 < steps)
    def _():
        for c in copies(step + 1, 1 - slot):
            c.start()

    for c in copies(step, slot):
        c.wait()

    def heads(j: int) -> list[tuple[int, jax.Array]]:
        h, pair, lane = slabs[j]
        w = buf[slot, j]
        halves = (w << 16, w & jnp.uint32(0xFFFF0000))
        return [
            ((h * groups + g) * head_dim + lane, pltpu.bitcast(half, jnp.float32))
            for g, half in enumerate(halves, start=pair * _PACKING)
            if g < groups
        ]

    info = jnp.finfo(q_ref.dtype)
    amax = None
    for j in range(len(slabs)):
        for _, x in heads(j):
            m = jnp.abs(x)
            amax = m if amax is None else jnp.maximum(amax, m)
    assert amax is not None
    scale = jnp.max(amax, axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    for j in range(len(slabs)):
        for col, x in heads(j):
            q = jnp.clip(x / scale, float(info.min), float(info.max))
            q_ref[:, col : col + _LANES] = q.astype(q_ref.dtype)
    s_ref[...] = scale


@functools.partial(
    jax.jit,
    static_argnames=(
        "quant_dtype",
        "num_q_heads_per_kv_head",
        "head_dim",
        "interpret",
    ),
)
def quantize_head_major_output(
    o_hbm: jax.Array,
    *,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    num_q_heads_per_kv_head: int | None = None,
    head_dim: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
) -> tuple[jax.Array, jax.Array]:
    """Quantize packed head-major RPA output to token-major `(q, scale)`.

    Token counts that are not multiples of 8 are zero-padded to the next
    8-token boundary and sliced back after quantization.
    """
    kv_heads, tokens, pairs, packing, padded_dim = o_hbm.shape
    groups = (
        pairs * packing
        if num_q_heads_per_kv_head is None
        else int(num_q_heads_per_kv_head)
    )
    unpadded_dim = padded_dim if head_dim is None else int(head_dim)
    if (
        o_hbm.dtype != jnp.bfloat16
        or packing != _PACKING
        or pairs * _PACKING < groups
        or unpadded_dim % _LANES
        or padded_dim < unpadded_dim
    ):
        raise ValueError(
            f"needs a bf16 [kv_heads, tokens, pairs, 2, d] buffer with lane-aligned "
            f"heads: {o_hbm.dtype}{o_hbm.shape}, {groups} heads of {unpadded_dim}"
        )
    pad = -tokens % 8
    if pad:
        o_hbm = jnp.pad(o_hbm, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)))
    slabs = tuple(
        (h, pair, lane)
        for h in range(kv_heads)
        for pair in range(-(-groups // _PACKING))
        for lane in range(0, unpadded_dim, _LANES)
    )
    vmem_limit = _vmem_capacity_bytes()
    padded = tokens + pad
    bt = block_tokens(padded, len(slabs), vmem_limit_bytes=vmem_limit)
    width = kv_heads * groups * unpadded_dim
    q, scale = pl.pallas_call(
        functools.partial(
            _quantize_output_kernel,
            slabs=slabs,
            groups=groups,
            head_dim=unpadded_dim,
            bt=bt,
        ),
        grid=(padded // bt,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=[
            pl.BlockSpec((bt, width), lambda i: (i, 0)),
            pl.BlockSpec((bt, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded, width), quant_dtype),
            jax.ShapeDtypeStruct((padded, 1), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, len(slabs), bt, _LANES), jnp.uint32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
        name="rpa_output_quantize",
    )(o_hbm)
    return (q[:tokens], scale[:tokens]) if pad else (q, scale)

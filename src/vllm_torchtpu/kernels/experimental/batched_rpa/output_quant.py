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
"""The batched RPA's head-major output as the o_proj's per-token FP8 rows.

The kernels write their output into the query buffer, `[kv_heads, tokens,
q_per_kv // 2, 2, head_dim]` in bf16, where each 32-bit word holds one lane of
two query heads of one token. An FP8 o_proj then needs the token-major
`[tokens, num_q_heads * head_dim]` rows quantized per token. Left to XLA that
is a transpose (which it plans around the quantization's f32 math), an f32
copy, a row-maximum pass and a cast into a column-major operand.

This pass reads the buffer once. For each block of tokens, strided DMAs gather
every (KV head, head pair, 128-lane tile) of the block into a contiguous
`[tokens, 128]` buffer of 32-bit words, eight tokens per vreg; shifts split the
words into the two heads' f32 values, and the block's rows are written
quantized, row-major, with their scales.

Arithmetic is `quantized_matmul.util.quantize_tensor` on the bf16 rows: the
f32 scale is the row's maximum magnitude / the dtype's max (1 where it is 0)
and each value is divided by it in f32, clipped and cast.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128
_PACKING = 2  # bf16 query heads per 32-bit word
# Words of one DMA slot ((KV head, head pair, lane tile) slabs x tokens x 128):
# 2 MiB per slot, two slots.
_SLOT_WORDS = 512 * 1024
_MAX_BLOCK_TOKENS = 512
_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def block_tokens(tokens: int, num_slabs: int) -> int:
    """The largest power of two that divides `tokens` and keeps a slot within
    `_SLOT_WORDS`."""
    bt = min(tokens & -tokens, _MAX_BLOCK_TOKENS)
    while bt > 8 and num_slabs * bt * _LANES > _SLOT_WORDS:
        bt //= 2
    return bt


def _kernel(o_hbm, q_ref, s_ref, buf, sem, *, slabs, groups, head_dim, bt):
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

    def heads(j):
        """(column, f32 values) of the slab's query heads (low half: the first),
        without the query buffer's padding head."""
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
    scale = jnp.max(amax, axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    for j in range(len(slabs)):
        for col, x in heads(j):
            q = jnp.clip(x / scale, float(info.min), float(info.max))
            q_ref[:, col : col + _LANES] = q.astype(q_ref.dtype)
    s_ref[...] = scale


@functools.partial(
    jax.jit,
    static_argnames=("num_q_heads_per_kv_head", "head_dim", "quant_dtype", "interpret"),
)
def quantize_head_major_output(
    o: jax.Array,  # [kv_heads, tokens, q_per_kv // 2, 2, padded head_dim] bf16
    *,
    num_q_heads_per_kv_head: int,
    head_dim: int,
    quant_dtype: jnp.dtype,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`quantize_tensor` of the token-major output rows: `(q [tokens,
    num_q_heads * head_dim] in quant_dtype, scale [tokens, 1] f32)`.

    A token count that 8-token blocks do not tile (an odd bucket) runs on zero
    rows padded to the next multiple of 8, and the outputs are cut back.
    """
    kv_heads, tokens, pairs, packing, padded_dim = o.shape
    groups = num_q_heads_per_kv_head
    if (
        o.dtype != jnp.bfloat16
        or packing != _PACKING
        or pairs * _PACKING < groups
        or head_dim % _LANES
        or padded_dim < head_dim
    ):
        raise ValueError(
            f"needs a bf16 [kv_heads, tokens, pairs, 2, d] buffer with lane-aligned "
            f"heads: {o.dtype}{o.shape}, {groups} heads of {head_dim}"
        )
    pad = -tokens % 8
    if pad:
        o = jnp.pad(o, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)))
    slabs = tuple(
        (h, pair, lane)
        for h in range(kv_heads)
        for pair in range(-(-groups // _PACKING))
        for lane in range(0, head_dim, _LANES)
    )
    padded = tokens + pad
    bt = block_tokens(padded, len(slabs))
    width = kv_heads * groups * head_dim
    q, scale = pl.pallas_call(
        functools.partial(
            _kernel, slabs=slabs, groups=groups, head_dim=head_dim, bt=bt
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
            # Each step starts the next step's copies.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="rpa_output_quantize",
    )(o)
    return (q[:tokens], scale[:tokens]) if pad else (q, scale)

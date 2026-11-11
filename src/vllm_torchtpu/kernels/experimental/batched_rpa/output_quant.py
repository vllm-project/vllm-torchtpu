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
"""Per-token FP8 quantization of head-major batched-RPA outputs."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import vmem_limit_bytes as _vmem_cap

_LANES = 128
_PACKING = 2  # bf16 query heads per 32-bit word
_SLOT_WORDS = 512 * 1024
_MAX_BLOCK_TOKENS = 512
_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def _block_vmem_bytes(
    bt: int, num_slabs: int, width: int, out_itemsize: int, in_bytes: int = 0
) -> int:
    return (
        2 * bt * (num_slabs * _LANES * 4 + width * out_itemsize + _LANES * 4) + in_bytes
    )


def block_tokens(
    tokens: int,
    num_slabs: int,
    vmem_limit_bytes: int | None = None,
    *,
    width: int | None = None,
    out_itemsize: int = 1,
) -> int:
    """Return the largest power-of-two block dividing ``tokens`` that fits VMEM."""
    if tokens <= 0 or tokens % 8 != 0:
        raise ValueError(f"tokens must be a positive multiple of 8, got {tokens}")
    limit = (
        _vmem_cap(_VMEM_LIMIT_BYTES) if vmem_limit_bytes is None else vmem_limit_bytes
    )
    budget = limit * 3 // 4
    row_width = num_slabs * _PACKING * _LANES if width is None else width

    def over(b: int) -> bool:
        return (
            num_slabs * b * _LANES > _SLOT_WORDS
            or _block_vmem_bytes(b, num_slabs, row_width, out_itemsize) > budget
        )

    bt = min(tokens & -tokens, _MAX_BLOCK_TOKENS)
    while bt > 8 and over(bt):
        bt //= 2
    if over(bt):
        raise ValueError(f"bt={bt}, slabs={num_slabs} exceeds VMEM limit {limit}")
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
        h, pair, lane = slabs[j]
        w = buf[slot, j]
        return [
            ((h * groups + g) * head_dim + lane, pltpu.bitcast(half, jnp.float32))
            for g, half in enumerate(
                (w << 16, w & jnp.uint32(0xFFFF0000)), pair * _PACKING
            )
            if g < groups
        ]

    info = jnp.finfo(q_ref.dtype)
    amax = functools.reduce(
        jnp.maximum, (jnp.abs(x) for j in range(len(slabs)) for _, x in heads(j))
    )
    scale = jnp.max(amax, axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    for j in range(len(slabs)):
        for col, x in heads(j):
            q_ref[:, col : col + _LANES] = jnp.clip(
                x / scale, float(info.min), float(info.max)
            ).astype(q_ref.dtype)
    s_ref[...] = scale


@functools.partial(
    jax.jit,
    static_argnames=("num_q_heads_per_kv_head", "head_dim", "quant_dtype", "interpret"),
)
def quantize_head_major_output(
    o: jax.Array,
    *,
    num_q_heads_per_kv_head: int,
    head_dim: int,
    quant_dtype: jnp.dtype,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Quantize head-major output into ``(q [tokens, width], scale [tokens, 1])``."""
    kv_heads, tokens, pairs, packing, padded_dim = o.shape
    groups = num_q_heads_per_kv_head
    if (
        o.dtype != jnp.bfloat16
        or packing != _PACKING
        or pairs * _PACKING < groups
        or head_dim % _LANES
        or padded_dim < head_dim
    ):
        raise ValueError(f"Expected lane-aligned bf16 buffer, got {o.dtype}{o.shape}")
    width = kv_heads * groups * head_dim
    if tokens == 0:
        return jnp.empty((0, width), quant_dtype), jnp.empty((0, 1), jnp.float32)
    pad = -tokens % 8
    if pad:
        o = jnp.pad(o, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)))
    slabs = tuple(
        (h, pair, lane)
        for h in range(kv_heads)
        for pair in range(-(-groups // _PACKING))
        for lane in range(0, head_dim, _LANES)
    )
    padded, vmem_limit = tokens + pad, _vmem_cap(_VMEM_LIMIT_BYTES)
    bt = block_tokens(
        padded,
        len(slabs),
        vmem_limit,
        width=width,
        out_itemsize=jnp.dtype(quant_dtype).itemsize,
    )
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
            dimension_semantics=("arbitrary",), vmem_limit_bytes=vmem_limit
        ),
        interpret=interpret,
        name="rpa_output_quantize",
    )(o)
    return (q[:tokens], scale[:tokens]) if pad else (q, scale)

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

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class _Heads:
    """The head geometry and constants both prologue kernels compute with."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    eps: float
    k_scale: float
    v_scale: float


def _block_values(qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, *, heads, kv_dtype):
    """A token block's values, shared by both kernels: `q_head(j)`, query head
    j's normed, rotated values (f32, rounded to bf16 by the caller), and
    `kv_rows()`, the quantized K/V rows in `prepare_inputs`' order (k, v of
    each KV head)."""
    nq, nkv, head_dim, eps = (
        heads.num_q_heads,
        heads.num_kv_heads,
        heads.head_dim,
        heads.eps,
    )
    cos = cos_ref[...].astype(jnp.float32)
    sin = sin_ref[...].astype(jnp.float32)
    wq, wk = wq_ref[...], wk_ref[...]

    def head(j):
        return qkv_ref[:, j * head_dim : (j + 1) * head_dim]

    def q_head(j):
        return _rotary(_rms(head(j), wq, eps), cos, sin)

    def kv_rows():
        rows = []
        for h in range(nkv):
            k = _rotary(_rms(head(nq + h), wk, eps), cos, sin)
            k = k.astype(jnp.bfloat16).astype(jnp.float32)
            v = _rms(head(nq + nkv + h), None, eps)
            rows += [
                _quantize(k, heads.k_scale, kv_dtype),
                _quantize(v, heads.v_scale, kv_dtype),
            ]
        return rows

    return q_head, kv_rows


def _kernel(qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, q_ref, kv_ref, *, heads):
    group = heads.num_q_heads // heads.num_kv_heads
    q_head, kv_rows = _block_values(
        qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, heads=heads, kv_dtype=kv_ref.dtype
    )
    for j in range(heads.num_q_heads):
        h, rest = divmod(j, group)
        q_ref[h, :, rest // _Q_PACKING, rest % _Q_PACKING, :] = q_head(j).astype(
            q_ref.dtype
        )
    rows = kv_rows()
    slots = kv_ref.shape[1]
    zeros = jnp.zeros((kv_ref.shape[0], heads.head_dim), kv_ref.dtype)
    rows += [zeros] * (slots * _KV_PACKING - len(rows))  # prepare_inputs' padding
    for r, row in enumerate(rows):
        kv_ref[:, r // _KV_PACKING, r % _KV_PACKING, :] = row


_LANES = 128


def _kernel_dma_stores(
    qkv_ref,
    cos_ref,
    sin_ref,
    wq_ref,
    wk_ref,
    q_hbm,
    kv_hbm,
    q_buf,
    kv_buf,
    sem,
    *,
    heads,
    bt,
    steps,
):
    """`_kernel` writing its outputs through 32-bit words.

    The packed outputs put two query heads (q) or four K/V rows (kv) of one
    token in each 32-bit word, so a token's word row is one sublane of a tile
    and a store of eight tokens is a strided one. Each (row group, 128-lane
    tile) is built here as `[tokens, 128]` words in a contiguous buffer and
    written with a strided DMA (double buffered across steps).
    """
    step = pl.program_id(0)
    slot = step % 2
    num_kv_heads, head_dim = heads.num_kv_heads, heads.head_dim
    group = heads.num_q_heads // num_kv_heads
    pairs = group // _Q_PACKING
    slots = kv_hbm.shape[1]
    lanes = range(0, head_dim, _LANES)
    q_words = q_hbm.bitcast(jnp.uint32)  # [kv_heads, tokens, pairs, 1, d]
    kv_words = kv_hbm.bitcast(jnp.uint32)  # [tokens, slots, 1, d]
    q_slabs = [
        (h, p, lane)
        for h in range(num_kv_heads)
        for p in range(pairs)
        for lane in lanes
    ]
    kv_slabs = [(s, lane) for s in range(slots) for lane in lanes]

    def copies(block, buf_slot):
        t0 = pl.multiple_of(block * bt, bt)
        out = [
            pltpu.make_async_copy(
                q_buf.at[buf_slot, j],
                q_words.at[h, pl.ds(t0, bt), p, 0, pl.ds(lane, _LANES)],
                sem.at[buf_slot],
            )
            for j, (h, p, lane) in enumerate(q_slabs)
        ]
        out += [
            pltpu.make_async_copy(
                kv_buf.at[buf_slot, j],
                kv_words.at[pl.ds(t0, bt), s, 0, pl.ds(lane, _LANES)],
                sem.at[buf_slot],
            )
            for j, (s, lane) in enumerate(kv_slabs)
        ]
        return out

    if steps > 2:

        @pl.when(step >= 2)
        def _():  # the copies that last read this slot's buffers
            for c in copies(step - 2, slot):
                c.wait()

    q_head, kv_rows = _block_values(
        qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, heads=heads, kv_dtype=kv_hbm.dtype
    )

    def bits(x):
        return pltpu.bitcast(x, jnp.uint32)

    for h in range(num_kv_heads):
        for p in range(pairs):
            first, second = (
                q_head(h * group + g).astype(jnp.bfloat16).astype(jnp.float32)
                for g in (2 * p, 2 * p + 1)
            )
            # bf16 bits of the pair's first head in the low half, as packed.
            words = (bits(first) >> 16) | (bits(second) & jnp.uint32(0xFFFF0000))
            for i, lane in enumerate(lanes):
                j = (h * pairs + p) * len(lanes) + i
                q_buf[slot, j] = words[:, lane : lane + _LANES]
    rows = kv_rows()
    for s in range(slots):
        words = None
        for r in range(_KV_PACKING):  # row r of the slot in byte r, as packed
            if s * _KV_PACKING + r >= len(rows):
                break  # prepare_inputs' zero padding rows
            byte = jax.lax.bitcast_convert_type(rows[s * _KV_PACKING + r], jnp.uint8)
            byte = byte.astype(jnp.uint32) << (8 * r)
            words = byte if words is None else words | byte
        for i, lane in enumerate(lanes):
            kv_buf[slot, s * len(lanes) + i] = words[:, lane : lane + _LANES]

    for c in copies(step, slot):
        c.start()

    @pl.when(step == steps - 1)
    def _():
        for c in copies(step, slot):
            c.wait()
        if steps > 1:
            for c in copies(step - 1, 1 - slot):
                c.wait()


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
        "dma_stores",
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
    dma_stores: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`qkv_to_rpa_inputs` on a token count that is a multiple of 16
    (`dma_stores`: `_kernel_dma_stores`, same outputs)."""
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
    heads = _Heads(num_q_heads, num_kv_heads, head_dim, eps, k_scale, v_scale)
    kernel = functools.partial(_kernel, heads=heads)
    row_spec = lambda cols: pl.BlockSpec((bt, cols), lambda i: (i, 0))  # noqa: E731
    weight_spec = pl.BlockSpec((1, head_dim), lambda i: (0, 0))
    out_shape = [
        jax.ShapeDtypeStruct(
            (num_kv_heads, tokens, group // _Q_PACKING, _Q_PACKING, head_dim),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct((tokens, slots, _KV_PACKING, head_dim), kv_dtype),
    ]
    operands = (
        qkv,
        cos,
        sin,
        q_norm_weight.astype(jnp.float32).reshape(1, head_dim),
        k_norm_weight.astype(jnp.float32).reshape(1, head_dim),
    )
    if dma_stores:
        lanes = head_dim // _LANES
        return pl.pallas_call(
            functools.partial(
                _kernel_dma_stores, heads=heads, bt=bt, steps=tokens // bt
            ),
            grid=(tokens // bt,),
            in_specs=[
                row_spec(width),
                row_spec(half),
                row_spec(half),
                weight_spec,
                weight_spec,
            ],
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_shape=out_shape,
            scratch_shapes=[
                pltpu.VMEM(
                    (2, num_kv_heads * (group // _Q_PACKING) * lanes, bt, _LANES),
                    jnp.uint32,
                ),
                pltpu.VMEM((2, slots * lanes, bt, _LANES), jnp.uint32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            compiler_params=pltpu.CompilerParams(
                # Each step waits for the copies of the step two before.
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=_VMEM_LIMIT_BYTES,
            ),
            name="qkv_to_rpa_inputs",
            interpret=interpret,
        )(*operands)
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
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT_BYTES
        ),
        name="qkv_to_rpa_inputs",
        interpret=interpret,
    )(*operands)


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

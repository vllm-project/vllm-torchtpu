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
`[tokens, slots, 4, head_dim]` (HEAD_ALONG_SUBLANE) or `[2 * kv_heads,
head_dim // 4, 4, padded tokens]` (SEQ_ALONG_LANE), the layouts the kernel
reads, so nothing is relaid out in between. It covers bf16 queries, an FP8 KV
cache and static per-tensor KV scales; `prepare_inputs` also handles per-token
scales, which this does not.

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

from vllm_torchtpu.kernels.experimental.batched_rpa import configs

_KV_PACKING = 4  # FP8 rows per packed K/V slot
_Q_PACKING = 2  # bf16 query heads per packed row pair
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_VMEM_BUDGET_BYTES = 46 * 1024 * 1024
_BLOCK_TOKENS = (512, 256, 128, 64, 32, 16)
_LANES = 128


def _block_vmem_bytes(
    bt: int,
    width: int,
    num_q_heads: int,
    slots: int,
    d: int,
    fp8_queries: bool,
    kv_lanes: int | None = None,
) -> int:
    """VMEM of a `bt`-token block; `kv_lanes`: its SEQ_ALONG_LANE K/V block's
    token lanes when they exceed `bt` (one block of a page-padded bucket)."""
    kv = (kv_lanes or bt) * slots * _KV_PACKING * d
    io = bt * width * 2 + num_q_heads * bt * d * 2 + kv
    io += num_q_heads * bt * d if fp8_queries else 0
    return 2 * io + 8 * bt * d * 4  # double-buffered blocks + f32 temporaries


def _seq_along_lane_one_block(
    tokens: int,
    width: int,
    num_q_heads: int,
    slots: int,
    d: int,
    fp8_queries: bool,
    kv_lanes: int,
) -> bool:
    """Whether `tokens` (a multiple of 16) take one block whose SEQ_ALONG_LANE
    K/V span all `kv_lanes`: a block of at most 512 tokens that fits the VMEM
    budget. Otherwise the tokens are padded to `kv_lanes` and tiled in blocks
    of whole lane tiles."""
    vmem = _block_vmem_bytes(
        tokens, width, num_q_heads, slots, d, fp8_queries, kv_lanes
    )
    return tokens <= _BLOCK_TOKENS[0] and vmem <= _VMEM_BUDGET_BYTES


def choose_block_tokens(
    tokens: int,
    width: int,
    num_q_heads: int,
    slots: int,
    d: int,
    fp8_queries: bool = False,
    lane_multiple: bool = False,
) -> int:
    """The largest token block that divides `tokens` and fits the VMEM budget
    (with `lane_multiple`, a whole number of 128-token lane tiles)."""
    for bt in _BLOCK_TOKENS:
        if lane_multiple and bt % _LANES:
            continue
        vmem = _block_vmem_bytes(bt, width, num_q_heads, slots, d, fp8_queries)
        if tokens % bt == 0 and vmem <= _VMEM_BUDGET_BYTES:
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
    # Shared-KV pages (`wrapper.ragged_paged_attention(shared_kv_rotary_dims=
    # ...)`): the K/V rows are the value and the key's rotated dims, and the
    # queries carry the factor by which the key is the value elsewhere.
    shared_kv_rotary_dims: int = 0


def shared_kv_query_factor(
    k_norm_weight: jax.Array, rotary_dims: int, k_scale: float, v_scale: float
) -> jax.Array:
    """`[1, head_dim]` f32 factor of the queries that attend shared-KV pages.

    Off the leading `rotary_dims` of each half of head_dim, the key is
    `k_norm_weight` times the value, and the page holds the value quantized at
    `v_scale` where the kernel dequantizes the key at `k_scale`: the queries
    carry `k_norm_weight * v_scale / k_scale` there and 1 on the rotated dims.
    """
    d = k_norm_weight.shape[-1]
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, d), 1) % (d // 2)
    w = k_norm_weight.reshape(1, d).astype(jnp.float32) * (v_scale / k_scale)
    return jnp.where(lane < rotary_dims, 1.0, w)


def _block_values(qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, *, heads, kv_dtype):
    """A token block's values, shared by both kernels: `q_head(j)`, query head
    j's normed, rotated values (f32, rounded to bf16 by the caller), and
    `kv_rows()`, the quantized K/V rows in `prepare_inputs`' order (k, v of
    each KV head). With `shared_kv_rotary_dims` the rows are the value and the
    key's rotated dims, zero-padded to head_dim, and each query is scaled by
    `shared_kv_query_factor` off the rotated dims, where the key is the k norm
    weight times the value."""
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

    rot = heads.shared_kv_rotary_dims
    factor = None
    if rot:
        factor = shared_kv_query_factor(wk, rot, heads.k_scale, heads.v_scale)

    def q_head(j):
        q = _rotary(_rms(head(j), wq, eps), cos, sin)
        return q if factor is None else q * factor

    def kv_rows(transpose: bool = False):
        rows = []
        for h in range(nkv):
            k = _rotary(_rms(head(nq + h), wk, eps), cos, sin)
            k = k.astype(jnp.bfloat16).astype(jnp.float32)
            v = _rms(head(nq + nkv + h), None, eps)
            if rot:  # the value, then the key's rotated dims
                half = head_dim // 2
                pad = jnp.zeros((k.shape[0], head_dim - 2 * rot), k.dtype)
                k = jnp.concatenate([k[:, :rot], k[:, half : half + rot], pad], 1)
                if transpose:
                    k, v = k.T, v.T
                rows += [
                    _quantize(v, heads.v_scale, kv_dtype),
                    _quantize(k, heads.k_scale, kv_dtype),
                ]
                continue
            if transpose:  # in f32: quantizing commutes with it
                k, v = k.T, v.T
            rows += [
                _quantize(k, heads.k_scale, kv_dtype),
                _quantize(v, heads.v_scale, kv_dtype),
            ]
        return rows

    return q_head, kv_rows


def _kernel(
    qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, q_ref, kv_ref, *, heads, seq_along_lane
):
    """One token block. With `seq_along_lane` the new K/V block is `[2 *
    kv_heads, head_dim // 4, 4, lanes]` (each K and V row with its head_dim on
    sublanes in fours and the tokens on lanes; lanes past the block's tokens
    are zero), else `[tokens, slots, 4, head_dim]`."""
    group = heads.num_q_heads // heads.num_kv_heads
    q_head, kv_rows = _block_values(
        qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, heads=heads, kv_dtype=kv_ref.dtype
    )
    for j in range(heads.num_q_heads):
        h, rest = divmod(j, group)
        q_ref[h, :, rest // _Q_PACKING, rest % _Q_PACKING, :] = q_head(j).astype(
            q_ref.dtype
        )
    if seq_along_lane:
        bt, lanes = qkv_ref.shape[0], kv_ref.shape[-1]
        if bt < lanes:
            tail = (*kv_ref.shape[:3], lanes - bt)
            kv_ref[:, :, :, bt:] = jnp.zeros(tail, kv_ref.dtype)
        for r, row in enumerate(kv_rows(transpose=True)):
            kv_ref[r, :, :, :bt] = row.reshape(*kv_ref.shape[1:3], bt)
        return
    rows = kv_rows()
    slots = kv_ref.shape[1]
    zeros = jnp.zeros((kv_ref.shape[0], heads.head_dim), kv_ref.dtype)
    rows += [zeros] * (slots * _KV_PACKING - len(rows))  # prepare_inputs' padding
    for r, row in enumerate(rows):
        kv_ref[:, r // _KV_PACKING, r % _KV_PACKING, :] = row


def _kernel_dma_stores(
    qkv_ref,
    cos_ref,
    sin_ref,
    wq_ref,
    wk_ref,
    *refs,
    heads,
    bt,
    steps,
    fp8_queries,
    kv_lanes,
):
    """`_kernel` writing its outputs through 32-bit words.

    The packed outputs put two query heads (q) or four K/V rows (kv) of one
    token in each 32-bit word, so a token's word row is one sublane of a tile
    and a store of eight tokens is a strided one. Each (row group, 128-lane
    tile) is built here as `[tokens, 128]` words in a contiguous buffer and
    written with a strided DMA (double buffered across steps). With
    `fp8_queries` it also writes `wrapper.quantize_queries`' FP8 queries
    (four heads per word) for the fp8_compute kernel. With `kv_lanes` the
    K/V are SEQ_ALONG_LANE: each K and V row of a step is one `[head_dim / 4,
    kv_lanes]` word slab (zero past the step's tokens), written in one DMA.
    """
    if fp8_queries:
        q_hbm, kv_hbm, qf_hbm, q_buf, kv_buf, qf_buf, sem = refs
    else:
        (q_hbm, kv_hbm, q_buf, kv_buf, sem), qf_hbm, qf_buf = refs, None, None
    step = pl.program_id(0)
    slot = step % 2
    num_kv_heads, head_dim = heads.num_kv_heads, heads.head_dim
    group = heads.num_q_heads // num_kv_heads
    pairs = group // _Q_PACKING
    slots = kv_hbm.shape[1]
    lanes = range(0, head_dim, _LANES)
    q_words = q_hbm.bitcast(jnp.uint32)  # [kv_heads, tokens, pairs, 1, d]
    # [tokens, slots, 1, d], or with kv_lanes [2 * kv_heads, d / 4, 1, lanes]
    kv_words = kv_hbm.bitcast(jnp.uint32)
    q_slabs = [
        (h, p, lane)
        for h in range(num_kv_heads)
        for p in range(pairs)
        for lane in lanes
    ]
    if kv_lanes is None:
        kv_slabs = [(s, lane) for s in range(slots) for lane in lanes]
    else:
        kv_slabs = list(range(2 * num_kv_heads))
    quads = group // _KV_PACKING if fp8_queries else 0
    qf_slabs = [
        (h, quad, lane)
        for h in range(num_kv_heads)
        for quad in range(quads)
        for lane in lanes
    ]

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
        if kv_lanes is None:
            out += [
                pltpu.make_async_copy(
                    kv_buf.at[buf_slot, j],
                    kv_words.at[pl.ds(t0, bt), s, 0, pl.ds(lane, _LANES)],
                    sem.at[buf_slot],
                )
                for j, (s, lane) in enumerate(kv_slabs)
            ]
        else:
            l0 = pl.multiple_of(block * kv_lanes, kv_lanes)
            out += [
                pltpu.make_async_copy(
                    kv_buf.at[buf_slot, r],
                    kv_words.at[r, :, 0, pl.ds(l0, kv_lanes)],
                    sem.at[buf_slot],
                )
                for r in kv_slabs
            ]
        if fp8_queries:
            qf_words = qf_hbm.bitcast(jnp.uint32)  # [kv_heads, tokens, quads, 1, d]
            out += [
                pltpu.make_async_copy(
                    qf_buf.at[buf_slot, j],
                    qf_words.at[h, pl.ds(t0, bt), quad, 0, pl.ds(lane, _LANES)],
                    sem.at[buf_slot],
                )
                for j, (h, quad, lane) in enumerate(qf_slabs)
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

    def byte_words(rows):
        """Up to four FP8 arrays as words, row r in byte r, as packed."""
        words = None
        for r, row in enumerate(rows):
            byte = jax.lax.bitcast_convert_type(row, jnp.uint8)
            byte = byte.astype(jnp.uint32) << (8 * r)
            words = byte if words is None else words | byte
        return words

    def store(buf, j0, words):
        for i, lane in enumerate(lanes):
            buf[slot, j0 + i] = words[:, lane : lane + _LANES]

    # Query heads in groups of four with fp8_queries (one FP8 word), else pairs.
    per = _KV_PACKING if fp8_queries else _Q_PACKING
    for h in range(num_kv_heads):
        for g0 in range(0, group, per):
            values = [
                q_head(h * group + g).astype(jnp.bfloat16).astype(jnp.float32)
                for g in range(g0, g0 + per)
            ]
            for k in range(0, per, _Q_PACKING):
                first, second = values[k : k + _Q_PACKING]
                # bf16 bits of the pair's first head in the low half, as packed.
                words = (bits(first) >> 16) | (bits(second) & jnp.uint32(0xFFFF0000))
                store(q_buf, (h * pairs + (g0 + k) // _Q_PACKING) * len(lanes), words)
            if fp8_queries:
                fp8 = [_quantize(v, 1.0, qf_hbm.dtype) for v in values]
                store(qf_buf, (h * quads + g0 // per) * len(lanes), byte_words(fp8))
    if kv_lanes is None:
        rows = kv_rows()
        for s in range(slots):
            # prepare_inputs' zero padding rows are left as zero bytes.
            words = byte_words(rows[s * _KV_PACKING : (s + 1) * _KV_PACKING])
            store(kv_buf, s * len(lanes), words)
    else:
        for r, row in enumerate(kv_rows(transpose=True)):
            kv_buf[slot, r, :, :bt] = bits(row)  # four head_dim rows per word
            if bt < kv_lanes:
                kv_buf[slot, r, :, bt:] = jnp.zeros(
                    (head_dim // _KV_PACKING, kv_lanes - bt), jnp.uint32
                )

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
        "fp8_queries",
        "seq_along_lane_lanes",
        "shared_kv_rotary_dims",
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
    fp8_queries: bool = False,
    seq_along_lane_lanes: int | None = None,
    shared_kv_rotary_dims: int = 0,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """`qkv_to_rpa_inputs` on a token count that is a multiple of 16
    (`dma_stores`: `_kernel_dma_stores`, same outputs; `fp8_queries` adds the
    FP8 queries, `quantize_queries(q, kv_dtype)`). `seq_along_lane_lanes`
    writes SEQ_ALONG_LANE K/V of that many token lanes: in one block when the
    tokens fit one (zero past them; `block_tokens` does not apply), else in
    128-lane multiples of the tokens, which then must equal it."""
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
    kv_lanes = seq_along_lane_lanes
    # SEQ_ALONG_LANE K/V blocks span whole 128-token lane tiles, or all lanes.
    one_block = kv_lanes is not None and _seq_along_lane_one_block(
        tokens, width, num_q_heads, slots, head_dim, fp8_queries, kv_lanes
    )
    if one_block:
        bt = tokens
    else:
        bt = block_tokens or choose_block_tokens(
            tokens,
            width,
            num_q_heads,
            slots,
            head_dim,
            fp8_queries,
            kv_lanes is not None,
        )
    if tokens % bt or (
        kv_lanes is not None and not one_block and (bt % _LANES or tokens != kv_lanes)
    ):
        raise ValueError(f"block of {bt} tokens does not tile {tokens} tokens")
    if shared_kv_rotary_dims and (kv_lanes is None or num_kv_heads != 1):
        layout = "token-major" if kv_lanes is None else "SEQ_ALONG_LANE"
        raise ValueError(
            "shared_kv_rotary_dims writes SEQ_ALONG_LANE pages of one KV head, got "
            f"{num_kv_heads} KV heads in {layout} pages"
        )
    heads = _Heads(
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps,
        k_scale,
        v_scale,
        shared_kv_rotary_dims,
    )
    kernel = functools.partial(
        _kernel, heads=heads, seq_along_lane=kv_lanes is not None
    )
    kv_block = None  # SEQ_ALONG_LANE: the K/V lanes a step writes
    if kv_lanes is not None:
        kv_rows_shape = (2 * num_kv_heads, head_dim // _KV_PACKING, _KV_PACKING)
        kv_shape = (*kv_rows_shape, kv_lanes)
        kv_block = kv_lanes if one_block else bt
        kv_spec = pl.BlockSpec((*kv_rows_shape, kv_block), lambda i: (0, 0, 0, i))
    else:
        kv_shape = (tokens, slots, _KV_PACKING, head_dim)
        kv_spec = pl.BlockSpec(
            (bt, slots, _KV_PACKING, head_dim), lambda i: (i, 0, 0, 0)
        )
    row_spec = lambda cols: pl.BlockSpec((bt, cols), lambda i: (i, 0))  # noqa: E731
    weight_spec = pl.BlockSpec((1, head_dim), lambda i: (0, 0))
    out_shape = [
        jax.ShapeDtypeStruct(
            (num_kv_heads, tokens, group // _Q_PACKING, _Q_PACKING, head_dim),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct(kv_shape, kv_dtype),
    ]
    operands = (
        qkv,
        cos,
        sin,
        q_norm_weight.astype(jnp.float32).reshape(1, head_dim),
        k_norm_weight.astype(jnp.float32).reshape(1, head_dim),
    )
    if fp8_queries and (not dma_stores or group % _KV_PACKING):
        raise ValueError(
            f"fp8_queries needs dma_stores and q heads per kv head in fours: {group}"
        )
    if dma_stores:
        lanes = head_dim // _LANES
        kv_buf = (
            (2, slots * lanes, bt, _LANES)
            if kv_block is None
            else (2, 2 * num_kv_heads, head_dim // _KV_PACKING, kv_block)
        )
        scratch = [
            pltpu.VMEM(
                (2, num_kv_heads * (group // _Q_PACKING) * lanes, bt, _LANES),
                jnp.uint32,
            ),
            pltpu.VMEM(kv_buf, jnp.uint32),
        ]
        if fp8_queries:
            out_shape.append(
                jax.ShapeDtypeStruct(
                    (num_kv_heads, tokens, group // _KV_PACKING, _KV_PACKING, head_dim),
                    kv_dtype,
                )
            )
            scratch.append(
                pltpu.VMEM(
                    (2, num_kv_heads * (group // _KV_PACKING) * lanes, bt, _LANES),
                    jnp.uint32,
                )
            )
        return pl.pallas_call(
            functools.partial(
                _kernel_dma_stores,
                heads=heads,
                bt=bt,
                steps=tokens // bt,
                fp8_queries=fp8_queries,
                kv_lanes=kv_block,
            ),
            grid=(tokens // bt,),
            in_specs=[
                row_spec(width),
                row_spec(half),
                row_spec(half),
                weight_spec,
                weight_spec,
            ],
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * len(out_shape),
            out_shape=out_shape,
            scratch_shapes=[*scratch, pltpu.SemaphoreType.DMA((2,))],
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
            kv_spec,
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
    *,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int | None = None,
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """(q_hbm, new_kv_hbm) as `wrapper.prepare_inputs(..., kv_layout,
    page_size)` returns them, and with `fp8_queries` the FP8 queries the
    fp8_compute kernel reads. With `shared_kv_rotary_dims` (SEQ_ALONG_LANE,
    one KV head) the new K/V are shared-KV page rows and the queries carry
    `shared_kv_query_factor`, as
    `wrapper.ragged_paged_attention(shared_kv_rotary_dims=...)` reads them.

    A token count that 16-token blocks do not tile (an odd bucket) runs on
    zero rows padded to the next multiple of 16; SEQ_ALONG_LANE pads them to
    `prepare_inputs`' new-K/V length, a multiple of max(128, `page_size`),
    unless they fit one block (`_seq_along_lane_one_block`). The query outputs
    are cut back to `tokens`; SEQ_ALONG_LANE K/V keep `prepare_inputs`' padded
    length, zero past `tokens`.
    """
    tokens = qkv.shape[0]
    pad = -tokens % _BLOCK_TOKENS[-1]
    lanes = None
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if page_size is None:
            raise ValueError("SEQ_ALONG_LANE K/V need the cache's page_size")
        align = max(_LANES, page_size)
        lanes = -(-tokens // align) * align
        nq, nkv, d = (kwargs[k] for k in ("num_q_heads", "num_kv_heads", "head_dim"))
        slots = -(-2 * nkv // _KV_PACKING)
        fp8_queries = kwargs.get("fp8_queries", False)
        width = qkv.shape[1]
        if not _seq_along_lane_one_block(
            tokens + pad, width, nq, slots, d, fp8_queries, lanes
        ):
            pad = lanes - tokens  # whole lane tiles of tokens
    if pad:
        rows = ((0, pad), (0, 0))
        qkv, cos, sin = (jnp.pad(a, rows) for a in (qkv, cos, sin))
    out = _qkv_to_rpa_inputs(
        qkv,
        cos,
        sin,
        q_norm_weight,
        k_norm_weight,
        seq_along_lane_lanes=lanes,
        **kwargs,
    )
    if not pad:
        return out
    q, kv, *q_fp8 = out
    kv = kv if lanes is not None else kv[:tokens]
    return (q[:, :tokens], kv, *(q8[:, :tokens] for q8 in q_fp8))

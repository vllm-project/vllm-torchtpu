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
"""Fused QKV RMSNorm, rotary, FP8 KV quantization, and RPA layout packing."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs
from vllm_torchtpu.tpu_info import get_host_safe_tpu_info

_KV_PACKING = 4  # FP8 rows per packed K/V slot
_Q_PACKING = 2  # bf16 query heads per packed row pair
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_VMEM_BUDGET_BYTES = 46 * 1024 * 1024
_BLOCK_TOKENS = (512, 256, 128, 64, 32, 16)
_LANES = 128


def _vmem_limits() -> tuple[int, int]:
    info = get_host_safe_tpu_info()
    if info is None:
        return _VMEM_LIMIT_BYTES, _VMEM_BUDGET_BYTES
    limit = min(_VMEM_LIMIT_BYTES, int(info.vmem_capacity_bytes))
    budget = min(_VMEM_BUDGET_BYTES, max(limit - 2 * 1024 * 1024, limit * 7 // 8))
    return limit, budget


def _block_vmem_bytes(
    bt: int,
    width: int,
    num_q_heads: int,
    slots: int,
    d: int,
    fp8_queries: bool,
    kv_lanes: int | None = None,
) -> int:
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
    vmem = _block_vmem_bytes(
        tokens, width, num_q_heads, slots, d, fp8_queries, kv_lanes
    )
    return tokens <= _BLOCK_TOKENS[0] and vmem <= _vmem_limits()[1]


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
    budget = _vmem_limits()[1]
    for bt in _BLOCK_TOKENS:
        if lane_multiple and bt % _LANES:
            continue
        vmem = _block_vmem_bytes(bt, width, num_q_heads, slots, d, fp8_queries)
        if tokens % bt == 0 and vmem <= budget:
            return bt
    raise ValueError(f"no token block for {tokens} tokens of width {width}")


def _rms(x: jax.Array, w: jax.Array | None, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return x32 if w is None else x32 * w


def _rotary(y: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
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
    shared_kv_rotary_dims: int = 0


def shared_kv_query_factor(
    k_norm_weight: jax.Array, rotary_dims: int, k_scale: float, v_scale: float
) -> jax.Array:
    """`[1, head_dim]` f32 factor of the queries that attend shared-KV pages."""
    d = k_norm_weight.shape[-1]
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, d), 1) % (d // 2)
    w = k_norm_weight.reshape(1, d).astype(jnp.float32) * (v_scale / k_scale)
    return jnp.where(lane < rotary_dims, 1.0, w)


def _block_values(qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, *, heads, kv_dtype):
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
    kv_words = kv_hbm.bitcast(jnp.uint32)
    q_slabs = [
        (h, p, lane)
        for h in range(num_kv_heads)
        for p in range(pairs)
        for lane in lanes
    ]
    kv_pair_slabs = [(s, lane) for s in range(slots) for lane in lanes]
    kv_row_slabs = list(range(2 * num_kv_heads))
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
                for j, (s, lane) in enumerate(kv_pair_slabs)
            ]
        else:
            l0 = pl.multiple_of(block * kv_lanes, kv_lanes)
            out += [
                pltpu.make_async_copy(
                    kv_buf.at[buf_slot, r],
                    kv_words.at[r, :, 0, pl.ds(l0, kv_lanes)],
                    sem.at[buf_slot],
                )
                for r in kv_row_slabs
            ]
        if fp8_queries and qf_hbm is not None and qf_buf is not None:
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

    per = _KV_PACKING if fp8_queries else _Q_PACKING
    for h in range(num_kv_heads):
        for g0 in range(0, group, per):
            values = [
                q_head(h * group + g).astype(jnp.bfloat16).astype(jnp.float32)
                for g in range(g0, g0 + per)
            ]
            for k in range(0, per, _Q_PACKING):
                first, second = values[k : k + _Q_PACKING]
                words = (bits(first) >> 16) | (bits(second) & jnp.uint32(0xFFFF0000))
                store(q_buf, (h * pairs + (g0 + k) // _Q_PACKING) * len(lanes), words)
            if fp8_queries and qf_hbm is not None and qf_buf is not None:
                fp8 = [_quantize(v, 1.0, qf_hbm.dtype) for v in values]
                store(qf_buf, (h * quads + g0 // per) * len(lanes), byte_words(fp8))
    if kv_lanes is None:
        rows = kv_rows()
        for s in range(slots):
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

    def row_spec(cols: int) -> pl.BlockSpec:
        return pl.BlockSpec((bt, cols), lambda i: (i, 0))

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
    in_specs = [
        row_spec(width),
        row_spec(half),
        row_spec(half),
        weight_spec,
        weight_spec,
    ]
    vmem_limit = _vmem_limits()[0]
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
            in_specs=in_specs,
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * len(out_shape),
            out_shape=out_shape,
            scratch_shapes=[*scratch, pltpu.SemaphoreType.DMA((2,))],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=vmem_limit
            ),
            name="qkv_to_rpa_inputs",
            interpret=interpret,
        )(*operands)
    return pl.pallas_call(
        kernel,
        grid=(tokens // bt,),
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec(
                (num_kv_heads, bt, group // _Q_PACKING, _Q_PACKING, head_dim),
                lambda i: (0, i, 0, 0, 0),
            ),
            kv_spec,
        ],
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        name="qkv_to_rpa_inputs",
        interpret=interpret,
    )(*operands)


def qkv_to_rpa_inputs(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    *,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int | None = None,
    **kwargs,
) -> tuple[jax.Array, ...]:
    """Convert fused ``qkv`` rows into batched-RPA ``(q_hbm, new_kv[, q_fp8])``."""
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

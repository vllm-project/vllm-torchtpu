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
"""Fused QKV norm, RoPE, and KV quantization prologue for batched RPA.

`qkv_to_rpa_inputs` converts fused `[tokens, (num_q_heads + 2 * num_kv_heads) *
head_dim]` projection rows directly into the packed `HEAD_ALONG_SUBLANE` query
and new-KV buffers consumed by `batched_rpa`:
- per-head weighted RMSNorm on Q and K (and optional unweighted RMSNorm on V
  when `v_rms_norm=True`),
- NeoX rotary embedding on Q and K,
- static per-tensor FP8 quantization on K and V, and
- optional unit-scale FP8 query emission (`emit_fp8_queries=True`) for
  `fp8_compute` prefill.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128
_KV_PACKING = 4  # FP8 rows per packed 32-bit K/V slot
_Q_PACKING = 2  # bf16 query heads per packed 32-bit word
_BLOCK_TOKENS = (512, 256, 128, 64, 32, 16)
_DEFAULT_CPU_VMEM_BYTES = 1 << 24  # 16 MiB fallback when TPU info is unavailable
_COMPILER_VMEM_HEADROOM_BYTES = 1 << 21  # 2 MiB reserved for compiler temporaries


def _vmem_capacity_bytes(default: int = _DEFAULT_CPU_VMEM_BYTES) -> int:
    try:
        info = pltpu.get_tpu_info()
    except Exception:
        return default
    if info is None:
        return default
    return int(info.vmem_capacity_bytes)


def _block_vmem_bytes(
    bt: int,
    width: int,
    num_q_heads: int,
    slots: int,
    d: int,
    emit_fp8_queries: bool,
) -> int:
    io = bt * width * 2 + num_q_heads * bt * d * 2 + bt * slots * _KV_PACKING * d
    if emit_fp8_queries:
        io += num_q_heads * bt * d
    return 2 * io + 8 * bt * d * 4  # double-buffered blocks + f32 temporaries


def choose_block_tokens(
    tokens: int,
    width: int,
    num_q_heads: int,
    slots: int,
    d: int,
    emit_fp8_queries: bool = False,
    vmem_limit_bytes: int | None = None,
) -> int:
    """Return the largest token block dividing `tokens` within the VMEM budget."""
    limit = (
        _vmem_capacity_bytes() if vmem_limit_bytes is None else int(vmem_limit_bytes)
    )
    budget = max(limit - _COMPILER_VMEM_HEADROOM_BYTES, limit // 2)
    for bt in _BLOCK_TOKENS:
        vmem = _block_vmem_bytes(bt, width, num_q_heads, slots, d, emit_fp8_queries)
        if tokens % bt == 0 and vmem <= budget:
            return bt
    raise ValueError(f"no token block for {tokens} tokens of width {width}")


def _rms(x: jax.Array, w: jax.Array | None, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return x32 if w is None else x32 * w


def _rotary(y: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply NeoX rotary embedding on the leading `2 * cos.shape[-1]` lanes."""
    half = cos.shape[-1]
    y1, y2 = y[..., :half], y[..., half : 2 * half]
    parts = [y1 * cos - y2 * sin, y2 * cos + y1 * sin]
    if 2 * half < y.shape[-1]:
        parts.append(y[..., 2 * half :])
    return jnp.concatenate(parts, axis=-1)


def _quantize(x: jax.Array, scale: float, dtype: jnp.dtype) -> jax.Array:
    if scale != 1.0:
        x = x / scale
    info = jnp.finfo(dtype)
    return jnp.clip(x, float(info.min), float(info.max)).astype(dtype)


@dataclasses.dataclass(frozen=True)
class _Heads:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    eps: float
    k_scale: float
    v_scale: float
    v_rms_norm: bool = False


def _block_values(qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, *, heads, kv_dtype):
    nq = heads.num_q_heads
    nkv = heads.num_kv_heads
    head_dim = heads.head_dim
    eps = heads.eps
    cos = cos_ref[...].astype(jnp.float32)
    sin = sin_ref[...].astype(jnp.float32)
    wq, wk = wq_ref[...], wk_ref[...]

    def head(j: int) -> jax.Array:
        return qkv_ref[:, j * head_dim : (j + 1) * head_dim]

    def q_head(j: int) -> jax.Array:
        return _rotary(_rms(head(j), wq, eps), cos, sin)

    def kv_rows() -> list[jax.Array]:
        rows = []
        for h in range(nkv):
            k = _rotary(_rms(head(nq + h), wk, eps), cos, sin)
            # Match eager/XLA path where rotary output rounds to bf16 before KV cast.
            k = k.astype(jnp.bfloat16).astype(jnp.float32)
            raw_v = head(nq + nkv + h)
            v = (
                _rms(raw_v, None, eps)
                if heads.v_rms_norm
                else raw_v.astype(jnp.float32)
            )
            rows += [
                _quantize(k, heads.k_scale, kv_dtype),
                _quantize(v, heads.v_scale, kv_dtype),
            ]
        return rows

    return q_head, kv_rows


def _prologue_kernel(
    qkv_ref,
    cos_ref,
    sin_ref,
    wq_ref,
    wk_ref,
    *refs,
    heads: _Heads,
    bt: int,
    steps: int,
    emit_fp8_queries: bool,
):
    """Pack Q (2 bf16 heads/word) and K/V (4 FP8 rows/word) via strided 32-bit DMAs."""
    if emit_fp8_queries:
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
    kv_words = kv_hbm.bitcast(jnp.uint32)  # [tokens, slots, 1, d]
    q_slabs = [
        (h, p, lane)
        for h in range(num_kv_heads)
        for p in range(pairs)
        for lane in lanes
    ]
    kv_slabs = [(s, lane) for s in range(slots) for lane in lanes]
    quads = group // _KV_PACKING if emit_fp8_queries else 0
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
        out += [
            pltpu.make_async_copy(
                kv_buf.at[buf_slot, j],
                kv_words.at[pl.ds(t0, bt), s, 0, pl.ds(lane, _LANES)],
                sem.at[buf_slot],
            )
            for j, (s, lane) in enumerate(kv_slabs)
        ]
        if emit_fp8_queries:
            assert qf_hbm is not None and qf_buf is not None
            qf_words = qf_hbm.bitcast(jnp.uint32)
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
        def _():
            for c in copies(step - 2, slot):
                c.wait()

    q_head, kv_rows = _block_values(
        qkv_ref, cos_ref, sin_ref, wq_ref, wk_ref, heads=heads, kv_dtype=kv_hbm.dtype
    )

    def bits(x: jax.Array) -> jax.Array:
        return pltpu.bitcast(x, jnp.uint32)

    def byte_words(rows: list[jax.Array]) -> jax.Array:
        words = None
        for r, row in enumerate(rows):
            byte = jax.lax.bitcast_convert_type(row, jnp.uint8)
            byte = byte.astype(jnp.uint32) << (8 * r)
            words = byte if words is None else words | byte
        assert words is not None
        return words

    def store(buf, j0: int, words: jax.Array) -> None:
        for i, lane in enumerate(lanes):
            buf[slot, j0 + i] = words[:, lane : lane + _LANES]

    per = _KV_PACKING if emit_fp8_queries else _Q_PACKING
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
            if emit_fp8_queries:
                assert qf_hbm is not None and qf_buf is not None
                fp8 = [_quantize(v, 1.0, qf_hbm.dtype) for v in values]
                store(qf_buf, (h * quads + g0 // per) * len(lanes), byte_words(fp8))
    rows = kv_rows()
    for s in range(slots):
        words = byte_words(rows[s * _KV_PACKING : (s + 1) * _KV_PACKING])
        store(kv_buf, s * len(lanes), words)

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
        "v_rms_norm",
        "emit_fp8_queries",
        "block_tokens",
        "interpret",
    ),
)
def _qkv_to_rpa_inputs(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    k_scale: float,
    v_scale: float,
    kv_dtype: jnp.dtype = jnp.float8_e4m3fn,
    v_rms_norm: bool = False,
    emit_fp8_queries: bool = False,
    block_tokens: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
) -> tuple[jax.Array, ...]:
    tokens, width = qkv.shape
    group = num_q_heads // num_kv_heads
    if width != (num_q_heads + 2 * num_kv_heads) * head_dim:
        raise ValueError(f"qkv width {width} does not match the head counts")
    if qkv.dtype != jnp.bfloat16 or group % _Q_PACKING or head_dim % _LANES:
        raise ValueError(
            f"needs bf16 qkv, an even number of q heads per kv head and a "
            f"lane-aligned head_dim: {qkv.dtype}, {group}, {head_dim}"
        )
    if emit_fp8_queries and group % _KV_PACKING:
        raise ValueError(
            f"emit_fp8_queries requires q heads per kv head in multiples of "
            f"{_KV_PACKING}, got {group}"
        )
    half = cos.shape[-1]
    if cos.shape != (tokens, half) or sin.shape != cos.shape or 2 * half > head_dim:
        raise ValueError(f"rotary rows {cos.shape} / {sin.shape} for {qkv.shape}")

    vmem_limit = _vmem_capacity_bytes()
    slots = -(-2 * num_kv_heads // _KV_PACKING)
    bt = block_tokens or choose_block_tokens(
        tokens,
        width,
        num_q_heads,
        slots,
        head_dim,
        emit_fp8_queries=emit_fp8_queries,
        vmem_limit_bytes=vmem_limit,
    )
    if tokens % bt:
        raise ValueError(f"block of {bt} tokens does not tile {tokens} tokens")

    heads = _Heads(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
        k_scale=k_scale,
        v_scale=v_scale,
        v_rms_norm=v_rms_norm,
    )

    def _row_spec(cols: int) -> pl.BlockSpec:
        return pl.BlockSpec((bt, cols), lambda i: (i, 0))

    weight_spec = pl.BlockSpec((1, head_dim), lambda i: (0, 0))
    out_shape = [
        jax.ShapeDtypeStruct(
            (num_kv_heads, tokens, group // _Q_PACKING, _Q_PACKING, head_dim),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct((tokens, slots, _KV_PACKING, head_dim), kv_dtype),
    ]
    lanes = head_dim // _LANES
    scratch = [
        pltpu.VMEM(
            (2, num_kv_heads * (group // _Q_PACKING) * lanes, bt, _LANES),
            jnp.uint32,
        ),
        pltpu.VMEM((2, slots * lanes, bt, _LANES), jnp.uint32),
    ]
    if emit_fp8_queries:
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
    operands = (
        qkv,
        cos,
        sin,
        q_norm_weight.astype(jnp.float32).reshape(1, head_dim),
        k_norm_weight.astype(jnp.float32).reshape(1, head_dim),
    )
    return pl.pallas_call(
        functools.partial(
            _prologue_kernel,
            heads=heads,
            bt=bt,
            steps=tokens // bt,
            emit_fp8_queries=emit_fp8_queries,
        ),
        grid=(tokens // bt,),
        in_specs=[
            _row_spec(width),
            _row_spec(half),
            _row_spec(half),
            weight_spec,
            weight_spec,
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * len(out_shape),
        out_shape=out_shape,
        scratch_shapes=[*scratch, pltpu.SemaphoreType.DMA((2,))],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
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
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    k_scale: float,
    v_scale: float,
    kv_dtype: jnp.dtype = jnp.float8_e4m3fn,
    v_rms_norm: bool = False,
    emit_fp8_queries: bool = False,
    block_tokens: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Convert fused QKV rows into packed batched-RPA inputs.

    Token counts that are not multiples of 16 are zero-padded to the next
    16-token boundary and sliced back after the kernel completes.
    """
    tokens = qkv.shape[0]
    pad = -tokens % _BLOCK_TOKENS[-1]
    if pad:
        rows = ((0, pad), (0, 0))
        qkv, cos, sin = (jnp.pad(a, rows) for a in (qkv, cos, sin))
    out = _qkv_to_rpa_inputs(
        qkv,
        cos,
        sin,
        q_norm_weight,
        k_norm_weight,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_dtype=kv_dtype,
        v_rms_norm=v_rms_norm,
        emit_fp8_queries=emit_fp8_queries,
        block_tokens=block_tokens,
        interpret=interpret,
    )
    if not pad:
        return out  # type: ignore[return-value]
    q, kv, *q_fp8 = out
    return (q[:, :tokens], kv[:tokens], *(q8[:, :tokens] for q8 in q_fp8))  # type: ignore[return-value]

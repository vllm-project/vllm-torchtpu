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
"""Packed FP8 storage layout and pack/unpack helpers for GDN recurrent state.

Unscaled formats (`f32`, `bf16`) store `[n_v, d_k, d_v]` directly. Packed `fp8`
packs one sequence slot into a single `uint32` array `[packed_rows, d_v]` so DMA
transfers carry values and scales together with 8-sublane alignment: `n_v`
contiguous per-head data blocks (four `float8_e4m3fn` elements packed per
`uint32` along `d_k`), followed by `align_to(n_v, 8)` float32 power-of-two
per-column scale rows bitcast to `uint32`.
"""

import enum
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import fp8_matmul

# Upper bound for scaled FP8 columns (float8_e4m3fn max is 448).
_COLUMN_MAX = 256.0
# TPU vector-register sublane count (row alignment granularity for VMEM slices).
_SUBLANES = 8


class RecurrentStateFormat(enum.StrEnum):
    """Storage format of the recurrent state in HBM and VMEM windows.

    Unscaled formats store `[n_v, d_k, d_v]` directly (`F32`, `BF16`). `FP8`
    stores `float8_e4m3fn` values packed into `uint32` alongside per-column
    power-of-two scales.
    """

    F32 = "f32"
    BF16 = "bf16"
    FP8 = "fp8"


def is_scaled(fmt: RecurrentStateFormat) -> bool:
    """Whether `fmt` packs quantized values alongside per-column scales."""
    return fmt == RecurrentStateFormat.FP8


def resolve_format(
    fmt: RecurrentStateFormat | str | None,
    recurrent_dtype: jax.typing.DTypeLike | None = None,
) -> RecurrentStateFormat:
    """Resolves `fmt`, falling back to `recurrent_dtype` when `fmt` is None."""
    if fmt is not None:
        return RecurrentStateFormat(fmt)
    if recurrent_dtype is None:
        return RecurrentStateFormat.F32
    dtype = jnp.dtype(recurrent_dtype)
    if dtype == jnp.float32:
        return RecurrentStateFormat.F32
    if dtype == jnp.bfloat16:
        return RecurrentStateFormat.BF16
    raise NotImplementedError(f"Unsupported recurrent_dtype: {recurrent_dtype}")


def data_rows_per_head(fmt: RecurrentStateFormat, d_k: int) -> int:
    """Returns `uint32` data rows per value head for `fmt`."""
    if is_scaled(fmt) and d_k % (4 * _SUBLANES) != 0:
        raise ValueError(
            f"d_k must be a multiple of {4 * _SUBLANES} for {fmt.value}; got {d_k}."
        )
    match fmt:
        case RecurrentStateFormat.F32:
            return d_k
        case RecurrentStateFormat.BF16:
            if d_k % 2 != 0:
                raise ValueError(f"d_k must be even for {fmt.value}; got {d_k}.")
            return d_k // 2
        case RecurrentStateFormat.FP8:
            return d_k // 4
    raise ValueError(f"Unknown recurrent state format: {fmt!r}.")


def scale_rows(fmt: RecurrentStateFormat, n_v: int) -> int:
    """Returns 8-aligned `uint32` scale rows per sequence slot for `fmt`."""
    if not is_scaled(fmt):
        return 0
    return pl.cdiv(n_v, _SUBLANES) * _SUBLANES


def packed_rows(fmt: RecurrentStateFormat, n_v: int, d_k: int) -> int:
    """Returns total `uint32` rows per sequence slot for `fmt`."""
    return n_v * data_rows_per_head(fmt, d_k) + scale_rows(fmt, n_v)


def slot_bytes(fmt: RecurrentStateFormat, n_v: int, d_k: int, d_v: int) -> int:
    """Returns stored bytes of one sequence slot across all `n_v` heads."""
    if not is_scaled(fmt):
        itemsize = 2 if fmt == RecurrentStateFormat.BF16 else 4
        return n_v * d_k * d_v * itemsize
    return packed_rows(fmt, n_v, d_k) * d_v * 4


def slot_granules(
    fmt: RecurrentStateFormat, n_v: int, d_k: int, d_v: int, num_lanes: int
) -> int:
    """Returns 512-byte granules (`num_lanes * 4` bytes) per sequence slot."""
    lane_bytes = num_lanes * 4
    total_bytes = slot_bytes(fmt, n_v, d_k, d_v)
    if total_bytes % lane_bytes != 0:
        raise ValueError(
            f"slot_bytes ({total_bytes}) must be divisible by lane_bytes "
            f"({lane_bytes}) for {fmt.value} (n_v={n_v}, d_k={d_k}, d_v={d_v})."
        )
    return total_bytes // lane_bytes


def slot_window_shape(
    fmt: RecurrentStateFormat, n_v: int, d_k: int, d_v: int
) -> tuple[int, ...]:
    """Returns one sequence's `uint32` VMEM window shape for `fmt`."""
    if is_scaled(fmt):
        return (packed_rows(fmt, n_v, d_k), d_v)
    return (n_v, data_rows_per_head(fmt, d_k), d_v)


def _unscaled_dtype(fmt: RecurrentStateFormat) -> jnp.dtype:
    if fmt == RecurrentStateFormat.F32:
        return jnp.dtype(jnp.float32)
    if fmt == RecurrentStateFormat.BF16:
        return jnp.dtype(jnp.bfloat16)
    raise ValueError(f"Expected unscaled recurrent state format; got {fmt!r}.")


def _pow2_scale_from_amax_bf16(
    amax: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Computes bfloat16 power-of-two `(scale, inverse_scale)` from `amax`."""
    return fp8_matmul.pow2_scales_from_amax(amax, _COLUMN_MAX)


def _pow2_column_scale_bf16(
    x_bf16: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Returns exact bfloat16 power-of-two `(scale, inverse_scale)` on `-2`."""
    return _pow2_scale_from_amax_bf16(jnp.max(jnp.abs(x_bf16), axis=-2, keepdims=True))


def _max_abs_8_heads(heads: Sequence[jax.Array], d_v: int) -> jax.Array:
    """Reduces `<= 8` bfloat16 `[d_k, d_v]` heads to one `[8, d_v]` max tile."""
    # Fold each head's d_k down to 8 sublane rows (two 8-row halves per step),
    # then run a 3-stage butterfly so row i holds max(abs(heads[i])).
    m: list[jax.Array] = []
    for x in heads:
        d_k = x.shape[0]
        m16 = jnp.max(jnp.abs(x.reshape(d_k // 16, 16, d_v)), axis=0)
        m.append(jnp.maximum(m16[:8, :], m16[8:, :]))
    if len(m) < _SUBLANES:
        zero = jnp.zeros((_SUBLANES, d_v), jnp.bfloat16)
        m.extend([zero] * (_SUBLANES - len(m)))

    # 3-stage sublane butterfly folds 8 heads -> 4 -> 2 -> 1 across sublanes so
    # row i holds max(abs(heads[i])) while keeping all 8 sublanes active.
    iota = jax.lax.broadcasted_iota(jnp.int32, (_SUBLANES, d_v), 0)
    sel4 = (iota & 4) == 0
    p = [
        jnp.where(
            sel4,
            jnp.maximum(m[i], jnp.roll(m[i], -4, axis=0)),
            jnp.maximum(m[i + 4], jnp.roll(m[i + 4], 4, axis=0)),
        )
        for i in range(4)
    ]
    sel2 = (iota & 2) == 0
    q = [
        jnp.where(
            sel2,
            jnp.maximum(p[i], jnp.roll(p[i], -2, axis=0)),
            jnp.maximum(p[i + 2], jnp.roll(p[i + 2], 2, axis=0)),
        )
        for i in range(2)
    ]
    sel1 = (iota & 1) == 0
    return jnp.where(
        sel1,
        jnp.maximum(q[0], jnp.roll(q[0], -1, axis=0)),
        jnp.maximum(q[1], jnp.roll(q[1], 1, axis=0)),
    )


def _mask_u32(raw_u32: jax.Array, has_init: jax.Array | None) -> jax.Array:
    # Mask raw uint32 bits before bitcasting/dequantizing so uninitialized HBM
    # NaN bit patterns cannot propagate when `has_init` is False.
    if has_init is None:
        return raw_u32
    return jnp.where(has_init, raw_u32, jnp.uint32(0))


def unpack_head(
    raw_u32: jax.Array,
    scale_f32: jax.Array,
    fmt: RecurrentStateFormat,
) -> jax.Array:
    """Unpacks one FP8 head `(raw_u32, scale_f32)` to float32 `[d_k, d_v]`."""
    if fmt != RecurrentStateFormat.FP8:
        raise ValueError(f"unpack_head requires FP8; got {fmt!r}.")
    val_bf16 = pltpu.bitcast(raw_u32, jnp.float8_e4m3fn).astype(jnp.bfloat16)
    # Exact in bfloat16 because scale_f32 is a power of two (shifts only exponent).
    return (val_bf16 * scale_f32.astype(jnp.bfloat16)).astype(jnp.float32)


def load_scale_tile(
    slot_ref: jax.Ref,
    seq_idx: int,
    ckpt_idx: int,
    v0: int,
    *,
    n_v: int,
    d_k: int,
    fmt: RecurrentStateFormat,
    has_init: jax.Array | None = None,
) -> jax.Array:
    """Loads one `[8, d_v]` float32 column-scale tile starting at head `v0`."""
    if fmt != RecurrentStateFormat.FP8:
        raise ValueError(f"load_scale_tile requires FP8; got {fmt!r}.")
    if v0 % _SUBLANES != 0:
        raise ValueError(f"v0 must be a multiple of {_SUBLANES}; got {v0}.")
    if not (0 <= v0 < n_v):
        raise ValueError(f"v0 must be in 0..{n_v - 1}; got {v0}.")
    data_rows = n_v * data_rows_per_head(fmt, d_k)
    row0 = data_rows + v0
    raw_scale = _mask_u32(
        slot_ref[seq_idx, ckpt_idx, pl.ds(row0, _SUBLANES), :],
        has_init,
    )
    return jax.lax.bitcast_convert_type(raw_scale, jnp.float32)


def load_head(
    slot_ref: jax.Ref,
    seq_idx: int,
    ckpt_idx: int,
    v: int,
    *,
    d_k: int,
    fmt: RecurrentStateFormat,
    scale_row: jax.Array,
    has_init: jax.Array | None = None,
) -> jax.Array:
    """Loads one packed FP8 head `v` from `slot_ref` as `[d_k, d_v]` float32."""
    if fmt != RecurrentStateFormat.FP8:
        raise ValueError(f"load_head requires FP8; got {fmt!r}.")
    rph = data_rows_per_head(fmt, d_k)
    row0 = v * rph
    raw_v = _mask_u32(slot_ref[seq_idx, ckpt_idx, pl.ds(row0, rph), :], has_init)
    return unpack_head(raw_v, scale_row, fmt)


def store_head_block(
    slot_ref: jax.Ref,
    seq_idx: int,
    ckpt_idx: int,
    v0: int,
    heads_blk: Sequence[jax.Array],
    *,
    n_v: int,
    d_k: int,
    d_v: int,
    fmt: RecurrentStateFormat,
) -> None:
    """Packs and writes `<= 8` FP8 heads starting at `v0` into `slot_ref`."""
    if fmt != RecurrentStateFormat.FP8:
        raise ValueError(f"store_head_block requires FP8; got {fmt!r}.")
    if v0 % _SUBLANES != 0:
        raise ValueError(f"v0 must be a multiple of {_SUBLANES}; got {v0}.")
    blk = len(heads_blk)
    if not (1 <= blk <= _SUBLANES):
        raise ValueError(f"heads_blk length must be in 1..{_SUBLANES}; got {blk}.")
    if v0 < 0 or v0 + blk > n_v:
        raise ValueError(
            f"Head range {v0}..{v0 + blk - 1} is out of bounds for n_v={n_v}."
        )
    rph = data_rows_per_head(fmt, d_k)
    data_rows = n_v * rph
    bf16_heads = [h.astype(jnp.bfloat16) for h in heads_blk]
    amax_8 = _max_abs_8_heads(bf16_heads, d_v)
    scale_bf16, inv_bf16 = _pow2_scale_from_amax_bf16(amax_8)
    for i, x_bf16 in enumerate(bf16_heads):
        row0 = (v0 + i) * rph
        raw_v = pltpu.bitcast(
            (x_bf16 * inv_bf16[i : i + 1, :]).astype(jnp.float8_e4m3fn),
            jnp.uint32,
        )
        slot_ref[seq_idx, ckpt_idx, pl.ds(row0, rph), :] = raw_v

    scale_u32 = jax.lax.bitcast_convert_type(scale_bf16.astype(jnp.float32), jnp.uint32)
    # Zero trailing padding rows so partial tail blocks match host pack_state.
    if blk < _SUBLANES:
        iota = jax.lax.broadcasted_iota(jnp.int32, (_SUBLANES, d_v), 0)
        scale_u32 = jnp.where(iota < blk, scale_u32, jnp.uint32(0))
    sc_row0 = data_rows + v0
    slot_ref[seq_idx, ckpt_idx, pl.ds(sc_row0, _SUBLANES), :] = scale_u32


def unpack_slot_to_carry(
    slot_ref: jax.Ref,
    carry_ref: jax.Ref,
    has_init: jax.Array,
    *,
    n_v: int,
    d_k: int,
    fmt: RecurrentStateFormat,
) -> None:
    """Streams one sequence slot from `slot_ref[0, 0]` to float32 `carry_ref`."""
    if not is_scaled(fmt):
        dtype = _unscaled_dtype(fmt)
        for v in range(n_v):
            raw_v = _mask_u32(slot_ref[0, 0, v], has_init)
            carry_ref[0, v] = pltpu.bitcast(raw_v, dtype).astype(jnp.float32)
        return
    for v0 in range(0, n_v, _SUBLANES):
        blk = min(_SUBLANES, n_v - v0)
        scale_tile = load_scale_tile(
            slot_ref, 0, 0, v0, n_v=n_v, d_k=d_k, fmt=fmt, has_init=has_init
        )
        for i in range(blk):
            v = v0 + i
            carry_ref[0, v] = load_head(
                slot_ref,
                0,
                0,
                v,
                d_k=d_k,
                fmt=fmt,
                scale_row=scale_tile[i : i + 1, :],
                has_init=has_init,
            )


def pack_carry_to_slot(
    carry_ref: jax.Ref,
    slot_ref: jax.Ref,
    *,
    n_v: int,
    d_k: int,
    d_v: int,
    fmt: RecurrentStateFormat,
) -> None:
    """Writes `carry_ref[0]` into `slot_ref[0, 0]` using storage format `fmt`."""
    if not is_scaled(fmt):
        dtype = _unscaled_dtype(fmt)
        for v in range(n_v):
            slot_ref[0, 0, v] = pltpu.bitcast(carry_ref[0, v].astype(dtype), jnp.uint32)
        return
    for v0 in range(0, n_v, _SUBLANES):
        blk = min(_SUBLANES, n_v - v0)
        store_head_block(
            slot_ref,
            0,
            0,
            v0,
            [carry_ref[0, v0 + i] for i in range(blk)],
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            fmt=fmt,
        )


class RecurrentStateStream:
    """Streams per-head recurrent state between VMEM slots and the kernel.

    Unscaled formats (`f32`, `bf16`) read and write one head at a time. Packed
    `fp8` caches each 8-head scale tile on reads and buffers up to 8 updated
    heads per checkpoint window before packing and writing them together, so
    callers must visit heads in ascending order (`0..n_v - 1`) per sequence.
    """

    def __init__(
        self,
        in_slot_ref: jax.Ref,
        out_slot_ref: jax.Ref,
        *,
        fmt: RecurrentStateFormat,
        n_v: int,
        d_k: int,
        d_v: int,
        window_size: int,
        seq_has_init: Sequence[jax.Array],
        carry_ref: jax.Ref | None = None,
        seq_is_first_tile: Sequence[jax.Array] | None = None,
        flush_cond: jax.Array | None = None,
    ):
        """Initializes the per-head VMEM state stream.

        Args:
            in_slot_ref: Input `uint32` VMEM slot ref holding incoming sequence state.
            out_slot_ref: Output `uint32` VMEM slot ref receiving checkpoint states.
            fmt: Storage format (`F32`, `BF16`, or packed `FP8`).
            n_v: Number of value heads per sequence.
            d_k: Key/query head dimension.
            d_v: Value head dimension.
            window_size: Number of speculative checkpoints written per sequence.
            seq_has_init: Per-sequence booleans indicating whether `in_slot_ref`
                holds valid initial state (False masks raw bits to zero).
            carry_ref: Optional inter-tile float32 VMEM carry scratchpad.
            seq_is_first_tile: Per-sequence booleans selecting `in_slot_ref` vs
                `carry_ref`; required when `carry_ref` is set.
            flush_cond: Optional scalar boolean gating FP8 writes to `out_slot_ref`.
        """
        if carry_ref is not None and seq_is_first_tile is None:
            raise ValueError("seq_is_first_tile is required when carry_ref is set.")
        self._in_slot_ref = in_slot_ref
        self._out_slot_ref = out_slot_ref
        self._fmt = fmt
        self._is_scaled = is_scaled(fmt)
        self._n_v = n_v
        self._d_k = d_k
        self._d_v = d_v
        self._window_size = window_size
        self._seq_has_init = seq_has_init
        self._carry_ref = carry_ref
        self._seq_is_first_tile = seq_is_first_tile
        self._flush_cond = flush_cond
        self._scale_key: tuple[int, int] | None = None
        self._scale_tile: jax.Array | None = None
        self._ckpt_heads: list[list[jax.Array]] = [[] for _ in range(window_size)]
        self._ckpt_keys: list[tuple[int, int] | None] = [None] * window_size

    def load_head(self, s: int, v: int) -> jax.Array:
        """Loads head `v` of sequence `s` as `[d_k, d_v]` float32."""
        if not (0 <= v < self._n_v):
            raise ValueError(f"Head index v must be in 0..{self._n_v - 1}; got v={v}.")
        if self._is_scaled:
            hbm_v = load_head(
                self._in_slot_ref,
                s,
                0,
                v,
                d_k=self._d_k,
                fmt=self._fmt,
                scale_row=self._scale_row(s, v),
                has_init=self._seq_has_init[s],
            )
        else:
            raw_v = _mask_u32(self._in_slot_ref[s, 0, v], self._seq_has_init[s])
            hbm_v = pltpu.bitcast(raw_v, _unscaled_dtype(self._fmt)).astype(jnp.float32)
        if self._carry_ref is not None:
            assert self._seq_is_first_tile is not None
            return jnp.where(self._seq_is_first_tile[s], hbm_v, self._carry_ref[s, v])
        return hbm_v

    def store_ckpt_head(self, s: int, v: int, w_idx: int, state: jax.Array) -> None:
        """Writes head `v` for checkpoint `w_idx`, flushing 8-head FP8 blocks."""
        if not (0 <= v < self._n_v):
            raise ValueError(f"Head index v must be in 0..{self._n_v - 1}; got v={v}.")
        if self._carry_ref is not None and w_idx == self._window_size - 1:
            self._carry_ref[s, v] = state
        if not self._is_scaled:
            self._out_slot_ref[s, w_idx, v] = pltpu.bitcast(
                state.astype(_unscaled_dtype(self._fmt)), jnp.uint32
            )
            return

        v0 = (v // _SUBLANES) * _SUBLANES
        blk = min(_SUBLANES, self._n_v - v0)
        buf = self._ckpt_heads[w_idx]
        key = (s, v0)
        active_key = self._ckpt_keys[w_idx]
        if active_key is not None and active_key != key:
            raise ValueError(
                "RecurrentStateStream expects contiguous ascending heads per "
                f"(seq, block); expected {active_key}, got {key} (v={v})."
            )
        if v != v0 + len(buf):
            raise ValueError(
                f"RecurrentStateStream expects heads in ascending order; got v={v}"
                f" at block offset {len(buf)} (v0={v0})."
            )
        self._ckpt_keys[w_idx] = key
        buf.append(state.astype(jnp.bfloat16))
        if len(buf) < blk:
            return
        self._ckpt_heads[w_idx] = []
        self._ckpt_keys[w_idx] = None
        self._flush_scaled_block(s, w_idx, v0, buf)

    def _scale_row(self, s: int, v: int) -> jax.Array:
        """Returns head `v`'s `[1, d_v]` scale row, caching each 8-head tile."""
        v0 = (v // _SUBLANES) * _SUBLANES
        if self._scale_key != (s, v0):
            self._scale_tile = load_scale_tile(
                self._in_slot_ref,
                s,
                0,
                v0,
                n_v=self._n_v,
                d_k=self._d_k,
                fmt=self._fmt,
                has_init=self._seq_has_init[s],
            )
            self._scale_key = (s, v0)
        assert self._scale_tile is not None
        return self._scale_tile[v - v0 : v - v0 + 1]

    def _flush_scaled_block(
        self, s: int, w_idx: int, v0: int, heads_blk: Sequence[jax.Array]
    ) -> None:
        """Packs and stores one `<= 8`-head block, honoring `flush_cond`."""

        def _write() -> None:
            store_head_block(
                self._out_slot_ref,
                s,
                w_idx,
                v0,
                heads_blk,
                n_v=self._n_v,
                d_k=self._d_k,
                d_v=self._d_v,
                fmt=self._fmt,
            )

        if self._flush_cond is not None:

            @pl.when(self._flush_cond)
            def _() -> None:
                _write()

        else:
            _write()


def _host_bitcast_dim_m2(x: jax.Array, ty: jax.typing.DTypeLike) -> jax.Array:
    """Bitcasts `x` to `ty` along axis `-2` in both eager and jitted JAX."""
    # pltpu.bitcast packs sub-32-bit values along the sublane axis (-2), while
    # jax.lax.bitcast_convert_type packs along the last axis (-1), so transpose
    # -2 <-> -1 around the bitcast to match on-device layout identically.
    dst_dtype = jnp.dtype(ty)
    src_bits = x.dtype.itemsize * 8
    dst_bits = dst_dtype.itemsize * 8
    x = jnp.moveaxis(x, -2, -1)
    if src_bits < dst_bits:
        *leading, n = x.shape
        packing = dst_bits // src_bits
        y = jax.lax.bitcast_convert_type(
            x.reshape(*leading, n // packing, packing), dst_dtype
        )
    elif src_bits > dst_bits:
        y = jax.lax.bitcast_convert_type(x, dst_dtype)
        *leading, n, packing = y.shape
        y = y.reshape(*leading, n * packing)
    else:
        y = jax.lax.bitcast_convert_type(x, dst_dtype)
    return jnp.moveaxis(y, -1, -2)


def pack_state(state: jax.Array, fmt: RecurrentStateFormat | str) -> jax.Array:
    """Packs host `[..., n_v, d_k, d_v]` state into the kernel's HBM layout.

    Args:
        state: Unpacked recurrent state `[..., n_v, d_k, d_v]`.
        fmt: Target storage format (`"f32"`, `"bf16"`, or `"fp8"`).

    Returns:
        `[..., n_v, d_k, d_v]` in float32/bfloat16 for unscaled formats, or
        `[..., packed_rows, d_v]` in `uint32` for packed FP8.
    """
    fmt = RecurrentStateFormat(fmt)
    state = state.astype(jnp.float32)
    if fmt == RecurrentStateFormat.F32:
        return state
    # Match TPU bfloat16 rounding identically across host CPU and device.
    x_bf16 = jax.lax.reduce_precision(state, exponent_bits=8, mantissa_bits=7).astype(
        jnp.bfloat16
    )
    if fmt == RecurrentStateFormat.BF16:
        return x_bf16

    *batch, n_v, d_k, d_v = state.shape
    rph = data_rows_per_head(fmt, d_k)
    scale_bf16, inv_bf16 = _pow2_column_scale_bf16(x_bf16)
    raw_u32 = _host_bitcast_dim_m2(
        (x_bf16 * inv_bf16).astype(jnp.float8_e4m3fn), jnp.uint32
    )
    raw_flat = raw_u32.reshape(*batch, n_v * rph, d_v)
    scale_u32 = jax.lax.bitcast_convert_type(
        jnp.squeeze(scale_bf16.astype(jnp.float32), axis=-2), jnp.uint32
    )
    pad_rows = scale_rows(fmt, n_v) - n_v
    if pad_rows:
        pad_width = [(0, 0)] * len(batch) + [(0, pad_rows), (0, 0)]
        scale_u32 = jnp.pad(scale_u32, pad_width)
    return jnp.concatenate([raw_flat, scale_u32], axis=-2)


def unpack_state(
    packed: jax.Array,
    fmt: RecurrentStateFormat | str,
    *,
    n_v: int,
    d_k: int,
) -> jax.Array:
    """Unpacks HBM `packed` back to float32 `[..., n_v, d_k, d_v]`."""
    fmt = RecurrentStateFormat(fmt)
    if not is_scaled(fmt):
        return packed.astype(jnp.float32)
    if packed.dtype != jnp.uint32:
        raise ValueError(
            f"Expected uint32 packed buffer for {fmt.value}; got {packed.dtype}."
        )

    *batch, total_rows, d_v = packed.shape
    expected_rows = packed_rows(fmt, n_v, d_k)
    if total_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} packed rows for {fmt.value} (n_v={n_v},"
            f" d_k={d_k}); got {total_rows}."
        )
    rph = data_rows_per_head(fmt, d_k)
    data_rows = n_v * rph
    raw_u32 = packed[..., :data_rows, :].reshape(*batch, n_v, rph, d_v)
    scale_f32 = jax.lax.bitcast_convert_type(
        packed[..., data_rows : data_rows + n_v, :], jnp.float32
    )[..., :, None, :]

    val = _host_bitcast_dim_m2(raw_u32, jnp.float8_e4m3fn).astype(jnp.float32)
    return val * scale_f32

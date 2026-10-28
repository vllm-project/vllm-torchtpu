# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""TurboQuant KV-cache quantization and Pallas VREG dequantization on TPU.

Implements TurboQuant (https://arxiv.org/abs/2504.19874) matching vLLM's upstream
config (`vllm/model_executor/layers/quantization/turboquant/`,
https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/):

* **MSE keys (`4bit_nc`, `k3v4_nc`, `3bit_nc`):** `k / ||k||` is rotated by an
  orthonormal Walsh-Hadamard matrix `H` and quantized to 3- or 4-bit Lloyd-Max
  centroids (`_nc` presets fold the norm correction in at store time, saving
  `||k|| * rsqrt(sum(c^2) + 1e-16)` as `bf16` instead of raw `||k||`). Because
  `q . k == (q H) . (k H)`, queries are pre-rotated once (`q @ H`) instead of
  inverse-rotating keys.
* **FP8 keys (`k8v4`):** Unrotated `float8_e4m3fn` bytes.
* **Values:** Per-token asymmetric 3- or 4-bit uniform quantization with `bf16`
  `(v_scale, v_zero)` metadata.

### Layout (`(num_blocks, num_kv_heads, page_words, 128)`) & Execution Flow

* **Slot:** One token position (`0 .. page_size - 1`) for one KV head
  (`slot_size_aligned` bytes).
* **Lane & Word:** A **word** is one 128-byte row (`u8[_WORD_SIZE]`), with one
  byte per TPU vector register **lane** (`0..127`).
* **Page:** `page_size` slots for one KV head, stored as `u8[page_words, 128]`
  partitioned into `[k_plane | v_plane | meta_plane]` and padded to 8-word
  multiples for TPU `T(8, 128)` sublane tiling.
* **Flow:** `_quantize_kv_components` + `_pack_new_tq_tokens_128` quantize and
  pack new tokens on XLA; inside Pallas RPA v3 (`RPAd`/`RPAm`),
  `_update_tq_page_vreg` merges new tokens into pages in VMEM/VREGs and
  `_dequant_tq_page_vreg` dequantizes pages in VREGs before MXU attention.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_WORD_SIZE = 128  # Bytes (and lanes) per TPU VREG row (`u8[128]`).
_SUBLANE_WORDS = 8  # VMEM sublane tile height for `T(8, 128)` tiling.

# Suffix `_nc` denotes norm correction: rescaling reconstructed unit-sphere
# Lloyd-Max centroid vectors by `1 / ||k_centroids||_2` before applying `||k||_2`.
TURBOQUANT_KV_CACHE_DTYPES: tuple[str, ...] = (
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)


def is_turboquant_kv_cache_dtype(cache_dtype: object) -> bool:
    return (
        isinstance(cache_dtype, str)
        and cache_dtype.lower().strip() in TURBOQUANT_KV_CACHE_DTYPES
    )


# Matches the epsilons in vLLM's Triton store/decode kernels.
_KEY_NORM_EPS = 1e-8
_CENTROID_NORM_EPS = 1e-16
_VALUE_SCALE_MIN = 1e-8


@dataclasses.dataclass(frozen=True)
class TQLayout:
    """Byte geometry of one TurboQuant page and slot (one token, one KV head)."""

    head_dim: int
    key_bits: int  # 8 = FP8 keys; 3 or 4 = Lloyd-Max MSE keys.
    value_bits: int  # 3 or 4.
    norm_correction: bool  # True for `*_nc` presets; renormalizes centroid vectors.
    slot_size_aligned: int  # Even-aligned bytes per (token, head) slot.

    def __post_init__(self) -> None:
        if self.head_dim < _WORD_SIZE or self.head_dim % _WORD_SIZE != 0:
            raise ValueError(
                f"TurboQuant head_dim must be a positive multiple of {_WORD_SIZE}, "
                f"got {self.head_dim}"
            )
        if not self.is_key_fp8 and (self.head_dim & (self.head_dim - 1)):
            raise ValueError(
                "TurboQuant MSE key presets require power-of-two "
                f"head_dim >= {_WORD_SIZE}, got {self.head_dim}"
            )

    @classmethod
    def from_tq_config(cls, tq_config: Any) -> TQLayout:
        return cls(
            head_dim=tq_config.head_dim,
            key_bits=tq_config.key_quant_bits,
            value_bits=tq_config.value_quant_bits,
            norm_correction=tq_config.norm_correction,
            slot_size_aligned=tq_config.slot_size_aligned,
        )

    @property
    def is_key_fp8(self) -> bool:
        return self.key_bits == 8

    @property
    def supported_page_sizes(self) -> tuple[int, ...]:
        return (16, 32, 64) if self.is_key_fp8 else (16, 64)

    def validate_page_size(self, page_size: int) -> None:
        if page_size not in self.supported_page_sizes:
            raise ValueError(
                f"Unsupported TurboQuant page_size={page_size} for "
                f"key_bits={self.key_bits}; supported page sizes are "
                f"{self.supported_page_sizes}"
            )

    @property
    def key_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.key_bits / 8)

    @property
    def value_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_bits / 8)

    def page_words_128(self, page_size: int) -> int:
        """Number of 128-byte words needed to store `page_size` slots per head."""
        self.validate_page_size(page_size)
        raw_words = math.ceil(page_size * self.slot_size_aligned / _WORD_SIZE)
        return ((raw_words + _SUBLANE_WORDS - 1) // _SUBLANE_WORDS) * _SUBLANE_WORDS

    def page_size_from_words_128(self, page_words: int) -> int:
        """Recovers `page_size` (slots per page) from `page_words_128(page_size)`."""
        for candidate in self.supported_page_sizes:
            if self.page_words_128(candidate) == page_words:
                return candidate
        raise ValueError(
            f"Cannot recover supported page_size {self.supported_page_sizes} "
            f"from page_words={page_words} (slot_size_aligned={self.slot_size_aligned})"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class TQTables:
    """Constant tables used by quantize/dequantize (host numpy arrays).

    Hashed by identity so it can be a static `jax.jit` argument.
    """

    hadamard: np.ndarray  # f32[head_dim, head_dim], orthonormal and symmetric.
    centroids: np.ndarray  # f32[2**bits], sorted.
    midpoints: np.ndarray  # f32[2**bits - 1], sorted.


def hadamard_matrix(head_dim: int) -> np.ndarray:
    """Orthonormal Sylvester Hadamard matrix, as in vLLM's `_build_hadamard`."""
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {head_dim}")
    hadamard = np.array([[1.0]], dtype=np.float64)
    while hadamard.shape[0] < head_dim:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    return (hadamard / math.sqrt(head_dim)).astype(np.float32)


def make_tables(layout: TQLayout, centroids: np.ndarray) -> TQTables:
    """Builds the constant tables from the Lloyd-Max centroids."""
    sorted_c = np.sort(np.asarray(centroids, dtype=np.float32))
    return TQTables(
        hadamard=(
            hadamard_matrix(layout.head_dim)
            if not layout.is_key_fp8
            else np.zeros((0, 0), np.float32)
        ),
        centroids=sorted_c,
        midpoints=((sorted_c[:-1] + sorted_c[1:]) / 2).astype(np.float32),
    )


def rotate(inputs: jax.Array, hadamard: jax.Array) -> jax.Array:
    """`inputs @ H` in `f32` (uses `bf16` MXU with `f32` accum for `bf16` inputs)."""
    if inputs.dtype == jnp.bfloat16:
        return jnp.einsum(
            "...d,de->...e",
            inputs,
            hadamard.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
    return jnp.einsum(
        "...d,de->...e",
        inputs.astype(jnp.float32),
        hadamard,
        precision=jax.lax.Precision.HIGHEST,
    )


def _plane_word_counts(page_size: int, layout: TQLayout) -> tuple[int, int, int]:
    """Number of 128-byte words per head for `(k_plane, v_plane, meta_plane)`."""
    layout.validate_page_size(page_size)
    k_bytes = layout.head_dim if layout.is_key_fp8 else layout.key_mse_bytes
    k_words = (page_size * k_bytes) // _WORD_SIZE
    v_words = (page_size * layout.value_data_bytes) // _WORD_SIZE
    raw_words = math.ceil(page_size * layout.slot_size_aligned / _WORD_SIZE)
    meta_words = raw_words - k_words - v_words
    if page_size == 64:
        # `page_size == 64` pages already allocate 8 metadata words due to
        # 8-row sublane alignment (`page_words_128(64)`). Storing each byte of
        # `(vs, vz[, kn])` in its own word (`4` or `6` words) lets
        # `_dequant_tq_page_vreg` reconstruct 2 pages (128 tokens) with zero
        # slice/concat overhead.
        meta_words = 4 if layout.is_key_fp8 else 6
    return k_words, v_words, meta_words


# ======================================================================================
# Pallas VREG in-kernel page dequantization
# ======================================================================================


def _bf16_u16_to_f32_vreg(lo_u32: jax.Array, hi_u32: jax.Array) -> jax.Array:
    """Reconstructs `f32` in VREGs from low/high `bfloat16` byte planes (`u32`)."""
    return jax.lax.bitcast_convert_type((lo_u32 | (hi_u32 << 8)) << 16, jnp.float32)


def _lookup_centroids_vreg(
    idx: jax.Array,
    bits: int,
    centroids: np.ndarray,
    *,
    scale: jax.Array | None = None,
) -> jax.Array:
    """VREG packed-pair select of `centroids[idx]` with IEEE-754 bitwise sign."""
    pos_f32_u32 = np.asarray(centroids, dtype=np.float32)[1 << (bits - 1) :].view(
        np.uint32
    )
    pos_u16 = [
        int(((int(u) + 0x7FFF + ((int(u) >> 16) & 1)) >> 16) & 0xFFFF)
        for u in pos_f32_u32
    ]
    pair_vals: list[Any] = [
        jnp.uint32(pos_u16[2 * pair_i] | (pos_u16[2 * pair_i + 1] << 16))
        for pair_i in range(len(pos_u16) // 2)
    ]
    for bit_idx in range(1, bits - 1):
        bit_set = (idx & jnp.uint32(1 << bit_idx)) != 0
        pair_vals = [
            jnp.where(bit_set, pair_vals[2 * pair_i + 1], pair_vals[2 * pair_i])
            for pair_i in range(len(pair_vals) // 2)
        ]
    u32_mag = jnp.where(
        (idx & jnp.uint32(1)) != 0,
        pair_vals[0] & jnp.uint32(0xFFFF0000),
        pair_vals[0] << 16,
    )
    sign_bit = (idx & jnp.uint32(1 << (bits - 1))) << (32 - bits)
    signed_val = jax.lax.bitcast_convert_type(u32_mag | sign_bit, jnp.float32)
    return signed_val * scale if scale is not None else signed_val


def _unpack_bits_vreg(
    planes: list[jax.Array],
    bits: int,
    *,
    mask_lo: bool = True,
    out_dtype: Any = jnp.uint32,
) -> jax.Array:
    """Unpacks `u8[P*bits//8, 128]` slices -> `out_dtype[len(planes)*P, 128]`."""
    if bits == 4:
        parts = []
        for plane in planes:
            plane_u32 = plane.astype(jnp.uint32)
            parts.append((plane_u32 & jnp.uint32(0xF)) if mask_lo else plane_u32)
            parts.append(plane_u32 >> jnp.uint32(4))
        return jnp.concatenate(parts, axis=0).astype(out_dtype)
    assert bits == 3, bits
    unpacked = []
    for plane in planes:
        packed = plane.astype(jnp.uint32)
        grp = packed.shape[0] // 3
        word = (
            packed[:grp]
            | (packed[grp : 2 * grp] << 8)
            | (packed[2 * grp : 3 * grp] << 16)
        )
        unpacked.extend((word >> (3 * slot_g)) & 0x7 for slot_g in range(8))
    res = jnp.concatenate(unpacked, axis=0)
    return res if out_dtype == jnp.uint32 else res.astype(out_dtype)


def _dequant_tq_page_vreg(
    pages_h: Any,  # u8[page_words, 128], list of them, or list of Pallas VMEM refs
    kv_dtype: jnp.dtype,
    *,
    layout: TQLayout,
    tables: TQTables,
    return_k_norm: bool = False,
) -> Any:
    """Dequantizes `(page, kv_head)` blocks from `u8[W, 128]` into VREGs."""
    pages_list = list(pages_h) if isinstance(pages_h, (list, tuple)) else [pages_h]
    if not isinstance(pages_list[0], jax.Array):
        pages_list = [ph[...] for ph in pages_list]
    num_pages = len(pages_list)
    page_words = pages_list[0].shape[0]
    p_sz = layout.page_size_from_words_128(page_words)
    max_pages_per_step = max(1, _WORD_SIZE // p_sz)
    folds = layout.head_dim // _WORD_SIZE
    k_words, v_words, meta_words = _plane_word_counts(p_sz, layout)
    k_wpf = k_words // folds
    v_wpf = v_words // folds
    m_start = k_words + v_words

    def _bcast_col(row: jax.Array) -> jax.Array:
        return jnp.transpose(jnp.broadcast_to(row, (_WORD_SIZE, _WORD_SIZE)))

    lane_lt_64 = (
        jax.lax.broadcasted_iota(jnp.int32, (1, _WORD_SIZE), 1) < 64
        if meta_words > 1 and num_pages >= 2
        else None
    )
    use_post_k_norm = (
        return_k_norm
        and not layout.is_key_fp8
        and meta_words > 1
        and (num_pages * p_sz) % _WORD_SIZE == 0
    )

    dk_parts: list[jax.Array] = []
    dv_parts: list[jax.Array] = []
    kn_row_parts: list[jax.Array] = []
    for start_i in range(0, num_pages, max_pages_per_step):
        sub_pages = pages_list[start_i : start_i + max_pages_per_step]
        sub_n = len(sub_pages)
        tot_slots = sub_n * p_sz
        sub_k = [fp[:k_words] for fp in sub_pages]
        sub_v = [fp[k_words:m_start] for fp in sub_pages]
        sub_m = [fp[m_start:page_words] for fp in sub_pages]

        if meta_words == 1:
            vs_lo_l, vs_hi_l, vz_lo_l, vz_hi_l, kn_lo_l, kn_hi_l = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for m_tile in sub_m:
                mt = _bcast_col(m_tile[0:1])
                vs_lo_l.append(mt[0:p_sz])
                vs_hi_l.append(mt[64 : 64 + p_sz])
                vz_lo_l.append(mt[p_sz : 2 * p_sz])
                vz_hi_l.append(mt[64 + p_sz : 64 + 2 * p_sz])
                if not layout.is_key_fp8:
                    kn_lo_l.append(mt[2 * p_sz : 3 * p_sz])
                    kn_hi_l.append(mt[64 + 2 * p_sz : 64 + 3 * p_sz])
            lo_u32 = jnp.concatenate(vs_lo_l + vz_lo_l + kn_lo_l, axis=0).astype(
                jnp.uint32
            )
            hi_u32 = jnp.concatenate(vs_hi_l + vz_hi_l + kn_hi_l, axis=0).astype(
                jnp.uint32
            )
            f32_meta = _bf16_u16_to_f32_vreg(lo_u32, hi_u32)
            v_scale = f32_meta[0:tot_slots]
            v_zero = f32_meta[tot_slots : 2 * tot_slots]
            k_norm = (
                None if layout.is_key_fp8 else f32_meta[2 * tot_slots : 3 * tot_slots]
            )
        else:
            half_m = meta_words // 2
            if sub_n == 2:
                assert lane_lt_64 is not None
                m_u32 = jnp.where(
                    lane_lt_64, sub_m[0][:meta_words], sub_m[1][:meta_words]
                ).astype(jnp.uint32)
            else:
                m_u32 = sub_m[0][:meta_words].astype(jnp.uint32)
            f32_rows = _bf16_u16_to_f32_vreg(m_u32[:half_m], m_u32[half_m:meta_words])
            v_scale = _bcast_col(f32_rows[0:1])[:tot_slots]
            v_zero = _bcast_col(f32_rows[1:2])[:tot_slots]
            if layout.is_key_fp8:
                k_norm = None
            elif use_post_k_norm:
                kn_row_parts.append(f32_rows[2:3])
                k_norm = None
            else:
                k_norm = _bcast_col(f32_rows[2:3])[:tot_slots]

        dk_folds = []
        for fold_i in range(folds):
            f_s, f_e = fold_i * k_wpf, (fold_i + 1) * k_wpf
            k_slices = [kp[f_s:f_e] for kp in sub_k]
            if layout.is_key_fp8:
                kp_all = jnp.concatenate(k_slices, axis=0)
                dk_folds.append(
                    jax.lax.bitcast_convert_type(kp_all, jnp.float8_e4m3fn).astype(
                        kv_dtype
                    )
                )
            else:
                k_idx_f = _unpack_bits_vreg(k_slices, layout.key_bits, mask_lo=False)
                kf_scaled = _lookup_centroids_vreg(
                    k_idx_f,
                    layout.key_bits,
                    tables.centroids,
                    scale=k_norm,
                )
                dk_folds.append(kf_scaled.astype(kv_dtype))
        dk_parts.append(
            dk_folds[0] if folds == 1 else jnp.concatenate(dk_folds, axis=1)
        )

        dv_folds = []
        for fold_i in range(folds):
            f_s, f_e = fold_i * v_wpf, (fold_i + 1) * v_wpf
            v_idx_f32 = _unpack_bits_vreg(
                [vp[f_s:f_e] for vp in sub_v],
                layout.value_bits,
                mask_lo=True,
                out_dtype=jnp.float32,
            )
            dv_folds.append((v_idx_f32 * v_scale + v_zero).astype(kv_dtype))
        dv_parts.append(
            dv_folds[0] if folds == 1 else jnp.concatenate(dv_folds, axis=1)
        )

    dk = dk_parts[0] if len(dk_parts) == 1 else jnp.concatenate(dk_parts, axis=0)
    dv = dv_parts[0] if len(dv_parts) == 1 else jnp.concatenate(dv_parts, axis=0)
    k_norm_row = (
        (
            kn_row_parts[0]
            if len(kn_row_parts) == 1
            else jnp.concatenate(kn_row_parts, axis=1)
        )
        if kn_row_parts
        else None
    )
    return (dk, dv, k_norm_row) if return_k_norm else (dk, dv)


# ======================================================================================
# Per-token quantization & Pallas VREG page update
# ======================================================================================


def _f32_to_bf16_bytes(val: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Splits `f32[..., 1]` into `(lo_u8[..., 1], hi_u8[..., 1])` as `bf16` bytes."""
    u16 = jax.lax.bitcast_convert_type(val.astype(jnp.bfloat16), jnp.uint16)
    return (u16 & jnp.uint16(0xFF)).astype(jnp.uint8), (u16 >> jnp.uint16(8)).astype(
        jnp.uint8
    )


def _quantize_kv_components(
    key: jax.Array,  # [T, H, D]
    value: jax.Array,  # [T, H, D]
    layout: TQLayout,
    tables: TQTables,
    *,
    rotated_key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quantizes K/V directly into lane-aligned `(k_unpacked, v_idx, meta_bytes)`."""
    val_f32 = value.astype(jnp.float32)
    levels = (1 << layout.value_bits) - 1
    v_min = jnp.min(val_f32, axis=-1, keepdims=True)
    v_max = jnp.max(val_f32, axis=-1, keepdims=True)
    scale = jnp.maximum((v_max - v_min) / levels, _VALUE_SCALE_MIN)
    v_idx = jnp.clip(jnp.floor((val_f32 - v_min) / scale + 0.5), 0, levels).astype(
        jnp.uint8
    )
    vs_lo, vs_hi = _f32_to_bf16_bytes(scale)
    vz_lo, vz_hi = _f32_to_bf16_bytes(v_min)

    if layout.is_key_fp8:
        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
        k_fp8 = jnp.clip(key.astype(jnp.float32), -fp8_max, fp8_max).astype(
            jnp.float8_e4m3fn
        )
        k_unpacked = jax.lax.bitcast_convert_type(k_fp8, jnp.uint8)
        meta_bytes = jnp.concatenate([vs_lo, vs_hi, vz_lo, vz_hi], axis=-1)
    else:
        key_f32 = key.astype(jnp.float32)
        norm = jnp.sqrt(jnp.sum(key_f32 * key_f32, axis=-1, keepdims=True))
        if rotated_key is not None:
            rot_k = rotated_key.astype(jnp.float32) / (norm + _KEY_NORM_EPS)
        else:
            rot_k = rotate(
                key_f32 / (norm + _KEY_NORM_EPS), jnp.asarray(tables.hadamard)
            )
        half_levels = 1 << (layout.key_bits - 1)
        pos_mids = jnp.asarray(tables.midpoints[half_levels:], dtype=jnp.float32)
        mag_idx = jnp.sum(jnp.abs(rot_k)[..., None] >= pos_mids, axis=-1).astype(
            jnp.uint8
        )
        if layout.norm_correction:
            pos_c_sq = jnp.asarray(
                tables.centroids[half_levels:] ** 2, dtype=jnp.float32
            )
            c_sq = jnp.sum(pos_c_sq[mag_idx], axis=-1, keepdims=True)
            norm = norm * jax.lax.rsqrt(c_sq + _CENTROID_NORM_EPS)
        k_unpacked = mag_idx | jnp.where(
            rot_k < 0.0, jnp.uint8(half_levels), jnp.uint8(0)
        )
        kn_lo, kn_hi = _f32_to_bf16_bytes(norm)
        meta_bytes = jnp.concatenate(
            [vs_lo, vs_hi, vz_lo, vz_hi, kn_lo, kn_hi], axis=-1
        )

    return k_unpacked, v_idx, meta_bytes


def _pack_new_tq_tokens_128(
    nk_u: jax.Array,  # u8[T, H, D]
    nv_i: jax.Array,  # u8[T, H, D]
    n_meta: jax.Array,  # u8[T, H, 4 or 6]
    page_size: int,
    layout: TQLayout,
) -> jax.Array:
    """Packs per-token quantized components into `u8[T, W_new, H, 128]`."""
    num_tokens, num_kv_heads, _ = nk_u.shape
    folds = layout.head_dim // _WORD_SIZE
    _, _, meta_words = _plane_word_counts(page_size, layout)

    def _to_fold_words(arr: jax.Array) -> jax.Array:
        return (
            arr[:, None, :, :]
            if folds == 1
            else arr.reshape(num_tokens, num_kv_heads, folds, _WORD_SIZE).transpose(
                0, 2, 1, 3
            )
        )

    # Keep `H` on sublane axis `-2` (`T(8, 128)`) throughout quantization and
    # packing. For non-FP8 presets (`key_bits <= 4`, `value_bits <= 4`), pack
    # `nk_u` (low nibble) and `nv_i` (high nibble) into one `u8` word per fold
    # so the total word count is `<= 4`.
    if layout.is_key_fp8:
        kv_words_list = [_to_fold_words(nk_u), _to_fold_words(nv_i)]
    else:
        kv_words_list = [
            _to_fold_words((nk_u & jnp.uint8(0x0F)) | (nv_i << jnp.uint8(4)))
        ]

    b_shape = (num_tokens, 1, num_kv_heads, _WORD_SIZE)
    m_cols = [
        jnp.broadcast_to(n_meta[:, None, :, col_i : col_i + 1], b_shape)
        for col_i in range(n_meta.shape[-1])
    ]
    lo_cols, hi_cols = m_cols[0::2], m_cols[1::2]
    if meta_words == 1:
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, 1, 1, _WORD_SIZE), 3)

        def _pack_half(cols: list[jax.Array], base: int) -> jax.Array:
            out = cols[-1]
            for idx in range(len(cols) - 2, -1, -1):
                out = jnp.where(lane < base + (idx + 1) * page_size, cols[idx], out)
            return out

        meta_words_list = [
            jnp.where(lane < 64, _pack_half(lo_cols, 0), _pack_half(hi_cols, 64))
        ]
    else:
        assert page_size == 64, page_size
        meta_words_list = lo_cols + hi_cols

    return jnp.concatenate(kv_words_list + meta_words_list, axis=1)


def _update_tq_page_vreg(
    old_page: jax.Array,  # u8[page_words, 128]
    new_tok: jax.Array
    | list[jax.Array],  # list of u8[1 or page_size, 128] or u8[1 or page_size, W, 128]
    valid_mask: jax.Array,  # bool[page_size, 1]
    *,
    slot_idx: jax.Array | None = None,
    layout: TQLayout,
) -> jax.Array:
    """Merges new tokens' quantized components into a single KV head's page in VREGs."""
    page_words = old_page.shape[0]
    p_sz = layout.page_size_from_words_128(page_words)
    folds = layout.head_dim // _WORD_SIZE
    k_words, v_words, meta_words = _plane_word_counts(p_sz, layout)
    k_wpf = k_words // folds
    v_wpf = v_words // folds
    m_start = k_words + v_words
    m_off = (2 * folds) if layout.is_key_fp8 else folds
    tok_words = (
        new_tok
        if isinstance(new_tok, list)
        else [new_tok[:, w_idx] for w_idx in range(new_tok.shape[1])]
    )
    is_single = tok_words[0].shape[0] == 1

    def _merge_bits_vreg(
        old_plane: jax.Array, new_idx: jax.Array, bits: int
    ) -> jax.Array:
        if bits == 4:
            half = p_sz // 2
            m_lo, m_hi = valid_mask[:half], valid_mask[half:]
            old_u32 = old_plane.astype(jnp.uint32)
            new_u32 = new_idx.astype(jnp.uint32)
            n_lo = (new_u32 if is_single else new_u32[:half]) & jnp.uint32(0xF)
            n_hi = (
                (new_u32 if is_single else new_u32[half:]) & jnp.uint32(0xF)
            ) << jnp.uint32(4)
            merged = jnp.where(m_lo, (old_u32 & jnp.uint32(0xF0)) | n_lo, old_u32)
            return jnp.where(m_hi, (merged & jnp.uint32(0x0F)) | n_hi, merged).astype(
                jnp.uint8
            )
        assert bits == 3, bits
        grp = p_sz // 8
        old_u32 = old_plane.astype(jnp.uint32)
        word = (
            old_u32[:grp]
            | (old_u32[grp : 2 * grp] << 8)
            | (old_u32[2 * grp : 3 * grp] << 16)
        )
        new_u32 = new_idx.astype(jnp.uint32) & jnp.uint32(0x7)
        for slot_g in range(8):
            m_k = valid_mask[slot_g * grp : (slot_g + 1) * grp]
            n_k = (
                new_u32 if is_single else new_u32[slot_g * grp : (slot_g + 1) * grp]
            ) << (3 * slot_g)
            clr_mask = jnp.uint32(~(0x7 << (3 * slot_g)) & 0xFFFFFF)
            word = jnp.where(m_k, (word & clr_mask) | n_k, word)
        return jnp.concatenate(
            [
                ((word >> (8 * byte_i)) & jnp.uint32(0xFF)).astype(jnp.uint8)
                for byte_i in range(3)
            ],
            axis=0,
        )

    def _reduce_or_axis0(val: jax.Array) -> jax.Array:
        while val.shape[0] > 1:
            half = val.shape[0] // 2
            val = val[:half] | val[half:]
        return val

    k_planes, v_planes = [], []
    for fold_i in range(folds):
        old_kp = old_page[fold_i * k_wpf : (fold_i + 1) * k_wpf]
        old_vp = old_page[k_words + fold_i * v_wpf : k_words + (fold_i + 1) * v_wpf]
        if layout.is_key_fp8:
            new_ku, new_vi = tok_words[fold_i], tok_words[folds + fold_i]
            k_planes.append(
                jnp.where(
                    valid_mask, new_ku.astype(jnp.uint32), old_kp.astype(jnp.uint32)
                ).astype(jnp.uint8)
            )
        else:
            kv_nib = tok_words[fold_i].astype(jnp.uint32)
            new_ku, new_vi = kv_nib, kv_nib >> jnp.uint32(4)
            k_planes.append(_merge_bits_vreg(old_kp, new_ku, layout.key_bits))
        v_planes.append(_merge_bits_vreg(old_vp, new_vi, layout.value_bits))

    old_meta = old_page[m_start:]
    n_m = (2 * p_sz) if layout.is_key_fp8 else (3 * p_sz)
    col64 = (
        jax.lax.broadcasted_iota(jnp.int32, (1 if is_single else p_sz, _WORD_SIZE), 1)
        & 63
    )
    col_slot = col64 if p_sz == 64 else (col64 % p_sz)
    in_bounds = True if p_sz == 64 else (col64 < n_m)

    if is_single:
        assert slot_idx is not None
        col_match = (col_slot == slot_idx) & in_bounds
        meta_planes = [
            jnp.where(
                col_match,
                tok_words[m_off + mw].astype(jnp.uint32),
                old_meta[mw : mw + 1].astype(jnp.uint32),
            ).astype(jnp.uint8)
            for mw in range(meta_words)
        ]
    else:
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (p_sz, _WORD_SIZE), 0)
        match = valid_mask & (row_iota == col_slot) & in_bounds
        msk = _reduce_or_axis0(jnp.where(match, jnp.uint32(0xFF), jnp.uint32(0)))
        meta_planes = [
            (
                (old_meta[mw : mw + 1].astype(jnp.uint32) & ~msk)
                | _reduce_or_axis0(
                    jnp.where(match, tok_words[m_off + mw].astype(jnp.uint32), 0)
                )
            ).astype(jnp.uint8)
            for mw in range(meta_words)
        ]

    if old_meta.shape[0] > meta_words:
        meta_planes.append(old_meta[meta_words:])
    return jnp.concatenate(k_planes + v_planes + meta_planes, axis=0)


# ======================================================================================
# Ragged paged attention entry point
# ======================================================================================


def tq_ragged_paged_attention(
    queries: jax.Array,  # [T, Hq, D]
    keys: jax.Array,  # [T, Hk, D]
    values: jax.Array,  # [T, Hk, D]
    kv_cache: jax.Array,  # u8[num_pages, Hk, page_words, 128]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    layout: TQLayout,
    tables: TQTables,
    rpa_fn: Callable,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Ragged paged attention over a compressed TurboQuant `uint8` KV cache.

    When `rpa_fn` is the Pallas RPA v3 kernel (`ragged_paged_attention`), the
    kernel fetches compressed `uint8` pages directly into VMEM (`btq_x2_ref`),
    updates touched pages in VREGs, dequantizes active pages in VREGs on the
    fly, and DMAs updated pages back to HBM. When `rpa_fn` is a reference
    callable such as `ref_ragged_paged_attention` (CPU/TPU unit tests), the same
    `_pack_new_tq_tokens_128`, `_update_tq_page_vreg`, and
    `_dequant_tq_page_vreg` functions are executed via `vmap`.

    Returns `(new_kv_cache, outputs)`.
    """
    from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
        merge_kv,
        ragged_paged_attention,
    )

    assert kv_cache.ndim == 4 and kv_cache.shape[3] == _WORD_SIZE, kv_cache.shape
    page_size = layout.page_size_from_words_128(kv_cache.shape[2])
    num_tokens = keys.shape[0]
    compute_dtype = queries.dtype

    if layout.is_key_fp8:
        q_in, k_in, k_rot = queries, keys, None
    else:
        hadamard = jnp.asarray(tables.hadamard)
        if not skip_kv_update and queries.dtype == keys.dtype:
            hq = queries.shape[1]
            qk_rot = rotate(jnp.concatenate([queries, keys], axis=1), hadamard)
            q_in = qk_rot[:, :hq].astype(compute_dtype)
            k_rot = qk_rot[:, hq:]
            k_in = k_rot.astype(compute_dtype)
        else:
            q_in = rotate(queries, hadamard).astype(compute_dtype)
            k_rot = None if skip_kv_update else rotate(keys, hadamard)
            k_in = (keys if k_rot is None else k_rot).astype(compute_dtype)

    dequant_fn = functools.partial(_dequant_tq_page_vreg, layout=layout, tables=tables)
    dequant_1 = functools.partial(dequant_fn, kv_dtype=compute_dtype)
    v_in = values.astype(compute_dtype)
    tq_new_kv = None
    update_fn = None
    if not skip_kv_update:
        nk_u, nv_i, n_meta = _quantize_kv_components(
            keys, values, layout, tables, rotated_key=k_rot
        )
        tq_new_kv = _pack_new_tq_tokens_128(nk_u, nv_i, n_meta, page_size, layout)
        update_fn = functools.partial(_update_tq_page_vreg, layout=layout)

    # Production passes the fused Pallas `ragged_paged_attention` kernel (which
    # accepts the `tq_*` VMEM hooks); CPU unit tests pass `ref_ragged_paged_attention`.
    if rpa_fn is ragged_paged_attention:
        outputs, kv_cache = rpa_fn(
            q_in,
            k_in,
            v_in,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            skip_kv_update=skip_kv_update,
            tq_page_size=page_size,
            tq_dequant_page_fn=dequant_fn,
            tq_new_kv=tq_new_kv,
            tq_update_page_fn=update_fn,
        )
        return kv_cache, outputs

    if not skip_kv_update:
        max_num_seqs = kv_lens.shape[0]
        pages_per_seq = page_indices.shape[0] // max_num_seqs
        num_seqs = distribution[2]
        clamped_cu = jnp.where(
            jnp.arange(max_num_seqs + 1) <= num_seqs, cu_q_lens, cu_q_lens[num_seqs]
        )
        tok_idx = jnp.arange(num_tokens, dtype=jnp.int32)
        seq_idx = jnp.minimum(
            jnp.searchsorted(clamped_cu[1:], tok_idx, side="right").astype(jnp.int32),
            max_num_seqs - 1,
        )
        q_start = clamped_cu[seq_idx]
        kv_pos = (
            kv_lens[seq_idx] - (clamped_cu[seq_idx + 1] - q_start) + (tok_idx - q_start)
        )
        local_page = seq_idx * pages_per_seq + jnp.clip(
            kv_pos // page_size, 0, pages_per_seq - 1
        )
        slot_in_pages = jnp.where(tok_idx < cu_q_lens[num_seqs], kv_pos % page_size, -1)
        slot_iota = jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)

        def _apply_tok(tok_i, cache):
            assert tq_new_kv is not None
            slot_in_p = slot_in_pages[tok_i]
            page_i = page_indices[local_page[tok_i]]
            valid_mask = (slot_in_p >= 0) & (slot_iota == slot_in_p)
            old_p = cache[page_i]
            new_p = jax.vmap(
                lambda op, nt: _update_tq_page_vreg(
                    op, nt, valid_mask, slot_idx=slot_in_p, layout=layout
                ),
                in_axes=(0, 2),
            )(old_p, tq_new_kv[tok_i][None])
            return cache.at[page_i].set(jnp.where(slot_in_p >= 0, new_p, old_p))

        kv_cache = jax.lax.fori_loop(0, num_tokens, _apply_tok, kv_cache)
        hist_k, hist_v = jax.vmap(jax.vmap(dequant_1))(kv_cache[page_indices])
        is_decode = (tok_idx < cu_q_lens[distribution[0]])[:, None, None]
        slot_clamped = jnp.maximum(slot_in_pages, 0)
        k_in = jnp.where(is_decode, hist_k[local_page, :, slot_clamped, :], k_in)
        v_in = jnp.where(is_decode, hist_v[local_page, :, slot_clamped, :], v_in)
    else:
        hist_k, hist_v = jax.vmap(jax.vmap(dequant_1))(kv_cache[page_indices])

    n_pages, n_heads, _, head_dim = hist_k.shape
    merged = merge_kv(
        hist_k.swapaxes(1, 2).reshape(n_pages * page_size, n_heads, head_dim),
        hist_v.swapaxes(1, 2).reshape(n_pages * page_size, n_heads, head_dim),
    )
    pool = merged.reshape(n_pages, page_size, *merged.shape[1:])
    pool_page_indices = jnp.arange(n_pages, dtype=page_indices.dtype)

    outputs, _ = rpa_fn(
        q_in,
        k_in,
        v_in,
        pool,
        kv_lens,
        pool_page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        skip_kv_update=skip_kv_update,
    )
    return kv_cache, outputs

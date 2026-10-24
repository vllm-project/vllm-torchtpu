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
  centroids (`||k||` stored as `bf16`). Because `q . k == (q H) . (k H)`,
  queries are pre-rotated once (`q @ H`) instead of inverse-rotating keys.
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
    def key_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.key_bits / 8)

    @property
    def value_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_bits / 8)

    def page_words_128(self, page_size: int) -> int:
        """Number of 128-byte words needed to store `page_size` slots per head."""
        raw_words = math.ceil(page_size * self.slot_size_aligned / _WORD_SIZE)
        return ((raw_words + _SUBLANE_WORDS - 1) // _SUBLANE_WORDS) * _SUBLANE_WORDS

    def page_size_from_words_128(self, page_words: int) -> int:
        """Recovers `page_size` (slots per page) from `page_words_128(page_size)`."""
        for candidate in (16, 32, 64, 128, 256, 512):
            if self.page_words_128(candidate) == page_words:
                return candidate
        return (
            (page_words * _WORD_SIZE) // self.slot_size_aligned // _SUBLANE_WORDS
        ) * _SUBLANE_WORDS


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
    """`inputs @ H` in `f32` (`Precision.HIGHEST` avoids `bf16` midpoint drift)."""
    return jnp.einsum(
        "...d,de->...e",
        inputs.astype(jnp.float32),
        hadamard,
        precision=jax.lax.Precision.HIGHEST,
    )


def _compute_local_slot_mapping(
    num_tokens: int,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Returns `(local_slot, local_page)` in the `0..N-1` gathered page table."""
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_seqs = distribution[2]
    total_q = cu_q_lens[num_seqs]
    clamped_cu = jnp.where(jnp.arange(max_num_seqs + 1) <= num_seqs, cu_q_lens, total_q)

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
    local_slot = local_page * page_size + kv_pos % page_size
    return jnp.where(tok_idx < total_q, local_slot, -1), local_page


def _plane_word_counts(page_size: int, layout: TQLayout) -> tuple[int, int, int]:
    """Number of 128-byte words per head for `(k_plane, v_plane, meta_plane)`."""
    k_bytes = layout.head_dim if layout.is_key_fp8 else layout.key_mse_bytes
    k_words = (page_size * k_bytes) // _WORD_SIZE
    v_words = (page_size * layout.value_data_bytes) // _WORD_SIZE
    raw_words = math.ceil(page_size * layout.slot_size_aligned / _WORD_SIZE)
    return k_words, v_words, raw_words - k_words - v_words


# ======================================================================================
# Pallas VREG in-kernel page dequantization
# ======================================================================================


def _bf16_u16_to_f32_vreg(lo_u32: jax.Array, hi_u32: jax.Array) -> jax.Array:
    """Reconstructs `f32` in VREGs from low/high `bfloat16` byte planes (`u32`)."""
    return jax.lax.bitcast_convert_type((lo_u32 | (hi_u32 << 8)) << 16, jnp.float32)


def _lookup_centroids_vreg(
    idx: jax.Array, bits: int, centroids: np.ndarray
) -> jax.Array:
    """VREG binary-tree select of `centroids[idx]` exploiting antisymmetry."""
    c_arr = np.asarray(centroids, dtype=np.float32)
    half = 1 << (bits - 1)
    top_bit = (idx & half) != 0
    fold_idx = jnp.where(top_bit, ~idx, idx)
    vals = [jnp.full(idx.shape, float(val), dtype=jnp.float32) for val in c_arr[:half]]
    for bit_idx in range(bits - 1):
        bit_set = (fold_idx & (1 << bit_idx)) != 0
        vals = [
            jnp.where(bit_set, vals[2 * pair_i + 1], vals[2 * pair_i])
            for pair_i in range(len(vals) // 2)
        ]
    return jnp.where(top_bit, -vals[0], vals[0])


def _unpack_bits_vreg(plane: jax.Array, bits: int) -> jax.Array:
    """Unpacks `u8[P*bits//8, 128]` -> `u32[P, 128]` in TPU VREGs."""
    packed = plane.astype(jnp.uint32)
    if bits == 4:
        return jnp.concatenate([packed & 0xF, (packed >> 4) & 0xF], axis=0)
    assert bits == 3, bits
    grp = packed.shape[0] // 3
    word = (
        packed[:grp] | (packed[grp : 2 * grp] << 8) | (packed[2 * grp : 3 * grp] << 16)
    )
    return jnp.concatenate(
        [(word >> (3 * slot_g)) & 0x7 for slot_g in range(8)], axis=0
    )


def _unpack_bits_multi_vreg(planes: list[jax.Array], bits: int) -> jax.Array:
    """Unpacks `u8[P*bits//8, 128]` planes -> `u32[len(planes)*P, 128]` in VREGs."""
    if len(planes) == 1:
        return _unpack_bits_vreg(planes[0], bits)
    if bits == 4:
        half = planes[0].shape[0]
        packed = jnp.concatenate(planes, axis=0).astype(jnp.uint32)
        lo_nib, hi_nib = packed & 0xF, (packed >> 4) & 0xF
        parts = []
        for page_i in range(len(planes)):
            parts.extend(
                [
                    lo_nib[page_i * half : (page_i + 1) * half],
                    hi_nib[page_i * half : (page_i + 1) * half],
                ]
            )
        return jnp.concatenate(parts, axis=0)
    return jnp.concatenate([_unpack_bits_vreg(plane, bits) for plane in planes], axis=0)


def _dequant_tq_page_vreg(
    pages_h: Any,  # u8[page_words, 128], list of them, or list of Pallas VMEM refs
    kv_dtype: jnp.dtype,
    *,
    layout: TQLayout,
    tables: TQTables,
) -> tuple[jax.Array, jax.Array]:
    """Dequantizes `(page, kv_head)` blocks from `u8[W, 128]` to `(k, v)` in VREGs."""
    pages_list = list(pages_h) if isinstance(pages_h, (list, tuple)) else [pages_h]
    num_pages = len(pages_list)
    page_words = pages_list[0].shape[0]
    p_sz = layout.page_size_from_words_128(page_words)
    tot_slots = num_pages * p_sz
    folds = layout.head_dim // _WORD_SIZE
    k_words, v_words, meta_words = _plane_word_counts(p_sz, layout)
    k_wpf = k_words // folds
    v_wpf = v_words // folds
    m_start = k_words + v_words

    # If plane boundaries are 8-row aligned (`#tpu.tiled<(8,128)(4,1)>`), slice
    # VMEM refs lazily per plane so K is not loaded into VREGs until V finishes.
    lazy_vmem = (
        not isinstance(pages_list[0], jax.Array)
        and (k_wpf % _SUBLANE_WORDS == 0)
        and (v_wpf % _SUBLANE_WORDS == 0)
        and (page_words % _SUBLANE_WORDS == 0)
    )
    if not lazy_vmem and not isinstance(pages_list[0], jax.Array):
        pages_list = [ph[...] for ph in pages_list]

    def _load_slice(ph: Any, start: int, end: int) -> jax.Array:
        return ph.at[start:end, :][...] if lazy_vmem else ph[start:end]

    def _bcast_col(row: jax.Array) -> jax.Array:
        return jnp.transpose(jnp.broadcast_to(row, (_WORD_SIZE, _WORD_SIZE)))

    m_tiles = [_load_slice(ph, m_start, page_words) for ph in pages_list]

    if meta_words == 1:
        vs_lo_l, vs_hi_l, vz_lo_l, vz_hi_l, kn_lo_l, kn_hi_l = [], [], [], [], [], []
        for m_tile in m_tiles:
            mt = _bcast_col(m_tile[0:1])
            vs_lo_l.append(mt[0:p_sz])
            vs_hi_l.append(mt[64 : 64 + p_sz])
            vz_lo_l.append(mt[p_sz : 2 * p_sz])
            vz_hi_l.append(mt[64 + p_sz : 64 + 2 * p_sz])
            if not layout.is_key_fp8:
                kn_lo_l.append(mt[2 * p_sz : 3 * p_sz])
                kn_hi_l.append(mt[64 + 2 * p_sz : 64 + 3 * p_sz])
        lo_u32 = jnp.concatenate(vs_lo_l + vz_lo_l + kn_lo_l, axis=0).astype(jnp.uint32)
        hi_u32 = jnp.concatenate(vs_hi_l + vz_hi_l + kn_hi_l, axis=0).astype(jnp.uint32)
        f32_meta = _bf16_u16_to_f32_vreg(lo_u32, hi_u32)
        v_scale = f32_meta[0:tot_slots]
        v_zero = f32_meta[tot_slots : 2 * tot_slots]
        k_norm = (
            v_scale if layout.is_key_fp8 else f32_meta[2 * tot_slots : 3 * tot_slots]
        )
    else:
        lo_v = jnp.concatenate([m_tile[0:1] for m_tile in m_tiles], axis=0).astype(
            jnp.uint32
        )
        hi_v = jnp.concatenate([m_tile[1:2] for m_tile in m_tiles], axis=0).astype(
            jnp.uint32
        )
        f32_v_all = _bf16_u16_to_f32_vreg(lo_v, hi_v)
        vs_l, vz_l = [], []
        for page_i in range(num_pages):
            mt_v = _bcast_col(f32_v_all[page_i : page_i + 1])
            vs_l.append(mt_v[0:p_sz])
            vz_l.append(mt_v[64 : 64 + p_sz])
        v_scale = jnp.concatenate(vs_l, axis=0)
        v_zero = jnp.concatenate(vz_l, axis=0)
        k_norm = v_scale

    # Step 1: Dequantize V before loading K so V's temporary registers (v_scale,
    # v_zero, and V bit-planes) can be freed, reducing peak VREG pressure.
    dv_folds = []
    for fold_i in range(folds):
        vp_list = [
            _load_slice(ph, k_words + fold_i * v_wpf, k_words + (fold_i + 1) * v_wpf)
            for ph in pages_list
        ]
        v_idx_f = _unpack_bits_multi_vreg(vp_list, layout.value_bits)
        dv_folds.append(
            (v_idx_f.astype(jnp.float32) * v_scale + v_zero).astype(kv_dtype)
        )
    dv = dv_folds[0] if folds == 1 else jnp.concatenate(dv_folds, axis=1)

    # Step 2: Extract k_norm (if meta_words > 1 and non-fp8) and dequantize K.
    if meta_words > 1 and not layout.is_key_fp8:
        mts2 = [_bcast_col(m_tile[2:3]) for m_tile in m_tiles]
        kn_lo = jnp.concatenate([mt[0:p_sz] for mt in mts2], axis=0).astype(jnp.uint32)
        kn_hi = jnp.concatenate([mt[64 : 64 + p_sz] for mt in mts2], axis=0).astype(
            jnp.uint32
        )
        k_norm = _bf16_u16_to_f32_vreg(kn_lo, kn_hi)

    k_folds = []
    for fold_i in range(folds):
        kp_list = [
            _load_slice(ph, fold_i * k_wpf, (fold_i + 1) * k_wpf) for ph in pages_list
        ]
        if layout.is_key_fp8:
            kp_all = jnp.concatenate(kp_list, axis=0)
            k_folds.append(
                jax.lax.bitcast_convert_type(kp_all, jnp.float8_e4m3fn).astype(kv_dtype)
            )
        else:
            k_idx_f = _unpack_bits_multi_vreg(kp_list, layout.key_bits)
            k_folds.append(
                _lookup_centroids_vreg(k_idx_f, layout.key_bits, tables.centroids)
            )

    if layout.is_key_fp8:
        dk = k_folds[0] if folds == 1 else jnp.concatenate(k_folds, axis=1)
    else:
        if layout.norm_correction:
            sq_acc = sum((kf * kf for kf in k_folds[1:]), k_folds[0] * k_folds[0])
            sq_sum = jnp.sum(sq_acc, axis=-1, keepdims=True)
            k_scale = k_norm * jax.lax.rsqrt(sq_sum + _CENTROID_NORM_EPS)
        else:
            k_scale = k_norm
        dk_folds = [(kf * k_scale).astype(kv_dtype) for kf in k_folds]
        dk = dk_folds[0] if folds == 1 else jnp.concatenate(dk_folds, axis=1)
    return dk, dv


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
        k_unpacked = jnp.sum(
            rot_k[..., None] >= jnp.asarray(tables.midpoints), axis=-1
        ).astype(jnp.uint8)
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

    lane = jax.lax.broadcasted_iota(jnp.int32, (1, 1, 1, _WORD_SIZE), 3)
    b_shape = (num_tokens, 1, num_kv_heads, _WORD_SIZE)
    m_cols = [
        jnp.broadcast_to(n_meta[:, None, :, col_i : col_i + 1], b_shape)
        for col_i in range(n_meta.shape[-1])
    ]
    vs_lo, vs_hi, vz_lo, vz_hi = m_cols[:4]

    if meta_words == 1:
        if layout.is_key_fp8:
            lo_128 = jnp.where(lane < page_size, vs_lo, vz_lo)
            hi_128 = jnp.where(lane < 64 + page_size, vs_hi, vz_hi)
        else:
            kn_lo, kn_hi = m_cols[4:6]
            lo_128 = jnp.where(
                lane < page_size, vs_lo, jnp.where(lane < 2 * page_size, vz_lo, kn_lo)
            )
            hi_128 = jnp.where(
                lane < 64 + page_size,
                vs_hi,
                jnp.where(lane < 64 + 2 * page_size, vz_hi, kn_hi),
            )
        meta_words_list = [jnp.where(lane < 64, lo_128, hi_128)]
    else:
        assert page_size == 64, page_size
        meta_words_list = [
            jnp.where(lane < 64, vs_lo, vz_lo),
            jnp.where(lane < 64, vs_hi, vz_hi),
        ]
        if not layout.is_key_fp8:
            kn_lo, kn_hi = m_cols[4:6]
            meta_words_list.append(jnp.where(lane < 64, kn_lo, kn_hi))

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
            n_lo = (new_idx if is_single else new_idx[:half]) & jnp.uint8(0xF)
            n_hi = (
                (new_idx if is_single else new_idx[half:]) & jnp.uint8(0xF)
            ) << jnp.uint8(4)
            merged = jnp.where(m_lo, (old_plane & jnp.uint8(0xF0)) | n_lo, old_plane)
            return jnp.where(m_hi, (merged & jnp.uint8(0x0F)) | n_hi, merged)
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

    def _reduce_or_axis0(val: jax.Array, msk: jax.Array) -> tuple[jax.Array, jax.Array]:
        while val.shape[0] > 1:
            half = val.shape[0] // 2
            val, msk = val[:half] | val[half:], msk[:half] | msk[half:]
        return val, msk

    k_planes, v_planes = [], []
    for fold_i in range(folds):
        old_kp = old_page[fold_i * k_wpf : (fold_i + 1) * k_wpf]
        old_vp = old_page[k_words + fold_i * v_wpf : k_words + (fold_i + 1) * v_wpf]
        if layout.is_key_fp8:
            new_ku, new_vi = tok_words[fold_i], tok_words[folds + fold_i]
            k_planes.append(jnp.where(valid_mask, new_ku, old_kp))
        else:
            kv_nib = tok_words[fold_i]
            new_ku, new_vi = kv_nib, kv_nib >> jnp.uint8(4)
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
            jnp.where(col_match, tok_words[m_off + mw], old_meta[mw : mw + 1])
            for mw in range(meta_words)
        ]
    else:
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (p_sz, _WORD_SIZE), 0)
        match = valid_mask & (row_iota == col_slot) & in_bounds
        meta_planes = []
        for mw in range(meta_words):
            val, msk = _reduce_or_axis0(
                jnp.where(match, tok_words[m_off + mw], jnp.uint8(0)),
                jnp.where(match, jnp.uint8(0xFF), jnp.uint8(0)),
            )
            meta_planes.append((old_meta[mw : mw + 1] & ~msk) | val)

    if old_meta.shape[0] > meta_words:
        meta_planes.append(old_meta[meta_words:])
    return jnp.concatenate(k_planes + v_planes + meta_planes, axis=0)


# ======================================================================================
# Ragged paged attention entry point
# ======================================================================================


def merge_kv_pages(keys: jax.Array, values: jax.Array) -> jax.Array:
    """Packs `[num_pages, page_size, num_kv_heads, head_dim]` K/V into RPA v3 layout."""
    from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import merge_kv

    n_pages, page_size, n_heads, head_dim = keys.shape
    merged = merge_kv(
        keys.reshape(n_pages * page_size, n_heads, head_dim),
        values.reshape(n_pages * page_size, n_heads, head_dim),
    )
    return merged.reshape(n_pages, page_size, *merged.shape[1:])


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
        q_in = rotate(queries, hadamard).astype(compute_dtype)
        k_rot = rotate(keys.astype(jnp.float32), hadamard)
        k_in = k_rot.astype(compute_dtype)

    dequant_fn = functools.partial(_dequant_tq_page_vreg, layout=layout, tables=tables)
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
    use_fused_pallas_kernel = rpa_fn is ragged_paged_attention
    if use_fused_pallas_kernel:
        outputs, kv_cache = rpa_fn(
            q_in,
            k_in,
            values.astype(compute_dtype),
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
        local_slot, local_page = _compute_local_slot_mapping(
            num_tokens, kv_lens, page_indices, cu_q_lens, distribution, page_size
        )
        slot_iota = jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)

        def _apply_tok(tok_i, cache):
            assert tq_new_kv is not None
            slot = local_slot[tok_i]
            page_i = page_indices[local_page[tok_i]]
            slot_in_p = slot % page_size
            valid_mask = (slot >= 0) & (slot_iota == slot_in_p)
            old_p = cache[page_i]
            new_p = jax.vmap(
                lambda op, nt: _update_tq_page_vreg(
                    op, nt, valid_mask, slot_idx=slot_in_p, layout=layout
                ),
                in_axes=(0, 2),
            )(old_p, tq_new_kv[tok_i][None])
            return cache.at[page_i].set(jnp.where(slot >= 0, new_p, old_p))

        kv_cache = jax.lax.fori_loop(0, num_tokens, _apply_tok, kv_cache)

    dequant_1 = functools.partial(dequant_fn, kv_dtype=compute_dtype)
    hist_k, hist_v = jax.vmap(jax.vmap(dequant_1))(kv_cache[page_indices])
    pool = merge_kv_pages(hist_k.swapaxes(1, 2), hist_v.swapaxes(1, 2))
    pool_page_indices = jnp.arange(page_indices.shape[0], dtype=page_indices.dtype)

    outputs, _ = rpa_fn(
        q_in,
        k_in,
        values.astype(compute_dtype),
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

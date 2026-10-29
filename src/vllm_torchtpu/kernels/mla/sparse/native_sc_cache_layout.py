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
"""Host helpers for the native SparseCore GLM-5.2 cache layout.

These conversions define the standalone native-SC kernel's byte contract. They
are intended for tests and offline cache construction; converting a serving
cache immediately before attention would defeat the layout's purpose.
"""

from __future__ import annotations

import numpy as np

NOPE_SUBROWS = 4
LANE_BYTES = 128
WORD_BYTES = np.dtype(np.uint32).itemsize
ROPE_WORDS = LANE_BYTES // WORD_BYTES


def _require_u8(name: str, value: np.ndarray, suffix: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.shape[-len(suffix) :] != suffix:
        raise ValueError(
            f"{name} must be uint8 with trailing shape {suffix}, got "
            f"{value.dtype} {value.shape}"
        )
    return value


def pack_nope(nope_u8: np.ndarray) -> np.ndarray:
    """Pack ``[..., token, 4, 128] u8`` as ``[..., token, 128] u32``.

    Native word ``w`` contains the four bytes ``nope_u8[..., :, w]``. The
    transpose is needed for a conventional row-major host array; the deployed
    TPU ``u8[4,128]`` tile has the same raw HBM byte image already.
    """
    nope_u8 = _require_u8("nope_u8", nope_u8, (NOPE_SUBROWS, LANE_BYTES))
    lane_major = np.ascontiguousarray(np.swapaxes(nope_u8, -2, -1))
    return lane_major.view(np.uint32).reshape(*nope_u8.shape[:-2], LANE_BYTES)


def unpack_nope(nope_i32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_nope`."""
    nope_i32 = np.asarray(nope_i32)
    if nope_i32.dtype != np.uint32 or nope_i32.shape[-1] != LANE_BYTES:
        raise ValueError(
            f"nope_i32 must be uint32[..., 128], got {nope_i32.dtype} {nope_i32.shape}"
        )
    lane_major = (
        np.ascontiguousarray(nope_i32)
        .view(np.uint8)
        .reshape(*nope_i32.shape[:-1], LANE_BYTES, NOPE_SUBROWS)
    )
    return np.ascontiguousarray(np.swapaxes(lane_major, -2, -1))


def pack_rope_banded(rope_u8: np.ndarray) -> np.ndarray:
    """Pack native-SC ROPE words into four-token, 128-word HBM rows.

    ``word[token, w].byte[band] = rope_u8[token, 32 * band + w]``. Four
    tokens occupy four consecutive 32-word quarters of the returned row, so
    an SC gather reads exactly 128 bytes per selected token.
    """
    rope_u8 = _require_u8("rope_u8", rope_u8, (NOPE_SUBROWS, LANE_BYTES))
    groups = rope_u8.shape[-3]
    banded = np.ascontiguousarray(
        rope_u8.reshape(*rope_u8.shape[:-1], NOPE_SUBROWS, ROPE_WORDS).swapaxes(-2, -1)
    )
    return banded.view(np.uint32).reshape(*rope_u8.shape[:-3], groups, LANE_BYTES)


def unpack_rope_banded(rope_i32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rope_banded`."""
    rope_i32 = np.asarray(rope_i32)
    if rope_i32.dtype != np.uint32 or rope_i32.shape[-1] != LANE_BYTES:
        raise ValueError(
            "rope_i32 must be uint32[..., groups, 128], got "
            f"{rope_i32.dtype} {rope_i32.shape}"
        )
    groups = rope_i32.shape[-2]
    banded = (
        np.ascontiguousarray(rope_i32)
        .view(np.uint8)
        .reshape(*rope_i32.shape[:-2], groups, NOPE_SUBROWS, ROPE_WORDS, NOPE_SUBROWS)
    )
    return np.ascontiguousarray(
        banded.swapaxes(-2, -1).reshape(
            *rope_i32.shape[:-2], groups, NOPE_SUBROWS, LANE_BYTES
        )
    )


def pack_rope_packed(rope_u8: np.ndarray) -> np.ndarray:
    """Pack ``[..., token, D] u8`` as ``[..., token / k, 128] u32`` (``k = 512 / D``).

    ``word[token, w].byte[band] = rope_u8[token, (D / 4) * band + w]``: token
    ``t`` owns ``D / 4`` consecutive words of the flat cache (e.g. 16 words for
    64-byte RoPE, eight tokens per 128-word row).
    """
    rope_u8 = np.asarray(rope_u8)
    if rope_u8.dtype != np.uint8 or rope_u8.ndim < 2:
        raise ValueError(
            "rope_u8 must be uint8[..., tokens, token_bytes], got "
            f"{rope_u8.dtype} {rope_u8.shape}"
        )
    token_bytes = rope_u8.shape[-1]
    row_bytes = WORD_BYTES * LANE_BYTES
    if token_bytes <= 0 or token_bytes % WORD_BYTES or row_bytes % token_bytes:
        raise ValueError(
            f"token_bytes ({token_bytes}) must be a positive multiple of "
            f"{WORD_BYTES} dividing {row_bytes}"
        )
    rope_words = token_bytes // WORD_BYTES
    tokens = rope_u8.shape[-2]
    tokens_per_row = LANE_BYTES // rope_words
    if tokens % tokens_per_row:
        raise ValueError(f"{tokens} tokens do not fill {tokens_per_row}-token rows")
    banded = np.ascontiguousarray(
        rope_u8.reshape(*rope_u8.shape[:-1], WORD_BYTES, rope_words).swapaxes(-2, -1)
    )
    return banded.view(np.uint32).reshape(
        *rope_u8.shape[:-2], tokens // tokens_per_row, LANE_BYTES
    )


def pack_native_caches(
    nope_u8: np.ndarray, rope_u8: np.ndarray, page_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Pack complete paged caches into the standalone native-SC ABI."""
    nope_u8 = np.asarray(nope_u8)
    rope_u8 = np.asarray(rope_u8)
    total_tokens = (
        nope_u8.shape[0] if nope_u8.ndim == 3 else nope_u8.shape[0] * nope_u8.shape[1]
    )
    total_pages = total_tokens // page_size
    if nope_u8.ndim == 3:
        nope_u8 = nope_u8.reshape(total_pages, page_size, NOPE_SUBROWS, LANE_BYTES)
    if rope_u8.ndim in (2, 3):
        rope_u8 = rope_u8.reshape(
            total_pages, page_size // NOPE_SUBROWS, NOPE_SUBROWS, LANE_BYTES
        )
    return pack_nope(nope_u8), pack_rope_banded(rope_u8)


def pack_selected_rope_for_tc(rope_u8: np.ndarray) -> np.ndarray:
    """Host oracle for the gather's four-token register delta-swap."""
    rope_u8 = np.asarray(rope_u8, dtype=np.uint8).reshape(-1, NOPE_SUBROWS, LANE_BYTES)
    swapped = np.ascontiguousarray(rope_u8.swapaxes(1, 2))
    return swapped.view(np.uint32).reshape(-1, LANE_BYTES)


def flatten_nope_rows(nope_i32: np.ndarray) -> np.ndarray:
    nope_i32 = np.asarray(nope_i32)
    if nope_i32.dtype != np.uint32 or nope_i32.shape[-1] != LANE_BYTES:
        raise ValueError("expected uint32 NOPE cache with 128-word rows")
    return nope_i32.reshape(-1, LANE_BYTES)


def flatten_rope_rows(rope_i32: np.ndarray, rope_words: int = ROPE_WORDS) -> np.ndarray:
    rope_i32 = np.asarray(rope_i32)
    if rope_i32.dtype != np.uint32 or rope_i32.shape[-1] != LANE_BYTES:
        raise ValueError("expected uint32 ROPE cache with 128-word group rows")
    if rope_words <= 0 or LANE_BYTES % rope_words != 0:
        raise ValueError(f"rope_words ({rope_words}) must divide {LANE_BYTES}")
    return rope_i32.reshape(-1, rope_words)


def source_bytes_per_token(
    nope_i32: np.ndarray, rope_i32: np.ndarray
) -> tuple[int, int]:
    """Return the logical allocation bytes per token for both caches."""
    nope_i32 = np.asarray(nope_i32)
    rope_i32 = np.asarray(rope_i32)
    if nope_i32.ndim == rope_i32.ndim and nope_i32.shape[:-2] != rope_i32.shape[:-2]:
        raise ValueError(
            f"cache page dimensions differ: {nope_i32.shape} vs {rope_i32.shape}"
        )
    nope_tokens = int(np.prod(nope_i32.shape[:-1], dtype=np.int64))
    rope_rows = int(np.prod(rope_i32.shape[:-1], dtype=np.int64))
    if (
        nope_tokens <= 0
        or rope_rows <= 0
        or nope_tokens % rope_rows != 0
        or LANE_BYTES % (nope_tokens // rope_rows) != 0
    ):
        raise ValueError(
            f"cache token counts incompatible: {nope_tokens} NOPE tokens vs "
            f"{rope_rows} ROPE rows"
        )
    return (nope_i32.nbytes // nope_tokens, rope_i32.nbytes // nope_tokens)

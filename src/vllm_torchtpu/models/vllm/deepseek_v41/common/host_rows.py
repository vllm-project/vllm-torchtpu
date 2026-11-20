# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host gather of Engram rows, run by the tpu_sync host signal server thread.

``gather_rows()`` is a numba C callback with tpu_sync's host signal handler
ABI (``tpu_sync/core/host_signal.h``). For each row id the device posted it
copies the row's FP8 bytes and then its scale bytes into pinned host staging,
packed row after row. It runs on the native server thread without the GIL.
"""

from functools import cache

import numba
import numpy as np
import torch
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

# Table description the callback's `user` word points to: weight address,
# scales address, row count, weight words per row, scale words per row.
_META_WORDS = 5
# Negative results the server reports as a failed request.
TOO_MANY_ROWS = -1
ROW_OUT_OF_RANGE = -2

_SIGNATURE = types.int64(
    types.intp, types.voidptr, types.int64, types.voidptr, types.int64
)


@intrinsic
def _void_pointer(typingctx, address):
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return types.voidptr(address), codegen


def _gather(user, words, count, staging, staging_bytes):
    meta = numba.carray(_void_pointer(user), (_META_WORDS,), dtype=np.int64)
    rows = meta[2]
    weight_words = meta[3]
    scale_words = meta[4]
    row_words = weight_words + scale_words
    if count * row_words * 8 > staging_bytes:
        return TOO_MANY_ROWS
    ids = numba.carray(words, (count,), dtype=np.int32)
    weight = numba.carray(
        _void_pointer(meta[0]), (rows * weight_words,), dtype=np.uint64
    )
    scales = numba.carray(
        _void_pointer(meta[1]), (rows * scale_words,), dtype=np.uint64
    )
    out = numba.carray(staging, (count * row_words,), dtype=np.uint64)
    for i in range(count):
        row = np.int64(ids[i])
        if row < 0 or row >= rows:
            return ROW_OUT_OF_RANGE
        base = i * row_words
        source = row * weight_words
        for j in range(weight_words):
            out[base + j] = weight[source + j]
        source = row * scale_words
        base += weight_words
        for j in range(scale_words):
            out[base + j] = scales[source + j]
    return count * row_words * 8


@cache
def gather_rows():
    """The compiled callback; its ``address`` is the handler to register."""
    return numba.cfunc(_SIGNATURE, nopython=True)(_gather)


def table_meta(weight: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    """The callback's description of one head shard's tables.

    ``weight`` and ``scales`` are the shard's ``[rows, bytes]`` uint8 host
    tables. The caller keeps them and the returned array alive while the
    server may call the callback.
    """
    for name, table in (("weight", weight), ("scales", scales)):
        if not (
            table.device.type == "cpu"
            and table.dim() == 2
            and table.is_contiguous()
            and table.dtype == torch.uint8
            and table.shape[0] > 0
        ):
            raise ValueError(
                f"Engram {name} table must be a non-empty contiguous uint8 "
                f"[rows, bytes] CPU tensor; got {table.dtype} {tuple(table.shape)}"
            )
    if weight.shape[0] != scales.shape[0]:
        raise ValueError("Engram weight and scale row counts differ")
    for table in (weight, scales):
        if table.shape[1] % 8 or table.data_ptr() % 8:
            raise NotImplementedError(
                "Engram host gathers copy whole 8-byte words per row"
            )
    return np.array(
        [
            weight.data_ptr(),
            scales.data_ptr(),
            weight.shape[0],
            weight.shape[1] // 8,
            scales.shape[1] // 8,
        ],
        dtype=np.int64,
    )

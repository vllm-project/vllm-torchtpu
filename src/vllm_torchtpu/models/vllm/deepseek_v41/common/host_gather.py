# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather paired host embedding rows without separate tensor dispatches."""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cache

import numpy as np
import torch
from numba import njit

# Random rows of a host table are DRAM-latency bound; every copy thread keeps
# its own misses in flight. Two threads took ~0.77 ms per 288-token decode
# layer, so the first layer's gather outlasted the encoder head it overlaps.
_GATHER_THREADS = 8


@cache
def _gather_pool() -> ThreadPoolExecutor:
    # The caller copies the first part; persistent workers copy the others.
    return ThreadPoolExecutor(
        max_workers=_GATHER_THREADS - 1, thread_name_prefix="engram-gather"
    )


os.register_at_fork(after_in_child=_gather_pool.cache_clear)


@njit(nogil=True, cache=True)
def _gather_weight_scale(
    weight: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    raw_out: np.ndarray,
    scale_out: np.ndarray,
) -> None:
    for token in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            # Keep address arithmetic wide even when stored row IDs are int32.
            row = np.int64(indices[token, head])
            if row < 0 or row >= weight.shape[0]:
                raise IndexError("Engram row index is outside the host table")
            i = token * indices.shape[1] + head
            for j in range(weight.shape[1]):
                raw_out[i, j] = weight[row, j]
            for j in range(scales.shape[1]):
                scale_out[i, j] = scales[row, j]


def gather_weight_scale(
    weight: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    raw_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    """Gather paired rows with parallel copy threads and compiled row copies."""
    tensors = (weight, scales, indices, raw_out, scale_out)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("Paired host gather requires CPU tensors")
    if indices.dtype not in (torch.int32, torch.int64) or any(
        tensor.dtype != torch.uint8 for tensor in (weight, scales, raw_out, scale_out)
    ):
        raise ValueError(
            "Paired host gather requires byte tables and int32/int64 indices"
        )
    if (
        weight.ndim != 2
        or scales.ndim != 2
        or indices.ndim not in (1, 2)
        or weight.shape[0] != scales.shape[0]
        or raw_out.shape != (indices.numel(), weight.shape[1])
        or scale_out.shape != (indices.numel(), scales.shape[1])
    ):
        raise ValueError("Paired host gather table and output shapes do not match")
    index_array = _host_array(
        indices, np.dtype(np.int32 if indices.dtype == torch.int32 else np.int64)
    )
    if indices.ndim == 1:
        index_array = index_array[:, None]
    with torch.profiler.record_function("engram::gather_weight_scale"):
        weight_array = _host_array(weight, np.dtype(np.uint8))
        scale_array = _host_array(scales, np.dtype(np.uint8))
        raw_array = _host_array(raw_out, np.dtype(np.uint8))
        scale_out_array = _host_array(scale_out, np.dtype(np.uint8))
        tokens, heads = index_array.shape
        parts = min(_GATHER_THREADS, tokens)
        if parts < 2:
            _gather_weight_scale(
                weight_array, scale_array, index_array, raw_array, scale_out_array
            )
            return
        bounds = [tokens * part // parts for part in range(parts + 1)]
        pool = _gather_pool()
        futures = [
            pool.submit(
                _gather_weight_scale,
                weight_array,
                scale_array,
                index_array[start:end],
                raw_array[start * heads : end * heads],
                scale_out_array[start * heads : end * heads],
            )
            for start, end in zip(bounds[1:-1], bounds[2:])
        ]
        try:
            _gather_weight_scale(
                weight_array,
                scale_array,
                index_array[: bounds[1]],
                raw_array[: bounds[1] * heads],
                scale_out_array[: bounds[1] * heads],
            )
        finally:
            # Even on failure, no writer may outlive the borrowed tensor storage.
            wait(futures)
        for future in futures:
            future.result()


def _host_array(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    # Tensor.numpy() emits detach/to events. Arguments keep this borrowed
    # storage alive until the synchronous, GIL-releasing gather returns.
    strides = tuple(stride * dtype.itemsize for stride in tensor.stride())
    span = (
        sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
        + dtype.itemsize
        if tensor.numel()
        else 0
    )
    storage = (ctypes.c_uint8 * span).from_address(tensor.data_ptr())
    return np.ndarray(tensor.shape, dtype=dtype, buffer=storage, strides=strides)

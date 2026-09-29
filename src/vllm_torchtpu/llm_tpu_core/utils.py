# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware utilities and math helpers for llm_tpu_core.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from typing import Any

from jax._src import dtypes

GBYTES = 1024 * 1024 * 1024
TPU_HEAD_SIZE_ALIGNMENT = 128
TPU_SECOND_LAST_MINOR = 8

_megacore = False


def get_dtype_packing(dtype: Any) -> int:
    """Returns number of packed elements per 32-bit word for dtype."""
    bits = dtypes.itemsize_bits(dtype)
    return 32 // bits


def get_padded_head_dim(head_dim: int) -> int:
    """Pads head_dim up to the nearest multiple of 128 for kernel performance."""
    if head_dim == 64:
        return 64
    return (head_dim + 127) // 128 * 128


def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    """Pads num_heads up to sharding_size alignment."""
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads


def align_to(unpadded_dim: int, pad_multiple: int) -> int:
    """Aligns an unpadded dimension to the next multiple of pad_multiple."""
    return (unpadded_dim + pad_multiple - 1) // pad_multiple * pad_multiple


def enable_megacore() -> None:
    """Enables megacore configuration globally."""
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    """Returns True if megacore configuration is enabled."""
    return _megacore


def largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of `x` that is <= `cap`."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    """Calculates effective number of KV heads under tensor parallelism."""
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def get_device_name(num_devices: int | None = None) -> str:
    """Returns canonical TPU hardware name string (e.g. 'TPU v5e')."""
    import jax

    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("Expected TPU devices")
    suffix = ""
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
        suffix = "e"
    elif kind.endswith("e"):
        kind = kind[:-1]
        suffix = "e"
    elif kind.endswith("p"):
        kind = kind[:-1]
        suffix = "p"
    elif kind.endswith("i"):
        kind = kind[:-1]
        suffix = "i"

    if kind.startswith("TPU7"):
        kind = "TPU v7"
    elif kind.startswith("TPU8"):
        kind = "TPU v8"
    assert kind[:-1] == "TPU v", kind
    kind += suffix
    if num_devices is not None:
        kind += f"-{num_devices}"
    return kind

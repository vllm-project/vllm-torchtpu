# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/cache_regions.py
#   at fc3c6941341286a89562b1b19c79e67e9488396e.
#   Owned by models/qwen38_flash from this copy on.
"""Byte-region geometry shared by Qwen state readers and writers."""

import math

import jax.numpy as jnp


def region_rows(pool, elements, dtype):
    token_bytes = math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    ntok = math.ceil(elements * jnp.dtype(dtype).itemsize / token_bytes)
    rows = ntok * token_bytes // (pool.shape[-1] * jnp.dtype(dtype).itemsize)
    return ntok, rows

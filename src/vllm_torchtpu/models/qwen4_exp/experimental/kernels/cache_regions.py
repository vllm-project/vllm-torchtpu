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

# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/cache_regions.py
#   at fc3c6941341286a89562b1b19c79e67e9488396e.
#   Owned by models/qwen4_exp from this copy on.
"""Byte-region geometry shared by Qwen state readers and writers."""

import math

import jax.numpy as jnp


def region_rows(pool, elements, dtype):
    token_bytes = math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    ntok = math.ceil(elements * jnp.dtype(dtype).itemsize / token_bytes)
    rows = ntok * token_bytes // (pool.shape[-1] * jnp.dtype(dtype).itemsize)
    return ntok, rows

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
"""Byte-region geometry shared by Qwen state readers and writers."""

import math

import jax.numpy as jnp


def region_rows(pool, elements, dtype):
    token_bytes = math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    ntok = (elements * jnp.dtype(dtype).itemsize + token_bytes - 1) // token_bytes
    rows = ntok * token_bytes // (pool.shape[-1] * jnp.dtype(dtype).itemsize)
    return ntok, rows

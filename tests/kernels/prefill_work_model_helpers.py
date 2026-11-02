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
"""Shape-only fixtures for sparse-prefill work accounting."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import prefill_cost_model


def work_model(
    dtype=jnp.bfloat16,
    *,
    heads=24,
    dim=256,
    budget=512,
    page_size=896,
    columns=1,
    kv_heads=2,
    ratio=4,
):
    shape = jax.ShapeDtypeStruct
    packing = 4 // jnp.dtype(dtype).itemsize
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    with patch.object(pltpu, "get_tpu_info", return_value=info):
        return prefill_cost_model(
            shape((256, heads, dim), jnp.bfloat16),
            shape((256, budget), jnp.int32),
            shape((256, columns), jnp.int32),
            shape(
                (4, page_size, (2 * kv_heads + packing - 1) // packing, packing, dim),
                dtype,
            ),
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=256,
        )

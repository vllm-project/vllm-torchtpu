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
"""FP32 matmul at ``Precision.HIGHEST`` for one call site.

TPU FP32 matmuls default to fewer BF16 passes. This op requests full FP32
precision for a single matmul (for example a router whose top-k selection is
sensitive to logit rounding) without changing the process-wide
``torch.set_float32_matmul_precision`` setting.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

# The op, built once outside torch.compile tracing (Dynamo traces through a
# functools.cache wrapper and cannot build a jax_op inside a graph).
_ops: dict[str, Any] = {}


def prebuild_fp32_matmul_op() -> None:
    """Build and cache the op outside compile-time tracing."""
    if "fp32_matmul" in _ops:
        return

    def matmul(x: jax.Array, weight: jax.Array) -> jax.Array:
        return jnp.matmul(
            x.astype(jnp.float32),
            weight.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    op = pallas.jax_op("pallas::fp32_matmul_highest", matmul)
    op.register_fake(
        lambda x, weight: torch.empty(
            *x.shape[:-1], weight.shape[-1], device=x.device, dtype=torch.float32
        )
    )
    _ops["fp32_matmul"] = op


def fp32_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x [..., K] @ weight [K, N]`` in FP32 with full FP32 precision.

    Call ``prebuild_fp32_matmul_op`` (for example in the module's ``__init__``)
    before any compiled forward uses this.
    """
    if x.device.type != "tpu":
        return torch.matmul(x.float(), weight.float())
    return _ops["fp32_matmul"](x, weight)

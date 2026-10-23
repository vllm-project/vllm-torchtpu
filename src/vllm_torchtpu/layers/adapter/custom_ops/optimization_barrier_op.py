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
"""Torch custom op wrapping ``jax.lax.optimization_barrier`` (XLA fusion fence).

It lives under layers/adapter/custom_ops so the compile-cache key hashes it,
and it is a generic XLA fence, not Omni-specific.
"""

from functools import cache as run_once
from typing import Any

import jax
import torch
from torch_tpu._internal import pallas


@run_once
def optimization_barrier_op() -> Any:
    """Build the ``pallas::optimization_barrier`` custom op on first TPU use."""

    def _optimization_barrier(x: jax.Array) -> jax.Array:
        return jax.lax.optimization_barrier(x)

    op = pallas.jax_op("pallas::optimization_barrier", _optimization_barrier)
    op.register_fake(lambda x: torch.empty_like(x))
    return op


def optimization_barrier(x: torch.Tensor) -> torch.Tensor:
    """Block XLA fusion passes from fusing across ``x``."""
    return optimization_barrier_op()(x)

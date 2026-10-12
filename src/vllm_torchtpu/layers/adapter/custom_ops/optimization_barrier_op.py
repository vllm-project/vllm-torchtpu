# SPDX-License-Identifier: Apache-2.0
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

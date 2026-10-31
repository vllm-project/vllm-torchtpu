# SPDX-License-Identifier: Apache-2.0
"""Expose both weight layouts before batch specialization of the FX graph."""

import jax
import torch
from torch_tpu._internal import pallas


def stable_weight_inputs_jax(x: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    del a, b
    return x.copy()


stable_weight_inputs = pallas.jax_op(
    "pallas::dsv4_stable_weight_inputs", stable_weight_inputs_jax
)


@stable_weight_inputs.register_fake
def _fake(x, a, b):
    return torch.empty_like(x)

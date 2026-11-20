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
"""Torch entry point for the fused SP shard add + norm + e4m3 quantize."""

import functools
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.models.minimax_m3.kernels import sp_norm


def _fake(shard: torch.Tensor, update: torch.Tensor, weight: torch.Tensor):
    return (
        torch.empty_like(shard),
        shard.new_empty(shard.shape, dtype=torch.float8_e4m3fn),
        shard.new_empty((shard.shape[0], 1), dtype=torch.float32),
    )


@functools.cache
def get_add_norm_quant_op(
    eps: float,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`sp_norm.add_norm_quant` as a torch op, built once per eps from module
    ``__init__`` (as `ops.msa.get_msa_op`)."""

    def _jax(
        shard: jax.Array, update: jax.Array, weight: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return sp_norm.add_norm_quant(shard, update, weight, eps=eps)

    name = f"pallas::minimax_m3_sp_add_norm_quant_e{eps!r}"
    op = pallas.jax_op(name.replace(".", "p").replace("-", "m"), _jax)
    op.register_fake(_fake)
    return op

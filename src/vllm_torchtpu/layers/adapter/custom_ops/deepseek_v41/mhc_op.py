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
"""DeepSeek-V4.1 Pallas custom-op bridge for mHC post-mix recombination."""

from collections.abc import Callable
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.mhc import mhc_post
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)

_MHC_POST_OP_NAME = "pallas::dsv41_mhc_post"
_mhc_post_op_cache: dict[str, Any] = {}


def _mhc_post_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    return mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _fake_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return residual.new_empty(residual.shape)


def get_mhc_post_op() -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]:
    """Return cached ``pallas::dsv41_mhc_post`` op for mHC residual recombination.

    Recombines ``x: bf16[..., D]`` into ``residual: bf16[..., M, D]`` using
    ``post_layer_mix: f32[..., M, 1]`` and ``comb_res_mix: f32[..., M, M]``.
    Call outside ``torch.compile`` while ``VllmModelWrapperContext`` is active.
    """
    mesh = get_pallas_op_mesh()
    # Avoid colliding in PyTorch's global custom-op registry across mesh modes:
    # registering a multi-device mesh under the same name would clobber mesh=None.
    op_name = (
        _MHC_POST_OP_NAME
        if mesh is None
        else f"{_MHC_POST_OP_NAME}_mesh_{'_'.join(mesh.axis_names)}_{mesh.size}"
    )
    op = _mhc_post_op_cache.get(op_name)
    if op is None:
        # Avoid inert PartitionSpec tuples on mesh=None calls; attach replicated
        # P() specs only when distributing across a multi-device mesh.
        input_partition_specs = (P(), P(), P(), P()) if mesh is not None else None
        op = pallas.jax_op(
            op_name,
            _mhc_post_jax,
            mesh=mesh,
            input_partition_specs=input_partition_specs,
        )
        op.register_fake(_fake_mhc_post)
        _mhc_post_op_cache[op_name] = op
    return op


__all__ = ["get_mhc_post_op"]

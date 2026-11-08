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
"""Sequence-parallel prefill for GLM-5.3-Flash under TP.

With TP every rank holds the whole residual stream -- four mHC streams per
token -- and runs every per-token op (the mHC seams, the norms) on all of
it. In a large step the residual stream is token-sharded instead: each rank
keeps `T / tp` tokens, a sublayer's input is all-gathered before its TP
matmuls and its partial output reduce-scattered back, which moves the same
bytes as the all-reduce it replaces. Small steps (decode) keep plain TP.

The branch is on the token count while tracing; the runner re-traces the
backbone per bucket group (`compilation.shape_variants`), so each bucket gets
the structure it was traced with.
"""

from __future__ import annotations

import functools

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from vllm_torchtpu import envs

_AXIS = "glm_sp"


def sp_min_tokens() -> int:
    """``GLM53_SP_MIN_TOKENS``: smallest step run sequence-parallel; 0 = off."""
    if get_tensor_model_parallel_world_size() == 1:
        return 0
    return envs.GLM53_SP_MIN_TOKENS


def sp_all_heads() -> bool:
    """``GLM53_SP_ALL_HEADS`` (default on, needs SP): the DSA layers project
    q / the output with every head from replicated weights, for this rank's
    tokens in an SP step and for all tokens otherwise (see
    `_ZeroRopeMLAWrapper._forward_all_heads`). 200K TTFT -11%, decode -8%."""
    return bool(sp_min_tokens()) and envs.GLM53_SP_ALL_HEADS


def use_sp(num_tokens: int) -> bool:
    min_tokens = sp_min_tokens()
    tp = get_tensor_model_parallel_world_size()
    return bool(min_tokens) and num_tokens >= min_tokens and num_tokens % tp == 0


@functools.cache
def _token_op(gather: bool):
    """All-gather or reduce-scatter of token shards over the TP world.

    Runs on a mesh in rank order, so mesh index i is TP rank i and the shard
    a rank holds is rows ``[i * T/tp, (i + 1) * T/tp)`` -- the slice
    `local_tokens` takes.
    """
    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh

    world = dist.get_world_size()
    if get_tensor_model_parallel_world_size() != world:
        raise NotImplementedError("GLM SP prefill needs TP to span the world")
    mesh = get_pallas_mesh(axis_names=(_AXIS,))

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            if gather:
                return jax.lax.all_gather(part, _AXIS, axis=0, tiled=True)
            return jax.lax.psum_scatter(part, _AXIS, scatter_dimension=0, tiled=True)

        # Specs say "replicated" on both sides although neither side is:
        # with P() the global shape is this rank's shape, so the torch
        # tensors the op hands back are sized for the rank (a sharded spec
        # gets the output placeholder sized for the global array).
        return jax.shard_map(
            local, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False
        )(x)

    op = pallas.jax_op(
        f"pallas::glm5_next_sp_{'ag' if gather else 'rs'}",
        collective,
        mesh=mesh,
        input_partition_specs=(P(),),
        output_partition_specs=(P(),),
    )

    def fake(x):
        rows = x.shape[0] * world if gather else x.shape[0] // world
        return x.new_empty((rows, x.shape[1]))

    op.register_fake(fake)
    return op


class TokenShards:
    """The `all_gather` / `reduce_scatter` pair KDA's `prefill_group` takes."""

    def __init__(self) -> None:
        self.world_size = get_tensor_model_parallel_world_size()
        self.rank_in_group = get_tensor_model_parallel_rank()
        # Built here, not on first use: registering a custom op inside the
        # compiled forward's trace does not work.
        self._ag = _token_op(True)
        self._rs = _token_op(False)

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        return self._ag(x)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        return self._rs(x)

    def local_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a replicated `[T, ...]` tensor."""
        return x.reshape(self.world_size, -1, *x.shape[1:])[
            self.rank_in_group
        ].contiguous()

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
"""Token-dimension TP collectives (all-gather, reduce-scatter, shard slice)."""

import functools

import torch
from vllm.distributed import get_tp_group

from vllm_torchtpu import envs
from vllm_torchtpu.utils import synchronize_tensors


@functools.cache
def _get_tp_rank_tensor(tp_rank: int, device: torch.device) -> torch.Tensor:
    t = torch.tensor([tp_rank], dtype=torch.int64, device=device)
    if device.type == "tpu" and not torch.compiler.is_compiling():
        synchronize_tensors([t], wait=True)
    return t


def tp_shard_slice(
    x: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_rank_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        tp_rank_idx is None
        or tp_rank_idx.device != x.device
        or not torch.compiler.is_compiling()
    ):
        tp_rank_idx = _get_tp_rank_tensor(tp_rank, x.device)
    shape_tail = tuple(x.size(i) for i in range(1, x.ndim))
    return x.reshape(tp_size, -1, *shape_tail).index_select(0, tp_rank_idx).squeeze(0)



def tp_token_shard_ok(num_tokens: int, tp_size: int) -> bool:
    """Whether a step of ``num_tokens`` tokens runs token-sharded over TP.

    The single predicate for every per-step token-sharding decision (Engram CPU
    prefetch, prelude/stage mHC residual, Engram shard, zeros fallback), so the
    host rows and the compiled slicing always agree. Steps below
    VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS stay replicated; <= 0 skips the
    comparison, which leaves the traced guards unchanged.
    """
    if not (tp_size > 1 and num_tokens > 0 and num_tokens % tp_size == 0):
        return False
    min_tokens = envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS
    return min_tokens <= 0 or num_tokens >= min_tokens


def tp_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    if world_size == 1:
        return x
    if dim < 0:
        dim += x.ndim
    x_contig = x.contiguous()
    shape_tail = tuple(int(x_contig.size(i)) for i in range(1, x_contig.ndim))
    if dim == 0 and shape_tail and shape_tail[-1] % 128 != 0:
        pad_cols = 128 - (shape_tail[-1] % 128)
        pad_zeros = torch.zeros(
            (*x_contig.shape[:-1], pad_cols),
            dtype=x_contig.dtype,
            device=x_contig.device,
        )
        x_padded = torch.cat((x_contig, pad_zeros), dim=-1)
        pad_tail = (*shape_tail[:-1], shape_tail[-1] + pad_cols)
        out_padded = torch.empty_like(
            x_padded.unsqueeze(0).expand(world_size, *((-1,) * x_padded.ndim)),
            memory_format=torch.contiguous_format,
        ).view(-1, *pad_tail)
        torch.distributed.all_gather_into_tensor(
            out_padded, x_padded, group=tp_group.device_group
        )
        return out_padded[..., : shape_tail[-1]].contiguous()
    output_tensor = torch.empty_like(
        x_contig.unsqueeze(0).expand(world_size, *((-1,) * x_contig.ndim)),
        memory_format=torch.contiguous_format,
    ).view(-1, *shape_tail)
    torch.distributed.all_gather_into_tensor(
        output_tensor, x_contig, group=tp_group.device_group
    )
    if dim == 0:
        return output_tensor
    output_tensor = output_tensor.view(world_size, -1, *shape_tail).movedim(0, dim)
    new_tail = list(shape_tail)
    new_tail[dim - 1] *= world_size
    return output_tensor.reshape(-1, *new_tail)


def tp_reduce_scatter(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    if world_size == 1:
        return x
    if dim < 0:
        dim += x.ndim
    input_tensor = x.movedim(0, dim).contiguous()
    if dim == 0:
        shape_tail = tuple(
            int(input_tensor.size(i)) for i in range(1, input_tensor.ndim)
        )
        output_tensor = torch.empty_like(
            input_tensor.view(world_size, -1, *shape_tail)[0],
            memory_format=torch.contiguous_format,
        )
        torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, group=tp_group.device_group
        )
        return output_tensor
    shape_tail = tuple(int(x.size(i)) for i in range(1, x.ndim))
    chunk_tail = list(shape_tail)
    chunk_tail[dim - 1] //= world_size
    output_tensor = torch.empty_like(
        x.view(-1, world_size, *chunk_tail)[:, 0].movedim(0, dim),
        memory_format=torch.contiguous_format,
    )
    torch.distributed.reduce_scatter_tensor(
        output_tensor, input_tensor, group=tp_group.device_group
    )
    return output_tensor.movedim(0, dim).contiguous()


def tp_all_to_all(x: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    if world_size == 1:
        return x
    x_contig = x.contiguous()
    output_tensor = torch.empty_like(
        x_contig, memory_format=torch.contiguous_format
    )
    torch.distributed.all_to_all_single(
        output_tensor, x_contig, group=tp_group.device_group
    )
    return output_tensor

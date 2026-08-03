# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel helpers shared by Kimi-Linear and Kimi-K3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.config import ParallelConfig
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter)
from vllm.model_executor.models.utils import sequence_parallel_chunk


def is_kimi_sequence_parallel(parallel_config: ParallelConfig) -> bool:
    """Whether Kimi should shard tokens between attention and its FFN.

    Pipeline parallelism is deliberately excluded because vLLM's pipeline
    boundary tensors are not sequence sharded. TPU currently has no MegaMoE
    path, so DP>1 is the configuration in which EP needs this optimization.
    """
    return (parallel_config.pipeline_parallel_size == 1
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
            and parallel_config.data_parallel_size > 1)


def shard(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pad and return this TP rank's contiguous token shard."""
    if hidden_states.ndim == 2:
        return sequence_parallel_chunk(hidden_states)

    # vLLM's generic helper pads with ``(0, 0, 0, pad_len)``, which is
    # specific to [tokens, hidden]. Kimi-K3 carries [tokens, 1, hidden]
    # states, where that tuple pads the singleton axis instead of tokens.
    # Remove that structural axis while applying the custom op, then restore
    # it; this also retains the op's compile-safe symbolic output shape.
    assert hidden_states.ndim == 3 and hidden_states.shape[1] == 1
    return sequence_parallel_chunk(hidden_states.squeeze(1)).unsqueeze(1)


def all_gather(hidden_states: torch.Tensor) -> torch.Tensor:
    """Reconstruct the padded token sequence on every TP rank."""
    communicator = get_tp_group().device_communicator
    custom = getattr(communicator, "custom_all_gather", None)
    if custom is not None:
        return custom(hidden_states)
    return tensor_model_parallel_all_gather(hidden_states, dim=0)


def reduce_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Sum TP partials and return this rank's padded token shard."""
    restore_structural_axis = hidden_states.ndim == 3
    if restore_structural_axis:
        assert hidden_states.shape[1] == 1
        hidden_states = hidden_states.squeeze(1)
    tp_size = get_tensor_model_parallel_world_size()
    padding = (-hidden_states.shape[0]) % tp_size
    if padding:
        hidden_states = F.pad(hidden_states, (0, 0, 0, padding))
    communicator = get_tp_group().device_communicator
    custom = getattr(communicator, "custom_reduce_scatter", None)
    if custom is not None:
        result = custom(hidden_states)
    else:
        result = tensor_model_parallel_reduce_scatter(hidden_states, dim=0)
    return result.unsqueeze(1) if restore_structural_axis else result

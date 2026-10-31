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
"""Torch bridge for the embedding lookup in ``kernels/embedding_rows.py``."""

import numpy as np
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_torchtpu.kernels.embedding_rows import can_pack, embedding_rows, pack_table
from vllm_torchtpu.utils import synchronize_tensors


def _fake_embedding_rows(ids: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (ids.shape[0], 2 * packed.shape[1]), dtype=torch.bfloat16, device=ids.device
    )


# Registered at import: the lookup runs inside the compiled forward.
_op = pallas.jax_op("pallas::embedding_rows", embedding_rows)
_op.register_fake(_fake_embedding_rows)


class PackedRowEmbedding(nn.Module):
    """``embed(ids)`` of a bf16 tp=1 embedding, from its packed table."""

    def __init__(self, packed: torch.Tensor):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _op(input_ids, self.packed)


def is_plain_lookup(embed: nn.Module) -> bool:
    """Whether ``embed(ids)`` is exactly ``embed.weight[ids]``."""
    if not isinstance(embed, VocabParallelEmbedding):
        return False
    w = embed.weight
    return (
        w.dtype == torch.bfloat16
        and isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
        and embed.tp_size == 1
        and w.shape[0] == embed.num_embeddings
        and can_pack(w.shape[1])
    )


def packed_embedding(embed: VocabParallelEmbedding) -> PackedRowEmbedding:
    """The packed replacement for ``embed``, built through the host once."""
    w = embed.weight.data
    bits = w.cpu().view(torch.int16).numpy().view(np.uint16)
    packed = torch.from_numpy(pack_table(bits)).to(w.device)
    synchronize_tensors([packed])
    return PackedRowEmbedding(packed)

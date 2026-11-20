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
"""Vocab-parallel embedding served from a load-time replica of the whole table.

`VocabParallelEmbedding` looks every token up in this rank's vocab shard, zeroes
the tokens other shards own and sums the shards: an all-reduce at decode, and a
reduce-scatter of a `[tokens, hidden]` partial under sequence-parallel prefill.
Exactly one shard contributes each row, so the sum is that row. With the shards
all-gathered once at load, every rank looks its tokens up locally instead: the
same rows (up to the sign of a zero), no collective, and a sequence-parallel
rank reads only its own tokens' rows.

The replica costs `vocab_padded * hidden * itemsize` bytes per rank, and only
serves tables without added (LoRA) vocabulary, where the shards concatenate to
the padded vocabulary in token order.
"""

from collections.abc import Callable
from typing import Any

import torch
from vllm.model_executor.layers import vocab_parallel_embedding as vpe

from vllm_torchtpu.utils import fresh_buffer

FULL_WEIGHT_ATTR = "_tpu_replicated_embedding_weight"


def can_replicate(embed: Any, tp: int) -> bool:
    """Whether `embed` is a plain `tp`-way vocab-parallel table.

    Plain: unquantized, no added vocabulary, so the shards concatenated in rank
    order are the padded vocabulary in token order.
    """
    return (
        tp > 1
        and isinstance(embed, vpe.VocabParallelEmbedding)
        and embed.tp_size == tp
        and type(embed.quant_method) is vpe.UnquantizedEmbeddingMethod
        and embed.num_embeddings_padded == embed.org_vocab_size_padded
        and not bool(getattr(embed, "use_fused_embedding", False))
    )


def arm(
    embed: vpe.VocabParallelEmbedding,
    all_gather_rows: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Gather the shards into `[vocab_padded, hidden]` and register it on `embed`.

    `all_gather_rows` concatenates every rank's shard on dim 0 in rank order.
    """
    shard = embed.weight.data
    full = all_gather_rows(shard)
    expected = (embed.num_embeddings_padded, shard.shape[1])
    if tuple(full.shape) != expected or full.dtype != shard.dtype:
        raise ValueError(
            f"gathered embedding {tuple(full.shape)} {full.dtype}, expected "
            f"{expected} {shard.dtype}"
        )
    # A fresh buffer, not a view of the gather output: torch_tpu would
    # otherwise re-materialize it before every forward.
    full = fresh_buffer(full)
    embed.register_buffer(FULL_WEIGHT_ATTR, full, persistent=False)
    return full


def arm_if_eligible(
    embed: Any,
    tp: int,
    all_gather_rows: Callable[[torch.Tensor], torch.Tensor],
) -> bool:
    if not can_replicate(embed, tp):
        return False
    arm(embed, all_gather_rows)
    return True


def full_weight(embed: object) -> torch.Tensor | None:
    return getattr(embed, FULL_WEIGHT_ATTR, None)


def lookup(full: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.embedding(token_ids.long(), full)

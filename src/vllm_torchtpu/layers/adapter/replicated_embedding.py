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
all-gathered once at load, a sequence-parallel prefill rank looks its own
tokens up locally instead: the same rows (up to the sign of a zero), and no
collective. Decode buckets keep the vocab-parallel lookup and its all-reduce:
short prompts prefilled in decode buckets came back with wrong tokens when
they read the replica too.

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
DECODE_LOOKUP_ATTR = "_tpu_replicated_embedding_decode"
SHARD_BOUND_NAMES = (
    "org_vocab_start_index",
    "org_vocab_end_index",
    "num_org_vocab_padding",
    "added_vocab_start_index",
    "added_vocab_end_index",
)


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
    *,
    decode: bool = False,
) -> bool:
    if not can_replicate(embed, tp):
        return False
    arm(embed, all_gather_rows)
    if decode:
        enable_decode_lookup(embed)
    return True


def retain_table(embed: object, out: torch.Tensor) -> torch.Tensor:
    """`out` unchanged, with `embed`'s replicated table lifted into the trace.

    The sequence-parallel prefill buckets read the table and the decode
    buckets do not; bucket hand-over needs both traces to lift it.
    """
    full = full_weight(embed)
    if full is None:
        return out
    # Avoid a module-level import: this module is imported where the
    # VocabParallelEmbedding.forward patch is installed (platform activation,
    # every process), and the attention adapter registers a vLLM attention
    # backend on import.
    from vllm_torchtpu.layers.adapter.attention import (  # noqa: PLC0415 -- see above
        _retain_buffers,
    )

    return _retain_buffers(out, full)


def enable_decode_lookup(embed: object) -> None:
    setattr(embed, DECODE_LOOKUP_ATTR, True)


def decode_lookup(embed: Any, token_ids: torch.Tensor) -> torch.Tensor | None:
    """Rows of `token_ids` from the replica, zero where the shards own no row.

    Matches the masked vocab-parallel lookup + all-reduce: ids outside
    `[0, org_vocab_size)` (negative, or in the padded tail) give zero rows.
    None unless `embed` is armed and `enable_decode_lookup` was called.
    """
    full = full_weight(embed)
    if full is None or not getattr(embed, DECODE_LOOKUP_ATTR, False):
        return None
    ids = token_ids.long()
    in_vocab = (ids >= 0) & (ids < embed.org_vocab_size)
    rows = torch.nn.functional.embedding(ids * in_vocab, full)
    rows = rows.masked_fill(~in_vocab.unsqueeze(-1), 0)
    from vllm_torchtpu.layers.adapter.attention import (  # noqa: PLC0415 -- same cycle as retain_table
        _retain_buffers,
    )

    # The sequence-parallel buckets lift the shard operands; lift them here
    # too so both traces hand buckets over with the same inputs.
    return _retain_buffers(rows, *shard_operands(embed))


def shard_operands(embed: Any) -> tuple[torch.Tensor | None, ...]:
    """What the vocab-parallel lookup reads: the shard and its five bounds."""
    return (embed.weight, *(getattr(embed, name, None) for name in SHARD_BOUND_NAMES))


def full_weight(embed: object) -> torch.Tensor | None:
    return getattr(embed, FULL_WEIGHT_ATTR, None)


def lookup(full: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.embedding(token_ids.long(), full)

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
"""The token step's view of vLLM's vocabulary-parallel embedding and LM head.

A program folds the embedding and the greedy LM head into its kernel
(`core.token_io`) only when the stock logits are exactly `hidden @ head.T` on
contiguous per-rank vocabulary shards: no bias, scale 1, no soft cap, no
LoRA, no added vocabulary. Anything else keeps the hidden step only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.megakernel.core.token_io import TokenIO, head_tile


@dataclass(frozen=True)
class GreedyHead:
    """One rank's embedding and LM head rows (vLLM's own tensors)."""

    embed: torch.Tensor  # [shard, H]
    head: torch.Tensor  # [shard, H]
    vocab: int  # the real vocabulary (ids at or past it are padding)
    shard: int


def _shard_of(layer: Any, tp_rank: int) -> int | str:
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return "no [shard, H] weight"
    if getattr(layer, "bias", None) is not None:
        return "has a bias"
    idx = getattr(layer, "shard_indices", None)
    shard = int(w.shape[0])
    if idx is not None and (
        int(idx.org_vocab_start_index) != tp_rank * shard
        or int(getattr(idx, "num_added_elements", 0)) != 0
    ):
        return "vocabulary shards are not contiguous rank blocks"
    return shard


def greedy_head(model: Any, embed: Any, tp_rank: int) -> GreedyHead | str:
    """The rank's rows of `embed` and of the module of `model` holding
    `lm_head` and `logits_processor`, or why the greedy head can't be folded."""
    owner = next(
        (
            m
            for m in ([model] + list(getattr(model, "modules", list)()))
            if getattr(m, "lm_head", None) is not None
            and getattr(m, "logits_processor", None) is not None
        ),
        None,
    )
    if owner is None:
        return "no module with an lm_head and a logits_processor"
    proc, head = owner.logits_processor, owner.lm_head
    if type(proc).__name__ != "LogitsProcessor":
        return f"logits processor {type(proc).__name__} (LoRA or custom) changes them"
    if float(getattr(proc, "scale", 1.0)) != 1.0 or getattr(proc, "soft_cap", None):
        return "the logits are scaled or soft-capped"
    shards = [_shard_of(layer, tp_rank) for layer in (embed, head)]
    if bad := [s for s in shards if isinstance(s, str)]:
        return f"embedding / LM head: {bad[0]}"
    if shards[0] != shards[1] or embed.weight.shape[1] != head.weight.shape[1]:
        return "embedding and LM head shards differ"
    vocab = int(getattr(proc, "org_vocab_size", getattr(head, "org_vocab_size", 0)))
    if vocab <= 0:
        return "unknown vocabulary size"
    return GreedyHead(embed.weight, head.weight, vocab, int(shards[0]))


def token_io_for(gh: GreedyHead, bank: int, rope_cache: torch.Tensor | None) -> TokenIO:
    """The kernel's `TokenIO` for `gh`, LM head tiles sized to `bank`."""
    H = int(gh.head.shape[1])
    rope = {}
    if rope_cache is not None:
        rope = dict(
            rope_rows=int(rope_cache.shape[0]),
            rope_cols=int(rope_cache.shape[1]),
            rope_dtype=_jax_dtype(rope_cache.dtype),
        )
    return TokenIO(
        vocab=gh.vocab, shard=gh.shard, tile=head_tile(H, gh.shard, bank), **rope
    )


def pack_head_tiles(gh: GreedyHead, tio: TokenIO) -> torch.Tensor:
    """The LM head rows as `[n_tiles, H, tile]` (one copy, made at load)."""
    w = gh.head
    H = int(w.shape[1])
    pad = tio.n_tiles * tio.tile - tio.shard
    wt = torch.nn.functional.pad(w.t(), (0, pad))
    out = wt.reshape(H, tio.n_tiles, tio.tile).permute(1, 0, 2).contiguous()
    return out


def _jax_dtype(dt: torch.dtype) -> Any:
    import jax.numpy as jnp

    return {torch.float32: jnp.float32, torch.bfloat16: jnp.bfloat16}[dt]


__all__ = ["GreedyHead", "greedy_head", "pack_head_tiles", "token_io_for"]

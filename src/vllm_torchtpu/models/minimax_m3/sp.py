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
"""Sequence-parallel residual stream for long MiniMax M3 steps.

With TPU_MINIMAX_M3_SP_MIN_TOKENS set, a step of at least that many tokens
(and a multiple of the TP size) keeps the residual stream split by token
across the TP group. Per layer:

    x = all_gather(norm(h + r))          # attention input, every token
    h = reduce_scatter(attn_partial(x))  # this rank's tokens
    x = all_gather(norm(h + r))          # FFN input, every token
    h = reduce_scatter(ffn_partial(x))

An all-reduce moves the bytes of a reduce-scatter plus an all-gather, so the
collectives alone cost the same in bf16; what shrinks is the residual add and
both RMSNorms (1/TP of the tokens each) and, with an fp8 gather
(TPU_MINIMAX_M3_SP_GATHER / _ATTN_GATHER = fp8), half of every all-gather's
bytes. The embedding's all-reduce becomes a reduce-scatter of the
vocab-parallel partials (exact: one rank holds each token's row), and the
output norm runs on the local tokens before one all-gather.

The branch is decided from the token count while tracing, which leaves a shape
guard, so the backbone warmup gives each side its own trace
(`compilation.shape_variants`). Both sides read the same parameters and
buffers in the same order, so a bucket handed between the traces keeps its
input signature.
"""

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu.compilation import shape_variants

FP8_MAX = 448.0
_SCALE_FLOOR = 1e-12


def active(num_tokens: int, min_tokens: int, tp_size: int) -> bool:
    """Whether a step of `num_tokens` takes the sequence-parallel stream."""
    if min_tokens <= 0 or tp_size <= 1:
        return False
    # Only the backbone warmup gives each side of this branch its own trace
    # (`shape_variants.aot_cache_tag` is set there); any other compile of the
    # model keeps the all-reduce stream.
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return num_tokens >= min_tokens and num_tokens % tp_size == 0


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """This rank's token rows of the TP sum of `x` [T, H] -> [T / tp, H]."""
    return tensor_model_parallel_reduce_scatter(x, 0)


def all_gather(x: torch.Tensor, fp8: bool) -> torch.Tensor:
    """Every rank's token rows, in rank order: [T / tp, H] -> [T, H] bf16.

    fp8: each row goes over the wire as e4m3 with one f32 scale (amax / 448),
    and is expanded back to `x.dtype` after the gather.
    """
    if not fp8:
        return tensor_model_parallel_all_gather(x, 0)
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=_SCALE_FLOOR) / FP8_MAX
    q = (xf / scale).to(torch.float8_e4m3fn)
    q = tensor_model_parallel_all_gather(q.view(torch.uint8), 0)
    scale = tensor_model_parallel_all_gather(scale, 0)
    return (q.view(torch.float8_e4m3fn).float() * scale).to(x.dtype)


def embed_partial(embed: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """The TPU VocabParallelEmbedding forward without its TP all-reduce: rows
    of tokens outside this rank's vocab shard are zero."""
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    masked, mask = vpe.get_masked_input_and_mask(
        input_ids,
        embed.org_vocab_start_index,
        embed.org_vocab_end_index,
        embed.num_org_vocab_padding,
        embed.added_vocab_start_index,
        embed.added_vocab_end_index,
    )
    out = embed.quant_method.embedding(embed, masked.long())
    out.masked_fill_(mask.unsqueeze(-1), 0)
    return out

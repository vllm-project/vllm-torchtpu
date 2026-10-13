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
"""N-gram PLE: the hashed n-gram embedding Qwen4Exp adds at one decoder layer.

Follows the HF reference (``Qwen4ExpTextNGramEmbedding`` and
``Qwen4ExpTextPLELayer``) and upstream vllm/models/qwen4_exp. The n-gram ids
are computed on the device from this step's input ids plus the tokens that
precede each request's chunk, so async scheduling does not need the newest
sampled token on the host.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.models.qwen4_exp.common.hyperconnection import GroupedGemmaRMSNorm
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
)
from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

_MASK64 = (1 << 64) - 1
_MAX_INT64 = (1 << 63) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007
_DEFAULT_SEED = 1234


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(
    vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    """Odd hash multipliers small enough that token * multiplier fits int64."""
    half_bound = max(1, _MAX_INT64 // max(vocab_size, 1) // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return multipliers


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def head_vocab_sizes(
    ngram_vocab_size_base: int, ngram_heads: int, ple_layer_index: int
) -> list[int]:
    """Each head hashes into its own prime-sized slice of the table."""
    sizes = []
    prime = ngram_vocab_size_base - 1
    for _ in range((ple_layer_index + 1) * ngram_heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
    return sizes[ple_layer_index * ngram_heads :]


def _token_requests(
    query_start_loc: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each token's position, request and offset within that request's chunk.

    Batch padding past the last request maps to the last request.
    """
    starts = query_start_loc.long()
    positions = torch.arange(num_tokens, device=starts.device)
    request = (positions.unsqueeze(1) >= starts[1:].unsqueeze(0)).sum(dim=1)
    request = request.clamp(max=starts.numel() - 2)
    return positions, request, positions - starts[request]


def ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    *,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    """Table rows for every head of every token, shape [num_tokens, heads].

    ``ngram_context[r]`` holds the tokens just before request ``r``'s chunk,
    oldest first, with EOS where the request has none.
    """
    tokens = input_ids.long()
    context = ngram_context.long()
    context_len = context.shape[-1]
    positions, request, chunk_pos = _token_requests(query_start_loc, tokens.shape[0])

    # Walk predecessors from newest to oldest. Everything behind an EOS reads
    # as EOS, which is how the reference resets the n-gram history.
    eos = torch.full_like(tokens, eos_token_id)
    crossed = torch.zeros_like(tokens, dtype=torch.bool)
    mixed = tokens * multipliers[0]
    blocks = []
    for shift in range(1, context_len + 1):
        from_step = tokens[(positions - shift).clamp(min=0)]
        from_context = context[
            request, (context_len - shift + chunk_pos).clamp(0, context_len - 1)
        ]
        token = torch.where(chunk_pos >= shift, from_step, from_context)
        token = torch.where(crossed, eos, token)
        crossed = crossed | (token == eos_token_id)
        # Each product fits int64 (see layer_multipliers), so the XOR stays
        # non-negative and the remainder matches the reference.
        mixed = torch.bitwise_xor(mixed, token * multipliers[shift])
        heads = slice((shift - 1) * heads_per_ngram, shift * heads_per_ngram)
        blocks.append(
            torch.remainder(mixed.unsqueeze(-1), vocab_sizes[heads]) + offsets[heads]
        )
    return torch.cat(blocks, dim=-1)


def dilated_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Depthwise causal conv over a ragged batch; updates ``conv_state``.

    ``weight`` is [channels, kernel]; ``conv_state[slot]`` holds a request's
    last (kernel - 1) * dilation inputs, oldest first. Taps that reach back
    past the chunk start read that window instead of ``x``. Every request,
    including padding requests, writes its window back to its slot. Returns
    the conv output before its activation.
    """
    num_tokens = x.shape[0]
    kernel = weight.shape[-1]
    history = (kernel - 1) * dilation
    starts = query_start_loc.long()
    num_reqs = starts.numel() - 1
    positions, request, chunk_pos = _token_requests(query_start_loc, num_tokens)

    slots = state_indices.long()
    windows = conv_state[slots].to(x.dtype)
    windows = torch.where(
        has_initial_state.view(-1, 1, 1), windows, torch.zeros_like(windows)
    )
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for tap in range(kernel):
        back = history - tap * dilation
        source = chunk_pos - back
        from_step = x[(positions - back).clamp(min=0)]
        from_window = windows[request, (history + source).clamp(0, history - 1)]
        rows = torch.where((source >= 0).unsqueeze(-1), from_step, from_window)
        out = out + rows.float() * weight[:, tap].float()

    # The new window is each request's last `history` inputs.
    query_lens = starts[1:] - starts[:-1]
    source = query_lens.unsqueeze(1) - history + torch.arange(history, device=x.device)
    from_step = x[(starts[:-1].unsqueeze(1) + source).clamp(0, num_tokens - 1)]
    from_window = windows[
        torch.arange(num_reqs, device=x.device).unsqueeze(1),
        (history + source).clamp(0, history - 1),
    ]
    new_windows = torch.where((source >= 0).unsqueeze(-1), from_step, from_window)
    conv_state.index_copy_(0, slots, new_windows.to(conv_state.dtype))
    return out.to(x.dtype)


@dataclass(frozen=True)
class PLEInputs:
    """Per-step PLE inputs for one batch.

    Attributes:
        input_ids: [num_tokens] token ids of this step.
        query_start_loc: [num_reqs + 1] chunk boundaries in ``input_ids``.
        ngram_context: [num_reqs, ngram_size - 1] tokens before each chunk.
        conv_state: [num_slots, (kernel - 1) * dilation, hc_count * hidden].
        state_indices: [num_reqs] each request's ``conv_state`` slot.
        has_initial_state: [num_reqs] False for requests starting from
            scratch, whose slot content is ignored.
    """

    input_ids: torch.Tensor
    query_start_loc: torch.Tensor
    ngram_context: torch.Tensor
    conv_state: torch.Tensor
    state_indices: torch.Tensor
    has_initial_state: torch.Tensor


class Qwen4ExpPLETable(PLEVocabParallelEmbedding):
    """N-gram table split by rows across TP ranks.

    Rows are dequantized right after the lookup (FP8 tables have one global
    scale), so the cross-rank sum runs in the activation dtype.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        padding_size: int,
        prefix: str,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
        )
        self.embedding_method = embedding_method
        self.output_dtype = torch.get_default_dtype()

    def allocate_embedding_weight(
        self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Storage for this rank's rows, called by the embedding method."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def local_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows owned by this rank; rows of other ranks come back as zeros."""
        flat = ids.reshape(-1).long()
        owned = None
        if self.tp_size > 1:
            # The table has no added or padding rows, so each rank owns one
            # row range. vllm-torchtpu registers its bounds as buffers so that
            # every TP rank compiles the same graph; fall back to the ints.
            start = getattr(
                self, "org_vocab_start_index", self.shard_indices.org_vocab_start_index
            )
            end = getattr(
                self, "org_vocab_end_index", self.shard_indices.org_vocab_end_index
            )
            owned = (flat >= start) & (flat < end)
            flat = torch.where(owned, flat - start, 0)
        rows = self.embedding_method.embedding(self, flat)
        rows = self.embedding_method.dequantize(self, rows, self.output_dtype)
        if owned is not None:
            rows = rows.masked_fill(~owned.unsqueeze(-1), 0)
        return rows.reshape(*ids.shape, self.embedding_dim)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        rows = self.local_rows(input_)
        if self.tp_size > 1:
            rows = tensor_model_parallel_all_reduce(rows)
        return rows


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashes each token's bigram and trigram into the PLE table."""

    layer_multipliers: torch.Tensor
    ngram_heads_vocab_sizes: torch.Tensor
    ngram_heads_offsets: torch.Tensor

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
        if embedding_dim % ngram_heads:
            raise ValueError(
                f"ple_embed_dim ({embedding_dim}) must be divisible by the "
                f"{ngram_heads} n-gram heads."
            )
        self.heads_per_ngram = config.heads_per_ngram
        eos_token_id = config.eos_token_id
        if isinstance(eos_token_id, list | tuple):
            eos_token_id = eos_token_id[0]
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = config.vocab_size
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        seed = getattr(config, "seed", None)
        multipliers = layer_multipliers(
            config.vocab_size,
            config.ngram_size,
            ple_layer_index,
            _DEFAULT_SEED if seed is None else int(seed),
        )
        sizes = head_vocab_sizes(
            config.ngram_vocab_size_base, ngram_heads, ple_layer_index
        )
        offsets = [sum(sizes[:head]) for head in range(ngram_heads)]
        self.register_buffer("layer_multipliers", torch.tensor(multipliers))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(sizes))
        self.register_buffer("ngram_heads_offsets", torch.tensor(offsets))
        divisor = config.make_ngram_vocab_size_divisible_by
        table_prefix = f"{prefix}.ngram_embedding"
        self.ngram_embedding = Qwen4ExpPLETable(
            math.ceil(sum(sizes) / divisor) * divisor,
            embedding_dim // ngram_heads,
            embedding_method=Qwen4ExpPLEEmbeddingMethod.from_quant_config(
                quant_config,
                table_prefix,
                getattr(config, "ple_embedding_dtype", None),
            ),
            padding_size=divisor,
            prefix=table_prefix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        ids = ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            eos_token_id=self.eos_token_id,
            heads_per_ngram=self.heads_per_ngram,
        )
        return self.ngram_embedding(ids).flatten(-2)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the hash buffers, the table scale and the split table rows.

        Can be called several times, each with part of the weights.
        """
        table = self.ngram_embedding
        table_is_fp8 = isinstance(table.embedding_method, Qwen4ExpPLEFp8EmbeddingMethod)
        shard_rows = math.ceil(table.org_vocab_size / self.split_ngram_parts)
        shard_prefix = "ngram_embedding.shard_"
        loaded: set[str] = set()
        for name, tensor in weights:
            if name in (
                "layer_multipliers",
                "ngram_heads_vocab_sizes",
                "ngram_heads_offsets",
            ):
                buffer = self.get_buffer(name)
                if buffer.shape != tensor.shape:
                    raise ValueError(
                        f"{name} has shape {tuple(tensor.shape)} in the checkpoint "
                        f"but the config implies {tuple(buffer.shape)}."
                    )
                buffer.copy_(tensor)
            elif name == "ngram_embedding.weight_scale" and table_is_fp8:
                scale = table.get_parameter("weight_scale")
                scale.data.copy_(tensor.reshape(scale.shape))
            elif name.startswith(shard_prefix) and name.endswith(".weight"):
                index = int(name[len(shard_prefix) : -len(".weight")])
                start = index * shard_rows
                rows = min(shard_rows, table.org_vocab_size - start)
                if index >= self.split_ngram_parts or tensor.shape != (
                    rows,
                    table.embedding_dim,
                ):
                    raise ValueError(
                        f"PLE table shard {index} has shape {tuple(tensor.shape)}; "
                        f"split_ngram_parts={self.split_ngram_parts} implies "
                        f"({rows}, {table.embedding_dim})."
                    )
                table.weight_loader(
                    table.get_parameter("weight"), tensor, checkpoint_start=start
                )
                name = "ngram_embedding.weight"
            else:
                raise ValueError(f"Unexpected PLE embedding weight {name!r}.")
            loaded.add(name)
        max_multiplier = int(self.layer_multipliers.max())
        if max_multiplier * (self.vocab_size - 1) > _MAX_INT64:
            raise ValueError(
                f"PLE hash multiplier {max_multiplier} overflows int64 for a "
                f"vocabulary of {self.vocab_size} tokens."
            )
        return loaded


class Qwen4ExpPLELayer(nn.Module):
    """Adds gated n-gram features, then a dilated short conv, to every stream."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        ple_layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden_size = config.hidden_size * config.hc_count
        embed_dim = config.ple_embed_dim
        self.conv_dilation = config.ngram_size
        self.conv_state_len = (config.ple_conv_kernel_size - 1) * self.conv_dilation
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            embed_dim,
            ple_layer_index,
            quant_config,
            prefix=f"{prefix}.ple_embedding",
        )
        self.key_proj = ReplicatedLinear(
            embed_dim,
            hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            embed_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        dtype = torch.get_default_dtype()
        self.norm_key, self.norm_query, self.norm_conv = (
            GroupedGemmaRMSNorm(
                hc_hidden_size, config.rms_norm_eps, config.hidden_size, dtype
            )
            for _ in range(3)
        )
        # Holds the checkpoint's [channels, 1, kernel] weight; forward runs the
        # conv with dilated_short_conv to carry state across steps.
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            config.ple_conv_kernel_size,
            groups=hc_hidden_size,
            dilation=self.conv_dilation,
            bias=False,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor, inputs: PLEInputs) -> torch.Tensor:
        """Returns the term to add to the hc_count * hidden_size residual."""
        embeddings = self.ple_embedding(
            inputs.input_ids, inputs.query_start_loc, inputs.ngram_context
        )
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        streams = (self.hc_count, self.hidden_size)
        key = self.norm_key(key).unflatten(-1, streams)
        query = self.norm_query(hidden_states).unflatten(-1, streams)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        conv_output = dilated_short_conv(
            self.norm_conv(gated_value),
            self.conv1d.weight[:, 0, :],
            self.conv_dilation,
            inputs.conv_state,
            inputs.state_indices,
            inputs.has_initial_state,
            inputs.query_start_loc,
        )
        return gated_value + F.silu(conv_output)

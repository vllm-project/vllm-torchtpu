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

Ported from upstream vllm/models/qwen4_exp (nvidia/ngram_embedding.py,
nvidia/ple_layer.py and the Triton kernels in nvidia/ops/ple.py) and checked
against the HF reference (``Qwen4ExpTextNGramEmbedding``,
``Qwen4ExpTextPLELayer``). The kernels become torch ops, and each function
names what it follows. The n-gram ids are computed on the device from this
step's input ids plus the tokens that precede each request's chunk, so async
scheduling does not need the newest sampled token on the host. The conv
state is passed in and returned, so the caller decides where it is stored.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig
from vllm.distributed import GroupCoordinator, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.models.qwen4_exp.common.hyperconnection import GroupedGemmaRMSNorm
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
)
from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding

from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
    build_normalized_gate_dot_dispatch,
)

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
    """Odd hash multipliers small enough that token * multiplier fits int64.

    Same values as upstream's ``Qwen4ExpNGramEmbedding._make_layer_multipliers``.
    """
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
    """Each head hashes into its own prime-sized slice of the table.

    Same values as upstream's ``Qwen4ExpNGramEmbedding._make_vocab_layout``,
    with trial division in place of its Miller-Rabin test: the primes sit just
    above ngram_vocab_size_base (2e7 in the released checkpoint).
    """
    sizes = []
    prime = ngram_vocab_size_base - 1
    for _ in range((ple_layer_index + 1) * ngram_heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
    return sizes[ple_layer_index * ngram_heads :]


def token_requests(
    query_start_loc: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token's request and its offset within that request's chunk.

    Marks the interior request starts and takes a cumsum, the form
    kda_attention_op._token_sequence_ids measured as cheaper on TPU than a
    [num_tokens, num_reqs] comparison. Starts at or past ``num_tokens`` land
    in a dropped overflow slot, and batch padding maps to the last request.
    """
    num_reqs = query_start_loc.numel() - 1
    query_start_loc = query_start_loc.to(torch.int32)
    starts = query_start_loc[1:num_reqs].clamp(max=num_tokens)
    # Keep the overflow slot inside a tensor op: a symbolic `num_tokens + 1`
    # would otherwise escape as a scalar output across the host PLE boundary.
    marks = F.pad(
        torch.zeros(num_tokens, dtype=torch.int32, device=query_start_loc.device),
        (0, 1),
    )
    marks.index_add_(0, starts, torch.ones_like(starts))
    request = marks[:num_tokens].cumsum(0, dtype=torch.int32).clamp(max=num_reqs - 1)
    positions = torch.arange(
        num_tokens, dtype=torch.int32, device=query_start_loc.device
    )
    return request, positions - query_start_loc[request]


def ngram_ids(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    request: torch.Tensor,
    chunk_pos: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    *,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    """Table rows for every head of every token, shape [num_tokens, heads].

    ``ngram_context[r]`` holds the tokens just before request ``r``'s chunk,
    oldest first, with EOS where the request has none; ``request`` and
    ``chunk_pos`` come from token_requests. Follows upstream's
    ``_ple_ngram_ids_kernel``, reading in-step predecessors as static shifts.
    """
    tokens = input_ids.long()
    context = ngram_context.long()
    num_tokens = tokens.shape[0]
    context_len = context.shape[-1]

    # Walk predecessors from newest to oldest. Everything behind an EOS reads
    # as EOS, which is how the reference resets the n-gram history.
    eos = torch.full_like(tokens, eos_token_id)
    crossed = torch.zeros_like(tokens, dtype=torch.bool)
    mixed = tokens * multipliers[0]
    blocks = []
    for shift in range(1, context_len + 1):
        from_step = F.pad(tokens, (shift, 0))[:num_tokens]
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
    windows: torch.Tensor,
    query_start_loc: torch.Tensor,
    request: torch.Tensor,
    chunk_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv over a ragged batch that carries each request's
    window between steps.

    ``weight`` is [channels, kernel]. ``windows[r]`` holds request ``r``'s last
    (kernel - 1) * dilation inputs, oldest first, and must be zero when the
    request has no history. Taps that reach back past the chunk start read the
    window instead of ``x``. Returns the conv output before its activation and
    each request's new window. Follows the taps and window shift of upstream's
    ``ple_conv`` prefill kernel and write-back, without touching storage.
    """
    num_tokens, channels = x.shape
    kernel = weight.shape[-1]
    history = (kernel - 1) * dilation
    num_reqs = query_start_loc.numel() - 1
    if windows.shape != (num_reqs, history, channels):
        raise ValueError(
            f"PLE conv windows have shape {tuple(windows.shape)}; expected "
            f"(num_reqs, (kernel - 1) * dilation, channels) = "
            f"({num_reqs}, {history}, {channels})."
        )
    query_start_loc = query_start_loc.to(torch.int32)

    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for tap in range(kernel):
        back = history - tap * dilation
        source = chunk_pos - back
        from_step = F.pad(x, (0, 0, back, 0))[:num_tokens]
        from_window = windows[request, (history + source).clamp(0, history - 1)]
        rows = torch.where((source >= 0).unsqueeze(-1), from_step, from_window)
        out = out + rows.float() * weight[:, tap].float()

    # The new window is each request's last `history` inputs.
    starts = query_start_loc[:-1]
    query_lens = query_start_loc[1:] - starts
    steps = torch.arange(history, dtype=torch.int32, device=x.device)
    source = query_lens.unsqueeze(1) - history + steps
    from_step = x[(starts.unsqueeze(1) + source).clamp(0, num_tokens - 1)]
    from_window = windows[
        torch.arange(num_reqs, dtype=torch.int32, device=x.device).unsqueeze(1),
        (history + source).clamp(0, history - 1),
    ]
    new_windows = torch.where((source >= 0).unsqueeze(-1), from_step, from_window)
    return out.to(x.dtype), new_windows


@dataclass(frozen=True)
class PLEInputs:
    """Per-step token inputs for one batch.

    Attributes:
        input_ids: [num_tokens] token ids of this step.
        query_start_loc: [num_reqs + 1] chunk boundaries in ``input_ids``.
        ngram_context: [num_reqs, ngram_size - 1] tokens before each chunk,
            oldest first, with EOS where the request has none.
    """

    input_ids: torch.Tensor
    query_start_loc: torch.Tensor
    ngram_context: torch.Tensor


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
        parallel_group: GroupCoordinator | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=parallel_group,
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
            # every TP rank compiles the same graph; without that patch (CPU
            # tests) the ints are used.
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
            rows = (
                tensor_model_parallel_all_reduce(rows)
                if self.parallel_group is None
                else self.parallel_group.all_reduce(rows)
            )
        return rows


class PLETableFactory(Protocol):
    """Choose storage before the embedding method allocates the table."""

    def __call__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        padding_size: int,
        prefix: str,
    ) -> PLEVocabParallelEmbedding: ...


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
        *,
        table_factory: PLETableFactory = Qwen4ExpPLETable,
    ) -> None:
        super().__init__()
        ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
        if embedding_dim % ngram_heads:
            raise ValueError(
                f"ple_embed_dim ({embedding_dim}) must be divisible by the "
                f"{ngram_heads} n-gram heads."
            )
        eos_token_id = config.eos_token_id
        if not isinstance(eos_token_id, int):
            raise ValueError(
                "Qwen4Exp PLE resets its n-gram history at a single EOS token, "
                f"but the text config has eos_token_id={eos_token_id!r}."
            )
        self.eos_token_id = eos_token_id
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.vocab_size = config.vocab_size
        # Optional checkpoint fields, with upstream's defaults. The released
        # checkpoint sets split_ngram_parts only.
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
        self.ngram_embedding = table_factory(
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
        ngram_context: torch.Tensor,
        request: torch.Tensor,
        chunk_pos: torch.Tensor,
    ) -> torch.Tensor:
        ids = self.compute_ids(input_ids, ngram_context, request, chunk_pos)
        return self.ngram_embedding(ids).flatten(-2)

    def compute_ids(
        self,
        input_ids: torch.Tensor,
        ngram_context: torch.Tensor,
        request: torch.Tensor,
        chunk_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Hash rows independently of resident or host table lookup."""
        if ngram_context.shape[-1] != self.ngram_size - 1:
            raise ValueError(
                f"ngram_context has {ngram_context.shape[-1]} tokens per "
                f"request; ngram_size={self.ngram_size} needs "
                f"{self.ngram_size - 1}."
            )
        return ngram_ids(
            input_ids,
            ngram_context,
            request,
            chunk_pos,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            eos_token_id=self.eos_token_id,
            heads_per_ngram=self.heads_per_ngram,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the hash buffers, the table scale and the split table rows.

        Can be called several times, each with part of the weights.
        """
        table = self.ngram_embedding
        table_is_fp8 = isinstance(table.quant_method, Qwen4ExpPLEFp8EmbeddingMethod)
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
                index_text = name[len(shard_prefix) : -len(".weight")]
                if not index_text.isdigit():
                    raise ValueError(
                        f"Unexpected PLE table weight {name!r}; table shards are "
                        "named ngram_embedding.shard_<index>.weight."
                    )
                index = int(index_text)
                if index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE table shard {index} is out of range for the "
                        f"config's split_ngram_parts={self.split_ngram_parts}."
                    )
                start = index * shard_rows
                rows = min(shard_rows, table.org_vocab_size - start)
                if tensor.shape != (rows, table.embedding_dim):
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
        *,
        table_factory: PLETableFactory = Qwen4ExpPLETable,
    ) -> None:
        super().__init__()
        if config.ple_conv_kernel_size < 2:
            raise NotImplementedError(
                "Qwen4Exp PLE on TPU needs ple_conv_kernel_size >= 2, got "
                f"{config.ple_conv_kernel_size}."
            )
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden_size = config.hidden_size * config.hc_count
        # transformers fills a missing ple_embed_dim with hidden_size.
        embed_dim = (
            config.hidden_size if config.ple_embed_dim is None else config.ple_embed_dim
        )
        self.conv_dilation = config.ngram_size
        self.conv_state_len = (config.ple_conv_kernel_size - 1) * self.conv_dilation
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            embed_dim,
            ple_layer_index,
            quant_config,
            prefix=f"{prefix}.ple_embedding",
            table_factory=table_factory,
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
        self.gate_dot = build_normalized_gate_dot_dispatch(
            self.hc_count, self.hidden_size
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs: PLEInputs,
        conv_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the term to add to the hc_count * hidden_size residual, and
        each request's new conv window (see dilated_short_conv)."""
        num_tokens = hidden_states.shape[0]
        if inputs.input_ids.shape[0] != num_tokens:
            raise ValueError(
                f"PLE got {inputs.input_ids.shape[0]} input ids for "
                f"{num_tokens} hidden states."
            )
        request, chunk_pos = token_requests(inputs.query_start_loc, num_tokens)
        embeddings = self.ple_embedding(
            inputs.input_ids, inputs.ngram_context, request, chunk_pos
        )
        gated_value = self.gated_values(hidden_states, embeddings)
        conv_output, new_windows = dilated_short_conv(
            self.norm_conv(gated_value),
            self.conv1d.weight[:, 0, :],
            self.conv_dilation,
            conv_windows,
            inputs.query_start_loc,
            request,
            chunk_pos,
        )
        return gated_value + F.silu(conv_output), new_windows

    def gated_values(
        self, hidden_states: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """PLE values before convolution; storage and history remain caller-owned."""
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        dot = self.gate_dot(key, hidden_states, self.norm_key, self.norm_query)
        gate = dot / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        return gated_value

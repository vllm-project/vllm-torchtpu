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
"""Protocol definitions and configuration dataclasses for AOT model export."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@dataclasses.dataclass(frozen=True)
class ShapeDtypeStruct:
    """Abstract tensor specification carrying shape, dtype, and optional sharding."""

    shape: tuple[int, ...]
    dtype: Any
    sharding: Any = None

    def __init__(
        self,
        shape: Sequence[int],
        dtype: Any,
        sharding: Any = None,
    ) -> None:
        object.__setattr__(self, "shape", tuple(int(d) for d in shape))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "sharding", sharding)


@dataclasses.dataclass(frozen=True)
class ModelExportConfig:
    """General model export configuration."""

    model_name: str
    target_tpu: str = "v6e"
    num_replicas: int = 1
    num_partitions: int = 1
    use_spmd_partitioning: bool = False
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    @property
    def tensor_parallel_size(self) -> int:
        """Size of the serving mesh's 'tp' axis."""
        return self.num_replicas * self.num_partitions


@dataclasses.dataclass(frozen=True)
class TextOnlyLLMExportConfig(ModelExportConfig):
    """Text-only autoregressive LLM export configuration.

    Attributes:
        vocab_size: Vocabulary size (rows of the embedding / LM head).
        hidden_size: Model hidden dimension.
        num_layers: Number of decoder layers (one KV cache per layer).
        num_q_heads: Number of attention query heads.
        num_kv_heads: Number of attention key/value heads.
        head_dim: Dimension of each attention head.
        block_size: Tokens per paged KV cache block.
        num_blocks: Number of blocks in each layer's KV cache.
        max_num_seqs: Maximum number of concurrently scheduled sequences.
        max_model_len: Maximum context length per sequence.
        eos_token_id: End-of-sequence token id.
        dtype: Activation and KV cache dtype (e.g. 'bf16').
        token_buckets: Token bucket sizes; one forward_bucket{B}.mlir each.
        req_buckets: Request bucket sizes; one compute_logits_req{R}.mlir and
            sample_tokens_req{R}.mlir each.
        emit_byte_tokenizer: Whether to emit a synthetic byte-level tokenizer.json
            (valid only for vocab_size <= 256 in small test models).
        hf_checkpoint_dir: Optional path to the HuggingFace checkpoint directory
            containing tokenizer.json, generation_config.json, and safetensors
            headers.
    """

    vocab_size: int = 151936
    hidden_size: int = 1024
    num_layers: int = 28
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 16
    num_blocks: int = 512
    max_num_seqs: int = 64
    max_model_len: int = 2048
    eos_token_id: int = 151645
    dtype: str = "bf16"
    token_buckets: tuple[int, ...] = (16, 32, 64, 128, 256)
    req_buckets: tuple[int, ...] = (8, 16, 32, 64)
    emit_byte_tokenizer: bool = False
    hf_checkpoint_dir: str | None = None
    uses_mrope: bool = False

    def __post_init__(self) -> None:
        if self.emit_byte_tokenizer and self.vocab_size > 256:
            raise ValueError(
                f"emit_byte_tokenizer requires vocab_size <= 256, got {self.vocab_size}"
            )

    @property
    def max_num_pages(self) -> int:
        """Size of the flattened block table: max_num_seqs * pages per sequence."""
        pages_per_seq = max(
            1, (self.max_model_len + self.block_size - 1) // self.block_size
        )
        return self.max_num_seqs * pages_per_seq

    @property
    def max_cu_seqs(self) -> int:
        """Length of query_start_loc: one cumulative offset per sequence plus 1."""
        return self.max_num_seqs + 1


@runtime_checkable
class ExportableModel(Protocol):
    """Foundational exportable model protocol."""

    @property
    def pallas_interpret(self) -> bool:
        """When True, lowers Pallas kernels with interpret=True for CPU execution."""
        ...

    @property
    def export_config(self) -> ModelExportConfig:
        """Returns the model's export configuration."""
        ...

    def abstract_params(self) -> Mapping[str, Any]:
        """Ordered mapping of parameter name -> ShapeDtypeStruct."""
        ...


@runtime_checkable
class ExportableTextOnlyLLM(ExportableModel, Protocol):
    """Text-only autoregressive LLM export protocol.

    Models may optionally implement `abstract_states(self)`
    (with length equal to `export_config.num_layers`) to declare custom per-layer
    state tensors such as MLA compressed latent KV caches or recurrent linear-attention
    states. When `abstract_states` is not defined on the model, the exporter defaults
    to a uniform paged KV cache of shape `[num_blocks, block_size, num_kv_heads * 2,
    head_dim]` and dtype `export_config.dtype` for every layer.
    """

    @property
    def export_config(self) -> TextOnlyLLMExportConfig:
        """Returns the LLM's export configuration."""
        ...

    def forward(
        self,
        weights: Mapping[str, Any],
        kv_caches: Sequence[Any],
        *,
        input_ids: Any,
        positions: Any,
        kv_lens: Any,
        block_tables: Any,
        query_start_loc: Any,
        req_dist: Any,
    ) -> tuple[Sequence[Any], Any]:
        """Executes one flattened 1D continuous batching forward pass.

        Args:
            weights: Parameter arrays keyed by name (all params except the LM head).
            kv_caches: One per-layer state tensor per layer.
            input_ids: Token ids, i32[B].
            positions: Token positions, i32[B] or i32[K, B].
            kv_lens: KV length per sequence, i32[max_num_seqs].
            block_tables: Flattened per-sequence block tables, i32[max_num_pages].
            query_start_loc: Cumulative query offsets, i32[max_cu_seqs].
            req_dist: Request distribution metadata, i32[3].

        Returns:
            Tuple of (updated_kv_caches, hidden_states[B, hidden_size]).
        """
        ...

    def compute_logits(
        self,
        hidden_states: Any,
        lm_head_weight: Any,
    ) -> Any:
        """Computes logits[R, vocab_size] from hidden_states[R, hidden_size]."""
        ...

    def sample_tokens(
        self,
        logits: Any,
        *,
        temperature: Any | None = None,
        top_k: Any | None = None,
        top_p: Any | None = None,
        min_p: Any | None = None,
        uniform: Any | None = None,
        repetition_penalty: Any | None = None,
        presence_penalty: Any | None = None,
        frequency_penalty: Any | None = None,
        history: Any | None = None,
        output_start: Any | None = None,
    ) -> Any:
        """Samples token ids i64[R, 1] from vocabulary logits[R, vocab_size].

        When `temperature` is None, executes pure greedy sampling (`argmax`).
        When dynamic sampling parameters (`temperature`, `top_k`, `top_p`, `min_p`,
        `uniform`) or penalty parameters (`repetition_penalty`, `presence_penalty`,
        `frequency_penalty`, `history`, `output_start`) are passed, executes
        dynamic per-row stochastic or penalized sampling.
        """
        ...

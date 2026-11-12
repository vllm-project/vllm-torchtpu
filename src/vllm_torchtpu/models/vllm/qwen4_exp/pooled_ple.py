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
"""Native history and encoded-table transport around the common PLE layer."""

from collections.abc import Callable, Iterable
from functools import partial
from typing import cast

import torch
from transformers import Qwen4ExpTextConfig
from vllm.distributed import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
    build_host_lookup_ops,
    build_pooled_short_conv_op,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .parallel import PLEParallel
from .ple import PLEInputs, Qwen4ExpPLELayer, token_requests
from .ple_offload import host_lookup
from .ple_storage import HostPLETable, ResidentPLETable, SharedPLEOptions


class Qwen4ExpPooledPLE(Qwen4ExpPLELayer):
    """Reuse common parameters, hash, loader and gate math with native history.

    ``forward`` retains the common dense-window interface. Model integration
    explicitly calls ``forward_with_history`` after binding its owned cache.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        ple_layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        *,
        parallel: PLEParallel,
        table_group: GroupCoordinator,
        cpu_offload: bool = False,
        shared: SharedPLEOptions | None = None,
        native_host_dma: bool = False,
        num_speculative_tokens: int = 0,
    ) -> None:
        if (table_group.rank_in_group, table_group.world_size) != (
            parallel.rank,
            parallel.world_size,
        ):
            raise ValueError("PLE table group must match its TP/DP row ownership")
        if shared is not None and not cpu_offload:
            raise ValueError("Shared PLE storage requires host offload")
        factory = (
            partial(HostPLETable, parallel_group=table_group, shared=shared)
            if cpu_offload
            else partial(ResidentPLETable, parallel_group=table_group)
        )
        super().__init__(
            config, ple_layer_index, quant_config, prefix, table_factory=factory
        )
        self.parallel = parallel
        self.cpu_offload = cpu_offload
        self.native_host_dma = native_host_dma
        self.register_buffer("native_host_table", None, persistent=False)
        self._host_start: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None
        self._host_finish: (
            Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ]
            | None
        ) = None
        self.prefix = prefix
        self.cache_owners: tuple[torch.Tensor, ...] = ()
        self._loaded: set[str] = set()
        self._finalized = False
        self.num_speculative_tokens = num_speculative_tokens
        self._conv: Callable[..., torch.Tensor] | None = None

    @property
    def storage(self) -> ResidentPLETable | HostPLETable:
        table = self.ple_embedding.ngram_embedding
        assert isinstance(table, (ResidentPLETable, HostPLETable))
        return table

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.conv_state_len, self.hidden_size * self.hc_count

    def load_checkpoint(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._finalized:
            raise ValueError("PLE weights are already finalized")
        loaded = AutoWeightsLoader(self).load_weights(weights)
        self._loaded.update(loaded)
        return loaded

    def finalize_weights(self) -> None:
        if self._finalized:
            return
        required = set(dict(self.named_parameters())) | {
            "ple_embedding.layer_multipliers",
            "ple_embedding.ngram_heads_offsets",
            "ple_embedding.ngram_heads_vocab_sizes",
        }
        missing = required - self._loaded
        if missing:
            raise ValueError(f"Missing PLE checkpoint tensors: {sorted(missing)}")
        self.storage.finalize_weights()
        if self.native_host_dma:
            assert isinstance(self.storage, HostPLETable)
            device = self.ple_embedding.ngram_heads_offsets.device
            self.native_host_table = self.storage.allocate_native_host(device)
            self._host_start, self._host_finish = build_host_lookup_ops(
                self.storage.shard_indices.num_org_elements, self.parallel.world_size
            )
        self._finalized = True

    def bind_pool(self, pool: torch.Tensor) -> None:
        self.cache_owners = (pool,)
        self._conv = build_pooled_short_conv_op(self.conv_dilation, self.conv_state_len)

    def _lookup_ids(
        self, inputs: PLEInputs, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        if inputs.ngram_context.shape[0] == 0:
            # No local requests, but DP peers may still need this rank's table.
            ids = inputs.input_ids.new_full(
                (
                    inputs.input_ids.shape[0],
                    self.ple_embedding.ngram_heads_vocab_sizes.numel(),
                ),
                -1,
                dtype=torch.int32,
            )
        else:
            request, chunk_pos = token_requests(
                inputs.query_start_loc, inputs.input_ids.shape[0]
            )
            ids = self.ple_embedding.compute_ids(
                inputs.input_ids, inputs.ngram_context, request, chunk_pos
            )
            # Hash products use int64. Bounded row ids cross an int32 Pallas ABI.
            ids = ids.to(torch.int32)
            valid = torch.arange(ids.shape[0], device=ids.device) < valid_tokens
            ids = torch.where(valid[:, None], ids, -1)
        return self.parallel.gather_ids(ids)

    def _empty_raw(self, inputs: PLEInputs) -> torch.Tensor:
        width = (
            self.ple_embedding.ngram_embedding.embedding_dim
            * self.ple_embedding.ngram_heads_vocab_sizes.numel()
        )
        return inputs.input_ids.new_zeros(
            (inputs.input_ids.shape[0], width),
            dtype=torch.int8
            if self.storage.storage_dtype == torch.float8_e4m3fn
            else torch.int16,
        )

    def lookup_for_model(
        self, inputs: PLEInputs, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute ids on device, then cross the eager host lookup boundary."""
        assert self.cpu_offload
        if inputs.ngram_context.shape[0] == 0 and not self.parallel.across_dp:
            return self._empty_raw(inputs).unsqueeze(0)
        ids = self._lookup_ids(inputs, valid_tokens)
        return host_lookup(
            ids,
            inputs.input_ids.shape[0],
            self.ple_embedding.ngram_embedding.embedding_dim * ids.shape[-1],
            self.parallel.dp_group.world_size if self.parallel.across_dp else 1,
            self.storage.storage_dtype == torch.float8_e4m3fn,
            self.prefix,
        )

    def start_native_lookup(self, inputs: PLEInputs, valid_tokens: torch.Tensor):
        assert self._host_start is not None and self.native_host_table is not None
        ids = self._lookup_ids(inputs, valid_tokens)
        local_ids = ids - self.storage._vocab_start
        return self._host_start(self.native_host_table, local_ids), local_ids

    def finish_native_lookup(self, pending, ids, hidden):
        assert self._host_finish is not None
        raw, hidden = self._host_finish(pending, ids, hidden)
        return raw.reshape(1, ids.shape[0], -1), hidden

    def forward_with_history(
        self,
        hidden_states: torch.Tensor,
        inputs: PLEInputs,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        valid_tokens: torch.Tensor,
        raw_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self._finalized and self._conv is not None
        assert len(self.cache_owners) == 1
        if self.cpu_offload:
            assert raw_embeddings is not None, "Host PLE requires prepared raw rows"
            # Owner collectives may update the supplied storage in place.
            raw = raw_embeddings.reshape(-1, raw_embeddings.shape[-1]).clone()
        else:
            assert isinstance(self.storage, ResidentPLETable)
            raw = (
                self._empty_raw(inputs)
                if state_indices.shape[0] == 0 and not self.parallel.across_dp
                else self.storage.lookup_raw(self._lookup_ids(inputs, valid_tokens))
            )
        raw = self.parallel.combine(raw, hidden_states.shape[0])
        if raw.dtype == torch.int32:
            raw = raw.view(torch.int8)
        # Keep owner collectives above the local-empty return, including on DP.
        if state_indices.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        gated = self.gated_values(hidden_states, self.storage.decode(raw))
        state_kwargs = {}
        if getattr(self, "num_speculative_tokens", 0) > 0:
            per_layer = cast(
                dict[str, AttentionMetadata], get_forward_context().attn_metadata
            )
            from .pool_layout import resolve_ple_companion

            companion = resolve_ple_companion(
                self.prefix,
                [k for k in per_layer if k.endswith(".linear_attn")],
            )
            metadata = (
                per_layer[companion]
                if companion is not None and companion in per_layer
                else per_layer[self.prefix]
            )
            assert metadata.mamba_ckpt_indices is not None
            assert metadata.mamba_slot_read_offsets is not None
            assert metadata.mamba_request_distribution is not None
            state_kwargs = dict(
                checkpoint_indices=metadata.mamba_ckpt_indices,
                read_offsets=metadata.mamba_slot_read_offsets,
                request_distribution=metadata.mamba_request_distribution,
            )
        conv = self._conv(
            self.norm_conv(gated),
            self.conv1d.weight[:, 0],
            self.cache_owners[0],
            inputs.query_start_loc,
            state_indices,
            has_initial_states,
            **state_kwargs,
        )
        request, _ = token_requests(inputs.query_start_loc, hidden_states.shape[0])
        valid = (
            torch.arange(hidden_states.shape[0], device=hidden_states.device)
            < valid_tokens
        ) & (state_indices[request] > 0)
        return torch.where(valid[:, None], gated + conv, 0)

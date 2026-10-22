# SPDX-License-Identifier: Apache-2.0
"""Native history and encoded-table transport around the common PLE layer."""

from collections.abc import Callable, Iterable
from functools import partial

import torch
from vllm.distributed import GroupCoordinator
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
    build_pooled_short_conv_op,
    build_short_conv_op,
)

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
        self.prefix = prefix
        self.cache_owners: tuple[torch.Tensor, ...] = ()
        self._checkpoint_loader: AutoWeightsLoader | None = None
        self._loaded: set[str] = set()
        self._finalized = False
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
        if self._checkpoint_loader is None:
            self._checkpoint_loader = AutoWeightsLoader(self)
        loader = self._checkpoint_loader
        # Upstream retains per-call diagnostics and appends suffix filters.
        loader.ignore_unexpected_suffixes.clear()
        loader._skipped_aliases.clear()
        loader._loaded_params_are_complete = True
        loaded = loader.load_weights(weights)
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
        self._checkpoint_loader = None
        self._finalized = True

    def bind_history(self, history: torch.Tensor) -> None:
        if history.ndim != 3 or history.shape[1:] != self.state_shape:
            raise ValueError(f"PLE history must have shape [slots,{self.state_shape}]")
        if history.dtype != torch.bfloat16:
            raise ValueError("PLE history must use BF16")
        self.cache_owners = (history,)
        self._conv = build_short_conv_op(self.conv_dilation)

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
            # Own narrow-type inputs before in-place collective dtype views.
            raw = raw_embeddings.reshape(
                -1,
                self.ple_embedding.ngram_embedding.embedding_dim
                * (self.ple_embedding.ngram_size - 1)
                * self.ple_embedding.heads_per_ngram,
            ).clone()
        else:
            assert isinstance(self.storage, ResidentPLETable)
            raw = (
                self._empty_raw(inputs)
                if state_indices.shape[0] == 0 and not self.parallel.across_dp
                else self.storage.lookup_raw(self._lookup_ids(inputs, valid_tokens))
            )
        raw = self.parallel.combine(raw, hidden_states.shape[0])
        # Keep owner collectives above the local-empty return, including on DP.
        if state_indices.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        gated = self.gated_values(hidden_states, self.storage.decode(raw))
        conv = self._conv(
            self.norm_conv(gated),
            self.conv1d.weight[:, 0],
            self.cache_owners[0],
            inputs.query_start_loc,
            state_indices,
            has_initial_states,
        )
        request, _ = token_requests(inputs.query_start_loc, hidden_states.shape[0])
        valid = (
            torch.arange(hidden_states.shape[0], device=hidden_states.device)
            < valid_tokens
        ) & (state_indices[request] >= 0)
        return torch.where(valid[:, None], gated + conv, 0)

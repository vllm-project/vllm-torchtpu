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
    build_pooled_speculative_short_conv_op,
    build_short_conv_op,
)

from .parallel import PLEParallel
from .ple import PLEInputs, Qwen4ExpPLELayer, token_requests
from .ple_offload import host_lookup, host_lookup_finish, host_lookup_start
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
        self._loaded: set[str] = set()
        self._finalized = False
        self._conv: Callable[..., torch.Tensor] | None = None
        self._speculative_conv: Callable[..., torch.Tensor] | None = None

    @property
    def storage(self) -> ResidentPLETable:
        table = self.ple_embedding.ngram_embedding
        assert isinstance(table, ResidentPLETable)
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
        # Registered before Dynamo traces forward, like the ordinary op.
        self._speculative_conv = build_pooled_speculative_short_conv_op(
            self.conv_dilation, self.conv_state_len
        )

    def _lookup_ids(
        self, inputs: PLEInputs, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        request, chunk_pos = token_requests(
            inputs.query_start_loc, inputs.input_ids.shape[0]
        )
        ids = self.ple_embedding.compute_ids(
            inputs.input_ids, inputs.ngram_context, request, chunk_pos
        )
        # Hash products use int64. The bounded row ids cross an int32 Pallas ABI.
        ids = ids.to(torch.int32)
        valid = torch.arange(ids.shape[0], device=ids.device) < valid_tokens
        return self.parallel.gather_ids(torch.where(valid[:, None], ids, -1))

    def lookup_for_model(
        self, inputs: PLEInputs, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute ids on device, then cross the eager host lookup boundary."""
        assert self.cpu_offload
        ids = self._lookup_ids(inputs, valid_tokens)
        return host_lookup(
            ids,
            inputs.input_ids.shape[0],
            self.ple_embedding.ngram_embedding.embedding_dim * ids.shape[-1],
            self.parallel.dp_group.world_size if self.parallel.across_dp else 1,
            self.storage.storage_dtype == torch.float8_e4m3fn,
            self.prefix,
        )

    def lookup_start(self, inputs: PLEInputs, valid_tokens: torch.Tensor) -> torch.Tensor:
        """Device ids, then start their copy to the host (see ple_offload)."""
        assert self.cpu_offload
        return host_lookup_start(self._lookup_ids(inputs, valid_tokens), self.prefix)

    def lookup_finish(
        self, token: torch.Tensor, inputs: PLEInputs, after: torch.Tensor
    ) -> torch.Tensor:
        e = self.ple_embedding
        parts = (e.ngram_size - 1) * e.heads_per_ngram
        return host_lookup_finish(
            token,
            after,
            inputs.input_ids.shape[0],
            e.ngram_embedding.embedding_dim * parts,
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
        speculative: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """``speculative`` is (read_ids, ckpt_ids, windowed) on the unified pool
        with speculative decoding; see ``pooled_short_conv_speculative``.

        With ``sequence_parallel`` the hidden states and the returned rows are
        this TP rank's token shard (see ``sequence_parallel``); the lookup,
        values and the convolution still cover every token of the step.
        """
        num_tokens = (
            inputs.input_ids.shape[0] if sequence_parallel else hidden_states.shape[0]
        )
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
            raw = self.storage.lookup_raw(self._lookup_ids(inputs, valid_tokens))
        raw = self.parallel.combine(raw, num_tokens)
        gated = self.gated_values(
            hidden_states,
            self.storage.decode(raw),
            num_tokens if sequence_parallel else None,
        )
        if speculative is None:
            conv = self._conv(
                self.norm_conv(gated),
                self.conv1d.weight[:, 0],
                self.cache_owners[0],
                inputs.query_start_loc,
                state_indices,
                has_initial_states,
            )
        else:
            assert self._speculative_conv is not None, "bind the pool first"
            read_ids, ckpt_ids, windowed = speculative
            conv = self._speculative_conv(
                self.norm_conv(gated),
                self.conv1d.weight[:, 0],
                self.cache_owners[0],
                inputs.query_start_loc,
                state_indices,
                has_initial_states,
                read_ids,
                ckpt_ids,
                windowed,
            )
        if state_indices.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        request, _ = token_requests(inputs.query_start_loc, num_tokens)
        valid = (
            torch.arange(num_tokens, device=hidden_states.device) < valid_tokens
        ) & (state_indices[request] >= 0)
        output = torch.where(valid[:, None], gated + conv, 0)
        if sequence_parallel:
            from . import sequence_parallel as sp

            output = sp.shard(output)
        return output

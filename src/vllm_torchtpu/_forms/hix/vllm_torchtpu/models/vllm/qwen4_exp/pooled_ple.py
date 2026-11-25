# SPDX-License-Identifier: Apache-2.0
"""Native history and encoded-table transport around the common PLE layer."""

from collections.abc import Callable, Iterable
from functools import partial

import torch
from vllm.distributed import GroupCoordinator
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
    build_pooled_fused_ple_op,
    build_pooled_short_conv_op,
    build_pooled_speculative_short_conv_op,
    build_short_conv_op,
)

from . import pdense_engagement
from .parallel import PLEParallel
from .ple import PLEInputs, Qwen4ExpPLELayer, ngram_ids, token_requests
from .ple_offload import host_lookup, host_lookup_finish, host_lookup_start
from .ple_storage import HostPLETable, ResidentPLETable, SharedPLEOptions


_FOLD_NOTED: list = []


def _note_fold(prefix: str, bounded: int) -> None:
    """Bind-time engagement line of VLLM_TPU_PLE_D14_FOLD, once per process (q38 L1-PLE)."""
    if _FOLD_NOTED:
        return
    _FOLD_NOTED.append(prefix)
    from vllm_torchtpu.logger import init_logger

    if bounded:
        init_logger(__name__).info(
            "q38 PLE-D14-FOLD on: %s bounded conv op (VLLM_TPU_PLE_DECODE_BOUNDED=%d) built "
            "with the fused conv + checkpoint program",
            prefix,
            bounded,
        )
    else:
        init_logger(__name__).warning(
            "q38 PLE-D14-FOLD inert: %s has no bounded conv op (VLLM_TPU_PLE_DECODE_BOUNDED=0)",
            prefix,
        )


_WRDEFER_NOTED: list = []


def _note_wrdefer(prefix: str, depth: str) -> None:
    """Bind-time engagement line of VLLM_TPU_PLE_WRDEFER, once per process (q38 L1 PLE-WRDEFER)."""
    if _WRDEFER_NOTED:
        return
    _WRDEFER_NOTED.append(prefix)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PLE-WRDEFER on: %s PLE checkpoint writer keeps %s DMAs in flight "
        "(VLLM_TPU_PLE_WRDEFER; D14 fold buckets write inside the fold program)",
        prefix,
        depth,
    )


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
        # VLLM_TPU_PLE_DECODE_BOUNDED (q38 L1 D07-b2): the uniform-window op.
        self._speculative_conv_bounded: Callable[..., torch.Tensor] | None = None
        self._fused: Callable[..., torch.Tensor] | None = None
        self._fused_tile = 0

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
        self._conv = build_pooled_short_conv_op(
            self.conv_dilation,
            self.conv_state_len,
            "mutant"
            if envs.Q38_PROBE_PLE_ONEHOT_MUTANT
            else bool(envs.VLLM_TPU_PLE_CONV_ONEHOT),
        )
        # Registered before Dynamo traces forward, like the ordinary op.
        self._speculative_conv = build_pooled_speculative_short_conv_op(
            self.conv_dilation, self.conv_state_len
        )
        # q38 L1 D07-b2: a second op for callers that guarantee uniform verify
        # windows (forward_with_history's ``uniform_windows``); unset = None.
        bounded = int(envs.VLLM_TPU_PLE_DECODE_BOUNDED or "0")
        # q38 L1-GLUEFIX: VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS (0 = the D07-b2 op).
        min_rows = int(envs.VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS or 0) if bounded else 0
        # q38 L1-PLE (PLE-INTO-D14): VLLM_TPU_PLE_D14_FOLD runs the bounded op's conv
        # and checkpoint writes as one Pallas program; it needs the bounded op.
        fold = bool(envs.VLLM_TPU_PLE_D14_FOLD) and bounded > 0
        if envs.VLLM_TPU_PLE_D14_FOLD:
            _note_fold(self.prefix, bounded)
        # q38 L1 PLE-WRDEFER: pool.write_region reads VLLM_TPU_PLE_WRDEFER at trace time.
        if (envs.VLLM_TPU_PLE_WRDEFER or "0") != "0":
            _note_wrdefer(self.prefix, envs.VLLM_TPU_PLE_WRDEFER)
        self._speculative_conv_bounded = (
            build_pooled_speculative_short_conv_op(
                self.conv_dilation, self.conv_state_len, bounded,
                *((min_rows,) if min_rows or fold else ()), *((True,) if fold else ()),  # q38 STACK-D v5: (min_rows, fold)
            )
            if bounded
            else None
        )
        self._fused = None
        # The TPU linear method stores dense weights flipped to [in, out] after loading
        # (WEIGHT_FLIPPED_ATTR); the fused kernels must contract the matching axis
        # (srv1-F re-run j-a8ea9be6 failed on the [out, in] assumption).
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        key_kn = bool(getattr(self.key_proj, WEIGHT_FLIPPED_ATTR, False))
        value_kn = bool(getattr(self.value_proj, WEIGHT_FLIPPED_ATTR, False))
        if envs.VLLM_TPU_PLE_FUSED and key_kn == value_kn:
            self._fused_tile = envs.VLLM_TPU_PLE_FUSED_TILE
            self._fused = build_pooled_fused_ple_op(
                self.hc_count,
                self.hidden_size,
                float(self.norm_conv.variance_epsilon),
                self.conv_dilation,
                self.conv_state_len,
                self._fused_tile,
                envs.VLLM_TPU_PLE_FUSED_TAP_MODE,
                envs.VLLM_TPU_PLE_FUSED_NORM,
                envs.VLLM_TPU_PLE_FUSED_PRECISION,
                key_kn,
            )
        # Host-side P-DENSE engagement (VLLM_TPU_PDENSE_COUNT; outside every traced forward): what this layer built.
        fused_ok = (
            self._fused is not None
            and self.key_proj.weight.dtype == torch.bfloat16
            and self.value_proj.weight.dtype == torch.bfloat16
        )
        pdense_engagement.register_ple(
            self,
            bool(envs.VLLM_TPU_PLE_CONV_ONEHOT) or bool(envs.Q38_PROBE_PLE_ONEHOT_MUTANT),
            self._fused_tile if fused_ok else 0,
        )

    def _fused_applies(
        self, num_tokens: int, speculative, sequence_parallel: bool, states: int
    ) -> bool:
        """The fused prefill path: whole token tiles, no speculation, no SP,
        bf16 projections; every other step keeps the served ops."""
        return (
            self._fused is not None
            and speculative is None
            and not sequence_parallel
            and states > 0
            and pdense_engagement.fused_rows(num_tokens, self._fused_tile)
            and self.key_proj.weight.dtype == torch.bfloat16
            and self.value_proj.weight.dtype == torch.bfloat16
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

    def host_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """``_lookup_ids`` on host (CPU) copies of the step's inputs, for
        VLLM_TPU_PLE_HOST_IDS (see ple_offload): the same functions and integer
        arithmetic as the device graph, so the rows are identical. None when the
        ids would need a cross-DP gather (the device path then runs)."""
        if self.parallel.across_dp:
            return None
        e = self.ple_embedding
        buffers = self.__dict__.get("_host_hash_buffers")
        if buffers is None:
            # One copy of the loaded hash constants (a one-time device read).
            buffers = (
                e.layer_multipliers.cpu(),
                e.ngram_heads_vocab_sizes.cpu(),
                e.ngram_heads_offsets.cpu(),
            )
            self._host_hash_buffers = buffers
        if ngram_context.shape[-1] != e.ngram_size - 1:
            raise ValueError(
                f"ngram_context has {ngram_context.shape[-1]} tokens per request; "
                f"ngram_size={e.ngram_size} needs {e.ngram_size - 1}."
            )
        request, chunk_pos = token_requests(query_start_loc, input_ids.shape[0])
        ids = ngram_ids(
            input_ids,
            ngram_context,
            request,
            chunk_pos,
            *buffers,
            eos_token_id=e.eos_token_id,
            heads_per_ngram=e.heads_per_ngram,
        ).to(torch.int32)
        valid = torch.arange(ids.shape[0]) < valid_tokens
        return torch.where(valid[:, None], ids, -1)

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
        uniform_windows: bool = False,
    ) -> torch.Tensor:
        """``speculative`` is (read_ids, ckpt_ids, windowed) on the unified pool
        with speculative decoding; see ``pooled_short_conv_speculative``.

        ``uniform_windows``: the caller guarantees that every live request of
        the step holds exactly one verify window (num_spec + 1 tokens) and that
        they come first, as the Qwen3.8 megakernel driver does before it runs
        its programs. With VLLM_TPU_PLE_DECODE_BOUNDED set, the speculative
        conv then runs the bounded op (q38 L1 D07-b2); otherwise it is ignored.

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
        if self._fused_applies(
            num_tokens, speculative, sequence_parallel, state_indices.shape[0]
        ):
            from vllm_torchtpu.models.vllm.qwen4_exp import pdense_engagement

            pdense_engagement.record_ple("ple_fused", num_tokens)
            return self._fused(
                hidden_states,
                self.storage.decode(raw),
                self.key_proj.weight,
                self.value_proj.weight,
                # norm_key before norm_query: the served step first uses them in this order, and a token
                # bucket handed between the fused and served traces is called positionally (fix 4).
                self.norm_key.weight,
                self.norm_query.weight,
                self.norm_conv.weight,
                # The whole conv1d weight [C, 1, taps], sliced inside the op: a torch-side
                # [:, 0] would be the first graph use of this parameter, so the fused
                # trace would lift it before the projection weights while the served
                # trace lifts it after the norms. Under shape-variant compilation the two
                # traces hand token buckets to each other only when their subgraph inputs
                # match in order (srv1-F3 j-afef8265: ShapeSpecializationError).
                self.conv1d.weight,
                self.cache_owners[0],
                inputs.query_start_loc,
                state_indices,
                has_initial_states,
                valid_tokens,
            )
        gated = self.gated_values(
            hidden_states,
            self.storage.decode(raw),
            num_tokens if sequence_parallel else None,
        )
        if speculative is None:
            if envs.VLLM_TPU_PLE_CONV_ONEHOT or envs.Q38_PROBE_PLE_ONEHOT_MUTANT:
                from vllm_torchtpu.models.vllm.qwen4_exp import pdense_engagement

                if pdense_engagement.ENABLED:
                    pdense_engagement.record_ple("ple_onehot", num_tokens)
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
            spec_conv = self._speculative_conv
            if uniform_windows and self._speculative_conv_bounded is not None:
                spec_conv = self._speculative_conv_bounded
            conv = spec_conv(
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

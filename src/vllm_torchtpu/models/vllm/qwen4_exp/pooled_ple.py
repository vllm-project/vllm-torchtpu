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
        # q38 PLE-CS: (streams, split, TP rank) with VLLM_TPU_PLE_CHANNEL_SHARD, else None.
        self._cs = channel_shard_spec(
            parallel.tp_group.rank_in_group,
            parallel.tp_group.world_size,
            self.hc_count,
            self.hidden_size,
        )

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
        if self._cs is not None:
            raise ValueError("VLLM_TPU_PLE_CHANNEL_SHARD needs the unified pool")
        self.cache_owners = (history,)
        self._conv = build_short_conv_op(self.conv_dilation)

    def bind_pool(self, pool: torch.Tensor) -> None:
        self.cache_owners = (pool,)
        self._conv = build_pooled_short_conv_op(self.conv_dilation, self.conv_state_len)
        # Registered before Dynamo traces forward, like the ordinary op.
        self._speculative_conv = build_pooled_speculative_short_conv_op(
            self.conv_dilation, self.conv_state_len
        )
        if self._cs is not None:
            self._bind_channel_shard(pool)

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
        if self._cs is not None:
            conv = self._channel_shard_conv(
                self.norm_conv(gated),
                inputs.query_start_loc,
                state_indices,
                has_initial_states,
                speculative,
                sequence_parallel,
            )
        elif speculative is None:
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

    def _bind_channel_shard(self, pool: torch.Tensor) -> None:
        """q38 PLE-CS: this rank's stream operand and the stream ops (see ``channel_shard_spec``)."""
        from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
            build_pooled_short_conv_cs_op,
            build_pooled_speculative_short_conv_cs_op,
        )

        streams, split, rank = self._cs
        # An operand, not a constant: every TP rank traces one graph (as vocab_start).
        self.register_buffer(
            "cs_rank",
            torch.tensor([rank], dtype=torch.int32, device=pool.device),
            persistent=False,
        )
        args = (self.conv_dilation, self.conv_state_len, streams, split)
        self._cs_conv = build_pooled_short_conv_cs_op(*args)
        self._cs_speculative_conv = build_pooled_speculative_short_conv_cs_op(*args)
        channels = self.hidden_size * self.hc_count
        width = channels // streams
        stream, part = divmod(rank, split)
        _logger().info(
            "q38 PLE-CS bound: %s TP rank %d computes HC stream %d (channels %d-%d of %d), "
            "all-gathers part %d of %d (%d channels) over TP %d",
            self.prefix,
            rank,
            stream,
            stream * width,
            (stream + 1) * width - 1,
            channels,
            part,
            split,
            width // split,
            streams * split,
        )

    def _channel_shard_conv(
        self,
        values: torch.Tensor,
        starts: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        speculative: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        """q38 PLE-CS: the conv of this rank's HC stream, its piece all-gathered over TP in
        rank order; bitwise the stock op's output (kernels/qwen4_exp/pool.py)."""
        if sequence_parallel:
            raise ValueError("VLLM_TPU_PLE_CHANNEL_SHARD does not run sequence-parallel steps")
        weight = self.conv1d.weight[:, 0]
        if speculative is None:
            piece = self._cs_conv(
                values,
                weight,
                self.cache_owners[0],
                starts,
                state_indices,
                has_initial_states,
                self.cs_rank,
            )
        else:
            read_ids, ckpt_ids, windowed = speculative
            piece = self._cs_speculative_conv(
                values,
                weight,
                self.cache_owners[0],
                starts,
                state_indices,
                has_initial_states,
                read_ids,
                ckpt_ids,
                windowed,
                self.cs_rank,
            )
        return self.parallel.tp_group.all_gather(piece, dim=-1)


def _logger():
    from vllm_torchtpu.logger import init_logger

    return init_logger(__name__)


def channel_shard_spec(
    tp_rank: int, tp_size: int, hc_count: int, hidden_size: int, cfg=None
) -> tuple[int, int, int] | None:
    """q38 PLE-CS: (streams, split, tp_rank) for VLLM_TPU_PLE_CHANNEL_SHARD, or None when unset.

    Rank r runs the PLE conv and its checkpoint windows on HC stream r // split (split = TP /
    streams) and contributes part r % split of that stream's output to the TP all-gather, whose
    rank order is then the channel order. A rank writes only its stream's checkpoint tokens, so
    the flag is refused (ValueError) on every engine but a PD decode engine (kv_role
    kv_consumer: its PLE pages arrive whole from prefill and no other engine reads them back) and
    in every mode where a PLE conv call could bypass the stream path: prefill-context
    parallelism, prefill sequence parallelism, VLLM_TPU_PLE_UNIFORM_COND, the HND pool layout,
    the decode megakernel. ``cfg`` defaults to the current vLLM config (read only when set).

    VLLM_TPU_PLE_CHANNEL_SHARD_BENCH_ADMIT=1 (STEP bench cars only) also admits an engine with
    no KV connector at all (``kv_transfer_config`` is None), such as the STEP bench's decode
    engine, which prefills its own requests: every PLE conv call of that engine, its local
    prefill chunks included, runs the stream ops, so each rank writes and later reads only its
    own stream, and no connector moves PLE pages to or from another engine. Every other refusal
    stays. On an engine that has a ``kv_transfer_config`` the key is a no-op (it is not read:
    the same admission or refusal as without it), and so it is without the flag.
    """
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp.pool import channel_shard_streams

    streams = channel_shard_streams()
    if not streams:
        return None
    if cfg is None:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
    from types import SimpleNamespace

    from vllm import envs as vllm_envs

    from vllm_torchtpu.megakernel_runtime import _is_megakernel_requested

    from .config import pd_transfer_config

    reasons = []
    if streams != hc_count:
        reasons.append(f"it splits the {hc_count} HC streams, not {streams}")
    # Bench admission: no KV connector at all, then the key (never read on a connector engine).
    bench = getattr(cfg, "kv_transfer_config", None) is None and bool(
        envs.VLLM_TPU_PLE_CHANNEL_SHARD_BENCH_ADMIT
    )
    transfer = pd_transfer_config(cfg)
    if not bench and (transfer is None or transfer.kv_role != "kv_consumer"):
        reasons.append("it runs only on a PD decode engine (kv_role kv_consumer)")
    if int(getattr(cfg.parallel_config, "prefill_context_parallel_size", 1) or 1) > 1:
        reasons.append("prefill context parallelism is on")
    if envs.QWEN4_EXP_PREFILL_SP:
        reasons.append("QWEN4_EXP_PREFILL_SP is set")
    if envs.VLLM_TPU_PLE_UNIFORM_COND:
        reasons.append("VLLM_TPU_PLE_UNIFORM_COND is set")
    if vllm_envs.VLLM_KV_CACHE_LAYOUT == "HND":
        reasons.append("the HND pool layout is not supported")
    if _is_megakernel_requested(SimpleNamespace(vllm_config=cfg)):
        reasons.append("the decode megakernel runs its own PLE")
    split = tp_size // streams if streams and tp_size % streams == 0 else 0
    if not split:
        reasons.append(f"TP {tp_size} is not a multiple of {streams} streams")
    elif hidden_size % split or (hidden_size // split) % 128:
        reasons.append(f"a stream of {hidden_size} channels does not split {split} ways in lanes")
    if reasons:
        raise ValueError(
            f"VLLM_TPU_PLE_CHANNEL_SHARD={streams} refused: " + "; ".join(reasons)
        )
    if bench:
        _logger().info(
            "q38 PLE-CS bench admission: TP rank %d, no KV connector "
            "(VLLM_TPU_PLE_CHANNEL_SHARD_BENCH_ADMIT=1): local prefill and decode run the "
            "stream ops; STEP bench only, never a served car",
            tp_rank,
        )
    return streams, split, tp_rank

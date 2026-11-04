"""Upstream Engram hashing, request history and TPU embedding lookup.

Shared by the v1 and v2 DeepSeek V4.1 implementations. Each implementation
subclasses ``NgramHashState`` to bind its own slot-cache write op and, when
its SWA cache module differs, the cache lifetime in ``ensure_cache``.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.models.deepseek_v41.common import engram as upstream

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.upstream_bindings import (
    bind_device_ops,
    device_hooks,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.model_loader_patches import register_row_slice

from ..config import engram_cpu_offload

if TYPE_CHECKING:
    from .host_channel import HostChannel

logger = init_logger(__name__)


class NgramHashState(upstream.NgramHashState):
    """Keep upstream tokenizer/layout/cache lifetime; replace Triton hashing."""

    def write_cache(
        self,
        cache: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Write mapped token ids into the slot cache; bound per implementation."""
        raise NotImplementedError

    def ensure_cache(self) -> bool:
        if torch.compiler.is_compiling():
            # The outer CausalLM initializes the cache before tracing.
            # Layer views inside the graph are temporary Python tensor objects;
            # comparing their weakrefs must not reset persistent token history.
            return (
                self._cache is not None
                if self.use_slot_cache
                else self.swa_cache_module.kv_cache.numel() > 0
            )
        kv_cache = self.swa_cache_module.kv_cache
        if self.use_slot_cache and kv_cache.numel():
            # Hybrid allocation can grow the physical SWA page beyond the
            # block size copied at model construction. Match the bound view
            # used by slot_mapping and block_table before upstream sizes its
            # generated-token history cache.
            block_size = kv_cache.shape[1]
            if self.block_size != block_size:
                self.block_size = block_size
                self._kv_cache_ref = None
        return super().ensure_cache()

    def forward(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        slot_mapping,
        block_table,
    ):
        count = input_ids.numel()
        layers, ngram = self.multipliers.shape
        if count == 0:
            return input_ids.new_empty(
                (0, layers, (ngram - 1) * self.primes.shape[-1]), dtype=torch.int32
            )
        if slot_mapping is not None:
            slot_mapping = slot_mapping[:count]
        if self.use_slot_cache:
            assert self._cache is not None and slot_mapping is not None
            mapped = self.token_map[input_ids.long()].masked_fill(
                dead_mask, upstream.DEAD_ID
            )
            valid = slot_mapping >= 0
            self.write_cache(self._cache, slot_mapping, mapped, valid)

        def fallback(past, reqs, window):
            if not self.use_slot_cache:
                return torch.full_like(window, self.pad_id)
            assert self._cache is not None
            logical = past.clamp(
                min=0, max=block_table.shape[1] * self.block_size - 1
            ).long()
            blocks = block_table[reqs, logical // self.block_size].long()
            slots = (blocks * self.block_size + logical % self.block_size).clamp(
                min=0, max=self._cache.numel() - 1
            )
            return self._cache[slots]

        tables = (self.token_map, self.multipliers, self.primes, self.offsets)
        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
            fallback,
        )

    def _hash(
        self,
        tables,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        fallback,
    ):
        token_map, multipliers, primes, offsets = tables
        count = input_ids.numel()
        _, ngram = multipliers.shape
        dev = input_ids.device
        rows = torch.arange(count, device=dev)
        reqs = torch.bucketize(rows, query_start_loc[1:], right=True).clamp(
            max=lookback_token_ids.shape[0] - 1
        )
        starts = query_start_loc[reqs].long().clamp(max=count - 1)
        chunk_positions = positions[starts]
        shifts = torch.arange(ngram, device=dev)
        past = positions[:, None] - shifts
        in_chunk = past >= chunk_positions[:, None]
        indices = (rows[:, None] - shifts).clamp(min=0)
        mapped_batch = token_map[input_ids.long()].masked_fill(
            dead_mask, upstream.DEAD_ID
        )
        batch = mapped_batch[indices]
        cols = chunk_positions[:, None] - 1 - past
        in_window = ~in_chunk & (cols >= 0) & (cols < self.lookback_depth)
        cols = cols.clamp(min=0, max=self.lookback_depth - 1).long()
        reqs_col = reqs[:, None]
        window_ids = lookback_token_ids[reqs_col, cols]
        known = in_window & (window_ids >= 0)
        mapped_lookback = token_map[lookback_token_ids.clamp(min=0).long()].masked_fill(
            lookback_dead_mask, upstream.DEAD_ID
        )
        window = mapped_lookback[reqs_col, cols]
        source = torch.where(
            in_chunk,
            batch,
            torch.where(known, window, fallback(past, reqs_col, window)),
        ).long()
        blocked = ((past < 0) | (source == upstream.DEAD_ID)).cumsum(dim=1) > 0
        source = source.masked_fill(blocked, self.pad_id)
        terms = source[:, None, :] * multipliers[None, :, :]
        rolling = terms[:, :, 0]
        rolling_steps = []
        for shift in range(1, ngram):
            rolling = rolling ^ terms[:, :, shift]
            rolling_steps.append(rolling)
        rolled = torch.stack(rolling_steps, dim=2)
        hashed = (rolled[:, :, :, None] % primes[None, :, :, :]).flatten(2) + offsets[
            None, :, :
        ]
        return hashed.to(torch.int32)


def _load_local_rows(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Upstream's head-shard loader for a tensor holding only this rank's rows."""
    if param.dtype == torch.uint8 or loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    else:
        loaded_weight = loaded_weight.to(param.dtype)
    assert loaded_weight.shape == param.shape, (
        f"engram shard {tuple(loaded_weight.shape)} does not fit param "
        f"{tuple(param.shape)}"
    )
    param.data.copy_(loaded_weight)


class ParallelEngramEmbedding(upstream.ParallelEngramEmbedding):
    """Reuse upstream head sharding and checkpoint loaders; support Host DRAM.

    Host-offloaded shards stay in CPU memory and are served by a native
    Host channel (``host_channel.py``) or, for ``TPU_DEEPSEEK_V41_ENGRAM_HBM``
    layers, by an HBM copy (``engram_hbm.py``).
    """

    @staticmethod
    def _weight_loader(param, loaded_weight):
        loader = upstream.ParallelEngramEmbedding._weight_loader
        if loaded_weight.shape[0] == param.shape[0]:
            # Already this rank's rows: the sharded Run:AI stream fetches only
            # them (see Engram._create_embedding). A full table has as many
            # rows only with a single shard, where both loaders agree.
            loader = _load_local_rows
        if param.device.type != "tpu":
            return loader(param, loaded_weight)
        from vllm_torchtpu.model_loader_patches import ExpertParamStager

        # Reuse upstream's shard selection and scale-byte interpretation.
        # A host stand-in detaches the shard from the giant mapped table;
        # the existing TPU stager bounds uploads and synchronizes publication.
        stager = ExpertParamStager()
        loader(stager.host_param(param), loaded_weight)
        stager.flush(param)

    def __init__(self, *args, cpu_offload: bool | None = None, **kwargs):
        # Upstream's own `cpu_offload` means a pinned table read through UVA
        # (CUDA only) and it resets the attribute in `__init__`; keep it False
        # and track the TPU host table separately.
        if cpu_offload is None:
            config = get_current_vllm_config_or_none()
            cpu_offload = (
                engram_cpu_offload(config)
                if config is not None
                else envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD
            )
        self.host_offload = bool(cpu_offload)
        # Upstream asks for the SM count solely to size its Triton lookup grid.
        # The TPU lookup below does not use that launch property.
        with device_hooks(
            torch.cuda,
            get_device_properties=lambda _: SimpleNamespace(multi_processor_count=0),
        ):
            super().__init__(*args, cpu_offload=False, **kwargs)
        upstream.set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        if self.weight_scale_inv is not None:
            upstream.set_weight_attrs(
                self.weight_scale_inv, {"dummy_weight_value": 127}
            )

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.host_offload:
            return super()._allocate_weights()
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
            ),
        )

    def lookup(self, indices, out, background=False):
        weight, scales = self._storage()
        work_device = weight.device
        if indices.device != work_device:
            indices = indices.to(work_device)
        heads = (
            torch.arange(self.part_n_hash_cols, device=work_device) + self.head_start
        )
        selected = indices[:, heads.clamp(max=self.n_hash_cols - 1)].long()
        valid = (
            (heads < self.n_hash_cols)[None, :]
            & (selected >= self.vocab_start_idx)
            & (selected < self.vocab_end_idx)
        )
        rows = (selected - self.vocab_start_idx).clamp(
            min=0, max=self.part_num_embeddings - 1
        )
        values = weight[rows].float()
        scale = (
            (scales[rows].float() - 127).exp2().repeat_interleave(self.block_size, -1)
        )
        out.copy_((values * scale).masked_fill(~valid[:, :, None], 0).to(out.dtype))


# E8M0 scale byte for 2**0. Scale bytes encode 2**(byte - 127).
_UNIT_SCALE = 127
# E8M0 byte 255 is NaN and never appears in a real table; it marks rows
# outside the shard and padded heads, which dequantize to zero.
_INVALID_SCALE = 255


def dequant_fp8_rows(raw, scale, block_size):
    """``[..., dim]`` FP8 bytes x ``[..., dim // block_size]`` E8M0 bytes -> bf16.

    Matches ``ParallelEngramEmbedding.lookup``'s ``values.float() *
    exp2(scale - 127)`` rounded once to bf16. The power of two is built from
    its float32 exponent bits (``byte << 23``), which is exact for bytes
    1..254 and does not depend on the backend's ``exp2`` accuracy; the product
    of a float32 and a power of two is then exact, so only the final bf16
    rounding remains. Byte 0 (2**-127, a float32 subnormal) maps to 0.
    Byte 255 (``_INVALID_SCALE``) yields +0 whatever the raw bytes are,
    including FP8 NaN.
    """
    values = raw.view(torch.float8_e4m3fn).float()
    power = (scale.to(torch.int32) << 23).view(torch.float32)
    power = power.repeat_interleave(block_size, -1)
    invalid = (scale == _INVALID_SCALE).repeat_interleave(block_size, -1)
    return torch.where(invalid, 0.0, values * power).to(torch.bfloat16)


def _unpack_bytes(words, num_bytes):
    """Little-endian bytes of int32 ``words`` (1-D) -> first ``num_bytes`` uint8.

    Inverse of the host's ``uint8_buffer.view(torch.int32)`` on a
    little-endian host: Host channels gather packed rows as int32 words.
    """
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=words.device)
    unpacked = (words.unsqueeze(-1) >> shifts) & 0xFF
    return unpacked.to(torch.uint8).reshape(-1)[:num_bytes]


def _gather_engram_rows(rows, dim):
    # Retain the existing 128-lane width alignment for narrow fixtures.
    # This does not fix the native Engram NaN: it also reproduces at width
    # 128, while an isolated 3-D BF16 gather passes. Released heads are 256.
    width = rows.shape[-1]
    padding = (-width) % 128
    if padding:
        rows = torch.nn.functional.pad(rows, (0, padding))
    result = upstream.tensor_model_parallel_all_gather(rows, dim=dim)
    return result[..., :width].contiguous() if padding else result


class Engram(upstream.Engram):
    """Inherit projections and collective head gather from upstream.

    Host-offloaded tables are looked up inside the compiled forward:
    ``start_host_embeddings`` (before layer 0) requests this step's rows and
    ``forward`` waits for them, so the gather and upload overlap the layers
    in between. No step leaves the device for Engram.
    """

    embed = bind_device_ops(
        upstream.Engram.embed, tensor_model_parallel_all_gather=_gather_engram_rows
    )
    # Set by publish_host_table on Host-offloaded layers only.
    host_channel: "HostChannel | None" = None

    def _create_embedding(self, layout, layer_hash_index):
        embedding = ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
        )
        # Each table is ~98 GB in one tensor, over the Run:AI streamer's
        # per-tensor memory limit, and a rank keeps only its heads' rows.
        layer_id = layout.layer_ids[layer_hash_index]
        if embedding.part_num_embeddings:
            for leaf in ("weight", "scale"):
                register_row_slice(
                    f"layers.{layer_id}.engram.embed.{leaf}",
                    embedding.vocab_start_idx,
                    embedding.part_num_embeddings,
                )
        return embedding

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        table = self.embed_tokens
        if not table.host_offload:
            super()._init_staging(max_tokens, head_dim)
            return
        self._prefetch_stream = None
        self._prefetch_done = None
        # Every DP rank's tokens reach every owner, one row per local head.
        self._host_capacity = max_tokens * table.dp_size * table.part_n_hash_cols
        self.host_channel: HostChannel | None = None
        self._uses_hbm: bool | None = None
        self._hbm_table: tuple | None = None
        self._hbm_rank: torch.Tensor | None = None
        self._host_pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self._host_rows: torch.Tensor | None = None

    @property
    def uses_hbm_lookup(self) -> bool:
        """``TPU_DEEPSEEK_V41_ENGRAM_HBM``: serve rows from an HBM copy of the
        host head shard instead of the Host channel. Decided once."""
        if self._uses_hbm is None:
            from .engram_hbm import should_use_hbm_lookup

            self._uses_hbm = should_use_hbm_lookup(
                host_offload=self.embed_tokens.host_offload,
                dp_size=int(self.embed_tokens.dp_size),
                layer_hash_index=int(self.layer_hash_index),
            )
        return self._uses_hbm

    def publish_host_table(
        self, predecessor: "HostChannel | None"
    ) -> "HostChannel | None":
        """After loading: place the HBM copy or start the Host channel.

        Returns this layer's channel, or ``predecessor`` if it has none, so a
        later layer's worker gathers only after the earlier upload landed and
        the two never compete for Host bandwidth while L1 is waiting.
        """
        if not self.embed_tokens.host_offload:
            return predecessor
        if self.uses_hbm_lookup:
            self.upload_hbm_table()
            return predecessor
        if self.host_channel is None:
            from .host_channel import HostChannel

            self.host_channel = HostChannel(
                self.embed_tokens, self._host_capacity, predecessor
            )
            logger.info(
                "[DSV41] Engram Host channel: layer=%d heads=[%d, %d) rows=%d "
                "capacity=%d",
                self.layer_hash_index,
                self.embed_tokens.head_start,
                self.embed_tokens.head_start + self.embed_tokens.part_n_hash_cols,
                self.embed_tokens.part_num_embeddings,
                self._host_capacity,
            )
        return self.host_channel

    def upload_hbm_table(self):
        """Copy this rank's host head shard to HBM once; returns
        (table, scales, shard metadata, rows per rank)."""
        if self._hbm_table is not None:
            return self._hbm_table
        from vllm_torchtpu.distributed.ep_mesh import ep_mesh_index

        from .engram_hbm import gather_shard_meta, upload_table

        emb = self.embed_tokens
        weight, scales = emb._storage()
        meta = gather_shard_meta(
            int(emb.head_start),
            int(emb.vocab_start_idx),
            int(emb.vocab_end_idx),
            int(weight.shape[0]),
        )
        rows = max(m[3] for m in meta)
        device = torch.device("tpu")
        table, table_scales = upload_table(weight, scales, device, rows)
        self._hbm_rank = torch.tensor(
            [[cast(int, ep_mesh_index())]], dtype=torch.int32, device=device
        )
        self._hbm_table = (table, table_scales, meta, rows)
        logger.info(
            "[DSV41] Engram HBM table: layer=%d rows/rank=%d bytes=%d",
            self.layer_hash_index,
            rows,
            table.numel() + table_scales.numel(),
        )
        return self._hbm_table

    def start_host_embeddings(
        self,
        local_hashes: torch.Tensor,
        gathered_hashes: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Request this step's rows; returns ``hidden_states`` ordered after it.

        ``local_hashes``/``gathered_hashes``: this rank's and the DP group's
        hashes of this layer; HBM layers need only the local ones. Call in
        layer order, as channels were published.
        """
        if self.uses_hbm_lookup:
            from .engram_hbm import build_lookup_op

            table, table_scales, meta, rows = self.upload_hbm_table()
            emb = self.embed_tokens
            op = build_lookup_op(
                int(self.layer_hash_index),
                int(emb.n_hash_cols),
                int(emb.part_n_hash_cols),
                rows,
                int(emb.dim),
                int(emb.block_size),
                int(emb.dp_size),
                meta,
            )
            self._host_rows = op(local_hashes, table, table_scales, self._hbm_rank)
            return hidden_states
        channel = self.host_channel
        assert channel is not None, "publish_host_table runs before tracing"
        assert gathered_hashes is not None
        receipt, hidden_states = channel.start(gathered_hashes, hidden_states)
        self._host_pending = (gathered_hashes, receipt)
        return hidden_states

    def _finish_host_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._host_pending is None:
            return hidden_states
        channel = self.host_channel
        assert channel is not None
        hashes, receipt = self._host_pending
        self._host_pending = None
        self._host_rows, hidden_states = channel.finish(hashes, receipt, hidden_states)
        return hidden_states

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if not self.embed_tokens.host_offload:
            return super()._ready_rows(num_tokens)
        rows = self._host_rows
        assert rows is not None, "start_host_embeddings precedes the layer"
        if self.uses_hbm_lookup:
            # The device lookup already dequantized this step's rows.
            return rows[:num_tokens]
        table = self.embed_tokens
        packed = rows[:num_tokens]
        data = _unpack_bytes(packed.flatten(), packed.numel() * 4).reshape(
            *packed.shape[:-1], packed.shape[-1] * 4
        )
        return dequant_fp8_rows(
            data[..., : table.dim].contiguous(),
            data[..., table.dim :].contiguous(),
            table.block_size,
        )

    def _sp_embed(self, hash_ids):
        """Rows for this rank's SP token chunk with all hash heads.

        Each rank stages ``[T, local_heads, dim]`` rows for every token. A
        token-axis all-gather yields ``[tp, T_pad, local_heads, dim]`` whose
        rank-major head order matches upstream's non-SP ``dim=1`` gather.
        """
        rows = self._ready_rows(hash_ids.shape[0])
        cols = self.embed_tokens.n_hash_cols
        tp = self.embed_tokens.tp_size
        num_tokens = rows.shape[0]
        chunk = -(-num_tokens // tp)
        rank = upstream.get_tensor_model_parallel_rank()
        if tp == 1:
            return rows[:, :cols]
        pad = chunk * tp - num_tokens
        if pad:
            rows = torch.nn.functional.pad(rows, (0, 0, 0, 0, 0, pad))
        gathered = _gather_engram_rows(rows, dim=0).unflatten(0, (tp, chunk * tp))
        local = gathered[:, rank * chunk : (rank + 1) * chunk]
        return local.transpose(0, 1).flatten(1, 2)[:, :cols]

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if self.embed_tokens.host_offload:
            hidden_states = self._finish_host_embeddings(hidden_states)
        if self.use_sequence_parallel:
            tp = self.embed_tokens.tp_size
            chunk = -(-hash_ids.shape[0] // tp)
            assert hidden_states.shape[0] == chunk, (hidden_states.shape, chunk)
            if token_mask is not None:
                # Padded SP rows keep the gate shut.
                pad = chunk * tp - token_mask.shape[0]
                token_mask = torch.cat([token_mask, token_mask.new_zeros(pad)])
                rank = upstream.get_tensor_model_parallel_rank()
                token_mask = token_mask[rank * chunk : (rank + 1) * chunk]
            embedded = self._sp_embed(hash_ids)
        else:
            embedded = self.embed(hash_ids)
        kv = self.wkv(embedded.flatten(-2))
        keys = (
            kv[:, : self.hc_mult * self.dim]
            .unflatten(-1, (self.hc_mult, self.dim))
            .float()
        )
        values = kv[:, self.hc_mult * self.dim :].float()
        hidden = hidden_states.float()
        dot = (hidden * self.q_weight * self.k_weight * keys).sum(-1)
        dot *= torch.rsqrt(hidden.square().mean(-1) + self.eps)
        dot *= torch.rsqrt(keys.square().mean(-1) + self.eps) * self.dim**-0.5
        gate_input = dot.abs().clamp(min=self.clamp_value).sqrt()
        gate = torch.where(dot < 0, -gate_input, gate_input).sigmoid()
        if token_mask is not None:
            gate = gate * token_mask[:, None]
        return (hidden + gate[:, :, None] * values[:, None, :]).to(hidden_states.dtype)

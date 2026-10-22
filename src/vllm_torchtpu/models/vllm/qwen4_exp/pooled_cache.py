# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp binding over the native planner and existing unified pool allocator."""

import gc
from dataclasses import replace

import torch
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.kv_cache_materializer import (
    LayerKVCache,
    materialize_kv_cache_tensors,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.utils import compute_hbm_budget, synchronize_tensors

from .cache import TEMP_HEADROOM, logger
from .config import (
    hbm_pool_sizing,
    mtp_layer_prefixes,
    native_cache_capacity,
    num_speculative_tokens,
    pool_graph_copies,
    pool_headroom_bytes,
    pooled_pool_blocks,
)
from .pd import TransferGroup, build_transfer_groups
from .pool_layout import check_packed_compressed_groups, pooled_cache_specs
from .qsa import Qwen4ExpPooledQSABackend


class Qwen4ExpPooledCache:
    """The unified pool and its binding to the model's layers.

    With HBM sizing (the default, see ``hbm_pool_sizing``) a worker holds
    two pools in turn. Before the engine has planned the KV cache, the dummy
    forwards of profiling run on a *scratch* pool of just the pages they
    address, so profiling measures activations rather than a guessed pool.
    The worker then reports a pool budget derived from the measured HBM
    (``memory_budget``), the engine plans ``num_blocks`` from it, and
    ``initialize`` replaces the scratch pool with the planned one. Static
    sizing allocates its final pool on the first dummy forward.
    """

    def __init__(self, runner, model):
        self.runner, self.model = runner, model
        self.layout, self.specs = pooled_cache_specs(runner.vllm_config)
        self.groups = get_kv_cache_groups(runner.vllm_config, self.specs)
        check_packed_compressed_groups(self.layout, self.groups)
        self.block_bytes = _get_kv_cache_bytes_per_block(self.groups)
        self.hbm_sizing = hbm_pool_sizing(runner.vllm_config)
        # Pages of the current pool; for static sizing, the fixed capacity.
        self.pages = (
            0
            if self.hbm_sizing
            else native_cache_capacity(runner.vllm_config, self.specs)
        )
        self.final = False
        self.materialized = None
        self.native_config = None

    def get_specs(self):
        return self.specs

    def get_transfer_groups(self) -> tuple[TransferGroup, ...]:
        """Use the bound scheduler plan, including auxiliary and state groups."""
        if self.materialized is None or self.native_config is None:
            raise ValueError("Initialize Qwen caches before describing PD ownership")
        return build_transfer_groups(self.native_config.kv_cache_groups)

    def dummy_pages(self, seq_lens: list[int], active_rows: int) -> int:
        """Pages ``dummy_metadata`` hands out for these rows, null page included."""
        window = num_speculative_tokens(self.runner.vllm_config) + 1
        pages = 1
        for group in self.groups:
            spec = group.kv_cache_spec
            if isinstance(spec, CircularBufferSpec):
                pages += active_rows
            elif isinstance(spec, MambaSpec):
                pages += active_rows * window
            else:
                columns = self._table_columns(spec)
                pages += sum(
                    min(columns, max(1, -(-seq_len // spec.block_size)))
                    for seq_len in seq_lens[:active_rows]
                )
        return pages

    def _table_columns(self, spec) -> int:
        return get_block_table_width(
            spec.max_num_blocks_per_req(
                self.runner.vllm_config, self.runner.max_model_len
            ),
            spec.block_size,
        )

    def allocate(self, pages: int | None = None):
        """Make a pool exist that holds at least ``pages`` pages.

        Static sizing allocates its final pool once. HBM sizing, before
        ``initialize``, (re)allocates a scratch pool of just the pages asked
        for.
        """
        if self.final or not self.hbm_sizing:
            if self.materialized is None:
                self._materialize(self.pages, final=True)
            assert pages is None or pages <= self.pages, (
                f"dummy forward needs {pages} pages, the pool has {self.pages}"
            )
            return
        pages = max(pages or 0, 2)
        if self.materialized is not None and self.pages >= pages:
            return
        self._release()
        self._materialize(pages, final=False)

    def memory_budget(self) -> int:
        """Pool bytes this worker can hold, from HBM measured after profiling.

        Profiling ran on the scratch pool, whose bytes are added back since
        ``initialize`` frees them first. ``pooled_pool_blocks`` reserves the
        graph-time pool copies and the headroom.
        """
        runner = self.runner
        budget = compute_hbm_budget(
            [runner.device], runner.cache_config.gpu_memory_utilization
        )
        scratch = (
            self.pages * self.block_bytes
            if self.materialized is not None and not self.final
            else 0
        )
        min_blocks = 1 + sum(
            group.kv_cache_spec.max_num_blocks_per_req(
                runner.vllm_config, runner.max_model_len
            )
            for group in self.groups
        )
        blocks = pooled_pool_blocks(
            self.block_bytes,
            cap_available=budget.available + scratch,
            physical_available=budget.total_limit - budget.total_used + scratch,
            min_blocks=min_blocks,
        )
        logger.info(
            "Qwen4Exp unified pool sizing: hbm_limit=%.2fGiB used=%.2fGiB "
            "(scratch pool %.2fGiB) cap_available=%.2fGiB -> %d blocks "
            "(%.2fGiB, %d tokens per QSA group) with %d graph-time pool "
            "copies and %.2fGiB headroom",
            budget.total_limit / 1024**3,
            budget.total_used / 1024**3,
            scratch / 1024**3,
            (budget.available + scratch) / 1024**3,
            blocks,
            blocks * self.block_bytes / 1024**3,
            blocks * self.layout.block_size,
            pool_graph_copies(),
            pool_headroom_bytes() / 1024**3,
        )
        return blocks * self.block_bytes

    def _release(self) -> None:
        """Drop the pool and every binding to it so its HBM is freed."""
        if self.materialized is None:
            return
        for layer in self.model.model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.kv_cache = None
            else:
                layer.self_attn.cache_owners = ()
            if layer.ple is not None:
                layer.ple.cache_owners = ()
        context = self.runner.vllm_config.compilation_config.static_forward_context
        for prefix in mtp_layer_prefixes(self.runner.vllm_config):
            attention = context.get(prefix + ".self_attn")
            if attention is not None:
                attention.cache_owners = ()
        self.materialized = None
        gc.collect()
        synchronize_tensors()

    def _materialize(self, pages: int, *, final: bool) -> None:
        runner = self.runner
        config = get_kv_cache_config_from_groups(
            runner.vllm_config, self.groups, pages * self.block_bytes
        )
        assert config.num_blocks == pages, (config.num_blocks, pages)
        # EngineCore attaches the resolved layout after invoking the planner.
        # Profiling allocates earlier, so preserve that same final contract.
        config.kv_cache_layout = runner.cache_config.kv_cache_layout
        self.attn_groups = [
            [
                AttentionGroup(
                    backend=Qwen4ExpPooledQSABackend,
                    layer_names=list(group.layer_names),
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                    metadata_builders=[],
                )
            ]
            for gid, group in enumerate(config.kv_cache_groups)
        ]
        self.kernel_block_sizes = [g.kv_cache_spec.block_size for g in self.groups]
        budget = compute_hbm_budget(
            [runner.device], runner.cache_config.gpu_memory_utilization
        )
        required = max(t.size for t in config.kv_cache_tensors)
        if not self.hbm_sizing and required + TEMP_HEADROOM > budget.available:
            raise ValueError(
                f"Qwen4Exp unified pool needs {required} bytes plus {TEMP_HEADROOM} "
                f"temporary reserve, but only {budget.available} bytes are available; "
                "reduce max_num_seqs or max_model_len"
            )
        if self.hbm_sizing and final:
            # Each compiled forward can hold extra copies of the pool (see
            # pooled_pool_blocks); XLA needs them in physical HBM, past
            # vLLM's utilization cap. Fail here rather than in the compiler.
            physical = budget.total_limit - budget.total_used
            needed = required * (1 + pool_graph_copies())
            if needed > physical:
                raise ValueError(
                    f"Qwen4Exp unified pool of {required} bytes needs {needed} "
                    f"bytes with {pool_graph_copies()} graph-time copies, but "
                    f"only {physical} bytes of HBM are free; set "
                    "QWEN4_EXP_MAX_POOL_BYTES or QWEN4_EXP_POOL_HEADROOM_BYTES"
                )
        self.materialized = materialize_kv_cache_tensors(
            kv_cache_config=config,
            attn_groups=self.attn_groups,
            kernel_block_sizes=self.kernel_block_sizes,
            device=runner.device,
            cache_dtype=runner.kv_cache_dtype,
        )
        caches = self.materialized.kv_caches
        for i, layer in enumerate(self.model.model.layers):
            prefix = f"model.layers.{i}"
            if layer.layer_type == "linear_attention":
                layer.linear_attn.kv_cache = tuple(caches[prefix + ".linear_attn"])
            else:
                self._bind_qsa(layer.self_attn, prefix + ".self_attn", caches)
            if layer.ple is not None:
                layer.ple.bind_pool(caches[prefix + ".ple"][0])
        self.bind_draft()
        synchronize_tensors(self.materialized.raw_tensors)
        self.pages, self.final = pages, final
        if final:
            self.native_config = config
        logger.info(
            "Qwen4Exp unified pool%s: manager_tokens=%d page_bytes=%d "
            "blocks=%d pools=%d bytes=%d",
            "" if final else " (profiling scratch)",
            self.layout.block_size,
            self.layout.page_bytes,
            pages,
            len(self.materialized.raw_tensors),
            required,
        )

    def bind_draft(self) -> None:
        """Bind the MTP draft's QSA layers once both they and the pool exist.

        The draft registers its attention in the shared static forward
        context; a draft that loads after allocation binds when it is
        attached (``Qwen4ExpAdapter.bind_draft``).
        """
        if self.materialized is None:
            return
        caches = self.materialized.kv_caches
        context = self.runner.vllm_config.compilation_config.static_forward_context
        for prefix in mtp_layer_prefixes(self.runner.vllm_config):
            attention = context.get(prefix + ".self_attn")
            if attention is None:
                continue
            self._bind_qsa(attention, prefix + ".self_attn", caches)

    def _bind_qsa(self, attention, name, caches) -> None:
        owner = self.layout.compressed_owner(name)
        main_owner = self.layout.main_owner(name)
        main = caches[main_owner[0]]
        assert main is caches[owner[0]] is caches[name + ".raw_ring"], (
            "QSA cache groups must share one native carrier"
        )
        if main_owner[1] < 0:
            attention.bind_pool(main, owner)
        else:
            attention.bind_pool(main, owner, main_owner=main_owner)

    def initialize(self, config) -> dict[str, LayerKVCache]:
        if self.hbm_sizing and not self.final:
            # Replace the profiling scratch pool with the planned pool.
            self._release()
            self._materialize(config.num_blocks, final=True)
        else:
            self.allocate()
        assert config == self.native_config, (
            "Qwen4Exp native pool plan changed after profiling"
        )
        runner = self.runner
        runner.kv_cache_config = config
        runner.empty_slot_mappings = {
            gid: torch.empty(0, device=runner.device) for gid in range(len(self.groups))
        }
        runner.attn_groups = self.attn_groups
        runner._kernel_block_sizes = self.kernel_block_sizes
        runner.may_reinitialize_input_batch(config, self.kernel_block_sizes)
        for gid, group_list in enumerate(runner.attn_groups):
            group = group_list[0]
            group.metadata_builders.append(
                AttentionMetadataBuilder(
                    group.kv_cache_spec,
                    group.layer_names,
                    runner.vllm_config,
                    runner.device,
                    runner=runner,
                    kv_cache_group_id=gid,
                )
            )
        materialized = self.materialized
        assert materialized is not None
        runner.kv_caches = list(materialized.raw_tensors)
        runner.kv_cache_raw_tensors = list(materialized.raw_tensors)
        runner.bm_pool_layout = materialized.bm_pool_layout
        runner._build_mamba_copy_plan(
            config, materialized.layer_to_raw, materialized.bm_pool_layout
        )
        if num_speculative_tokens(runner.vllm_config) and config.has_mamba_layers:
            # Per-block read offsets (num_accepted - 1 of the last verify):
            # GDN and PLE resume from that checkpoint block, which is how
            # rejected drafts roll back. Mirrors the generic unified pool.
            runner.mamba_slot_read_offsets = torch.zeros(
                config.num_blocks, dtype=torch.int32
            ).to(runner.device)
        runner._precompile_mamba_state_seed_copies()
        if not runner.enforce_eager:
            runner._precompile_substitute_placeholder_token()
        return materialized.kv_caches

    def dummy_metadata(self, metadata, active_rows):
        """Disjoint owner pages even before runner cache initialization.

        Every group gets its own IDs. Reusing one dummy block across QSA,
        GDN and PLE would turn profiling into destructive cross-owner writes.
        """
        rows = metadata.seq_lens.shape[0]
        seq_lens_cpu = metadata.seq_lens.to("cpu")
        result = {}
        next_page = 1
        # Speculative decoding: state groups own one checkpoint block per
        # verify position. The compiled forward keeps no guards, so profiling
        # must trace the same checkpoint/read-offset branches serving takes.
        window = num_speculative_tokens(self.runner.vllm_config) + 1
        read_offsets = None
        if window > 1:
            read_offsets = torch.zeros(
                self.pages, dtype=torch.int32, device=self.runner.device
            )
            # Serving stages both distributions as views of one [6] tensor
            # (TPUModelRunner._combined_request_distribution_cpu); keep the
            # traced inputs' aliasing identical.
            combined = torch.cat([metadata.request_distribution] * 2)
            metadata = replace(metadata, request_distribution=combined[0:3])
            mamba_distribution = combined[3:6]
        for group in self.groups:
            spec = group.kv_cache_spec
            state = isinstance(spec, MambaSpec)
            ring = isinstance(spec, CircularBufferSpec)
            columns = (
                window
                if state
                else get_block_table_width(
                    spec.max_num_blocks_per_req(
                        self.runner.vllm_config, self.runner.max_model_len
                    ),
                    spec.block_size,
                )
            )
            table = torch.zeros((rows, columns), dtype=torch.int32, device="cpu")
            for row in range(active_rows):
                req_blocks = max(
                    1,
                    (int(seq_lens_cpu[row]) + spec.block_size - 1) // spec.block_size,
                )
                owned = 1 if ring else window if state else min(columns, req_blocks)
                table[row, :owned] = torch.arange(
                    next_page, next_page + owned, dtype=torch.int32, device="cpu"
                )
                next_page += owned
            table = table.to(self.runner.device)
            item = replace(
                metadata,
                block_tables=table.reshape(-1),
                mamba_state_indices=table[:, 0] if state else None,
            )
            if state and read_offsets is not None:
                # Serving derives state and checkpoint indices as separate
                # tensors; do not trace them as views of one base.
                item = replace(
                    item,
                    mamba_state_indices=table[:, 0].clone(),
                    mamba_slot_read_offsets=read_offsets,
                    mamba_ckpt_indices=table.clone(),
                    mamba_request_distribution=mamba_distribution,
                )
            result.update({name: item for name in group.layer_names})
        assert next_page <= self.pages, "dummy owner pages exceed the planned pool"
        return result

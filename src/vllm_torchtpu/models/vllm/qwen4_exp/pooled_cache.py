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
"""Qwen4Exp binding over the native planner and existing unified pool allocator."""

from dataclasses import replace

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.core.kv_cache_planning import (
    check_enough_kv_cache_memory,
    get_global_kv_cache_groups,
    get_kv_cache_config_from_groups,
)
from vllm.v1.core.kv_cache_utils import (
    resolve_cache_hit_alignment_tokens,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.kv_cache_materializer import LayerKVCache
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import (
    compute_hbm_budget,
    min_across_ep_dp,
    synchronize_tensors,
)

from .config import native_cache_capacity, require_unified_pool
from .pd import TransferGroup, build_transfer_groups
from .pool_allocation import materialize_pool, validate_pool_tiling
from .pool_layout import pooled_cache_specs
from .qsa import Qwen4ExpPooledQSABackend

TEMP_HEADROOM = 8 * 1024**3
logger = init_logger(__name__)


class Qwen4ExpPooledCache:
    def __init__(self, runner, model, draft_model=None):
        require_unified_pool()
        self.runner = runner
        self.model = None if model is None else model.get_language_model()
        self.requested_blocks = runner.cache_config.num_gpu_blocks_override
        self._budgeted_pages: int | None = None
        self._capacity_ceiling: int | None = None
        self.materialized = None
        self.draft_model = draft_model

    def get_specs(self, available_memory: int | None = None):
        cfg = self.runner.vllm_config
        layout, specs = pooled_cache_specs(cfg)
        pages = native_cache_capacity(cfg, specs)
        if (self.materialized is not None or self._budgeted_pages is not None) and (
            layout != self.layout
            or specs != self.specs
            or pages != self._capacity_ceiling
        ):
            raise ValueError("Qwen native cache geometry changed after allocation")
        self.layout, self.specs = layout, specs
        self._capacity_ceiling = pages
        cache = cfg.cache_config
        cache.block_size = self.layout.block_size
        cache.mamba_block_size = (
            self.layout.block_size if cache.enable_prefix_caching else None
        )
        cache.mamba_page_size_padded = self.layout.page_bytes
        self.groups = get_global_kv_cache_groups(cfg, [self.specs])
        if available_memory is not None and self._budgeted_pages is None:
            previous_override = cache.num_gpu_blocks_override
            cache.num_gpu_blocks_override = self.requested_blocks
            try:
                plan = get_kv_cache_config_from_groups(
                    cfg, self.groups, available_memory
                )
            finally:
                cache.num_gpu_blocks_override = previous_override
            if plan.num_blocks <= 0:
                raise ValueError("Qwen native pool budget cannot hold a cache block")
            physical_bytes = {tensor.size for tensor in plan.kv_cache_tensors}
            assert len(physical_bytes) == 1
            bytes_per_block, remainder = divmod(physical_bytes.pop(), plan.num_blocks)
            assert not remainder
            selected = (
                min(pages, plan.num_blocks)
                if self.requested_blocks is None
                else self.requested_blocks
            )
            layout.validate_capacity(selected)
            required = selected * bytes_per_block
            if required > available_memory:
                raise ValueError(
                    "Qwen native pool override exceeds available HBM budget"
                )
            check_enough_kv_cache_memory(cfg, self.specs, required)
            self._budgeted_pages = selected
        self.pages = (
            self._budgeted_pages
            if self._budgeted_pages is not None
            else self.requested_blocks
            if self.requested_blocks is not None
            else pages
        )
        layout.validate_capacity(self.pages)
        cache.num_gpu_blocks_override = self.pages
        return self.specs

    def get_worker_specs(self):
        """Select one physical capacity before the executor negotiates its plan."""
        if self._budgeted_pages is not None or self.materialized is not None:
            return self.get_specs()
        cfg = self.runner.vllm_config
        budget = compute_hbm_budget(
            [self.runner.device], cfg.cache_config.gpu_memory_utilization
        )
        available = budget.available - TEMP_HEADROOM
        if cfg.parallel_config.tensor_parallel_size > 1:
            value = torch.tensor([available], dtype=torch.int64, device="cpu")
            torch.distributed.all_reduce(
                value, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
            )
            available = int(value.item())
        available = min_across_ep_dp(cfg.parallel_config, available)
        return self.get_specs(available_memory=available)

    def get_transfer_groups(self) -> tuple[TransferGroup, ...]:
        """Use the bound scheduler plan, including auxiliary and state groups."""
        if self.materialized is None:
            raise ValueError("Initialize Qwen caches before describing PD ownership")
        return build_transfer_groups(self.native_config.kv_cache_groups)

    def allocate(self):
        if self.materialized is not None:
            return
        runner = self.runner
        self.get_worker_specs()
        config = get_kv_cache_config_from_groups(
            runner.vllm_config,
            self.groups,
            self.pages
            * self.layout.page_bytes
            * max(len(g.layer_names) for g in self.groups),
        )
        # Profiling precedes EngineCore's scheduler finalization. Resolve the
        # same metadata so full plan equality still guards physical ownership.
        config.kv_cache_layout = runner.cache_config.kv_cache_layout
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            config, runner.vllm_config
        )
        config.hash_block_size = hash_block_size
        config.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
            config, runner.vllm_config, scheduler_block_size, hash_block_size
        )
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
        if required + TEMP_HEADROOM > budget.available:
            raise ValueError(
                f"Qwen4Exp unified pool needs {required} bytes plus {TEMP_HEADROOM} "
                f"temporary reserve, but only {budget.available} bytes are available; "
                "reduce max_num_seqs or max_model_len"
            )
        if runner.device.type == "tpu":
            from jax.experimental.pallas import tpu as pltpu

            hardware = pltpu.get_tpu_info()
            validate_pool_tiling(self.layout, hardware)
        self.materialized = materialize_pool(
            runner.vllm_config, config, self.layout, runner.device
        )
        caches = self.materialized.kv_caches
        assert self.model is not None, "Bind a native language model before allocation"
        owners = [
            (f"model.layers.{i}", layer)
            for i, layer in enumerate(self.model.model.layers)
        ]
        if runner.vllm_config.speculative_config is not None:
            assert self.draft_model is not None, "MTP weights must precede profiling"
            owners.extend(
                (f"mtp.layers.{self.model.config.num_hidden_layers + i}", layer)
                for i, layer in enumerate(self.draft_model.model.layers)
            )
        for prefix, layer in owners:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.kv_cache = tuple(caches[prefix + ".linear_attn"])
            else:
                main = caches[prefix + ".self_attn"]
                index = caches[prefix + ".self_attn.compressed_k"]
                assert index is caches[prefix + ".self_attn.raw_ring"]
                assert main is not index
                layer.self_attn.bind_pool(main, index)
            if layer.ple is not None:
                layer.ple.bind_pool(caches[prefix + ".ple"][0])
        synchronize_tensors(self.materialized.raw_tensors)
        self.native_config = config
        logger.info(
            "Qwen4Exp unified pool: manager_tokens=%d page_bytes=%d "
            "blocks=%d pools=%d bytes=%d",
            self.layout.block_size,
            self.layout.page_bytes,
            self.pages,
            len(self.materialized.raw_tensors),
            required,
        )

    def initialize(self, config) -> dict[str, LayerKVCache]:
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
        # Native page bytes include the compact index. State copies address
        # only the main row; index bytes at a Mamba ID contain no live state.
        copy_groups = [
            replace(
                group,
                kv_cache_spec=replace(
                    group.kv_cache_spec, page_size_padded=self.layout.main_page_bytes
                ),
            )
            if isinstance(group.kv_cache_spec, MambaSpec)
            else group
            for group in config.kv_cache_groups
        ]
        if any(isinstance(g.kv_cache_spec, MambaSpec) for g in copy_groups):
            runner._build_mamba_copy_plan(
                replace(config, kv_cache_groups=copy_groups),
                materialized.layer_to_raw,
                materialized.bm_pool_layout,
            )
        else:
            runner._mamba_copy_plan = []
            runner._mamba_state_block_size = None
            runner._pool_block_split = 1
        if runner.vllm_config.speculative_config is not None:
            runner.mamba_slot_read_offsets = torch.zeros(
                config.num_blocks, dtype=torch.int32, device=runner.device
            )
        if runner._mamba_copy_plan:
            runner._precompile_mamba_state_seed_copies()
        if not runner.enforce_eager:
            runner._precompile_substitute_placeholder_token()
        return materialized.kv_caches

    def dummy_metadata(self, metadata, active_rows, *, use_max_model_len=True):
        """Disjoint owner pages even before runner cache initialization.

        Every group gets its own IDs. Reusing one dummy block across QSA,
        GDN and PLE would turn profiling into destructive cross-owner writes.
        """
        rows = metadata.seq_lens.shape[0]
        context_limit = (
            self.runner.max_model_len
            if use_max_model_len
            else self.runner.most_model_len
        )
        # Match the serving runner's two views of one six-element upload.
        # Aliasing these fields to the SAME three-element tensor specializes
        # the compiled graph to one distribution. Verify then incorrectly
        # uses RPA's prefill segment for GDN/PLE and never writes checkpoints.
        mamba_distribution = None
        if self.runner.vllm_config.speculative_config is not None:
            distribution = torch.cat(
                (metadata.request_distribution, metadata.request_distribution)
            )
            metadata = replace(metadata, request_distribution=distribution[:3])
            mamba_distribution = distribution[3:]
        result = {}
        next_page = 1
        for group in self.groups:
            spec = group.kv_cache_spec
            state = isinstance(spec, MambaSpec)
            ring = isinstance(spec, CircularBufferSpec)
            columns = (
                1 + spec.num_speculative_blocks
                if state
                else get_block_table_width(
                    spec.max_num_blocks_per_req(self.runner.vllm_config, context_limit),
                    spec.block_size,
                )
            )
            table = torch.zeros((rows, columns), dtype=torch.int32, device="cpu")
            for row in range(active_rows):
                owned = 1 if ring else columns
                table[row, :owned] = torch.arange(
                    next_page, next_page + owned, dtype=torch.int32, device="cpu"
                )
                next_page += owned
            table = table.to(self.runner.device)
            item = replace(
                metadata,
                block_tables=table.reshape(-1),
                # Match the independent, contiguous gathers produced by the
                # serving metadata walk. A strided column view can specialize
                # AOT input aliasing and make later request rows read padding.
                mamba_state_indices=table[:, 0].clone() if state else None,
                mamba_ckpt_indices=table.clone() if state and columns > 1 else None,
                mamba_slot_read_offsets=(
                    torch.zeros(
                        self.pages, dtype=torch.int32, device=self.runner.device
                    )
                    if state and self.runner.vllm_config.speculative_config is not None
                    else None
                ),
                mamba_request_distribution=mamba_distribution if state else None,
            )
            result.update({name: item for name in group.layer_names})
        assert next_page <= self.pages, "dummy owner pages exceed the planned pool"
        return result

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
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.kv_cache_materializer import LayerKVCache
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import compute_hbm_budget, synchronize_tensors

from .config import native_cache_capacity, require_unified_pool
from .pd import TransferGroup, build_transfer_groups
from .pool_allocation import materialize_pool, validate_pool_tiling
from .pool_layout import pooled_cache_specs
from .qsa import Qwen4ExpPooledQSABackend

TEMP_HEADROOM = 8 * 1024**3
logger = init_logger(__name__)


class Qwen4ExpPooledCache:
    def __init__(self, runner, model):
        require_unified_pool()
        self.runner, self.model = runner, model
        self.layout, self.specs = pooled_cache_specs(runner.vllm_config)
        self.pages = native_cache_capacity(runner.vllm_config, self.specs)
        self.groups = get_kv_cache_groups(runner.vllm_config, self.specs)
        self.materialized = None

    def get_specs(self):
        return self.specs

    def get_transfer_groups(self) -> tuple[TransferGroup, ...]:
        """Use the bound scheduler plan, including auxiliary and state groups."""
        if self.materialized is None:
            raise ValueError("Initialize Qwen caches before describing PD ownership")
        return build_transfer_groups(self.native_config.kv_cache_groups)

    def allocate(self):
        if self.materialized is not None:
            return
        runner = self.runner
        config = get_kv_cache_config_from_groups(
            runner.vllm_config,
            self.groups,
            self.pages
            * self.layout.page_bytes
            * max(len(g.layer_names) for g in self.groups),
        )
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
        if required + TEMP_HEADROOM > budget.available:
            raise ValueError(
                f"Qwen4Exp unified pool needs {required} bytes plus {TEMP_HEADROOM} "
                f"temporary reserve, but only {budget.available} bytes are available; "
                "reduce max_num_seqs or max_model_len"
            )
        if runner.device.type == "tpu":
            from jax.experimental.pallas import tpu as pltpu

            try:
                hardware = pltpu.get_tpu_info()
            except ValueError as error:
                if not str(error).startswith("Unsupported TPU device kind: "):
                    raise
                hardware = None
            validate_pool_tiling(self.layout, hardware)
        self.materialized = materialize_pool(
            runner.vllm_config, config, self.layout, runner.device
        )
        caches = self.materialized.kv_caches
        for i, layer in enumerate(self.model.model.layers):
            prefix = f"model.layers.{i}"
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
        result = {}
        next_page = 1
        for group in self.groups:
            spec = group.kv_cache_spec
            state = isinstance(spec, MambaSpec)
            ring = isinstance(spec, CircularBufferSpec)
            columns = (
                1
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
                owned = 1 if ring else columns
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
            result.update({name: item for name in group.layer_names})
        assert next_page <= self.pages, "dummy owner pages exceed the planned pool"
        return result

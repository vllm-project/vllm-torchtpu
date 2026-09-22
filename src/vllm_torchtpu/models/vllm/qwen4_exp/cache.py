# SPDX-License-Identifier: Apache-2.0
"""Native scheduler specs with independently allocated typed TPU owners."""
import copy
import math

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.layers.core.attention_metadata import \
    AttentionMetadataBuilder
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import compute_hbm_budget, synchronize_tensors

logger = init_logger(__name__)
TEMP_HEADROOM = 8 * 1024**3


class Qwen4ExpCache:

    def __init__(self, runner, model):
        self.runner, self.model = runner, model
        self.slots = runner.max_num_reqs + 1
        self.pages = 1 + runner.scheduler_config.max_num_seqs * (
            math.ceil(runner.max_model_len / 128) + 11)
        self.owners = {}
        self.specs = {}
        self.group_ids = {}
        self.physical_allocation_bytes = 0
        for i, layer in enumerate(model.model.layers):
            base = f"model.layers.{i}"
            if layer.layer_type == "linear_attention":
                self.specs[base + ".linear_attn"] = MambaSpec(
                    # APC is disabled: native MambaManager allocates by token
                    # span, so one state block must cover the whole request.
                    block_size=runner.max_model_len,
                    shapes=tuple(layer.linear_attn.get_state_shape()),
                    dtypes=layer.linear_attn.get_state_dtype(),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN)
            else:
                self.specs[base +
                           ".self_attn"] = layer.self_attn.get_kv_cache_spec(
                               runner.vllm_config)
                for role, spec in layer.self_attn.get_auxiliary_cache_specs(
                ).items():
                    self.specs[base + ".self_attn." + role] = spec
            if layer.ple is not None:
                self.specs[base + ".ple"] = MambaSpec(
                    block_size=runner.max_model_len,
                    shapes=layer.ple.get_state_shape(),
                    dtypes=layer.ple.get_state_dtype(),
                    tp_replicated=True,
                    mamba_type=MambaAttentionBackendEnum.SHORT_CONV)
        assert len(
            self.specs) == 73, "Qwen4Exp native owner collection is incomplete"

    def get_specs(self):
        c = self.runner.cache_config
        if c.num_gpu_blocks_override not in (None, self.pages):
            raise ValueError(
                f"Qwen4Exp S1 fixes native capacity to {self.pages} blocks")
        c.num_gpu_blocks_override = self.pages
        c.kv_cache_layout = "BLHNC"
        return self.specs

    def allocate(self):
        if self.owners:
            return
        runner = self.runner
        before = compute_hbm_budget([runner.device],
                                    runner.cache_config.gpu_memory_utilization)
        # Observed-layout allowance, including independently allocated states
        # and padding. Actual allocator deltas govern admission after materialization.
        estimate = (12 * self.pages * (2 * 128 * 256 + 32 * 128) * 2 +
                    36 * self.slots * (8 * 1280 + 6 * 128 * 128) * 4 +
                    math.ceil(self.slots / 8) * 8 * 9 * 10240 * 2 +
                    12 * self.slots * 8192)
        if before.available < estimate + TEMP_HEADROOM:
            raise ValueError(
                "Qwen4Exp cache allocation would consume reserved temporary headroom"
            )

        def alloc(name, shape, dtype, fill=0):
            result = torch.full(shape, fill, dtype=dtype, device=runner.device)
            self.owners[name] = result
            return result

        for i, layer in enumerate(self.model.model.layers):
            base = f"model.layers.{i}"
            if layer.layer_type == "linear_attention":
                layer.linear_attn.kv_cache = tuple(
                    alloc(base + ".linear_attn." +
                          role, (self.slots, *shape), dtype)
                    for role, shape, dtype in zip((
                        "conv",
                        "recurrent"), layer.linear_attn.get_state_shape(
                        ), layer.linear_attn.get_state_dtype()))
            else:
                base_q = base + ".self_attn."
                layer.self_attn.bind_kv_cache(
                    (alloc(base_q + "main_k", (self.pages, 128, 1, 256),
                           torch.bfloat16),
                     alloc(base_q + "main_v", (self.pages, 128, 1, 256),
                           torch.bfloat16),
                     alloc(base_q + "compressed_k", (self.pages, 32, 1, 128),
                           torch.bfloat16),
                     alloc(base_q + "raw_ring", (self.slots, 8, 128),
                           torch.bfloat16),
                     alloc(base_q + "raw_positions", (self.slots, 8),
                           torch.int64, -1)))
            if layer.ple is not None:
                layer.ple.bind_history(
                    alloc(base + ".ple.history", (self.slots, 9, 10240),
                          torch.bfloat16))
        assert len(
            self.owners) == 133, "Qwen4Exp must bind all 133 persistent fields"
        assert len({id(t)
                    for t in self.owners.values()
                    }) == 133, "Typed owners must not alias"
        synchronize_tensors(list(self.owners.values()))
        buffers = {
            t.untyped_storage().data_ptr()
            for t in self.owners.values()
        }
        assert len(
            buffers) == 133, "Qwen4Exp typed owners unexpectedly share storage"
        after = compute_hbm_budget([runner.device],
                                   runner.cache_config.gpu_memory_utilization)
        self.physical_allocation_bytes = after.total_used - before.total_used
        if after.available < TEMP_HEADROOM:
            raise ValueError(
                "Qwen4Exp physical cache allocation exceeded temporary headroom"
            )
        logger.info(
            "Qwen4Exp physical cache allocation: pages=%d slots=%d fields=%d "
            "allocator_delta=%d observed_layout_allowance=%d resident=%d free_under_cap=%d reserve=%d",
            self.pages, self.slots, len(self.owners),
            self.physical_allocation_bytes, estimate, after.total_used,
            after.available, TEMP_HEADROOM)

    def initialize(self, config):
        runner = self.runner
        config = copy.deepcopy(config)
        if config.num_blocks != self.pages:
            raise ValueError(
                "Qwen4Exp native planner changed the admitted capacity")
        seen = set()
        for gid, group in enumerate(config.kv_cache_groups):
            per = (group.kv_cache_spec.kv_cache_specs if isinstance(
                group.kv_cache_spec, UniformTypeKVCacheSpecs) else {
                    name: group.kv_cache_spec
                    for name in group.layer_names
                })
            for name, spec in per.items():
                if name not in self.specs or spec != self.specs[name]:
                    raise ValueError(
                        f"Unexpected native cache owner/spec: {name}")
                self.group_ids[name] = gid
                seen.add(name)
        if seen != set(self.specs):
            raise ValueError("Native cache planner omitted Qwen4Exp owners")
        for descriptor in config.kv_cache_tensors:
            for i, name in enumerate(descriptor.layers):
                end = (descriptor.offset + i * descriptor.layer_stride +
                       (self.pages - 1) * descriptor.block_stride +
                       self.specs[name].page_size_bytes)
                if end > descriptor.size:
                    raise ValueError(
                        f"Native cache last-block bound invalid for {name}")
        self.allocate()
        budget = compute_hbm_budget([runner.device],
                                    runner.cache_config.gpu_memory_utilization)
        if budget.available < TEMP_HEADROOM:
            raise ValueError(
                "Qwen4Exp profiling left insufficient temporary headroom")
        runner.kv_cache_config = config
        runner.empty_slot_mappings = {
            gid: torch.empty(0, device=runner.device)
            for gid in range(len(config.kv_cache_groups))
        }
        runner.may_reinitialize_input_batch(
            config,
            [g.kv_cache_spec.block_size for g in config.kv_cache_groups])
        runner.attn_groups = []
        for gid, group in enumerate(config.kv_cache_groups):
            # Native hybrid groups wrap per-layer specs. The metadata builder
            # needs the concrete type to preserve compact GDN state-slot IDs.
            specs = (list(group.kv_cache_spec.kv_cache_specs.values()) if
                     isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                     else [group.kv_cache_spec])
            metadata_spec = specs[0]
            assert all(
                isinstance(s, MambaSpec) == isinstance(metadata_spec,
                                                       MambaSpec)
                and s.block_size == metadata_spec.block_size for s in specs)
            builder = AttentionMetadataBuilder(metadata_spec,
                                               group.layer_names,
                                               runner.vllm_config,
                                               runner.device,
                                               runner=runner,
                                               kv_cache_group_id=gid)
            runner.attn_groups.append([
                AttentionGroup(backend=None,
                               layer_names=list(group.layer_names),
                               kv_cache_spec=metadata_spec,
                               kv_cache_group_id=gid,
                               metadata_builders=[builder])
            ])
        runner._mamba_num_blocks = self.slots
        runner._init_mamba_slot_pool(self.slots)
        runner.kv_caches = list(self.owners.values())
        # QSA layers of each role share the native group's request table;
        # main and compressed remain independently addressed inputs.
        for role in ("self_attn", "self_attn.compressed_k"):
            groups = {
                self.group_ids[f"model.layers.{i}.{role}"]
                for i in range(3, 48, 4)
            }
            assert len(
                groups) == 1, f"Qwen4Exp role {role} has inconsistent groups"
        logger.info(
            "Qwen4Exp native cache verified: %d specs, %d groups, %d descriptors, %d blocks",
            len(self.specs), len(config.kv_cache_groups),
            len(config.kv_cache_tensors), self.pages)

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
"""Derive launch bounds from the unchanged RPA buffers and schedule."""

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule

from .abi import WORD_BYTES, scalar_prefetch_words
from .tuned_block_sizes import DEFAULT_POLICY, DecodePolicy


def make_config(
    batch: int,
    heads: int,
    dim: int,
    kv_heads: int,
    ratio: int,
    groups: int,
    cache_dtype: jnp.dtype,
    *,
    policy: DecodePolicy = DEFAULT_POLICY,
    info: pltpu.TpuInfo | None = None,
) -> configs.RpaConfigs:
    return configs.RpaConfigs(
        block=policy.block,
        model=configs.ModelConfigs(
            num_q_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=dim,
            sm_scale=dim**-0.5,
            mask_value=float(jnp.finfo(jnp.float32).min),
        ),
        serve=configs.ServingConfigs(
            num_seqs=batch,
            page_size=ratio,
            total_q_tokens=batch,
            num_page_indices=batch * (groups + 1),
            dtype_q=jnp.dtype(jnp.float32),
            dtype_out=jnp.dtype(jnp.float32),
            dtype_kv=cache_dtype,
            skip_kv_update=True,
        ),
        vmem_limit_bytes=(info or pltpu.get_tpu_info()).vmem_capacity_bytes,
        mode=configs.RpaCase.DECODE,
    )


# Same conservative compiler margins as shared configs.py / wrapper.py.
SMEM_COMPILER_RESERVE_BYTES = 32 * 1024
VMEM_USABLE_FRACTION = 0.8


@dataclasses.dataclass(frozen=True)
class ResourceUsage:
    schedule_bytes: int
    scalar_bytes: int
    buffer_bytes: int
    scratch_bytes: int
    working_estimate_bytes: int
    gate_bytes: int
    worst_steps: int
    schedule_steps: int

    @property
    def vmem_estimate_bytes(self) -> int:
        return (
            self.buffer_bytes
            + self.scratch_bytes
            + self.working_estimate_bytes
            + self.gate_bytes
        )

    def fits(self, info: pltpu.TpuInfo) -> bool:
        return (
            self.worst_steps <= self.schedule_steps
            and self.scalar_bytes + self.schedule_bytes
            <= info.smem_capacity_bytes - SMEM_COMPILER_RESERVE_BYTES
            and self.vmem_estimate_bytes
            <= info.vmem_capacity_bytes * VMEM_USABLE_FRACTION
        )


def resource_usage(cfgs: configs.RpaConfigs) -> ResourceUsage:
    """Exact allocation sizes, separate from working-set/headroom estimates."""
    n = cfgs.serve.num_seqs
    tiles = pl.cdiv(cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz)
    worst_steps = pl.cdiv(n * tiles, cfgs.batch_size)
    metadata = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs)
    schedule_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(metadata))
    # Actual scalar prefetch: cu_q_lens, offsets, q_lens and virtual pages.
    scalar_bytes = scalar_prefetch_words(n, cfgs.serve.num_page_indices) * WORD_BYTES
    q_bytes = math.prod(cfgs.q_vmem_shape) * cfgs.serve.dtype_q.itemsize
    kv_bytes = math.prod(cfgs.kv_vmem_shape) * cfgs.serve.dtype_kv.itemsize
    buffers = (
        configs.NUM_INPUT_SLOTS * (q_bytes + kv_bytes)
        + configs.NUM_OUTPUT_SLOTS * q_bytes
    )
    scratches = (
        2 * math.prod(cfgs.lm_scratch_shape) + math.prod(cfgs.acc_scratch_shape)
    ) * cfgs.serve.dtype_out.itemsize
    # Same live Q/KV/QK working-set estimate as RPA.calculate_block_sizes,
    # plus the fused entry's resident BF16 gate and FP32 gate intermediates.
    heads, dim = cfgs.model.num_q_heads, cfgs.aligned_q_head_dim
    working = cfgs.batch_size * (
        heads * dim * cfgs.serve.dtype_q.itemsize
        + cfgs.bkv_sz * cfgs.model.num_kv_heads * dim * cfgs.serve.dtype_kv.itemsize
        + cfgs.bkv_sz * heads * cfgs.serve.dtype_out.itemsize
        + heads * dim * cfgs.serve.dtype_q.itemsize
    )
    gate = n * heads * dim * jnp.dtype(jnp.bfloat16).itemsize
    # Keep the existing RPA planner's 20% headroom for compiler layout/temps.
    return ResourceUsage(
        schedule_bytes=schedule_bytes,
        scalar_bytes=scalar_bytes,
        buffer_bytes=buffers,
        scratch_bytes=scratches,
        working_estimate_bytes=working,
        gate_bytes=gate,
        worst_steps=worst_steps,
        schedule_steps=cfgs.max_steps_ub,
    )


def fits_launch(cfgs: configs.RpaConfigs) -> bool:
    return resource_usage(cfgs).fits(pltpu.get_tpu_info())


def _config_for_batch(cfgs: configs.RpaConfigs, batch: int) -> configs.RpaConfigs:
    return dataclasses.replace(
        cfgs,
        serve=dataclasses.replace(
            cfgs.serve,
            num_seqs=batch,
            total_q_tokens=batch,
            num_page_indices=batch * cfgs.serve.pages_per_seq,
        ),
    )


@dataclasses.dataclass(frozen=True)
class DecodeLaunch:
    policy: DecodePolicy
    config: configs.RpaConfigs
    resources: ResourceUsage

    @property
    def chunk_queries(self) -> int:
        return self.config.serve.num_seqs

    def config_for_batch(self, batch: int) -> configs.RpaConfigs:
        if batch > self.chunk_queries:
            raise ValueError("Chunk exceeds planned query capacity")
        if batch == self.chunk_queries:
            return self.config
        # Rebuild actual scalar/page shapes, not the power-of-two upper bound.
        return _config_for_batch(self.config, batch)


def _plan_launch(
    cfgs: configs.RpaConfigs,
    info: pltpu.TpuInfo,
    policy: DecodePolicy = DEFAULT_POLICY,
) -> DecodeLaunch | None:
    size = cfgs.serve.num_seqs
    while size:
        cfgs = _config_for_batch(cfgs, size)
        resources = resource_usage(cfgs)
        if resources.fits(info):
            return DecodeLaunch(policy=policy, config=cfgs, resources=resources)
        size //= 2
    return None


def query_batch_size(
    batch: int,
    heads: int,
    dim: int,
    kv_heads: int,
    ratio: int,
    groups: int,
    cache_dtype: jnp.dtype,
) -> int:
    """Largest power-of-two launch fitting resources; zero means use RPA."""
    info = pltpu.get_tpu_info()
    capacity = 1 << (batch - 1).bit_length()
    cfgs = make_config(
        capacity, heads, dim, kv_heads, ratio, groups, cache_dtype, info=info
    )
    launch = _plan_launch(cfgs, info)
    return launch.chunk_queries if launch else 0

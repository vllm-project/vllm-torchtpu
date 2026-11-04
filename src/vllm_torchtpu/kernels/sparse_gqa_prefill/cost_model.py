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
"""Shape-derived TC/SC work estimates, in milliseconds.

Estimates combine scheduled work with hardware peak bandwidth and dot rates.
Attained efficiency and submission latency are not modeled; these estimates
are heuristics rather than runtime bounds.
"""

import math
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_gqa_prefill import slab_gather


@dataclass(frozen=True)
class CostModel:
    tc_fixed_ms: float
    tc_word_ms: float
    tc_visit_ms: float
    sc_fixed_ms: float
    sc_tile_ms: float
    sc_page_ms: float
    work: dict = field(default_factory=dict, compare=False, hash=False)

    def tc_ms(self, words, visits):
        return self.tc_fixed_ms + self.tc_word_ms * words + self.tc_visit_ms * visits

    def sc_ms(self, tiles, pages):
        return self.sc_fixed_ms + self.sc_tile_ms * tiles + self.sc_page_ms * pages


def build(query, blocks, table, cache, starts, *, kv_heads, ratio, info=None):
    """Account scheduled work without a file, shape table or context threshold."""
    if info is None:
        info = pltpu.get_tpu_info()
    q, heads, dim = query.shape
    qb, tile, chunk, group = 256, 256, 64, 8
    qblocks = (q + qb - 1) // qb
    chunks = (q + chunk - 1) // chunk
    packing = cache.shape[3]
    head_axes = cache.shape[2]
    dtype_bytes = jnp.dtype(cache.dtype).itemsize
    query_bytes = jnp.dtype(query.dtype).itemsize
    padded_heads = (heads // kv_heads + 7) // 8 * 8
    sparse = info.sparse_core
    assert sparse is not None
    from vllm_torchtpu.kernels.sparse_gqa_prefill.owner_window import window_data_pages

    staged_capacity = window_data_pages(cache, starts, table, None)
    granularity = math.lcm(
        4 * sparse.num_lanes * sparse.num_cores * sparse.num_subcores, head_axes
    )
    capacity = (blocks.shape[1] * ratio + ratio - 1 + tile - 1) // tile * tile
    slot_rows = (
        (chunk * capacity * head_axes + granularity - 1) // granularity * granularity
    )
    slot_bytes = slot_rows * packing * dim * dtype_bytes
    slab_bytes = (
        q
        * slab_gather.field_layout(blocks.shape[1], table.shape[1])["stride_words"]
        * 4
    )
    # Q/gate pack and read, attention write/read, and the canonical outer gate.
    query_io = q * heads * dim * query_bytes
    packed_io = q * kv_heads * padded_heads * dim * query_bytes
    plan_bytes = blocks.size * 4 + table.size * 4 + q * 4
    stage_page_bytes = math.prod(cache.shape[1:]) * dtype_bytes
    gather_tile_bytes = group * tile * head_axes * packing * dim * dtype_bytes
    # Each independent query has one QK and one PV. There is no cross-query
    # ownership-diagonal multiplier. These are logical operations, not MXU passes.
    sc_tile_flops = 4 * group * kv_heads * padded_heads * tile * dim
    tc_visit_flops = 4 * qb * heads * tile * dim
    tc_visit_bytes = tile * head_axes * packing * dim * dtype_bytes
    bandwidth = info.mem_bw_bytes_per_second * info.num_cores
    dot_rate = info.bf16_ops_per_second * info.num_cores
    ms_per_byte = 1000.0 / bandwidth
    ms_per_flop = 1000.0 / dot_rate
    # A serial sum does not assume ideal SC/TC overlap. Submission and efficiency
    # corrections are currently unknown, rather than hidden in a fixed latency.
    tc_fixed = (4 * query_io + blocks.size * 4 + table.size * 4) * ms_per_byte
    sc_fixed = (
        plan_bytes
        + 2 * slab_bytes
        + 2 * slot_bytes
        + 2 * chunks * slot_bytes
        + 6 * packed_io
        + 4 * query_io
        + stage_page_bytes
    ) * ms_per_byte
    work = dict(
        stage_grid_iterations=staged_capacity + 1,
        sparse_core_lanes=sparse.num_lanes,
        sparse_core_VMEM_bytes=sparse.vmem_capacity_bytes,
        sparse_core_DMA_granule_bytes=sparse.dma_granule_size_bytes,
        query_blocks=qblocks,
        microchunks=chunks,
        query_group=group,
        producer_read_write_bytes=2 * chunks * slot_bytes,
        capacity=capacity,
        slot_bytes=slot_bytes,
        zeroed_slots=2,
        metadata_slab_bytes=slab_bytes,
        plan_bytes=plan_bytes,
        page_bytes=stage_page_bytes,
        gather_tile_bytes=gather_tile_bytes,
        SC_logical_flops_per_group_tile=sc_tile_flops,
        TC_logical_flops_per_visit=tc_visit_flops,
        TC_read_bytes_per_visit=tc_visit_bytes,
        peak_bandwidth_bytes_per_second=bandwidth,
        peak_bf16_ops_per_second=dot_rate,
        submission_cost_measured=False,
        efficiency_measured=False,
        overlap_assumed=False,
    )
    return CostModel(
        tc_fixed,
        2
        * qblocks
        * ((qb + info.num_lanes - 1) // info.num_lanes * info.num_lanes)
        * 4
        * ms_per_byte,
        tc_visit_bytes * ms_per_byte + tc_visit_flops * ms_per_flop,
        sc_fixed,
        gather_tile_bytes * ms_per_byte + sc_tile_flops * ms_per_flop,
        2 * stage_page_bytes * ms_per_byte,
        work,
    )

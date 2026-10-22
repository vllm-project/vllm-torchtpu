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
"""Materialize native uniform placements as main/index carrier pairs."""

import math

import jax.numpy as jnp
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.kv_cache_materializer import MaterializedKVCache, layer_to_pool_index

from .pool_layout import INDEX_LANES, Qwen4ExpPoolLayout
from .qsa import Qwen4ExpPooledQSABackend, get_qsa_cache_specs


def validate_pool_tiling(
    layout: Qwen4ExpPoolLayout, hardware: pltpu.TpuInfo | None
) -> None:
    """Reject unknown/incompatible layouts before allocating or tracing."""
    if not layout.index_page_bytes:
        return
    if hardware is None:
        raise ValueError("Qwen4Exp compact pool requires known TPU layout facts")
    if (
        hardware.num_lanes != INDEX_LANES
        or layout.compressed_width % hardware.num_lanes
        or layout.index_rows % hardware.get_sublane_tiling(jnp.bfloat16)
    ):
        raise ValueError(
            "Qwen4Exp compact pool is incompatible with this TPU lane/sublane layout"
        )
    # This is a necessary lower bound for the changed pair-writer tile, not
    # an estimate of the complete fused program's VMEM peak. Existing scorer,
    # main, GDN and PLE kernels retain their own compiler resource checks.
    values = 128 * 2 * layout.compressed_width * 2
    metadata = 128 * (4 + 4 + 2 * 4)
    double_dma = 2 * layout.compressed_kernel_bytes
    if values + metadata + double_dma >= hardware.vmem_capacity_bytes:
        raise ValueError("Qwen4Exp compact index tile exceeds available VMEM")


def materialize_pool(
    cfg, config, layout: Qwen4ExpPoolLayout, device
) -> MaterializedKVCache:
    placements = layer_to_pool_index(config)
    blocks = config.num_blocks
    layout.validate_capacity(blocks)
    backing = {t.size for t in config.kv_cache_tensors}
    assert len(backing) == 1
    total = backing.pop()
    regions, remainder = divmod(total, blocks * layout.page_bytes)
    assert not remainder
    main_spec, _ = get_qsa_cache_specs(
        cfg.model_config.hf_text_config,
        cfg.parallel_config.tensor_parallel_size,
        cfg.cache_config.cache_dtype,
    )
    shape = Qwen4ExpPooledQSABackend.get_kv_cache_shape(
        blocks,
        layout.block_size,
        main_spec.num_kv_heads,
        main_spec.head_size,
        cache_dtype_str=cfg.cache_config.cache_dtype,
    )
    assert math.prod(shape[1:]) * main_spec.dtype.itemsize == layout.main_page_bytes
    allocate = torch.empty if main_spec.dtype == torch.float8_e4m3fn else torch.zeros
    mains = [
        allocate(shape, dtype=main_spec.dtype, device=device) for _ in range(regions)
    ]
    indices = (
        [
            torch.zeros(
                (blocks, layout.index_rows // 2, 1, 2, layout.compressed_width),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(regions)
        ]
        if layout.index_page_bytes
        else []
    )
    raw = mains + indices
    assert sum(t.untyped_storage().nbytes() for t in raw) == total
    kv_caches = {}
    layer_to_raw = {}
    state_names = {
        n
        for g in config.kv_cache_groups
        if isinstance(g.kv_cache_spec, MambaSpec)
        for n in g.layer_names
    }
    for name, region in placements.items():
        tensor = indices[region] if name.endswith(".raw_ring") else mains[region]
        layer_to_raw[name] = tensor
        kv_caches[name] = [tensor] if name in state_names else tensor
        if name.endswith(".self_attn"):
            kv_caches[name + ".compressed_k"] = indices[region]
    return MaterializedKVCache(
        kv_caches=kv_caches, raw_tensors=raw, layer_to_raw=layer_to_raw
    )

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
"""Config-only block-major layout contract for the Raiden offload path.

The block-major layout bundles the full-model KV cache into a single
contiguous array in HBM (`[kernel_blocks, F, *R]`). This aligns all layer
fragments of a logical block contiguously in memory, collapsing per-block
save/load/transfer operations from F independent DMAs into a single hardware DMA.

This module derives the layout contract ahead of physical tensor allocation:
`resolve_block_major_contract` derives fragment count, order, memory geometry,
and a deterministic logical fingerprint purely from the resolved engine and
KV cache configurations. Both the scheduler and worker ranks compute identical
contracts without requiring device pointers. When `VLLM_TPU_BLOCK_MAJOR_KV` is
disabled, it returns None; on unsupported geometries, it fails closed (raises)
to prevent cross-layout cache corruption.

The bundled attention op relies on the stock Pallas `jax_op` buffer donation
for in-place bundle updates, avoiding intermediate memory copies without
requiring dynamic capability probes.

The contract feeds three primary consumers:
- Offload namespace: Salts cache keys with the layout fingerprint to prevent
  cross-layout aliasing with layer-major entries in shared storage.
- Worker device view: Reconstructs the canonical array registered with Raiden
  for single-DMA transfers.
- Connector startup gate: Verifies layout compatibility and rejects unsupported
  model topologies early.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Schema version of the bundle memory layout (distinct from model-specific parameters).
# Included in the logical fingerprint and namespace salt to invalidate stale cache entries
# across layout format changes.
BLOCK_MAJOR_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class BlockMajorContract:
    """Immutable specification of the bundled KV layout agreed upon across ranks.

    Attributes:
        fragment_count: Number of physical fragments folded into the bundle's layer dimension (dim 1).
        fragment_row_bytes: Byte size of a single fragment's row for one kernel block.
        bundle_row_bytes: Total contiguous byte length of one bundled kernel block across all fragments (= F * fragment_row_bytes).
        logical_fingerprint: Deterministic SHA-256 digest of layout parameters; must match across all serving peers.
    """
    fragment_count: int
    fragment_row_bytes: int
    bundle_row_bytes: int
    logical_fingerprint: str


def _canonical_layout_fingerprint(value: Mapping[str, Any]) -> str:
    """Computes a deterministic SHA-256 fingerprint from a canonical JSON layout payload."""
    encoded = json.dumps(dict(value),
                         sort_keys=True,
                         separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def block_major_layer_indices(
    kv_cache_config: KVCacheConfig,
    fragment_row_bytes: int,
) -> dict[str, int]:
    """Read layer positions from vLLM's native block-outermost placement."""
    tensors = kv_cache_config.kv_cache_tensors
    sizes = {tensor.size for tensor in tensors}
    if len(sizes) != 1:
        raise ValueError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1: backing sizes are not uniform")
    size = sizes.pop()
    row_bytes, remainder = divmod(size, kv_cache_config.num_blocks)
    if remainder or row_bytes % fragment_row_bytes:
        raise ValueError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1: backing allocation does not "
            "tile whole scheduler blocks and fragment pages")
    result = {}
    for tensor in tensors:
        if tensor.block_stride != row_bytes:
            raise ValueError(
                "VLLM_TPU_BLOCK_MAJOR_KV=1: placement is not block-major")
        for index, name in enumerate(tensor.layers):
            offset = tensor.offset + index * tensor.layer_stride
            if offset % fragment_row_bytes or not 0 <= offset < row_bytes:
                raise ValueError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: fragment page offset "
                    f"does not match kernel-row bytes: {name}={offset}")
            result[name] = offset // fragment_row_bytes
    if set(result.values()) != set(range(row_bytes // fragment_row_bytes)):
        raise ValueError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1: placement does not tile kernel-row bytes (empty fragments)"
        )
    return result


def resolve_block_major_contract(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> BlockMajorContract | None:
    """Derives the block-major layout contract from engine and KV cache configurations.

    Args:
        vllm_config: Resolved vLLM engine configuration.
        kv_cache_config: Resolved KV cache configuration containing tensor specifications.

    Returns:
        A BlockMajorContract describing the bundled layout, or None if
        VLLM_TPU_BLOCK_MAJOR_KV is disabled.

    Raises:
        ValueError: If VLLM_TPU_BLOCK_MAJOR_KV is enabled but the KV cache geometry
            cannot be represented as a uniform bundle. Fails closed to prevent
            cross-layout cache corruption.
    """
    if not tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV:
        return None

    # Deferred import to avoid circular dependency with raiden_store.
    from vllm_torchtpu.offload.raiden_store import (is_multi_shapes_geometry,
                                                    resolve_kernel_geometry)

    (kernel_block_size, per_block_shape, kv_dtype,
     device_block_size) = resolve_kernel_geometry(vllm_config, kv_cache_config)
    if is_multi_shapes_geometry(per_block_shape):
        raise ValueError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1 not yet supported for multi-shapes KV cache."
        )
    assert device_block_size % kernel_block_size == 0
    factor = device_block_size // kernel_block_size
    if factor != 1:
        # The bundled RPA kernel and worker view require factor == 1 (device_block_size == kernel_block_size).
        # A factor > 1 bundle would pass flat byte-size validations while scrambling layer vs. sub-block
        # dimensional ordering during DMA transfers.
        raise ValueError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1: device_block_size "
            f"{device_block_size} != kernel_block_size {kernel_block_size} "
            f"(factor {factor}); the block-major bundle requires them to "
            "match")

    # Compute the physical byte size of a single fragment row for one kernel block.
    import numpy as np
    import torch
    fragment_row_bytes = (int(np.prod(per_block_shape)) *
                          torch.empty(0, dtype=kv_dtype).element_size())

    indices = block_major_layer_indices(kv_cache_config, fragment_row_bytes)
    fragment_count = len(set(indices.values()))
    fragment_layers = [[] for _ in range(fragment_count)]
    for name, index in indices.items():
        fragment_layers[index].append(name)
    fragment_order = tuple(f"{index}:{names[0]}(+{len(names) - 1})"
                           for index, names in enumerate(fragment_layers))

    logical_fingerprint = _canonical_layout_fingerprint({
        "layout":
        "block-major",
        "layout_version":
        BLOCK_MAJOR_LAYOUT_VERSION,
        "fragment_count":
        fragment_count,
        "fragment_order":
        list(fragment_order),
        "fragment_row_bytes":
        fragment_row_bytes,
        "kernel_block_size":
        kernel_block_size,
        "device_block_size":
        device_block_size,
        "dtype":
        str(kv_dtype),
        "per_block_shape":
        list(per_block_shape),
    })

    contract = BlockMajorContract(
        fragment_count=fragment_count,
        fragment_row_bytes=fragment_row_bytes,
        bundle_row_bytes=fragment_count * fragment_row_bytes,
        logical_fingerprint=logical_fingerprint,
    )
    logger.info(
        "Block-major contract: F=%d fragments x %d B kernel rows "
        "(bundle row %d B, kernel_block_size=%d, factor=%d), "
        "logical_fingerprint=%s", contract.fragment_count,
        contract.fragment_row_bytes, contract.bundle_row_bytes,
        kernel_block_size, factor, contract.logical_fingerprint)
    return contract

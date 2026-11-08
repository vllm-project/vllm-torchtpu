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
"""GLM-5.3-Flash on the unified hybrid KV pool (what prefix caching needs).

Prefix caching on a hybrid model needs every recurrent state to live in a
block the scheduler owns (``mamba_cache_mode='align'``), which on TPU means
the unified pool: every KV cache group shares one block-id space and one
page size, and a block holds whatever its owning group stores.

The stock unified pool hands every layer one attention-shaped tensor. That
does not fit GLM-5.3: its 11 sparse-MLA layers need their native ``(nope,
rope)`` split cache and each DSA indexer needs its pool-key cache. Instead,
each pool region (one layer of every group, see
`kv_cache_materializer.layer_to_pool_index`) is three native tensors that
together are exactly one manager page per block:

    nope      (blocks, block_size, 4, 128) uint8   512 B / token
    rope      (blocks, block_size // 4, 4, 128) uint8   128 B / token
    pool key  (blocks, block_size // 16, 4, 256) uint8   64 B / token

and the groups use them as follows:

- sparse MLA reads ``(nope, rope)`` as is: no kernel change.
- the pool-key cache is folded into the MLA group instead of being a group
  of its own. A separate group would get a whole (MLA-sized) page per block
  for 64 B / token of records, roughly doubling the attention footprint. Its
  layer rides the MLA group's block table (the runner adds it to that
  group's metadata) and reads the region's pool-key tensor.
- KDA keeps its recurrent and conv state in the nope tensor of the block it
  owns, one 512-byte token row per pool row: ``ssm_tokens`` rows of fp32
  state, then a power-of-two run of bf16 conv rows (`gdn_pool_layout`).
- the indexer's member state (the incomplete trailing pool) uses the first
  ``kpool`` token rows of the nope tensor; one row is exactly one
  ``[k | gate]`` bf16 member row.

A block owned by one group leaves the other tensors' share of that block
unused; that is the price of one block-id space.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    derive_pooled_gdn_state_layout,
    pooled_gdn_state_itemsize,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

GLM5_NEXT_ARCHITECTURES = frozenset(
    {"Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"}
)

# One nope token row: the KDA state regions and the member rows are measured
# in these.
NOPE_ROW_BYTES = 512


def is_glm5_next(vllm_config: VllmConfig) -> bool:
    return vllm_config.model_config.architecture in GLM5_NEXT_ARCHITECTURES


def glm5_unified_pool_enabled(vllm_config: VllmConfig) -> bool:
    from vllm_torchtpu.platforms.tpu_block_size_utils import unified_kv_layout_enabled

    return is_glm5_next(vllm_config) and unified_kv_layout_enabled(vllm_config)


def _mla_head_size(vllm_config: VllmConfig) -> int:
    config = vllm_config.model_config.hf_text_config
    return config.kv_lora_rank + config.qk_rope_head_dim


def sparse_mla_specs(vllm_config: VllmConfig, num_pages: int, block_size: int):
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend

    return PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        num_pages,
        block_size,
        _mla_head_size(vllm_config),
        vllm_config.cache_config.cache_dtype,
    )


def pool_key_shape(num_pages: int, block_size: int) -> tuple[int, ...]:
    from vllm_torchtpu.models.vllm.glm5_next.indexer import (
        POOL_KEY_KPOOL,
        POOL_KEY_RECORD_BYTES,
        Glm5NextPoolKeyBackend,
    )

    return tuple(
        Glm5NextPoolKeyBackend.get_kv_cache_shape(
            num_pages, block_size, 1, POOL_KEY_RECORD_BYTES // POOL_KEY_KPOOL
        )
    )


def _spec_bytes(spec) -> int:
    return math.prod(spec.shape) * spec.torch_dtype.itemsize


def page_bytes(vllm_config: VllmConfig, block_size: int) -> int:
    """Bytes of one manager block across the three region tensors."""
    nope, rope = sparse_mla_specs(vllm_config, 1, block_size)
    return (
        _spec_bytes(nope) + _spec_bytes(rope) + math.prod(pool_key_shape(1, block_size))
    )


def kda_state_rows(vllm_config: VllmConfig) -> int:
    """Nope token rows one KDA layer's state occupies in its block."""
    config = vllm_config.model_config.hf_text_config
    kda = config.linear_attn_config
    heads = kda["num_heads"] // vllm_config.parallel_config.tensor_parallel_size
    head_dim = kda["head_dim"]
    conv_elem = pooled_gdn_state_itemsize(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE)
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=heads * head_dim * head_dim * 4,
        conv_bytes=(kda["short_conv_kernel_size"] - 1)
        * 3
        * heads
        * head_dim
        * conv_elem,
        token_bytes=NOPE_ROW_BYTES,
    )
    return layout.required_tokens


def update_block_config(vllm_config: VllmConfig) -> None:
    """Pick the manager block size and the uniform page for the GLM pool.

    The block must hold one KDA layer's state in its nope rows. It is rounded
    up to a power of two: prefill chunks (8192) then split into whole blocks,
    which align mode requires of every chunk but a prompt's last.
    """
    cache_config = vllm_config.cache_config
    rows = kda_state_rows(vllm_config)
    floor = max(rows, cache_config.block_size or 0)
    block_size = 1 << (floor - 1).bit_length()
    page = page_bytes(vllm_config, block_size)
    logger.info(
        "GLM-5.3 unified pool: block_size %s -> %s (KDA state %s rows of %s B),"
        " page %s B (nope + rope + pool key)",
        cache_config.block_size,
        block_size,
        rows,
        NOPE_ROW_BYTES,
        page,
    )
    cache_config.block_size = block_size
    if cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = block_size
    cache_config.mamba_page_size_padded = page


def pool_key_pairs(layer_names) -> dict[str, str]:
    """Pool-key cache layer -> the MLA layer of the same decoder layer."""
    by_layer: dict[str, str] = {}
    pool_keys: list[str] = []
    for name in layer_names:
        if name.endswith(".self_attn.indexer.key_cache"):
            pool_keys.append(name)
        elif name.endswith(".self_attn.attn"):
            by_layer[name.split(".self_attn.", 1)[0]] = name
    pairs = {}
    for name in pool_keys:
        decoder = name.split(".self_attn.", 1)[0]
        if decoder not in by_layer:
            raise ValueError(f"no MLA layer next to pool-key cache {name}")
        pairs[name] = by_layer[decoder]
    return pairs


def materialize(
    *,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    attn_groups,
    pool_key_layers: dict[str, str],
    device: torch.device,
    forward_context: dict[str, Any],
    runner,
):
    """Allocate the region tensors and bind every layer to its share.

    Returns ``(kv_caches, raw_tensors, layer_to_raw)``; ``layer_to_raw`` maps
    the state (mamba) layers to the nope tensor their blocks live in, which
    is what the align-mode seed copies duplicate.
    """
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch
    from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index

    num_blocks = kv_cache_config.num_blocks
    block_size = vllm_config.cache_config.block_size
    expected_page = page_bytes(vllm_config, block_size)
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        # A mamba group's block_size is the tokens one state covers: the pool
        # block size in align mode, the whole context otherwise. Only its
        # page has to match.
        size_ok = isinstance(spec, MambaSpec) or spec.block_size == block_size
        if not size_ok or spec.page_size_bytes != expected_page:
            raise AssertionError(
                "GLM-5.3 unified pool expects every group at block_size "
                f"{block_size} and page {expected_page} B, got "
                f"{type(spec).__name__} block_size={spec.block_size} "
                f"page={spec.page_size_bytes}"
            )

    region_of = layer_to_pool_index(kv_cache_config)
    num_regions = len(set(region_of.values()))
    nope_spec, rope_spec = sparse_mla_specs(vllm_config, num_blocks, block_size)
    regions = []
    for _ in range(num_regions):
        nope = torch.zeros(nope_spec.shape, dtype=nope_spec.torch_dtype, device=device)
        rope = torch.zeros(rope_spec.shape, dtype=rope_spec.torch_dtype, device=device)
        pool_key = torch.zeros(
            pool_key_shape(num_blocks, block_size), dtype=torch.uint8, device=device
        )
        regions.append((nope, rope, pool_key))
    logger.info(
        "GLM-5.3 unified pool: %d regions x %d blocks of %d tokens; per "
        "region nope %s, rope %s, pool key %s",
        num_regions,
        num_blocks,
        block_size,
        tuple(nope_spec.shape),
        tuple(rope_spec.shape),
        pool_key_shape(num_blocks, block_size),
    )

    kv_caches: dict[str, Any] = {}
    layer_to_raw: dict[str, torch.Tensor] = {}
    for group_list in attn_groups:
        for group in group_list:
            spec = group.kv_cache_spec
            for name in group.layer_names:
                if name in pool_key_layers:
                    continue
                nope, rope, _ = regions[region_of[name]]
                if isinstance(spec, MambaSpec):
                    kv_caches[name] = [nope]
                    layer_to_raw[name] = nope
                elif isinstance(spec, MLAAttentionSpec):
                    module = forward_context[name]
                    if not getattr(module, "use_sparse", False):
                        raise NotImplementedError(
                            f"GLM-5.3 unified pool: {name} is not sparse MLA"
                        )
                    specs = sparse_mla_specs(vllm_config, num_blocks, block_size)
                    mla_dispatch.admit_layer(runner, module, specs, block_size)
                    module.mla_kv_spec = specs
                    kv_caches[name] = (nope, rope)
                else:
                    raise NotImplementedError(
                        f"GLM-5.3 unified pool: unexpected {type(spec)!r} for {name}"
                    )
    for pool_key_name, mla_name in pool_key_layers.items():
        kv_caches[pool_key_name] = regions[region_of[mla_name]][2]

    raw_tensors = [t for region in regions for t in region]
    return kv_caches, raw_tensors, layer_to_raw

# SPDX-License-Identifier: Apache-2.0
"""Model-owned geometry for QSA, GDN and PLE in attention-shaped pool pages.

The scheduler page fits every state type. QSA continues to use 128-token
kernel pages; compressed keys remain BF16 even when the pool carrier is FP8.
No allocation or device initialization takes place here.
"""

import math
from dataclasses import dataclass, replace

import torch
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

from .config import native_cache_specs
from .qsa import QSA_BLOCK_SIZE

# Explicit physical format shared by this model's page planner and GDN ops.
POOLED_GDN_CONV_STATE_DTYPE = torch.float32


@dataclass(frozen=True)
class Qwen4ExpPoolLayout:
    block_size: int
    page_bytes: int
    token_bytes: int
    kernel_block_size: int
    gdn_state_bytes: int
    ple_state_bytes: int
    compressed_kernel_rows: int
    compressed_kernel_bytes: int
    compressed_width: int

    @property
    def main_kernel_pages(self):
        return self.block_size // self.kernel_block_size

    @property
    def compressed_physical_pages(self):
        return self.page_bytes // self.compressed_kernel_bytes


def pooled_cache_specs(cfg):
    """Derive uniform physical pages without widening compressed-key coverage.

    Increasing a compressed page's *logical* span merely to fill the pool
    would multiply prefix-match alignment by the compression ratio. Instead,
    main and compressed pages cover the same token interval; compressed pages
    have unused tail bytes. Raw rings retain their non-cacheable lifecycle.
    """
    specs = native_cache_specs(cfg)
    text = cfg.model_config.hf_text_config
    main = specs[f"model.layers.{text.layer_types.index('full_attention')}.self_attn"]
    token_bytes = main.page_size_bytes // main.block_size
    gdn = specs[
        f"model.layers.{text.layer_types.index('linear_attention')}.linear_attn"
    ]
    assert isinstance(gdn, MambaSpec)
    state = derive_pooled_gdn_state_layout(
        ssm_bytes=math.prod(gdn.shapes[1]) * gdn.dtypes[1].itemsize,
        conv_bytes=math.prod(gdn.shapes[0]) * POOLED_GDN_CONV_STATE_DTYPE.itemsize,
        token_bytes=token_bytes,
    )
    ple_bytes = max(
        (spec.page_size_bytes for name, spec in specs.items() if name.endswith(".ple")),
        default=0,
    )
    required_rows = max(state.required_tokens, math.ceil(ple_bytes / token_bytes))
    block_size = math.ceil(required_rows / QSA_BLOCK_SIZE) * QSA_BLOCK_SIZE
    page_bytes = block_size * token_bytes
    compressed_rows = QSA_BLOCK_SIZE // text.indexer_compress_ratio
    # Preserve the pool's lane axis. Narrowing 256 lanes to 128 outside the
    # kernel can relayout the entire owner; a padded query/key contraction
    # instead pays for only the active key tiles and keeps native RPA storage.
    compressed_width = math.ceil(text.head_dim / 128) * 128
    if text.indexer_head_dim > compressed_width:
        raise ValueError(
            "Qwen4Exp pooled indexer width must fit the main KV lane width"
        )
    compressed_bytes = compressed_rows * compressed_width * 2
    if page_bytes % compressed_bytes:
        raise ValueError("Qwen4Exp pool page must contain whole BF16 indexer pages")
    layout = Qwen4ExpPoolLayout(
        block_size=block_size,
        page_bytes=page_bytes,
        token_bytes=token_bytes,
        kernel_block_size=QSA_BLOCK_SIZE,
        gdn_state_bytes=state.required_bytes,
        ple_state_bytes=ple_bytes,
        compressed_kernel_rows=compressed_rows,
        compressed_kernel_bytes=compressed_bytes,
        compressed_width=compressed_width,
    )
    pooled: dict[str, KVCacheSpec] = {}
    # Upstream uses one persistent Mamba state block per request without APC.
    # Its logical span is max_model_len, while the physical state still fits
    # in a single uniform pool page. APC uses position-indexed align blocks.
    mamba_mode = "align" if cfg.cache_config.enable_prefix_caching else "none"
    mamba_block_size = (
        block_size if mamba_mode == "align" else cfg.model_config.max_model_len
    )
    for name, spec in specs.items():
        if isinstance(spec, MambaSpec):
            dtypes = (
                (POOLED_GDN_CONV_STATE_DTYPE, spec.dtypes[1])
                if name.endswith(".linear_attn")
                else spec.dtypes
            )
            pooled[name] = replace(
                spec,
                block_size=mamba_block_size,
                page_size_padded=page_bytes,
                dtypes=dtypes,
                mamba_cache_mode=mamba_mode,
            )
        else:
            assert isinstance(spec, AttentionSpec)
            if name.endswith(".compressed_k"):
                assert isinstance(spec, MLAAttentionSpec)
                spec = replace(spec, state_content_bytes=compressed_width * 2)
            pooled[name] = replace(
                spec,
                block_size=spec.block_size
                if name.endswith(".raw_ring")
                else block_size,
                page_size_padded=page_bytes,
            )
    return layout, pooled

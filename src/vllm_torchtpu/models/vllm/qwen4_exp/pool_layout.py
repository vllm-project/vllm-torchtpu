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
"""Logical prefix blocks with separate main and compact BF16 index carriers."""

import math
from collections.abc import Collection
from dataclasses import dataclass, replace

from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

from .config import native_cache_specs
from .qsa import QSA_BLOCK_SIZE, get_qsa_cache_specs, qsa_ring_capacity
from .qsa_geometry import rope_axis_assignment

# Existing RPA/StreamIndex format ABI, independently checked against worker
# TPUInfo. Sixteen BF16 rows cover both pinned small/large sublane layouts.
INDEX_LANES = 128
INDEX_ROW_TILE = 16


def _layer_order_key(name: str) -> tuple[int, str]:
    for part in name.split("."):
        if part.isdigit():
            return (int(part), name)
    return (10**9, name)


def resolve_ple_companion(ple_name: str, linear_owners: Collection[str]) -> str | None:
    """Return the GDN state owner whose block/checkpoint indices PLE shares."""
    same = ple_name.removesuffix(".ple") + ".linear_attn"
    if same in linear_owners:
        return same
    if not linear_owners:
        return None
    return min(linear_owners, key=_layer_order_key)


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
    main_page_bytes: int
    index_page_bytes: int
    index_tag_fields: int = 4
    cache_tags: bool = True
    ring_carrier_rows: int = 8
    ring_capacity: int = 4

    def validate_capacity(self, blocks: int) -> None:
        if blocks <= 0 or blocks * max(self.block_size, self.index_rows // 2) >= 2**31:
            raise ValueError("Qwen4Exp pool capacity exceeds positive int32 addressing")

    @property
    def main_kernel_pages(self):
        return self.block_size // self.kernel_block_size

    @property
    def compressed_physical_pages(self):
        return self.index_page_bytes // self.compressed_kernel_bytes

    @property
    def index_rows(self):
        return self.index_page_bytes // (self.compressed_width * 2)

    @property
    def ring_rows(self):
        return 2 * self.ring_carrier_rows

    @property
    def carrier_row_bytes(self):
        return 4 * self.compressed_width

    @property
    def ple_rows(self):
        return (
            math.ceil(self.ple_state_bytes / self.carrier_row_bytes)
            if self.ple_state_bytes > 0
            else 0
        )

    @property
    def ple_page_bytes(self):
        return self.ple_rows * self.carrier_row_bytes

    @property
    def ring_slot_bytes(self):
        return (
            self.ring_carrier_rows * self.carrier_row_bytes
            if self.index_page_bytes
            else 0
        )


def pooled_cache_specs(cfg):
    """Co-own KV and compressed keys without changing original-token coverage.

    Native blocks retain uniform physical size. Each is materialized as a main
    carrier plus a compact BF16 sidecar; non-cacheable rings and PLE recurrent
    state use dedicated companion pools outside the 4 scheduler groups.
    """
    specs = native_cache_specs(cfg)
    text = cfg.model_config.hf_text_config
    main, _ = get_qsa_cache_specs(
        text, cfg.parallel_config.tensor_parallel_size, cfg.cache_config.cache_dtype
    )
    token_bytes = main.page_size_bytes // main.block_size
    gdn_specs = [s for n, s in specs.items() if n.endswith(".linear_attn")]
    gdn_bytes = 0
    if gdn_specs:
        gdn = gdn_specs[0]
        assert isinstance(gdn, MambaSpec)
        state = derive_pooled_gdn_state_layout(
            ssm_bytes=math.prod(gdn.shapes[1]) * gdn.dtypes[1].itemsize,
            conv_bytes=math.prod(gdn.shapes[0]) * gdn.dtypes[0].itemsize,
            token_bytes=token_bytes,
        )
        gdn_bytes = state.required_bytes
    ple_bytes = max(
        (s.page_size_bytes for n, s in specs.items() if n.endswith(".ple")),
        default=0,
    )
    required_rows = max(
        QSA_BLOCK_SIZE,
        cfg.cache_config.block_size or 0,
        math.ceil(gdn_bytes / token_bytes),
    )
    block_size = math.ceil(required_rows / QSA_BLOCK_SIZE) * QSA_BLOCK_SIZE
    main_bytes = block_size * token_bytes
    ratio = text.indexer_compress_ratio
    if ratio <= 0 or QSA_BLOCK_SIZE % ratio:
        raise ValueError("pooled compression ratio must divide the QSA kernel block")
    compressed_rows = QSA_BLOCK_SIZE // ratio
    if compressed_rows % 2:
        raise ValueError("pooled index pages require whole BF16 key pairs")
    width = math.ceil(text.indexer_head_dim / INDEX_LANES) * INDEX_LANES
    if (
        text.indexer_head_dim <= 0
        or width > math.ceil(text.head_dim / INDEX_LANES) * INDEX_LANES
    ):
        raise ValueError("pooled indexer width must fit the qualified main lane width")
    compressed_bytes = compressed_rows * width * 2
    has_qsa = any(n.endswith(".compressed_k") for n in specs)
    alignment = math.lcm(2, INDEX_ROW_TILE, compressed_rows)
    rows = block_size // ratio
    index_bytes = math.ceil(rows / alignment) * alignment * width * 2 if has_qsa else 0
    cache_tags = (
        rope_axis_assignment(
            text.rope_parameters, int(text.head_dim * text.partial_rotary_factor)
        )
        is not None
    )
    spec_cfg = getattr(cfg, "speculative_config", None)
    num_spec = (
        getattr(spec_cfg, "num_speculative_tokens", 0) or 0
        if spec_cfg is not None
        else 0
    )
    ring_cap = qsa_ring_capacity(ratio, num_spec)
    ring_logical_rows = math.ceil(ring_cap / 8) * 8
    ring_bf16_carriers = ring_logical_rows // 2
    tag_carriers = math.ceil(ring_logical_rows * 4 / width) * 4 if cache_tags else 0
    layout = Qwen4ExpPoolLayout(
        block_size=block_size,
        page_bytes=main_bytes + index_bytes,
        token_bytes=token_bytes,
        kernel_block_size=QSA_BLOCK_SIZE,
        gdn_state_bytes=gdn_bytes,
        ple_state_bytes=ple_bytes,
        compressed_kernel_rows=compressed_rows,
        compressed_kernel_bytes=compressed_bytes,
        compressed_width=width,
        main_page_bytes=main_bytes,
        index_page_bytes=index_bytes,
        index_tag_fields=4 if cache_tags else 0,
        cache_tags=cache_tags,
        ring_carrier_rows=ring_bf16_carriers + tag_carriers,
        ring_capacity=ring_cap,
    )
    pooled: dict[str, KVCacheSpec] = {}
    mamba_mode = "align" if cfg.cache_config.enable_prefix_caching else "none"
    mamba_block_size = (
        block_size if mamba_mode == "align" else cfg.model_config.max_model_len
    )
    for name, spec in specs.items():
        if name.endswith((".compressed_k", ".raw_ring", ".ple")):
            # Deterministic compressed keys share their main block's life;
            # raw_ring and PLE use dedicated companion pools outside the
            # scheduler's KVCacheGroups.
            continue
        if isinstance(spec, MambaSpec):
            pooled[name] = replace(
                spec,
                block_size=mamba_block_size,
                page_size_padded=layout.page_bytes,
                mamba_cache_mode=mamba_mode,
            )
        else:
            assert isinstance(spec, AttentionSpec)
            pooled[name] = replace(
                spec,
                block_size=block_size,
                page_size_padded=layout.page_bytes,
            )
    return layout, pooled

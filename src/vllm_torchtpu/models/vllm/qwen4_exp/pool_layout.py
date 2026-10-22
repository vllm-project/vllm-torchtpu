# SPDX-License-Identifier: Apache-2.0
"""Model-owned geometry for QSA, GDN and PLE in attention-shaped pool pages.

The scheduler page fits every state type. QSA continues to use 128-token
kernel pages; compressed keys remain BF16 even when the pool carrier is FP8.
No allocation or device initialization takes place here.
"""

import math
import os
from dataclasses import dataclass, field, replace

import torch
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

from .config import mtp_layer_prefixes, native_cache_specs
from .qsa import QSA_BLOCK_SIZE

# Explicit physical format shared by this model's page planner and GDN ops.
POOLED_GDN_CONV_STATE_DTYPE = torch.float32

PACK_COMPRESSED_K_ENV = "QWEN4_EXP_PACK_COMPRESSED_K"


def pack_compressed_k_enabled() -> bool:
    """Whether several QSA layers' compressed keys share one pool page.

    Off by default. ``QWEN4_EXP_PACK_COMPRESSED_K=1`` packs; see
    ``pooled_cache_specs``.
    """
    value = os.environ.get(PACK_COMPRESSED_K_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


PACK_MAIN_KV_ENV = "QWEN4_EXP_PACK_MAIN_KV"


def pack_main_kv_enabled() -> bool:
    """Whether two QSA layers' main KV heads share one pool page.

    Off by default. ``QWEN4_EXP_PACK_MAIN_KV=1`` pairs; see
    ``pooled_cache_specs``.
    """
    value = os.environ.get(PACK_MAIN_KV_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


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
    # Layers whose compressed keys share one pool page (1 = unpacked).
    compressed_slots: int = 1
    # (QSA attention prefix, compressed-key cache owner name, page slot);
    # empty when unpacked, where every layer owns ``<prefix>.compressed_k``.
    compressed_owners: tuple[tuple[str, str, int], ...] = field(default=())
    # (QSA attention prefix, main KV cache owner name, head-pair slot) of
    # layers sharing main KV pages; empty when every layer owns its pages.
    main_owners: tuple[tuple[str, str, int], ...] = field(default=())

    def compressed_owner(self, attention_prefix: str) -> tuple[str, int]:
        """Cache owner name and in-page slot of a QSA layer's compressed keys."""
        for prefix, owner, slot in self.compressed_owners:
            if prefix == attention_prefix:
                return owner, slot
        return attention_prefix + ".compressed_k", 0

    def main_owner(self, attention_prefix: str) -> tuple[str, int]:
        """Cache owner name and head-pair slot of a QSA layer's main KV.

        Slot -1: the layer owns whole rows of its own pages.
        """
        for prefix, owner, slot in self.main_owners:
            if prefix == attention_prefix:
                return owner, slot
        return attention_prefix, -1

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

    With ``QWEN4_EXP_PACK_COMPRESSED_K=1`` those tail bytes hold other
    layers' compressed keys instead: consecutive QSA layers (the MTP draft's
    included) form one scheduler group named after its first member, whose
    page stores member ``s`` at kernel pages
    ``[s * main_kernel_pages, (s + 1) * main_kernel_pages)``. Each member
    still covers exactly the group's token interval, so prefix alignment,
    hashing and eviction are unchanged; only the page count per token drops.

    ``QWEN4_EXP_PACK_MAIN_KV=1`` likewise pairs main KV pages. With one FP8
    KV head per device a token row has four head slots, K and V fill two.
    Consecutive target QSA layers pair up: the group is named after the
    first, member ``s`` stores K and V at head slots ``2s`` and ``2s + 1``.
    MTP draft layers keep pages of their own (the proposer addresses its
    layer's group by name), as does an odd last target layer.
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
    # A layer's compressed keys for one manager page fill only
    # ``main_kernel_pages`` kernel pages; the rest of the pool page is free.
    packed_slots = page_bytes // (block_size // QSA_BLOCK_SIZE * compressed_bytes)
    compressed_names = [name for name in specs if name.endswith(".compressed_k")]
    owners: list[tuple[str, str, int]] = []
    if pack_compressed_k_enabled() and packed_slots > 1:
        # Consecutive QSA layers (the MTP draft's last) share one owner
        # group; one block table and one page lifetime cover all of them,
        # since compressed keys of every layer track the same token span.
        for start in range(0, len(compressed_names), packed_slots):
            chunk = compressed_names[start : start + packed_slots]
            owners.extend(
                (name.removesuffix(".compressed_k"), chunk[0], slot)
                for slot, name in enumerate(chunk)
            )
    else:
        packed_slots = 1
    main_owners: list[tuple[str, str, int]] = []
    if pack_main_kv_enabled():
        packing = 4 if main.dtype == torch.float8_e4m3fn else 2
        if main.num_kv_heads != 1 or packing * main.num_head_slots != 4:
            raise ValueError(
                f"{PACK_MAIN_KV_ENV}=1 pairs layers in the two padding head "
                "slots of an FP8 main KV row, which needs one KV head per "
                f"device and --kv-cache-dtype fp8; got {main.num_kv_heads} "
                f"heads of {main.dtype}"
            )
        draft = {prefix + ".self_attn" for prefix in mtp_layer_prefixes(cfg)}
        targets = [
            name for name in specs if name.endswith(".self_attn") and name not in draft
        ]
        for start in range(0, len(targets) - 1, 2):
            owner = targets[start]
            main_owners.extend(
                (name, owner, slot)
                for slot, name in enumerate(targets[start : start + 2])
            )
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
        compressed_slots=packed_slots,
        compressed_owners=tuple(owners),
        main_owners=tuple(main_owners),
    )
    # Members other than their group's owner get no scheduler group.
    packed_away = {prefix + ".compressed_k" for prefix, owner, _ in owners} - {
        owner for _, owner, _ in owners
    }
    packed_away |= {name for name, owner, _ in main_owners if name != owner}
    pooled: dict[str, KVCacheSpec] = {}
    # Upstream uses one persistent Mamba state block per request without APC.
    # Its logical span is max_model_len, while the physical state still fits
    # in a single uniform pool page. APC uses position-indexed align blocks.
    mamba_mode = "align" if cfg.cache_config.enable_prefix_caching else "none"
    mamba_block_size = (
        block_size if mamba_mode == "align" else cfg.model_config.max_model_len
    )
    for name, spec in specs.items():
        if name in packed_away:
            continue
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


def check_packed_compressed_groups(layout: Qwen4ExpPoolLayout, groups) -> None:
    """Each packed compressed-key or main KV owner needs a block table of its own.

    Members address slots of their owner's pages, so two owners sharing one
    group (one block ID per token span) would overwrite each other's slots.
    """
    for kind, owners, env in (
        (
            "compressed keys",
            {owner for _, owner, _ in layout.compressed_owners},
            PACK_COMPRESSED_K_ENV,
        ),
        ("main KV", {owner for _, owner, _ in layout.main_owners}, PACK_MAIN_KV_ENV),
    ):
        for group in groups:
            names = set(group.layer_names)
            if names & owners and len(names) != 1:
                raise ValueError(
                    f"Qwen4Exp packed {kind} need one scheduler group per "
                    f"owner, got {sorted(names)}; unset {env}"
                )

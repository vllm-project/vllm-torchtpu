# SPDX-License-Identifier: Apache-2.0
"""Host-DRAM prefix-cache offload geometry for the Qwen4Exp unified pool.

The unified pool is one raw tensor of ``num_blocks`` uniform pages. Every
cache group -- QSA main KV (FP8 head-packed, one layer or a layer pair per
page), compressed keys (BF16, one layer per page or several packed), the
raw rings, and the GDN/PLE align checkpoints -- owns whole pages of it,
addressed by scheduler block ID. A
(group, block) offload key therefore names exactly one page, and moving that
page as opaque bytes restores the group's data bit for bit, whatever its
dtype or logical shape: there is no per-group geometry to agree on.

The pool is born as the carrier attention cache: the first attention group's
spec, with its manager block as the kernel block, laid out by
``Qwen4ExpPooledQSABackend`` (see ``kv_cache_materializer``'s
``_pool_attention_geometry`` and ``Qwen4ExpPooledCache.kernel_block_sizes``).
Raiden registers the pool with that same typed shape, so one Raiden row is
one manager page and every transfer is a whole-page DMA.

Under the gcd group-size rule (the TPU default) a group holds several layers
and a worker backs one pool region per layer slot it owns: one raw tensor of
``num_blocks`` pages each (two on a served PP8 TP1 stage). Block ``b`` of a
group is then row ``b`` of each region, so the connector registers every
region and a key moves that row of each; the bytes of every layer of the
group still round-trip whole. A pipeline stage registers its own regions,
and a key covers the block on every stage.
"""

import math
from dataclasses import replace

from vllm.v1.kv_cache_interface import AttentionSpec, EncoderOnlyAttentionSpec

QWEN4_EXP_ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
)


def is_qwen4_exp(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    architecture = getattr(model_config, "architecture", None)
    return isinstance(architecture, str) and architecture in QWEN4_EXP_ARCHITECTURES


def _carrier_spec(kv_cache_config) -> AttentionSpec:
    """The attention spec whose rows describe one pool page: the first one,
    by its stored rows when it is a VLLM_TPU_QSA_MAIN_KV_DCP main-KV spec
    (``tokens_per_state`` tokens per stored row)."""
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, AttentionSpec) and not isinstance(
            spec, EncoderOnlyAttentionSpec
        ):
            tokens_per_state = getattr(spec, "tokens_per_state", 1)
            if isinstance(tokens_per_state, int) and tokens_per_state > 1:
                return replace(
                    spec,
                    block_size=spec.block_size // tokens_per_state,
                    tokens_per_state=1,
                )
            return spec
    raise ValueError("Qwen4Exp unified pool has no attention carrier group")


def offload_pool_geometry(vllm_config, kv_cache_config):
    """Raiden row geometry of the pool for ``resolve_kernel_geometry``.

    Returns ``(kernel_block_size, per_block_shape, kv_dtype,
    device_block_size)`` with one kernel row per manager page.
    """
    import torch

    from .qsa import Qwen4ExpPooledQSABackend

    groups = kv_cache_config.kv_cache_groups
    pages = {group.kv_cache_spec.page_size_bytes for group in groups}
    if len(pages) != 1:
        raise ValueError(
            f"Qwen4Exp host offload needs one uniform pool page, got {sorted(pages)}"
        )
    (page_bytes,) = pages
    tensors = getattr(kv_cache_config, "kv_cache_tensors", None) or ()
    if len({tensor.size for tensor in tensors}) > 1:
        raise ValueError(
            "Qwen4Exp host offload needs every cache descriptor in one backing pool"
        )
    if tensors and tensors[0].size % (kv_cache_config.num_blocks * page_bytes):
        # The backing allocation is a whole number of regions of
        # num_blocks pages each (kv_cache_materializer.layer_to_pool_index).
        # Under the gcd group-size rule a block ID's layers sit in several
        # regions (one per layer slot this worker backs: 2 on a served PP8
        # stage); the connector registers every region and a key moves the
        # block's row of each, so a group's block still round-trips whole.
        raise ValueError(
            "Qwen4Exp host offload needs whole pool regions of "
            f"{kv_cache_config.num_blocks} x {page_bytes} bytes, got "
            f"{tensors[0].size} bytes"
        )
    spec = _carrier_spec(kv_cache_config)
    shape = Qwen4ExpPooledQSABackend.get_kv_cache_shape(
        num_blocks=1,
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        cache_dtype_str=spec.dtype,
    )
    per_block_shape = tuple(shape[1:])
    row_bytes = math.prod(per_block_shape) * torch.empty(0, dtype=spec.dtype).itemsize
    if row_bytes != page_bytes:
        raise ValueError(
            f"Qwen4Exp pool carrier rows hold {row_bytes} bytes, but pages "
            f"hold {page_bytes}"
        )
    return spec.block_size, per_block_shape, spec.dtype, spec.block_size


def pool_regions(kv_cache_config) -> int:
    """Raw pool regions behind one worker's unified pool (one per layer slot).

    The materializer allocates ``backing bytes / (num_blocks * page)`` raw
    tensors (``allocate_raw_kv_cache_tensors``); the offload connector must
    register exactly that many distinct storages, or a key would move only
    part of a block's layers.
    """
    page = max(g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups)
    tensors = kv_cache_config.kv_cache_tensors
    return tensors[0].size // (kv_cache_config.num_blocks * page) if tensors else 0


def merged_key_geometry(vllm_config, kv_cache_config) -> tuple[int, int] | None:
    """``(page_bytes, key_stride)`` of a merged-key pool, else None.

    QWEN4_EXP_MERGE_COMPRESSED_K: each raw region holds ``num_blocks`` pages,
    then a key region with one ``key_stride``-byte block of keys per block ID
    (``merged_key_span`` pages' worth of a layer's keys, at
    ``num_blocks * page + id * key_stride``; ``pd_manifest`` exposes the same
    pool). Host offload moves a main-KV block's key block with its page.
    """
    from .pool_layout import pooled_cache_specs

    layout, _ = pooled_cache_specs(vllm_config)
    if not layout.merged_compressed:
        return None
    if layout.split_compressed:
        raise ValueError(
            "Qwen4Exp host offload does not move keys split by page owner "
            "(QWEN4_EXP_PCP_SPLIT_COMPRESSED_K)"
        )
    pages = {g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups}
    if pages != {layout.page_bytes}:
        raise ValueError(
            f"Qwen4Exp merged keys expect {layout.page_bytes}-byte pages, got {sorted(pages)}"
        )
    key_stride = (
        layout.merged_key_span * layout.page_bytes // layout.compressed_columns_per_page
    )
    return layout.page_bytes, key_stride

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
"""

import math

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
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, AttentionSpec) and not isinstance(
            spec, EncoderOnlyAttentionSpec
        ):
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
    if tensors and tensors[0].size != kv_cache_config.num_blocks * page_bytes:
        # More than one pool region would put a group's block on several
        # rows; the unified pool keeps every group on a single region.
        raise ValueError(
            "Qwen4Exp host offload needs a single pool region of "
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

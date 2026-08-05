# SPDX-License-Identifier: Apache-2.0
"""Which axis of the KV cache occupies the TPU's 32-bit packing dimension.

TPU memory is addressed in 32-bit words, so a sub-32-bit dtype stores several
values per word: `kv_packing = 32 / bitwidth`, i.e. 2 for bf16 and 4 for fp8
(`get_dtype_packing`). Memory is charged per whole word, so whatever occupies
the packing dimension must supply `kv_packing` values to fill it; leftover
slots are allocated and wasted.

`KVCacheLayout` selects what fills it: KV heads, or tokens. Both layouts are
5-D and keep `num_blocks` at dim 0, so consumers that locate the block
dimension by index work against either. `kv_cache_shape` builds both, and
documents each at the point it is built.

See b/510425663 for the history behind the two layouts existing.
"""

import math
from enum import Enum

import torch
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.utils import TPU_HEAD_SIZE_ALIGNMENT


class KVCacheLayout(str, Enum):
    """Which axis occupies the 32-bit packing dimension."""

    HEAD_PACKED = "head_packed"
    TOKEN_PACKED = "token_packed"


def resolve_kv_cache_layout() -> KVCacheLayout:
    """The layout every caller allocates and accounts against.

    Constant, because the Pallas kernels can only read and write
    `HEAD_PACKED`. Do not make this configurable before they handle
    `TOKEN_PACKED`: the shape here and the shape the kernels assume must
    agree, and a mismatch produces wrong attention rather than an error.
    """
    # TODO(b/510425663): make selectable once the kernels support both.
    return KVCacheLayout.HEAD_PACKED


def get_dtype_packing(dtype: torch.dtype, packing_bits: int = 32) -> int:
    """Return number of dtype values packed into a packing_bits lane."""
    bits = dtype.itemsize * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"The bit width must divide {packing_bits}, but got {bits} for "
            f"dtype={dtype}.")
    return packing_bits // bits


def kv_vectors_per_token(num_kv_heads: int, head_size: int) -> int:
    """`Kx2` — how many head-sized vectors a single token stores.

    Normally a Key and a Value per head. The hd64 kernel is the exception: at
    `head_size == 64` it folds K and V together along head_dim (64 + 64 = 128),
    so only the heads themselves remain.
    """
    return num_kv_heads if head_size == 64 else num_kv_heads * 2


def padded_head_size(head_size: int) -> int:
    """Head vectors occupy whole 128-lane registers."""
    return cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT


def kv_cache_shape(
    layout: KVCacheLayout,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
) -> tuple[int, ...]:
    """The 5-D Pallas KV cache shape for one layer under `layout`.

    Parameters use vLLM's names, since vLLM calls the backend that calls this:
    `num_blocks` blocks of `block_size` **tokens** each. The Pallas kernels
    call the same two quantities `total_num_pages` and `page_size`; block and
    page mean the same thing, and only the vocabulary changes at this boundary.
    This module says "block" throughout.

    `num_blocks` stays at dim 0 in both layouts. vLLM relies on that: its
    offloading worker locates the block dimension with `shape.index(...)`
    (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`).
    """
    # Coerce so that the string form of a member -- which compares equal to it
    # but is not identical to it -- cannot fall through to the wrong branch.
    # Unknown values raise here rather than silently picking a layout.
    layout = KVCacheLayout(layout)

    kv_packing = get_dtype_packing(kv_dtype)
    heads = kv_vectors_per_token(num_kv_heads, head_size)
    head_dim = padded_head_size(head_size)

    if layout is KVCacheLayout.HEAD_PACKED:
        # cdiv rounds up because a partially filled word is not addressable;
        # slots past the real head count are allocated and unused.
        return (
            num_blocks,
            block_size,  # which token within the block
            cdiv(heads, kv_packing),  # words needed per token
            kv_packing,  # slot within a word: KV HEADS FILL THIS
            head_dim,
        )

    if layout is not KVCacheLayout.TOKEN_PACKED:
        # Reached only if a member is added to the enum without a branch here.
        # Falling through would hand back the token-packed shape, which is the
        # silent wrong-attention failure this module exists to prevent.
        raise ValueError(f"Unhandled KV cache layout: {layout}")

    if block_size % kv_packing:
        # Block sizes are 16-256 and kv_packing is 2 or 4, so this only trips
        # callers asking for per-token rather than per-block bytes -- notably
        # platforms/tpu_block_size_utils.py, which passes block_size=1. One
        # token is not a block, and rounding up would over-report by
        # kv_packing, so refuse rather than answer wrongly.
        raise ValueError(
            f"TOKEN_PACKED needs block_size divisible by {kv_packing} to fill "
            f"each 32-bit word with tokens, got block_size={block_size}.")
    # Dims 2 and 3 are the token axis regrouped, not extra space: they multiply
    # back to block_size. Nothing is rounded up, so no slot goes unused.
    return (
        num_blocks,
        heads,  # which K or V vector (head)
        block_size // kv_packing,  # groups of kv_packing tokens
        kv_packing,  # slot within a word: TOKENS FILL THIS
        head_dim,
    )


def kv_cache_page_size_bytes(
    layout: KVCacheLayout,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
) -> int:
    """Bytes for one block of one layer, including any layout padding.

    Keeps "page_size" in the name to match vLLM's
    `AttentionSpec.page_size_bytes` and the backend method wrapping this.
    """
    shape = kv_cache_shape(layout, 1, block_size, num_kv_heads, head_size,
                           kv_dtype)
    return math.prod(shape) * kv_dtype.itemsize

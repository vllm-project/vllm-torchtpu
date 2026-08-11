# SPDX-License-Identifier: Apache-2.0
"""JAX-side KV cache layout helpers for the v3 RPA kernel family.

`vllm_torchtpu.kv_cache_layout` owns the same two layouts on the torch side,
where it sizes the allocation vLLM makes. This module is its JAX counterpart:
it builds the same shapes for the kernels and the reference implementation, and
converts a cache between them. The two must agree exactly -- an allocator and a
kernel that disagree about the layout produce wrong attention rather than an
error -- so `KVCacheLayout` is imported from there rather than redefined, and
`tests/kernels/test_kv_cache_layout_jax.py` pins the shapes against each other.

Both layouts are 5-D and hold `num_blocks` at dim 0, but the middle dims mean
different things, which is the whole difficulty:

    HEAD_PACKED   [pages, page_size, Kx2 // p, p, head_dim]
    TOKEN_PACKED  [pages, Kx2, page_size // p, p, head_dim]

`describe_cache` normalises both to the same logical quantities, and
`to_token_major` presents either as plain `[tokens, Kx2, head_dim]`. Together
they let one implementation serve both layouts, which is why
`ref_ragged_paged_attention` needs no second copy of the attention maths.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to, get_dtype_packing)
from vllm_torchtpu.kv_cache_layout import KVCacheLayout


def kv_cache_shape(
    layout: KVCacheLayout,
    total_num_pages: int,
    page_size: int,
    num_kv_heads_x2: int,
    actual_head_dim: int,
    kv_dtype,
) -> tuple[int, ...]:
    """Shape of one layer's cache under `layout`.

    `page_size` is the number of tokens a page holds -- vLLM calls the same
    quantity `block_size`, and the kernels call it `page_size`.

    Takes `num_kv_heads_x2` already folded, so this module needs no opinion on
    the v3-vs-hd64 rule: v3 passes `actual_num_kv_heads * 2`, hd64 passes
    `actual_num_kv_heads` (it folds K and V along head_dim instead).
    """
    layout = KVCacheLayout(layout)
    kv_packing = get_dtype_packing(kv_dtype)
    head_dim = align_to(actual_head_dim, 128)

    if layout is KVCacheLayout.HEAD_PACKED:
        # Kx2 is rounded up to a whole word: a partially filled word is not
        # addressable, so slots past the real head count are still allocated.
        # A narrower dtype raises kv_packing and the rounding absorbs it, so
        # the page can cost the same in bytes as it did at bf16.
        return (total_num_pages, page_size,
                align_to(num_kv_heads_x2, kv_packing) // kv_packing,
                kv_packing, head_dim)

    if layout is not KVCacheLayout.TOKEN_PACKED:
        raise ValueError(f"Unhandled KV cache layout: {layout}")

    if page_size % kv_packing:
        raise ValueError(
            f"TOKEN_PACKED needs page_size divisible by {kv_packing} to fill "
            f"each 32-bit word with tokens, got {page_size=}.")
    # Kx2 is NOT rounded up here; tokens fill the word instead, and page sizes
    # are always large multiples of kv_packing. Every slot therefore holds real
    # data, and page bytes scale with itemsize at any head count.
    return (total_num_pages, num_kv_heads_x2, page_size // kv_packing,
            kv_packing, head_dim)


def describe_cache(cache: jax.Array, layout: KVCacheLayout):
    """Normalise either layout to the same logical quantities.

    Returns `(total_num_pages, page_size, num_kv_heads_x2, kv_packing,
    head_dim)` regardless of which dim each happens to live on. Callers that
    only need these numbers stay layout-agnostic; callers that index into the
    array need `to_token_major`.
    """
    layout = KVCacheLayout(layout)
    total_num_pages, dim1, dim2, kv_packing, head_dim = cache.shape

    # Dims 1 and 2 carry opposite meanings in the two layouts, and dim 2 is the
    # one split across the packing dim, so it is always `dim2 * kv_packing`
    # that reconstitutes the logical count.
    if layout is KVCacheLayout.HEAD_PACKED:
        # dim1 is page_size outright; dim2 counts words of heads.
        return (total_num_pages, dim1, dim2 * kv_packing, kv_packing, head_dim)
    if layout is KVCacheLayout.TOKEN_PACKED:
        # dim1 is Kx2 outright; dim2 counts words of tokens.
        return (total_num_pages, dim2 * kv_packing, dim1, kv_packing, head_dim)
    raise ValueError(f"Unhandled KV cache layout: {layout}")


def to_token_major(pages: jax.Array, layout: KVCacheLayout) -> jax.Array:
    """View a run of pages as `[num_tokens, Kx2, head_dim]`.

    Token `t` of page `i` lands at row `i * page_size + t` in both layouts, so
    everything downstream can index by token without knowing the layout.
    """
    layout = KVCacheLayout(layout)
    num_pages, dim1, dim2, kv_packing, head_dim = pages.shape

    if layout is KVCacheLayout.HEAD_PACKED:
        # [pages, page_size, Kx2//p, p, hd] -- token is already dim 1, and the
        # two packing dims are adjacent and in Kx2 order, so a reshape suffices.
        return pages.reshape(num_pages * dim1, dim2 * kv_packing, head_dim)

    if layout is not KVCacheLayout.TOKEN_PACKED:
        raise ValueError(f"Unhandled KV cache layout: {layout}")
    # [pages, Kx2, page_size//p, p, hd]: dims 2 and 3 are the token axis split
    # into groups and slots, so they rejoin as page_size, and Kx2 then has to
    # move back inside the token axis.
    page_size = dim2 * kv_packing
    return (pages.reshape(num_pages, dim1, page_size, head_dim).transpose(
        0, 2, 1, 3).reshape(num_pages * page_size, dim1, head_dim))


def from_token_major(tokens: jax.Array, layout: KVCacheLayout,
                     pages_shape: tuple[int, ...]) -> jax.Array:
    """Inverse of `to_token_major`, back into `pages_shape`."""
    layout = KVCacheLayout(layout)
    num_pages, dim1, dim2, kv_packing, head_dim = pages_shape

    if layout is KVCacheLayout.HEAD_PACKED:
        return tokens.reshape(pages_shape)

    if layout is not KVCacheLayout.TOKEN_PACKED:
        raise ValueError(f"Unhandled KV cache layout: {layout}")
    page_size = dim2 * kv_packing
    return (tokens.reshape(num_pages, page_size, dim1,
                           head_dim).transpose(0, 2, 1,
                                               3).reshape(pages_shape))


def head_packed_to_token_packed(
        cache: jax.Array,
        *,
        actual_num_kv_heads_x2: int | None = None) -> jax.Array:
    """Rebuild a head-packed cache in the token-packed layout.

    Lets a cache produced by today's trusted code be handed to the new one, so
    reads can be tested before writes exist.

    `actual_num_kv_heads_x2` drops head-packing's padding rows, which
    token-packing does not need. Leave it unset to keep them and stay an exact
    inverse of `token_packed_to_head_packed`.
    """
    _, page_size, num_kv_heads_x2, kv_packing, head_dim = describe_cache(
        cache, KVCacheLayout.HEAD_PACKED)
    if page_size % kv_packing:
        raise ValueError(
            f"TOKEN_PACKED needs page_size divisible by {kv_packing}, got "
            f"{page_size=}.")

    tokens = to_token_major(cache, KVCacheLayout.HEAD_PACKED)
    if actual_num_kv_heads_x2 is not None:
        if actual_num_kv_heads_x2 > num_kv_heads_x2:
            raise ValueError(
                f"{actual_num_kv_heads_x2=} exceeds the {num_kv_heads_x2=} "
                f"stored in this cache.")
        tokens = tokens[:, :actual_num_kv_heads_x2, :]
        num_kv_heads_x2 = actual_num_kv_heads_x2

    return from_token_major(tokens, KVCacheLayout.TOKEN_PACKED,
                            (cache.shape[0], num_kv_heads_x2,
                             page_size // kv_packing, kv_packing, head_dim))


def token_packed_to_head_packed(cache: jax.Array) -> jax.Array:
    """Rebuild a token-packed cache in the head-packed layout.

    Pads Kx2 up to a whole word with zeros, because head-packing addresses
    heads by word and cannot express a partial one. The result is therefore
    legitimately larger than the input whenever Kx2 is not already a multiple
    of `kv_packing`; that growth is the layout's cost, not a conversion error.
    """
    num_pages, page_size, num_kv_heads_x2, kv_packing, head_dim = (
        describe_cache(cache, KVCacheLayout.TOKEN_PACKED))

    tokens = to_token_major(cache, KVCacheLayout.TOKEN_PACKED)
    padded = align_to(num_kv_heads_x2, kv_packing)
    if padded != num_kv_heads_x2:
        tokens = jnp.pad(tokens,
                         ((0, 0), (0, padded - num_kv_heads_x2), (0, 0)),
                         constant_values=0)

    return from_token_major(
        tokens, KVCacheLayout.HEAD_PACKED,
        (num_pages, page_size, padded // kv_packing, kv_packing, head_dim))

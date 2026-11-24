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
"""Pure stateless JAX ops for packing per-request PLE state in a KV pool."""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import torch
import torch_tpu  # noqa: F401 - attaches torch.tpu.pallas

# TPU HBM flushes subnormal bf16 and canonicalizes fp8 NaN bit patterns, so
# raw values are split into 4-bit nibbles (0..15) rather than bit-cast directly.
_NIBBLE_BITS = 4


def read_ple_state(
    pool: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    block_size: int,
    is_lbhnc: bool,
    window_shape: tuple[int, int],
    window_dtype: torch.dtype,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reads each request's (conv_window, ngram_context) from ``pool``."""
    pages_per_block = _pages_per_block(pool.shape, block_size, is_lbhnc=is_lbhnc)
    return _read_ple_state_op(
        pool,
        block_ids.to(torch.int32),
        pages_per_block,
        window_shape[0],
        window_shape[1],
        str(window_dtype).removeprefix("torch."),
        context_len,
    )


def write_ple_state(
    pool: torch.Tensor,
    block_ids: torch.Tensor,
    conv_window: torch.Tensor,
    ngram_context: torch.Tensor,
    *,
    block_size: int,
    is_lbhnc: bool,
) -> None:
    """Writes each request's (conv_window, ngram_context) into ``pool`` in place."""
    pages_per_block = _pages_per_block(pool.shape, block_size, is_lbhnc=is_lbhnc)
    new_pool = _write_ple_state_op(
        pool,
        block_ids.to(torch.int32),
        conv_window,
        ngram_context.to(torch.int32),
        pages_per_block,
    )
    # Donation plus copy_ is aliased in place by XLA, as for pooled GDN.
    pool.copy_(new_pool)


@torch.tpu.pallas.jax_op(  # pyrefly: ignore[missing-attribute]
    "pallas::qwen4_exp_ple_state_read"
)
def _read_ple_state_op(
    pool: jax.Array,
    block_ids: jax.Array,
    pages_per_block: int,
    window_rows: int,
    window_channels: int,
    window_dtype: str,
    context_len: int,
) -> tuple[jax.Array, jax.Array]:
    return _read_states(
        pool,
        block_ids,
        pages_per_block=pages_per_block,
        window_shape=(window_rows, window_channels),
        window_dtype=jnp.dtype(window_dtype),
        context_len=context_len,
    )


@_read_ple_state_op.register_fake
def _(
    pool: torch.Tensor,
    block_ids: torch.Tensor,
    pages_per_block: int,
    window_rows: int,
    window_channels: int,
    window_dtype: str,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del pages_per_block
    dtype = getattr(torch, window_dtype)
    return (
        pool.new_empty((block_ids.shape[0], window_rows, window_channels), dtype=dtype),
        pool.new_empty((block_ids.shape[0], context_len), dtype=torch.int32),
    )


@torch.tpu.pallas.jax_op(  # pyrefly: ignore[missing-attribute]
    "pallas::qwen4_exp_ple_state_write", donate_argnums=(0,)
)
def _write_ple_state_op(
    pool: jax.Array,
    block_ids: jax.Array,
    window: jax.Array,
    context: jax.Array,
    pages_per_block: int,
) -> jax.Array:
    return _write_states(
        pool,
        block_ids,
        window,
        context,
        pages_per_block=pages_per_block,
    )


@_write_ple_state_op.register_fake
def _(
    pool: torch.Tensor,
    block_ids: torch.Tensor,
    window: torch.Tensor,
    context: torch.Tensor,
    pages_per_block: int,
) -> torch.Tensor:
    del block_ids, window, context, pages_per_block
    return torch.empty_like(pool)


def _read_states(
    pool: jax.Array,
    block_ids: jax.Array,
    *,
    pages_per_block: int,
    window_shape: tuple[int, int],
    window_dtype: jnp.dtype,
    context_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Each request's (conv_window, ngram_context), bit-exact as last written."""
    w_elems, state_elems = _state_elems(
        pool.dtype, window_shape, window_dtype, context_len
    )
    pages_needed = _state_pages(pool, pages_per_block, state_elems)
    offsets = jnp.arange(pages_needed, dtype=block_ids.dtype)
    page_ids = (block_ids[:, None] * pages_per_block + offsets[None, :]).reshape(-1)
    # Gather axis 0 directly before reshaping so XLA does not relayout the full pool.
    flat = pool[page_ids].reshape(block_ids.shape[0], -1)
    if _store_window_directly(pool.dtype, window_dtype):
        window = flat[:, :w_elems].reshape(block_ids.shape[0], *window_shape)
    else:
        window = _decode_nibbles(flat[:, :w_elems], window_dtype, window_shape)
    context = _decode_nibbles(flat[:, w_elems:state_elems], jnp.int32, (context_len,))
    return window, context


def _write_states(
    pool: jax.Array,
    block_ids: jax.Array,
    window: jax.Array,
    context: jax.Array,
    *,
    pages_per_block: int,
) -> jax.Array:
    """The pool with each request's state written to its block; other blocks unchanged.

    Rows that own no block point at block 0, the pool's null block.
    """
    num_reqs = block_ids.shape[0]
    if window.ndim != 3 or window.shape[0] != num_reqs:
        raise ValueError(
            f"conv_window must be [{num_reqs}, rows, channels], got {window.shape}."
        )
    if context.shape[0] != num_reqs or context.dtype != jnp.int32:
        raise ValueError(
            f"ngram_context must be int32 [{num_reqs}, n], got "
            f"{context.dtype} {context.shape}."
        )
    ctx_encoded = _encode_nibbles(context, pool.dtype)
    if _store_window_directly(pool.dtype, window.dtype):
        win_encoded = window.reshape(num_reqs, -1)
    else:
        win_encoded = _encode_nibbles(window, pool.dtype)
    data = jnp.concatenate([win_encoded, ctx_encoded], axis=-1)
    pages_needed = _state_pages(pool, pages_per_block, data.shape[1])
    padded_elems = pages_needed * math.prod(pool.shape[1:])
    pages = jnp.pad(data, ((0, 0), (0, padded_elems - data.shape[1]))).reshape(
        num_reqs, pages_needed, *pool.shape[1:]
    )

    # Sequential dynamic_update_slice keeps the pool update in place; a whole-page
    # scatter copies the full KV pool on TPU.
    def _update_block(
        carry: jax.Array, item: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        block_id, block_pages = item
        start = (block_id * pages_per_block, *(0,) * (pool.ndim - 1))
        return jax.lax.dynamic_update_slice(carry, block_pages, start), None

    updated, _ = jax.lax.scan(_update_block, pool, (block_ids, pages))
    return updated


def _pages_per_block(
    pool_shape: Sequence[int], block_size: int, *, is_lbhnc: bool = False
) -> int:
    """Kernel pages per scheduler block of a per-region 5-D attention pool."""
    if len(pool_shape) != 5:
        raise NotImplementedError(
            f"Qwen4Exp PLE state needs a per-region 5-D pool, got {pool_shape}."
        )
    page_tokens = pool_shape[-1] if is_lbhnc else pool_shape[1]
    if block_size % page_tokens:
        raise ValueError(
            f"Block size {block_size} is not a multiple of the pool's "
            f"{page_tokens}-token pages."
        )
    return block_size // page_tokens


def _store_window_directly(pool_dtype: jnp.dtype, window_dtype: jnp.dtype) -> bool:
    """Store window natively when pool matches window_dtype and is not fp8."""
    return pool_dtype == window_dtype and pool_dtype != jnp.float8_e4m3fn


def _state_pages(pool: jax.Array, pages_per_block: int, state_elems: int) -> int:
    """Leading pages per block required to store ``state_elems``."""
    page_elems = math.prod(pool.shape[1:])
    block_elems = pages_per_block * page_elems
    if state_elems > block_elems:
        raise ValueError(
            f"PLE state needs {state_elems} elements but a state block has "
            f"{block_elems}."
        )
    return math.ceil(state_elems / page_elems)


def _uint_dtype(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.dtype(f"uint{jax.dtypes.itemsize_bits(dtype)}")


def _nibbles_per_elem(dtype: jnp.dtype) -> int:
    return jax.dtypes.itemsize_bits(dtype) // _NIBBLE_BITS


def _encode_nibbles(values: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Split each element into planar 4-bit nibbles (0..15), exact in fp8 and bf16.

    Concatenating nibble planes instead of interleaving pairs avoids a trailing
    dim of size 2 that TPU pads out to 128 lanes.
    """
    bits = jax.lax.bitcast_convert_type(values, _uint_dtype(values.dtype))
    flat = bits.reshape(values.shape[0], -1)
    return jnp.concatenate(
        [
            ((flat >> (_NIBBLE_BITS * k)) & 0xF).astype(dtype)
            for k in range(_nibbles_per_elem(values.dtype))
        ],
        axis=-1,
    )


def _decode_nibbles(
    encoded: jax.Array, dtype: jnp.dtype, item_shape: tuple[int, ...]
) -> jax.Array:
    uint_dtype = _uint_dtype(dtype)
    parts = jnp.split(encoded, _nibbles_per_elem(dtype), axis=-1)
    acc = jnp.zeros(parts[0].shape, dtype=uint_dtype)
    for k, part in enumerate(parts):
        nibble = jnp.round(part.astype(jnp.float32)).astype(uint_dtype)
        acc = acc | (nibble << (_NIBBLE_BITS * k))
    return jax.lax.bitcast_convert_type(
        acc.reshape(encoded.shape[0], *item_shape), dtype
    )


def _state_elems(
    pool_dtype: jnp.dtype,
    window_shape: tuple[int, int],
    window_dtype: jnp.dtype,
    context_len: int,
) -> tuple[int, int]:
    window_count = math.prod(window_shape)
    context_elems = context_len * _nibbles_per_elem(jnp.int32)
    if _store_window_directly(pool_dtype, window_dtype):
        window_elems = window_count
    else:
        window_elems = window_count * _nibbles_per_elem(window_dtype)
    return window_elems, window_elems + context_elems

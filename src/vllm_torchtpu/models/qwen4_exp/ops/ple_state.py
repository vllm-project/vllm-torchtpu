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
# raw bytes are stored as 4-bit nibbles (0..15) rather than bit-cast directly.
_NIBBLES_PER_BYTE = 2


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
    blocks = _blocks(pool, pages_per_block, state_elems)
    flat = blocks[block_ids, : _num_rows(blocks, state_elems)].reshape(
        block_ids.shape[0], -1
    )
    if _store_window_directly(pool.dtype, window_dtype):
        window = flat[:, :w_elems].reshape(block_ids.shape[0], *window_shape)
    else:
        window = _from_bytes(
            _decode_bytes(flat[:, :w_elems]), window_dtype, window_shape
        )
    context = _from_bytes(
        _decode_bytes(flat[:, w_elems:state_elems]), jnp.int32, (context_len,)
    )
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
    ctx_encoded = _encode_bytes(_to_bytes(context), pool.dtype)
    if _store_window_directly(pool.dtype, window.dtype):
        win_encoded = window.reshape(num_reqs, -1)
    else:
        win_encoded = _encode_bytes(_to_bytes(window), pool.dtype)
    data = jnp.concatenate([win_encoded, ctx_encoded], axis=-1)
    blocks = _blocks(pool, pages_per_block, data.shape[1])
    num_rows = _num_rows(blocks, data.shape[1])
    row_shape = blocks.shape[2:]
    padded = num_rows * math.prod(row_shape)
    data = jnp.pad(data, ((0, 0), (0, padded - data.shape[1])))
    rows = data.reshape(num_reqs, num_rows, *row_shape)
    blocks = blocks.at[block_ids, :num_rows].set(rows)
    return blocks.reshape(pool.shape)


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


def _blocks(pool: jax.Array, pages_per_block: int, state_elems: int) -> jax.Array:
    """[num_blocks, rows, pack, D] view of the pool, in its own dtype."""
    num_blocks = pool.shape[0] // pages_per_block
    blocks = pool.reshape(num_blocks, -1, *pool.shape[-2:])
    if state_elems > blocks[0].size:
        raise ValueError(
            f"PLE state needs {state_elems} elements but a state block has "
            f"{blocks[0].size}."
        )
    return blocks


def _num_rows(blocks: jax.Array, state_elems: int) -> int:
    return math.ceil(state_elems / (blocks.shape[2] * blocks.shape[3]))


def _to_bytes(values: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(values, jnp.uint8).reshape(values.shape[0], -1)


def _from_bytes(
    data: jax.Array, dtype: jnp.dtype, item_shape: tuple[int, ...]
) -> jax.Array:
    shaped = data.reshape(data.shape[0], *item_shape, jnp.dtype(dtype).itemsize)
    return jax.lax.bitcast_convert_type(shaped, dtype)


def _encode_bytes(data: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Split each uint8 into two 4-bit nibbles (0..15), exact in fp8 and bf16."""
    hi = (data >> 4).astype(dtype)
    lo = (data & 0xF).astype(dtype)
    return jnp.stack([hi, lo], axis=-1).reshape(data.shape[0], -1)


def _decode_bytes(encoded: jax.Array) -> jax.Array:
    pairs = (
        jnp.round(encoded.astype(jnp.float32))
        .astype(jnp.uint8)
        .reshape(encoded.shape[0], -1, _NIBBLES_PER_BYTE)
    )
    return (pairs[..., 0] << 4) | pairs[..., 1]


def _state_elems(
    pool_dtype: jnp.dtype,
    window_shape: tuple[int, int],
    window_dtype: jnp.dtype,
    context_len: int,
) -> tuple[int, int]:
    window_count = math.prod(window_shape)
    context_elems = context_len * jnp.dtype(jnp.int32).itemsize * _NIBBLES_PER_BYTE
    if _store_window_directly(pool_dtype, window_dtype):
        window_elems = window_count
    else:
        window_elems = (
            window_count * jnp.dtype(window_dtype).itemsize * _NIBBLES_PER_BYTE
        )
    return window_elems, window_elems + context_elems

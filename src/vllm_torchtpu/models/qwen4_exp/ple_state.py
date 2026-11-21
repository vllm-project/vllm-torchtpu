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
"""Per-request PLE state, stored in a shared GDN mamba block of the KV pool."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.models.qwen4_exp.ple import (
    PLEInputs,
    PLEOutput,
    ragged_tail,
)


class PLEState(NamedTuple):
    """What one PLE layer carries per request between steps."""

    conv_window: torch.Tensor  # [num_reqs, (kernel - 1) * dilation, hc_hidden]
    ngram_context: torch.Tensor  # [num_reqs, ngram_size - 1], int32


class PLEStateCache(nn.Module, MambaBase):
    """Owns one PLE layer's mamba cache slot and pooled state read/write.

    Joins the GDN cache group by forwarding ``state_spec_layer``'s MambaBase
    spec (config.py admits PLE only on GDN layers).
    """

    def __init__(
        self,
        *,
        prefix: str,
        vllm_config: VllmConfig,
        state_spec_layer: MambaBase,
        window_shape: tuple[int, int],
        window_dtype: torch.dtype,
        context_len: int,
        eos_token_id: int,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Tuple keeps state_spec_layer out of named_modules() / state_dict().
        self._state_spec_layer = (state_spec_layer,)
        self._window_shape = window_shape
        self._context_len = context_len
        self._eos_token_id = eos_token_id
        self.kv_cache = (torch.tensor([]),)
        vllm_config.compilation_config.static_forward_context[prefix] = self
        self._read_op, self._write_op = _get_state_ops(
            window_shape, window_dtype, context_len, vllm_config.cache_config
        )

    def read(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[PLEInputs, torch.Tensor]:
        """Returns (PLEInputs, prev_conv_windows) for the current batch."""
        pool = self.kv_cache[0]
        if pool.numel() == 0:
            # Memory profiling runs before any KV cache exists: one request, no history.
            query_start_loc = hidden_states.shape[0] * torch.arange(
                2, dtype=torch.int32, device=hidden_states.device
            )
            windows = hidden_states.new_zeros(1, *self._window_shape)
            context = torch.full(
                (1, self._context_len),
                self._eos_token_id,
                dtype=torch.int32,
                device=input_ids.device,
            )
            return PLEInputs(input_ids, query_start_loc, context), windows

        metadata = self._attn_metadata()
        query_start_loc = metadata.query_start_loc
        num_reqs = query_start_loc.numel() - 1
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        has_history = metadata.seq_lens[:num_reqs] - query_lens > 0
        state = self._read_op(pool, metadata.mamba_state_indices[:num_reqs])
        windows = torch.where(has_history.view(-1, 1, 1), state.conv_window, 0)
        context = torch.where(
            has_history.view(-1, 1), state.ngram_context, self._eos_token_id
        )
        return PLEInputs(input_ids, query_start_loc, context), windows

    def write(
        self,
        inputs: PLEInputs,
        prev_windows: torch.Tensor,
        ple_out: PLEOutput,
    ) -> None:
        """Writes each request's next state (and any prefill checkpoint) to the pool."""
        pool = self.kv_cache[0]
        if pool.numel() == 0:
            return
        metadata = self._attn_metadata()
        query_start_loc = inputs.query_start_loc
        num_reqs = query_start_loc.numel() - 1
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        token_ids = inputs.input_ids.to(torch.int32)
        blocks_out = metadata.mamba_state_indices[:num_reqs]
        windows_out = ple_out.conv_window
        context_out = ragged_tail(
            token_ids, inputs.ngram_context, query_start_loc, query_lens
        )
        checkpoint_offsets = metadata.mamba_prefill_ckpt_offsets
        if checkpoint_offsets is not None:
            offsets = checkpoint_offsets[:num_reqs]
            blocks_out = torch.cat(
                [
                    metadata.mamba_prefill_ckpt_indices[:num_reqs].clamp(min=0),
                    blocks_out,
                ]
            )
            windows_out = torch.cat(
                [
                    ragged_tail(
                        ple_out.conv_input, prev_windows, query_start_loc, offsets
                    ),
                    windows_out,
                ]
            )
            context_out = torch.cat(
                [
                    ragged_tail(
                        token_ids, inputs.ngram_context, query_start_loc, offsets
                    ),
                    context_out,
                ]
            )
        self._write_op(pool, blocks_out, PLEState(windows_out, context_out))

    def _attn_metadata(self) -> Any:
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        return attn_metadata[self.prefix]

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return self._state_spec_layer[0].mamba_type

    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        return self._state_spec_layer[0].get_state_shape()

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return self._state_spec_layer[0].get_state_dtype()

    def get_prefill_checkpoint_alignment(self) -> int | None:
        return self._state_spec_layer[0].get_prefill_checkpoint_alignment()

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return self._state_spec_layer[0].get_kv_cache_spec(vllm_config)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self._state_spec_layer[0].get_attn_backend()


# TPU HBM flushes subnormal bf16 and canonicalizes fp8 NaN bit patterns, so
# raw bytes are stored as 4-bit nibbles (0..15) rather than bit-cast directly.
_NIBBLES_PER_BYTE = 2
_ReadStateFn = Callable[[torch.Tensor, torch.Tensor], PLEState]
_WriteStateFn = Callable[[torch.Tensor, torch.Tensor, PLEState], None]


def _get_state_ops(
    window_shape: tuple[int, int],
    window_dtype: torch.dtype,
    context_len: int,
    cache_config: CacheConfig,
) -> tuple[_ReadStateFn, _WriteStateFn]:
    key = (window_shape, window_dtype, context_len, id(cache_config))
    if key not in _ops:
        _ops[key] = _register_ops(window_shape, window_dtype, context_len, cache_config)
    return _ops[key]


_ops: dict[tuple, tuple[_ReadStateFn, _WriteStateFn]] = {}
_op_ids = itertools.count()


def _register_ops(
    window_shape: tuple[int, int],
    window_dtype: torch.dtype,
    context_len: int,
    cache_config: CacheConfig,
) -> tuple[_ReadStateFn, _WriteStateFn]:
    jax_dtype = jnp.dtype(str(window_dtype).removeprefix("torch."))

    # block_size and resolved KV layout are read at trace time (final after load).
    def pages_per_block(pool_shape: Sequence[int]) -> int:
        is_lbhnc = cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.LBHNC
        return _pages_per_block(pool_shape, cache_config.block_size, is_lbhnc=is_lbhnc)

    def read(pool: jax.Array, block_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _read_states(
            pool,
            block_ids,
            pages_per_block=pages_per_block(pool.shape),
            window_shape=window_shape,
            window_dtype=jax_dtype,
            context_len=context_len,
        )

    def write(
        pool: jax.Array, block_ids: jax.Array, window: jax.Array, context: jax.Array
    ) -> jax.Array:
        return _write_states(
            pool,
            block_ids,
            window,
            context,
            pages_per_block=pages_per_block(pool.shape),
        )

    name = f"pallas::qwen4_exp_ple_state_{next(_op_ids)}"
    read_op = pallas.jax_op(f"{name}_read", read)
    read_op.register_fake(
        lambda pool, block_ids: (
            pool.new_empty((block_ids.shape[0], *window_shape), dtype=window_dtype),
            pool.new_empty((block_ids.shape[0], context_len), dtype=torch.int32),
        )
    )
    write_op = pallas.jax_op(f"{name}_write", write, donate_argnums=(0,))
    write_op.register_fake(lambda pool, *_: torch.empty_like(pool))

    def read_state(pool: torch.Tensor, block_ids: torch.Tensor) -> PLEState:
        return PLEState(*read_op(pool, block_ids.to(torch.int32)))

    def write_state(
        pool: torch.Tensor, block_ids: torch.Tensor, state: PLEState
    ) -> None:
        new_pool = write_op(
            pool,
            block_ids.to(torch.int32),
            state.conv_window,
            state.ngram_context.to(torch.int32),
        )
        # Donation plus copy_ is aliased in place by XLA, as for pooled GDN.
        pool.copy_(new_pool)

    return read_state, write_state


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
    """[rows, ...] values as [rows, bytes] uint8."""
    return jax.lax.bitcast_convert_type(values, jnp.uint8).reshape(values.shape[0], -1)


def _from_bytes(rows: jax.Array, dtype: jnp.dtype, shape: tuple[int, ...]) -> jax.Array:
    itemsize = jnp.dtype(dtype).itemsize
    if itemsize > 1:
        rows = rows.reshape(rows.shape[0], -1, itemsize)
    return jax.lax.bitcast_convert_type(rows, dtype).reshape(rows.shape[0], *shape)


def _encode_bytes(data: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Encode uint8 bytes as 4-bit nibbles (0..15) so TPU HBM never corrupts bits."""
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

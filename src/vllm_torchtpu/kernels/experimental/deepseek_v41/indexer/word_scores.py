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
"""TPU7x Pallas kernel for ReLU-weighted index scoring from packed int32 word rows."""

import contextlib
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TPU7X_GENERATION = 7
ROW_WORDS = 128
DIM = 128
SUBLANES = 8
HALF_ROW_WORDS = ROW_WORDS // 2
SLOTS = ROW_WORDS // HALF_ROW_WORDS
MIN_ROWS = 1
MAX_BLOCK_ROWS = 512
SUB_BLOCK_ROWS = 256
BF16_BYTES = 2
WORD_BYTES = 4
SCORE_BYTES = 4
WEIGHT_BYTES_PER_HEAD = 4
SCOPED_VMEM_DIVISOR = 2
VMEM_HEADROOM_DIVISOR = 8
HIGH_HALF_SHIFT = 16
HIGH_HALF_MASK = -65536


def _check_tpu7x(func_name: str) -> None:
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation != TPU7X_GENERATION or tpu_info.num_lanes != ROW_WORDS:
        raise RuntimeError(
            f"{func_name} requires TPU7x with {ROW_WORDS} lanes, got "
            f"generation={tpu_info.generation}, num_lanes={tpu_info.num_lanes}"
        )


def _scoped_vmem_limit_bytes() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes) // SCOPED_VMEM_DIVISOR


def _max_vmem_tokens(heads: int, padded_rows: int) -> int:
    # Avoid counting unpadded 4 B/head for 3D weights [T, H, 1]: XLA tiles outer VMEM
    # refs to (8, 128) (512 B per sublane-aligned head per token).
    padded_heads = ((heads + SUBLANES - 1) // SUBLANES) * SUBLANES
    bytes_per_token = max(
        1,
        heads * BF16_BYTES * DIM
        + padded_heads * ROW_WORDS * WEIGHT_BYTES_PER_HEAD
        + SCORE_BYTES * SLOTS * padded_rows,
    )
    scoped_vmem_limit_bytes = _scoped_vmem_limit_bytes()
    vmem_headroom_bytes = scoped_vmem_limit_bytes // VMEM_HEADROOM_DIVISOR
    usable_vmem_bytes = scoped_vmem_limit_bytes - vmem_headroom_bytes
    return max(1, usable_vmem_bytes // bytes_per_token)


def _named_scope(name: str, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    return jax.named_scope(name) if enabled else contextlib.nullcontext()


def _bitcast_to_bf16(word_bits: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(word_bits, jnp.float32).astype(jnp.bfloat16)


def _mxu_dot(query_slice: jax.Array, key_tile: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        query_slice,
        key_tile,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _unpack_word_subtile_to_bf16_keys(
    words_ref: Any,
    keys_vmem_ref: Any,
    *,
    row_offset: jax.Array,
    enable_named_scope: bool = False,
) -> jax.Array:
    with _named_scope("unpack_word_tile", enable_named_scope):
        words = words_ref[0, pl.ds(row_offset, SUB_BLOCK_ROWS), :]
        tile_rows = words.shape[0]
        rolled_words = pltpu.roll(words, HALF_ROW_WORDS, 1)
        is_first_half = (
            jax.lax.broadcasted_iota(jnp.int32, words.shape, 1) < HALF_ROW_WORDS
        )
        keys_vmem_ref[pl.ds(0, tile_rows, stride=SLOTS), :] = jnp.where(
            is_first_half,
            words << HIGH_HALF_SHIFT,
            rolled_words & jnp.int32(HIGH_HALF_MASK),
        )
        keys_vmem_ref[pl.ds(1, tile_rows, stride=SLOTS), :] = jnp.where(
            is_first_half,
            rolled_words << HIGH_HALF_SHIFT,
            words & jnp.int32(HIGH_HALF_MASK),
        )
        return _bitcast_to_bf16(keys_vmem_ref[...])


def _compute_relu_weighted_scores(
    q_vmem_ref: Any,
    weights_vmem_ref: Any,
    cu_q_smem_ref: Any,
    key_tile: jax.Array,
    out_vmem_ref: Any,
    *,
    group_idx: jax.Array,
    key_offset: jax.Array,
    groups: int,
    sub_keys: int,
    enable_named_scope: bool = False,
) -> None:
    with _named_scope("mxu_dot_and_relu_reduce", enable_named_scope):
        tokens = q_vmem_ref.shape[0]
        group_start_token = jnp.clip(cu_q_smem_ref[group_idx], 0, tokens)
        group_end_token = jnp.clip(
            cu_q_smem_ref[group_idx + 1], group_start_token, tokens
        )
        zeros_sub = jnp.zeros((sub_keys,), dtype=jnp.float32)

        @pl.when(group_idx == 0)
        def _zero_head() -> None:
            @pl.loop(0, group_start_token)
            def _zero_head_token(token_idx: jax.Array) -> None:
                out_vmem_ref[token_idx, 0, pl.ds(key_offset, sub_keys)] = zeros_sub

        @pl.loop(group_start_token, group_end_token)
        def _token_loop(token_idx: jax.Array) -> None:
            logits = _mxu_dot(q_vmem_ref[token_idx], key_tile)
            out_vmem_ref[token_idx, 0, pl.ds(key_offset, sub_keys)] = (
                jnp.maximum(logits, 0) * weights_vmem_ref[token_idx]
            ).sum(0)

        @pl.when(group_idx == groups - 1)
        def _zero_tail() -> None:
            @pl.loop(group_end_token, tokens)
            def _zero_tail_token(token_idx: jax.Array) -> None:
                out_vmem_ref[token_idx, 0, pl.ds(key_offset, sub_keys)] = zeros_sub


def _score_word_rows_tile(
    q_vmem_ref: Any,
    weights_vmem_ref: Any,
    cu_q_smem_ref: Any,
    words_ref: Any,
    out_vmem_ref: Any,
    keys_vmem_ref: Any,
    *,
    groups: int,
    keys_per_block: int,
    sub_keys: int,
    num_subs: int,
    enable_named_scope: bool = False,
) -> None:
    group_idx, block_idx = pl.program_id(0), pl.program_id(1)

    @pl.loop(0, num_subs)
    def _sub_loop(sub_idx: jax.Array) -> None:
        row_offset = pl.multiple_of(sub_idx * SUB_BLOCK_ROWS, SUB_BLOCK_ROWS)
        key_offset = pl.multiple_of(
            block_idx * keys_per_block + sub_idx * sub_keys, sub_keys
        )
        key_tile = _unpack_word_subtile_to_bf16_keys(
            words_ref,
            keys_vmem_ref,
            row_offset=row_offset,
            enable_named_scope=enable_named_scope,
        )
        _compute_relu_weighted_scores(
            q_vmem_ref,
            weights_vmem_ref,
            cu_q_smem_ref,
            key_tile,
            out_vmem_ref,
            group_idx=group_idx,
            key_offset=key_offset,
            groups=groups,
            sub_keys=sub_keys,
            enable_named_scope=enable_named_scope,
        )


@functools.partial(jax.jit, static_argnames=("interpret", "enable_named_scope"))
def score_word_rows(
    q: jax.Array,
    weights: jax.Array,
    words: jax.Array,
    *,
    cu_q_lens: jax.Array | None = None,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Shapes: q=(T, H, 128) [even, odd] bf16, weights=(T, H, 1) fp32 -> fp32[T, 2*R].
    words=(G, R, 128) int32 packs 2 split-even/odd bf16 keys/row (R >= 1).
    Requires T % G == 0 if cu_q_lens is None, else monotonic clipped int32[G + 1].
    """
    if q.ndim != 3 or weights.ndim != 3 or words.ndim != 3:
        raise ValueError(
            "Expected 3D tensors: q [tokens, heads, 128], "
            "weights [tokens, heads, 1], words [groups, rows, 128]"
        )
    groups, rows, row_words = words.shape
    tokens, heads, q_dim = q.shape
    if (
        row_words != ROW_WORDS
        or groups < 1
        or q_dim != DIM
        or weights.shape != (tokens, heads, 1)
        or q.dtype != jnp.bfloat16
        or weights.dtype != jnp.float32
        or words.dtype != jnp.int32
    ):
        raise ValueError(
            "Expected BF16 q [tokens, heads, 128], FP32 weights [tokens, heads, 1], "
            "and int32 words [groups, rows, 128] with groups >= 1"
        )
    if cu_q_lens is None:
        if tokens % groups != 0:
            raise ValueError(
                f"When cu_q_lens is None, tokens ({tokens}) must be divisible by "
                f"groups ({groups})"
            )
    elif cu_q_lens.shape != (groups + 1,) or cu_q_lens.dtype != jnp.int32:
        raise ValueError(
            f"Expected int32 cu_q_lens of shape ({groups + 1},), got "
            f"dtype={cu_q_lens.dtype}, shape={cu_q_lens.shape}"
        )
    if rows < MIN_ROWS:
        raise ValueError(f"Word rows per group must be >= {MIN_ROWS}, got rows={rows}")
    _check_tpu7x("score_word_rows")
    if tokens == 0 or heads == 0:
        return jnp.zeros((tokens, SLOTS * rows), dtype=jnp.float32)
    if cu_q_lens is None:
        cu_q_lens = jnp.arange(groups + 1, dtype=jnp.int32) * (tokens // groups)

    padded_rows = ((rows + SUB_BLOCK_ROWS - 1) // SUB_BLOCK_ROWS) * SUB_BLOCK_ROWS
    if padded_rows != rows:
        words = jnp.pad(words, ((0, 0), (0, padded_rows - rows), (0, 0)))
    block_rows = MAX_BLOCK_ROWS if padded_rows % MAX_BLOCK_ROWS == 0 else SUB_BLOCK_ROWS
    num_blocks = padded_rows // block_rows
    num_subs = block_rows // SUB_BLOCK_ROWS
    keys_per_block = SLOTS * block_rows
    sub_keys = SLOTS * SUB_BLOCK_ROWS
    scratch_shapes = [pltpu.VMEM((sub_keys, ROW_WORDS), jnp.int32)]
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    compiler_params = pltpu.CompilerParams(vmem_limit_bytes=_scoped_vmem_limit_bytes())

    def _run_pallas(
        chunk_q: jax.Array,
        chunk_weights: jax.Array,
        chunk_cu_q_lens: jax.Array,
        chunk_words: jax.Array,
    ) -> jax.Array:
        chunk_tokens, chunk_groups = chunk_q.shape[0], chunk_words.shape[0]
        out_shape = jax.ShapeDtypeStruct(
            (chunk_tokens, 1, SLOTS * padded_rows), jnp.float32
        )
        words_hbm = pltpu.with_memory_space_constraint(
            chunk_words, pltpu.MemorySpace.HBM
        )
        tile_fn = functools.partial(
            _score_word_rows_tile,
            groups=chunk_groups,
            keys_per_block=keys_per_block,
            sub_keys=sub_keys,
            num_subs=num_subs,
            enable_named_scope=enable_named_scope,
        )

        def _emit(
            q_ref: Any,
            weights_ref: Any,
            cu_q_ref: Any,
            words_ref: Any,
            out_ref: Any,
            keys_ref: Any,
        ) -> None:
            body = functools.partial(
                tile_fn,
                q_ref,
                weights_ref,
                cu_q_ref,
                out_vmem_ref=out_ref,
                keys_vmem_ref=keys_ref,
            )
            words_block_spec = pl.BlockSpec(
                (1, block_rows, ROW_WORDS),
                lambda group_idx, block_idx: (group_idx, block_idx, 0),
            )
            pltpu.emit_pipeline(
                body,
                grid=(chunk_groups, num_blocks),
                in_specs=[words_block_spec],
                dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
                trace_scopes=enable_named_scope,
            )(words_ref)

        scores = pl.pallas_call(
            _emit,
            grid=(),
            in_specs=[vmem_spec, vmem_spec, smem_spec, hbm_spec],
            out_specs=vmem_spec,
            out_shape=out_shape,
            scratch_shapes=scratch_shapes,
            compiler_params=compiler_params,
            interpret=interpret,
            name="dsv41_index_word_scores",
        )(chunk_q, chunk_weights, chunk_cu_q_lens, words_hbm).reshape(
            chunk_tokens, SLOTS * padded_rows
        )
        return scores[:, : SLOTS * rows]

    max_tokens_per_chunk = _max_vmem_tokens(heads, padded_rows)

    def _scan_tokens(
        scan_q: jax.Array,
        scan_weights: jax.Array,
        scan_cu_q_lens: jax.Array,
        scan_words: jax.Array,
    ) -> jax.Array:
        num_full_chunks, remainder_tokens = divmod(
            scan_q.shape[0], max_tokens_per_chunk
        )

        def _tok_step(_: None, chunk_idx: jax.Array) -> tuple[None, jax.Array]:
            start_token = chunk_idx * max_tokens_per_chunk
            step_cu_q_lens = jnp.clip(
                scan_cu_q_lens - start_token, 0, max_tokens_per_chunk
            )
            step_q = jax.lax.dynamic_slice_in_dim(
                scan_q, start_token, max_tokens_per_chunk, 0
            )
            step_weights = jax.lax.dynamic_slice_in_dim(
                scan_weights, start_token, max_tokens_per_chunk, 0
            )
            return None, _run_pallas(step_q, step_weights, step_cu_q_lens, scan_words)

        _, full_scores = jax.lax.scan(
            _tok_step, None, jnp.arange(num_full_chunks, dtype=jnp.int32)
        )
        out_scores = full_scores.reshape(
            num_full_chunks * max_tokens_per_chunk, SLOTS * rows
        )
        if remainder_tokens == 0:
            return out_scores
        start_token = num_full_chunks * max_tokens_per_chunk
        remainder_cu_q_lens = jnp.clip(
            scan_cu_q_lens - start_token, 0, remainder_tokens
        )
        tail_scores = _run_pallas(
            scan_q[start_token:],
            scan_weights[start_token:],
            remainder_cu_q_lens,
            scan_words,
        )
        return jnp.concatenate([out_scores, tail_scores], 0)

    if tokens <= max_tokens_per_chunk:
        return _run_pallas(q, weights, cu_q_lens, words)
    return _scan_tokens(q, weights, cu_q_lens, words)

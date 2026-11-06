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
"""Index scores directly from gathered BF16 index-cache word rows on TPU7x.

Architecture & Memory Hierarchy
-------------------------------
1. **Packed Index-Cache Word-Row Layout**:
   A physical index-cache word row (`int32[128]`, 512 bytes) stores
   `SLOTS = 128 // (DIM // 2) = 2` BF16 index keys (`DIM = 128`) in
   split-element order: lanes `[0, 64)` hold key `2 * row`, lanes `[64, 128)`
   hold key `2 * row + 1`, the low 16 bits of each `int32` word hold even
   channel elements `0, 2, ..., 126`, and the high 16 bits hold odd channel
   elements `1, 3, ..., 127`. Query tensors `q` (`bf16[tokens, heads, 128]`)
   are pre-ordered by the caller into `[even_elements, odd_elements]` along the
   trailing `128`-lane dimension so MXU dot products match natural-order
   inner products without unpacking `q`.
2. **Invariant `SUB_BLOCK_ROWS = 256` (`512`-Key) Two-Column MXU Tiling**:
   External callers only need `rows >= MIN_ROWS` (`8`) and (when
   `not interpret`) `rows % MIN_ROWS == 0`. Internally, `score_word_rows` pads
   `words` along axis 1 to a multiple of `SUB_BLOCK_ROWS = 256`, selects HBM
   pipeline tile height `block_rows = 512 if padded_rows % 512 == 0 else 256`,
   streams `(1, block_rows, 128)` word tiles from `pltpu.MemorySpace.HBM` into
   VMEM via `pltpu.emit_pipeline`, and unpacks `SUB_BLOCK_ROWS = 256` rows
   (`512` keys = two full 256-wide MXU columns) per inner sub-tile before
   slicing `scores[:, : 2 * rows]` on return.
3. **128K Scoped VMEM Budget & `jax.lax.scan` Token Chunking**:
   Outer `pl.pallas_call(..., grid=())` keeps `q`, `weights`, and output
   `scores` in `pltpu.VMEM` (`cu_q_lens` in `pltpu.SMEM`). `_max_vmem_tokens`
   queries `pltpu.get_tpu_info().vmem_capacity_bytes` (`64 MiB` on TPU7x) and
   caps per-call tokens at `7 / 16` (`28.0 MiB`) of VMEM capacity so outer
   VMEM allocations plus `keys_vmem_ref` (`128 KiB`) and double-buffered `words`
   pipeline tiles (`<= 512 KiB`) stay within XLA's `32.0 MiB` scoped VMEM limit
   up to 128K context (`rows = 65536`, `131,072` keys). Larger batches are
   chunked via `jax.lax.scan` across uniform whole-group, uniform intra-group,
   and ragged `cu_q_lens` regimes. Because 3D `weights` (`fp32[T, H, 1]`) in
   `pltpu.VMEM` is lane-padded by XLA to `(8, 128)` (`512 B/head/token`, i.e.,
   `16 KiB/token` at `H = 32` and `32 KiB/token` at `H = 64`), `_max_vmem_tokens`
   bounds the `(queries, rows)` score tile at long contexts (`rows >= 2048`),
   while standalone short-context callers (`rows <= 256`) should keep
   `tokens <= 1,024` at `H = 32` (`tokens <= 512` at `H = 64`; in-stack
   `paged_indexer_topk` clamps `MAX_SCORING_TOKENS = 288`).
"""

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
HALF_ROW_WORDS = ROW_WORDS // 2
SLOTS = ROW_WORDS // HALF_ROW_WORDS
MIN_ROWS = 8
MAX_BLOCK_ROWS = 512
SUB_BLOCK_ROWS = 256
MAX_BATCHED_GROUP_TOKENS = 8
BF16_BYTES = 2
WORD_BYTES = 4
WEIGHT_BYTES_PER_HEAD = 4
VMEM_BUDGET_NUMERATOR = 7
VMEM_BUDGET_DENOMINATOR = 16
HIGH_HALF_SHIFT = 16
HIGH_HALF_MASK = -65536
_OUTER_COMPILER_PARAMS = pltpu.CompilerParams(dimension_semantics=())


def _check_tpu7x(func_name: str) -> None:
    """Validate that the active or abstract TPU device is TPU7x (128 lanes).

    Args:
        func_name: Public kernel entrypoint name for error reporting.

    Raises:
        RuntimeError: If `tpu_info.generation != 7` or `tpu_info.num_lanes != 128`.
    """
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation != TPU7X_GENERATION or tpu_info.num_lanes != ROW_WORDS:
        raise RuntimeError(
            f"{func_name} requires TPU7x with {ROW_WORDS} lanes, got "
            f"generation={tpu_info.generation}, num_lanes={tpu_info.num_lanes}"
        )


def _max_vmem_tokens(heads: int, padded_rows: int) -> int:
    """Derive per-call token cap so scoped VMEM stays < 32 MiB on TPU7x.

    Note:
        This budget models `WEIGHT_BYTES_PER_HEAD = 4` (`4 B/head/token`) and
        tightly bounds the `(queries, rows)` score tile at long contexts
        (`padded_rows >= 2048`). Because 3D `weights` (`fp32[T, H, 1]`) in
        outer `pltpu.VMEM` is lane-padded by XLA to `(8, 128)` (`512 B/head/token`,
        i.e., `16 KiB/token` at `H = 32` and `32 KiB/token` at `H = 64`),
        standalone short-context callers (`padded_rows <= 256`) should keep
        `tokens <= 1,024` at `H = 32` (`tokens <= 512` at `H = 64`; clamped to
        `MAX_SCORING_TOKENS = 288` in-stack by `paged_indexer_topk`).

    Args:
        heads: Number of query heads (`H`).
        padded_rows: Word rows per group after rounding up to `SUB_BLOCK_ROWS`.

    Returns:
        Maximum number of query tokens per `pl.pallas_call` invocation.
    """
    vmem_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    bytes_per_token = max(
        1,
        heads * (BF16_BYTES * DIM + WEIGHT_BYTES_PER_HEAD)
        + WORD_BYTES * SLOTS * padded_rows,
    )
    usable_vmem_bytes = (vmem_bytes * VMEM_BUDGET_NUMERATOR) // VMEM_BUDGET_DENOMINATOR
    return max(1, usable_vmem_bytes // bytes_per_token)


def _named_scope(name: str, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """Return a `jax.named_scope` context when `enabled` is True."""
    return jax.named_scope(name) if enabled else contextlib.nullcontext()


def _bitcast_to_bf16(word_bits: jax.Array) -> jax.Array:
    """Convert high-16-bit-aligned `int32` words to `bfloat16` via `float32`."""
    return jax.lax.bitcast_convert_type(word_bits, jnp.float32).astype(jnp.bfloat16)


def _mxu_dot(query_slice: jax.Array, key_tile: jax.Array) -> jax.Array:
    """Contract `[..., 128]` queries with `[sub_keys, 128]` keys in FP32 on MXU."""
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
    """Vector compute helper: unpack a `[256, 128]` word slice to BF16 keys.

    Data lifecycle — Used: `words_ref` (`int32[1, block_rows, 128]`).
    Changed: `keys_vmem_ref` (`int32[512, 128]` VMEM scratch).
    Created: `bf16[512, 128]` unpacked key tile in record order.
    """
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


def _compute_uniform_relu_weighted_scores(
    q_vmem_ref: Any,
    weights_vmem_ref: Any,
    key_tile: jax.Array,
    out_vmem_ref: Any,
    *,
    group_idx: jax.Array,
    key_offset: jax.Array,
    group_tokens: int,
    sub_keys: int,
    enable_named_scope: bool = False,
) -> None:
    """MXU compute helper: uniform-group MXU contraction and ReLU reduction.

    Batches `(group_tokens * heads, 128)` into a single MXU matmul when
    `group_tokens <= MAX_BATCHED_GROUP_TOKENS` (`8`), and switches to `@pl.loop`
    for larger uniform groups to bound live `logits` VMEM/VREG residency.
    """
    with _named_scope("mxu_dot_and_relu_reduce", enable_named_scope):
        heads = q_vmem_ref.shape[1]
        if group_tokens <= MAX_BATCHED_GROUP_TOKENS:
            q_group = q_vmem_ref[pl.ds(group_idx * group_tokens, group_tokens), :, :]
            all_logits = _mxu_dot(q_group.reshape(group_tokens * heads, DIM), key_tile)
            for token_offset in range(group_tokens):
                logits = all_logits[token_offset * heads : (token_offset + 1) * heads]
                token_idx = group_idx * group_tokens + token_offset
                out_vmem_ref[token_idx, 0, pl.ds(key_offset, sub_keys)] = (
                    jnp.maximum(logits, 0) * weights_vmem_ref[token_idx]
                ).sum(0)
        else:
            group_token_start = group_idx * group_tokens

            @pl.loop(0, group_tokens)
            def _tok_loop(token_offset: jax.Array) -> None:
                token_idx = group_token_start + token_offset
                logits = _mxu_dot(q_vmem_ref[token_idx], key_tile)
                out_vmem_ref[token_idx, 0, pl.ds(key_offset, sub_keys)] = (
                    jnp.maximum(logits, 0) * weights_vmem_ref[token_idx]
                ).sum(0)


def _compute_ragged_relu_weighted_scores(
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
    """MXU compute helper: ragged `cu_q_lens` MXU contraction and padding zero-fill.

    Precondition: `cu_q_lens` must be monotonically non-decreasing after
    clipping to `[0, tokens]` (`cu_q_lens[i] <= cu_q_lens[i + 1]`) so group
    token spans are disjoint under `dimension_semantics=(PARALLEL, PARALLEL)`
    and `_zero_tail` on `group_idx == groups - 1` only zeroes trailing padding.
    """
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
    group_tokens: int,
    keys_per_block: int,
    sub_keys: int,
    num_subs: int,
    enable_named_scope: bool = False,
) -> None:
    """Per-tile pipeline body: score one `(group_idx, block_idx)` word tile."""
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
        if cu_q_smem_ref is None:
            _compute_uniform_relu_weighted_scores(
                q_vmem_ref,
                weights_vmem_ref,
                key_tile,
                out_vmem_ref,
                group_idx=group_idx,
                key_offset=key_offset,
                group_tokens=group_tokens,
                sub_keys=sub_keys,
                enable_named_scope=enable_named_scope,
            )
        else:
            _compute_ragged_relu_weighted_scores(
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
    """Compute ReLU-weighted index scores for BF16 keys packed in `words`.

    Data lifecycle:
        - Used (read-only): `q` (`bf16[tokens, heads, 128]` in split-element
          `[even_elements, odd_elements]` order), `weights`
          (`fp32[tokens, heads, 1]`), `words` (`int32[groups, rows, 128]`,
          2 BF16 keys per word row), and `cu_q_lens` (`int32[groups + 1]` or
          `None`).
        - Changed: None.
        - Created: `fp32[tokens, 2 * rows]` ReLU-weighted dot-product scores
          (`0.0` on padded tokens outside `[cu_q_lens[0], cu_q_lens[groups])`).

    Args:
        q: Split-element BF16 query tensor of shape `(tokens, heads, 128)`.
        weights: FP32 per-head routing weights of shape `(tokens, heads, 1)`.
        words: Packed `int32` index-cache word rows of shape
            `(groups, rows, 128)` holding `2 * rows` BF16 keys per group.
        cu_q_lens: Optional cumulative query lengths of shape `(groups + 1,)`
            with dtype `int32`. Must be monotonically non-decreasing after
            clipping to `[0, tokens]` (`cu_q_lens[i] <= cu_q_lens[i + 1]`) so
            group token spans are disjoint under `(PARALLEL, PARALLEL)` grid
            execution and `_zero_tail` on `group_idx == groups - 1` only zeroes
            trailing padding. When `None`, `tokens` must be divisible by
            `groups` and each group of `tokens // groups` queries shares
            `words[group_idx]`.
        interpret: Whether to execute `pl.pallas_call` in CPU interpret mode.
        enable_named_scope: Whether to emit `jax.named_scope` markers (`trace_scopes`)
            for XProf `XLA TraceMe` profiling.

    Returns:
        FP32 tensor of shape `(tokens, 2 * rows)` containing ReLU-weighted
        index scores in record order (`key_idx = 2 * row + slot`).

    Raises:
        ValueError: If tensor ranks, shapes, dtypes, `cu_q_lens` constraints,
            or `rows >= 8` (and `rows % 8 == 0` when `not interpret`) are
            violated.
        RuntimeError: If the target or abstract TPU device is not TPU7x
            (`generation == 7` and `num_lanes == 128`).
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
    # NOTE(contract): Enforce MIN_ROWS = 8 TPU sublane divisibility upfront before
    # padding rows to SUB_BLOCK_ROWS = 256.
    if rows < MIN_ROWS or (not interpret and rows % MIN_ROWS != 0):
        raise ValueError(
            f"Word rows per group must be >= {MIN_ROWS} and (when not interpret) "
            f"divisible by {MIN_ROWS}, got rows={rows}"
        )
    _check_tpu7x("score_word_rows")
    if tokens == 0 or heads == 0:
        return jnp.zeros((tokens, SLOTS * rows), dtype=jnp.float32)

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
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    def _run_pallas(
        chunk_q: jax.Array,
        chunk_weights: jax.Array,
        chunk_cu_q_lens: jax.Array | None,
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
            group_tokens=(
                0 if chunk_cu_q_lens is not None else chunk_tokens // chunk_groups
            ),
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

        if chunk_cu_q_lens is None:

            def _uniform_kernel(
                q_ref: Any,
                weights_ref: Any,
                words_ref: Any,
                out_ref: Any,
                keys_ref: Any,
            ) -> None:
                _emit(q_ref, weights_ref, None, words_ref, out_ref, keys_ref)

            kernel = _uniform_kernel
            in_specs = [vmem_spec, vmem_spec, hbm_spec]
            call_args: tuple[jax.Array, ...] = (chunk_q, chunk_weights, words_hbm)
        else:
            kernel = _emit
            smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
            in_specs = [vmem_spec, vmem_spec, smem_spec, hbm_spec]
            call_args = (chunk_q, chunk_weights, chunk_cu_q_lens, words_hbm)

        scores = pl.pallas_call(
            kernel,
            grid=(),
            in_specs=in_specs,
            out_specs=vmem_spec,
            out_shape=out_shape,
            scratch_shapes=scratch_shapes,
            compiler_params=_OUTER_COMPILER_PARAMS,
            interpret=interpret,
            name="dsv41_index_word_scores",
        )(*call_args).reshape(chunk_tokens, SLOTS * padded_rows)
        return scores[:, : SLOTS * rows]

    max_tokens_per_chunk = _max_vmem_tokens(heads, padded_rows)

    def _scan_tokens(
        scan_q: jax.Array,
        scan_weights: jax.Array,
        scan_cu_q_lens: jax.Array | None,
        scan_words: jax.Array,
    ) -> jax.Array:
        num_full_chunks, remainder_tokens = divmod(
            scan_q.shape[0], max_tokens_per_chunk
        )

        def _tok_step(_: None, chunk_idx: jax.Array) -> tuple[None, jax.Array]:
            start_token = chunk_idx * max_tokens_per_chunk
            step_cu_q_lens = (
                None
                if scan_cu_q_lens is None
                else jnp.clip(scan_cu_q_lens - start_token, 0, max_tokens_per_chunk)
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
        remainder_cu_q_lens = (
            None
            if scan_cu_q_lens is None
            else jnp.clip(scan_cu_q_lens - start_token, 0, remainder_tokens)
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
    if cu_q_lens is not None:
        return _scan_tokens(q, weights, cu_q_lens, words)
    group_tokens = tokens // groups
    if group_tokens <= max_tokens_per_chunk:
        groups_per_step = max_tokens_per_chunk // group_tokens
        tokens_per_step = groups_per_step * group_tokens
        num_full_steps, remainder_groups = divmod(groups, groups_per_step)

        def _grp_step(_: None, step_idx: jax.Array) -> tuple[None, jax.Array]:
            start_group = step_idx * groups_per_step
            start_token = step_idx * tokens_per_step
            step_q = jax.lax.dynamic_slice_in_dim(q, start_token, tokens_per_step, 0)
            step_weights = jax.lax.dynamic_slice_in_dim(
                weights, start_token, tokens_per_step, 0
            )
            step_words = jax.lax.dynamic_slice_in_dim(
                words, start_group, groups_per_step, 0
            )
            return None, _run_pallas(step_q, step_weights, None, step_words)

        _, full_scores = jax.lax.scan(
            _grp_step, None, jnp.arange(num_full_steps, dtype=jnp.int32)
        )
        out_scores = full_scores.reshape(num_full_steps * tokens_per_step, SLOTS * rows)
        if remainder_groups == 0:
            return out_scores
        start_group = num_full_steps * groups_per_step
        start_token = num_full_steps * tokens_per_step
        tail_scores = _run_pallas(
            q[start_token:], weights[start_token:], None, words[start_group:]
        )
        return jnp.concatenate([out_scores, tail_scores], 0)

    def _one_grp(_: None, group_idx: jax.Array) -> tuple[None, jax.Array]:
        start_token = group_idx * group_tokens
        group_q = jax.lax.dynamic_slice_in_dim(q, start_token, group_tokens, 0)
        group_weights = jax.lax.dynamic_slice_in_dim(
            weights, start_token, group_tokens, 0
        )
        group_words = jax.lax.dynamic_slice_in_dim(words, group_idx, 1, 0)
        return None, _scan_tokens(group_q, group_weights, None, group_words)

    _, group_outputs = jax.lax.scan(_one_grp, None, jnp.arange(groups, dtype=jnp.int32))
    return group_outputs.reshape(tokens, SLOTS * rows)

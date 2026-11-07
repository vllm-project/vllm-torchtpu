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
"""TensorCore Pallas writer for both SparseCore-layout sparse-MLA caches.

One `pallas_call` packs a decode step's `kv_c_normed` and `k_pe` in VMEM and
DMAs each owned token straight into the nope and rope caches, in place. It
replaces the two `pack_tokens` + pad + SparseCore `scatter_rows` offloads the
SparseCore path issues per layer, so the TensorCore no longer waits on two
SparseCore round trips. The caches keep the SparseCore layout and the bytes
written are identical to `scatter.scatter`'s.

Mosaic DMAs into the tiled cache only in contiguous 512 B rows. A nope token
is one such row; a rope token is a quarter of one (`WORD_BYTES` tokens share a
row), so each owned rope token is a read-modify-write of its whole row: read
it, merge in every token of this call that lands in it, write it back.

The kernel is unrolled per token, so it is only used for decode-sized steps
(`MAX_TOKENS`); larger steps keep the SparseCore writer.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    WORD_BYTES,
    KVCacheLayout,
    SparseMLAKVCacheSpec,
)

# Largest step written by this kernel. Its scalar pass and rope merge unroll
# over the tokens (the merge is quadratic), so larger steps stay on the
# SparseCore writer. Measured on GLM-5.2 DCP16: faster than the SparseCore
# writer at 32 tokens, slower at 64 on a rank that owns rows.
MAX_TOKENS = 32

# Test-only faults (positive controls for the kernel's own test).
_FAULTS = (None, "skip_rope_read", "shift_lane_group")


def supports(
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    num_tokens: int,
) -> bool:
    """Whether `scatter_caches` can write this step into these caches."""
    if not 0 < num_tokens <= MAX_TOKENS:
        return False
    if nope_spec.layout is not KVCacheLayout.SPARSECORE:
        return False
    if rope_spec.layout is not KVCacheLayout.SPARSECORE:
        return False
    # A nope token must be exactly one cache row, and a rope row a whole
    # number of rope tokens.
    nope_words = nope_spec.token_bytes // WORD_BYTES
    rope_words = rope_spec.token_bytes // WORD_BYTES
    return nope_spec.shape[-1] == nope_words and rope_spec.shape[-1] % rope_words == 0


def _pack_words(values: jax.Array, token_bytes: int, word_dtype: Any) -> jax.Array:
    """In-kernel `kv_cache_utils.pack_tokens`: `[T, n]` -> `[T, token_bytes / 4]`.

    Word `w` takes byte `w` of each of the `WORD_BYTES` bands of the token.
    Bands past the token's `n` bytes are the zero pad and contribute nothing.
    """
    u8 = pltpu.bitcast(values, jnp.uint8)
    band = token_bytes // WORD_BYTES
    words = None
    for b in range(WORD_BYTES):
        if b * band >= u8.shape[-1]:
            break
        part = u8[:, b * band : (b + 1) * band].astype(word_dtype)
        if b:
            part = part << (8 * b)
        words = part if words is None else words | part
    return words


def _tc_scatter_kernel(
    dst_ref: Any,
    kv_c_ref: Any,
    k_pe_ref: Any,
    nope_in: Any,
    rope_in: Any,
    nope_out: Any,
    rope_out: Any,
    nope_buf: Any,
    rope_tok_buf: Any,
    rope_read_buf: Any,
    rope_row_buf: Any,
    sem: Any,
    *,
    num_tokens: int,
    page_size: int,
    num_rows: int,
    nope_token_bytes: int,
    rope_token_bytes: int,
    fault: str | None,
) -> None:
    del nope_in, rope_in  # aliased to nope_out / rope_out
    rope_tok_words = rope_token_bytes // WORD_BYTES
    rope_row_words = rope_row_buf.shape[-1]
    tokens_per_row = rope_row_words // rope_tok_words

    # Scalar pass: where each token lands. `dst < 0` is `SKIP_ROW` (padding,
    # or a row another DCP rank owns), the same predicate `scatter_rows` uses.
    tokens = []
    for i in range(num_tokens):
        dst = dst_ref[i]
        write = jnp.logical_and(dst >= 0, dst < num_rows)
        dst = jnp.clip(dst, 0, num_rows - 1)
        page, slot = dst // page_size, dst % page_size
        group = slot % tokens_per_row
        if fault == "shift_lane_group":
            group = (group + 1) % tokens_per_row
        tokens.append((write, page, slot, slot // tokens_per_row, group))

    any_write = tokens[0][0]
    for write, *_ in tokens[1:]:
        any_write = jnp.logical_or(any_write, write)

    # Most DCP ranks own no token of a decode step: skip everything.
    @pl.when(any_write)
    def _():
        nope_copies, read_copies, write_copies = [], [], []
        for i, (_, page, slot, row, _) in enumerate(tokens):
            nope_copies.append(
                pltpu.make_async_copy(
                    nope_buf.at[pl.ds(i, 1)],
                    nope_out.at[page, pl.ds(slot, 1)],
                    sem.at[0, i],
                )
            )
            read_copies.append(
                pltpu.make_async_copy(
                    rope_out.at[page, pl.ds(row, 1)],
                    rope_read_buf.at[pl.ds(i, 1)],
                    sem.at[1, i],
                )
            )
            write_copies.append(
                pltpu.make_async_copy(
                    rope_row_buf.at[pl.ds(i, 1)],
                    rope_out.at[page, pl.ds(row, 1)],
                    sem.at[2, i],
                )
            )
        read_rows = fault != "skip_rope_read"

        # Rope row reads first: they need nothing packed, so they overlap the
        # packing below.
        if read_rows:
            for (write, *_), copy in zip(tokens, read_copies):

                @pl.when(write)
                def _(copy=copy):
                    copy.start()

        word_dtype = nope_buf.dtype
        nope_buf[...] = _pack_words(kv_c_ref[...], nope_token_bytes, word_dtype)
        # One rope token, repeated into every lane group of a row so the merge
        # below is a select, never a lane shift.
        rope_words = _pack_words(k_pe_ref[...], rope_token_bytes, word_dtype)
        rope_tok_buf[...] = jnp.concatenate([rope_words] * tokens_per_row, axis=1)

        for (write, *_), copy in zip(tokens, nope_copies):

            @pl.when(write)
            def _(copy=copy):
                copy.start()

        if read_rows:
            for (write, *_), copy in zip(tokens, read_copies):

                @pl.when(write)
                def _(copy=copy):
                    copy.wait()

        # Merge every token of this step that lands in token i's rope row, not
        # just token i: rows written by two tokens then carry identical bytes,
        # whichever DMA lands last.
        lane_group = (
            jax.lax.broadcasted_iota(jnp.int32, (1, rope_row_words), 1)
            // rope_tok_words
        )
        for i, (write, page, _, row, _) in enumerate(tokens):

            @pl.when(write)
            def _(i=i, page=page, row=row):
                if read_rows:
                    merged = rope_read_buf[pl.ds(i, 1), :]
                else:
                    merged = jnp.zeros((1, rope_row_words), word_dtype)
                for j, (write_j, page_j, _, row_j, group_j) in enumerate(tokens):
                    same_row = jnp.logical_and(
                        write_j, jnp.logical_and(page_j == page, row_j == row)
                    )
                    group = jnp.where(same_row, group_j, -1)
                    merged = jnp.where(
                        lane_group == group, rope_tok_buf[pl.ds(j, 1), :], merged
                    )
                rope_row_buf[pl.ds(i, 1), :] = merged
                write_copies[i].start()

        for (write, *_), nope_copy, rope_copy in zip(tokens, nope_copies, write_copies):

            @pl.when(write)
            def _(nope_copy=nope_copy, rope_copy=rope_copy):
                nope_copy.wait()
                rope_copy.wait()


def scatter_caches(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    dst_rows: jax.Array,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    *,
    interpret: bool = False,
    _fault: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Writes one step's nope and rope tokens into both caches, in place.

    Byte-identical to `scatter.scatter` on each cache with the same
    `dst_rows`.

    Args:
      kv_cache_nope, kv_cache_rope: caches allocated from the specs, both
        SPARSECORE layout.
      kv_c_normed: [num_tokens, lkv_dim] fp8 latents.
      k_pe: [num_tokens, rope_dim] fp8 decoupled RoPE keys.
      dst_rows: [num_tokens] flattened `page * page_size + slot`; negative or
        out of range drops the token.
      nope_spec, rope_spec: layout descriptors the caches were allocated from.
      interpret: run the kernel in the Pallas TPU interpreter (CPU tests).
      _fault: test-only fault injection; None in production.

    Returns:
      The updated (nope, rope) caches, same shapes and dtypes as the inputs.
    """
    num_tokens = kv_c_normed.shape[0]
    assert supports(nope_spec, rope_spec, num_tokens), (
        f"TC cache writer cannot write {num_tokens} tokens into"
        f" {nope_spec.layout} / {rope_spec.layout} caches"
    )
    assert _fault in _FAULTS, f"unknown fault {_fault}"
    for name, cache, spec in (
        ("nope", kv_cache_nope, nope_spec),
        ("rope", kv_cache_rope, rope_spec),
    ):
        assert cache.shape == spec.shape and cache.dtype == spec.jax_dtype, (
            f"{name} cache {cache.shape} {cache.dtype} does not match spec"
            f" {spec.shape} {spec.jax_dtype}"
        )
    for name, values, spec in (
        ("kv_c_normed", kv_c_normed, nope_spec),
        ("k_pe", k_pe, rope_spec),
    ):
        assert values.shape[0] == num_tokens and values.dtype.itemsize == 1, (
            f"{name} must be [{num_tokens}, n] with 1-byte elements, got"
            f" {values.shape} {values.dtype}"
        )
        band = spec.token_bytes // WORD_BYTES
        assert values.shape[1] <= spec.token_bytes and values.shape[1] % band == 0, (
            f"{name} is {values.shape[1]}B; the {spec.token_bytes}B token packs"
            f" whole {band}B bands"
        )
    assert dst_rows.shape == (num_tokens,), (
        f"dst_rows {dst_rows.shape} must be ({num_tokens},)"
    )

    word_dtype = nope_spec.jax_dtype
    rope_row_words = rope_spec.shape[-1]
    kernel = functools.partial(
        _tc_scatter_kernel,
        num_tokens=num_tokens,
        page_size=nope_spec.page_size,
        num_rows=nope_spec.num_pages * nope_spec.page_size,
        nope_token_bytes=nope_spec.token_bytes,
        rope_token_bytes=rope_spec.token_bytes,
        fault=_fault,
    )
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(kv_cache_nope.shape, kv_cache_nope.dtype),
            jax.ShapeDtypeStruct(kv_cache_rope.shape, kv_cache_rope.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(kv_c_normed.shape, lambda i, _: (0, 0)),
                pl.BlockSpec(k_pe.shape, lambda i, _: (0, 0)),
                hbm,
                hbm,
            ],
            out_specs=[hbm, hbm],
            scratch_shapes=[
                # packed nope tokens
                pltpu.VMEM((num_tokens, nope_spec.shape[-1]), word_dtype),
                # packed rope tokens, repeated into every lane group
                pltpu.VMEM((num_tokens, rope_row_words), word_dtype),
                # rope rows read back from the cache
                pltpu.VMEM((num_tokens, rope_row_words), word_dtype),
                # rope rows merged, to write back
                pltpu.VMEM((num_tokens, rope_row_words), word_dtype),
                pltpu.SemaphoreType.DMA((3, num_tokens)),
            ],
        ),
        # Operand indices count the scalar-prefetch `dst_rows`.
        input_output_aliases={3: 0, 4: 1},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="tc_scatter_mla_kv_caches",
    )(
        dst_rows.astype(jnp.int32),
        kv_c_normed,
        k_pe,
        kv_cache_nope,
        kv_cache_rope,
    )

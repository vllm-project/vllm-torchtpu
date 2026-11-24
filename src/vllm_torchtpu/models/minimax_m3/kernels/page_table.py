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
"""16-bit page tables for the MSA kernels' SMEM (``TPU_MINIMAX_M3_PT16``).

The MSA kernels scalar-prefetch vLLM's whole flat block table (``int32
[max_num_seqs * pages_per_seq]``, page key -> physical page) into SMEM. At a
1M max_model_len that is 16 KiB per request at DCP2 (8 KiB at DCP4), which
caps max_num_seqs (`layers.adapter.attention.get_tpu_min_page_size` keeps the
table within half of the 1 MiB SMEM). A rank's KV pool has far fewer than
65536 pages (31332 at DCP2, 20166 at DCP4 served), so `pack` stores two page
ids per 32-bit word: with ``H`` words, key ``k < H`` is the low half of word
``k`` and key ``k >= H`` the high half of word ``k - H`` (the table's halves
side by side: packing is one elementwise pass over two contiguous slices; an
interleaved pair layout cost 37-95 us per pack in lane shuffles,
j-cfce17b9).

A packed table is ``uint32``; an unpacked one ``int32``. The dtype marks the
format, so the kernels and the code between need no extra argument: `load`
and `num_keys` read both. XLA-side code keeps the logical int32 table.
Tables up to `INT32_MAX_BYTES` stay int32 even with the flag on.
`pack` is a function of the step's block table only, so XLA CSEs it across
layers (as the other step metadata, `attention._dcp_rank_tensor`).
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs

MAX_PAGES = 1 << 16
# Largest int32 table the kernels keep: every MSA kernel fits a 768 KiB table
# in the 1 MiB SMEM (deviceless tpu7x compiles at the 1M served shapes; the
# 2048-token prefill LSE call needs the most besides it, ~140 KiB). Up to it
# the flag changes nothing (a packed lookup costs a few scalar ops: indexer
# decode +20-30%, attention +0.2-3% at equal tables, bench j-11320980).
INT32_MAX_BYTES = 768 * 1024
_HALF_BITS = 16
_HALF_MASK = (1 << _HALF_BITS) - 1


def enabled() -> bool:
    return bool(envs.TPU_MINIMAX_M3_PT16)


def is_packed(table) -> bool:
    """Whether `table` (array or ref) is a `pack`ed table."""
    return table.dtype == jnp.uint32


def num_keys(table) -> int:
    """Page keys `table` (array or ref) holds."""
    return table.shape[0] * (2 if is_packed(table) else 1)


def pack(table: jax.Array) -> jax.Array:
    """``int32 [N]`` page ids in ``[0, 65536)`` -> ``uint32 [ceil(N / 2)]``."""
    flat = table.reshape(-1).astype(jnp.uint32)
    if flat.shape[0] % 2:
        flat = jnp.pad(flat, (0, 1))
    half = flat.shape[0] // 2
    return (flat[:half] & _HALF_MASK) | (flat[half:] << _HALF_BITS)


def unpack(packed: jax.Array) -> jax.Array:
    """`pack`'s inverse (padded to an even length)."""
    lo = (packed & _HALF_MASK).astype(jnp.int32)
    hi = (packed >> _HALF_BITS).astype(jnp.int32)
    return jnp.concatenate([lo, hi])


def prefetch(table: jax.Array, num_pages: int) -> jax.Array:
    """The kernels' flat table for `table` (vLLM's logical block table) over a
    cache of `num_pages` pages: packed with the flag on, ids that fit and an
    int32 table over `INT32_MAX_BYTES`."""
    flat = table.reshape(-1)
    if enabled() and num_pages <= MAX_PAGES and flat.size * 4 > INT32_MAX_BYTES:
        return pack(flat)
    return flat


def run_start(table_ref, base):
    """`load_run`'s per-run state for a run from `base`: its first word and
    half shift (int32 tables: `base`)."""
    if not is_packed(table_ref):
        return base, None
    half = table_ref.shape[0]
    high = base >= half
    shift = jnp.where(high, _HALF_BITS, 0).astype(jnp.uint32)
    return base - jnp.where(high, half, 0), shift


def load_run(table_ref, start, g: int):
    """``table[base + g]`` for a static `g` of a run from `start =
    run_start(table_ref, base)` that does not cross the table's middle
    (`run_safe`): a load and two scalar ops per entry."""
    word_base, shift = start
    if shift is None:
        return table_ref[word_base + g]
    return ((table_ref[word_base + g] >> shift) & _HALF_MASK).astype(jnp.int32)


def run_safe(table, pages_per_seq: int, run: int) -> bool:
    """Whether every `run` entries of a `pages_per_seq`-entry row starting at
    a multiple of `run` stay in one half of `table` (array or ref)."""
    if not is_packed(table):
        return True
    return pages_per_seq % run == 0 and table.shape[0] % pages_per_seq == 0


def load(table_ref, key):
    """``table[key]`` (an int32 scalar) of a packed or plain SMEM table, in a
    kernel body or an index map."""
    if not is_packed(table_ref):
        return table_ref[key]
    half = table_ref.shape[0]
    high = key >= half
    word = table_ref[key - jnp.where(high, half, 0)]
    shift = jnp.where(high, _HALF_BITS, 0).astype(jnp.uint32)
    return ((word >> shift) & _HALF_MASK).astype(jnp.int32)

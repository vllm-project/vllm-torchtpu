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
"""`write_kv` must leave the KV cache exactly as RPA v3 leaves it.

It replaces an RPA v3 call whose attention output is discarded, so its
contract is the cache the kernel leaves behind: same slots, same merged K/V
rows, nothing else touched, updated in place. The CPU test checks it against an
independent NumPy writer; the TPU tests check it against the kernel itself and
check that the donated cache is not copied.
"""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_v3
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kv_write import write_kv

PAGE_SIZE = 16
MAX_NUM_SEQS = 6
NUM_PADDING_TOKENS = 5
# Pages no sequence owns: they must come out untouched, and they make the
# cache large next to the write's scratch, as in serving.
NUM_SPARE_PAGES = 512

# (q_len, kv_len) per live sequence, then the distribution. Sequence 0 of
# "prefill_range" sits in the prefill-only range, which RPA v3 skips without
# a chunk prefill size.
BATCHES = {
    "verify_windows": ([(16, 40), (16, 17), (16, 64)], (0, 0, 3)),
    "decode_and_mixed": ([(1, 9), (1, 33), (5, 21), (24, 24)], (2, 2, 4)),
    "prefill_range": ([(16, 48), (8, 30)], (0, 1, 2)),
}

CACHES = [
    pytest.param(8, 128, jnp.bfloat16, id="bf16-8kv-hd128"),
    pytest.param(1, 96, jnp.bfloat16, id="bf16-1kv-hd96"),
    pytest.param(1, 128, jnp.float8_e4m3fn, id="fp8-1kv-hd128"),
    pytest.param(4, 128, jnp.float8_e4m3fn, id="fp8-4kv-hd128"),
]


@dataclasses.dataclass
class Case:
    kv_cache: jax.Array
    keys: jax.Array
    values: jax.Array
    kv_lens: jax.Array
    page_indices: jax.Array
    cu_q_lens: jax.Array
    distribution: jax.Array

    def rpa_args(self):
        return (
            self.kv_lens,
            self.page_indices,
            self.cu_q_lens,
            self.distribution,
        )


def _make_case(batch, num_kv_heads, head_dim, dtype, seed=0):
    seqs, distribution = BATCHES[batch]
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-kv_len // PAGE_SIZE) for _, kv_len in seqs)
    num_pages = 1 + MAX_NUM_SEQS * pages_per_seq + NUM_SPARE_PAGES
    page_indices = rng.permutation(np.arange(1, num_pages)).astype(np.int32)[
        : MAX_NUM_SEQS * pages_per_seq
    ]

    q_lens = [q_len for q_len, _ in seqs]
    cu_q_lens = np.zeros(MAX_NUM_SEQS + 1, np.int32)
    cu_q_lens[1 : len(seqs) + 1] = np.cumsum(q_lens)
    cu_q_lens[len(seqs) + 1 :] = cu_q_lens[len(seqs)]
    kv_lens = np.zeros(MAX_NUM_SEQS, np.int32)
    kv_lens[: len(seqs)] = [kv_len for _, kv_len in seqs]

    num_tokens = sum(q_lens) + NUM_PADDING_TOKENS
    kv_shape = (num_tokens, num_kv_heads, head_dim)
    cache_shape = rpa_v3.get_kv_cache_shape(
        num_pages, PAGE_SIZE, num_kv_heads, head_dim, dtype
    )
    return Case(
        kv_cache=jnp.asarray(rng.standard_normal(cache_shape), dtype),
        keys=jnp.asarray(rng.standard_normal(kv_shape), dtype),
        values=jnp.asarray(rng.standard_normal(kv_shape), dtype),
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution, jnp.int32),
    )


def _reference_write(case):
    """RPA v3's cache write, re-derived sequence by sequence in NumPy."""
    cache = np.array(case.kv_cache)
    merged = np.asarray(rpa_v3.merge_kv(case.keys, case.values))
    kv_lens, page_indices, cu_q_lens = (
        np.asarray(case.kv_lens),
        np.asarray(case.page_indices),
        np.asarray(case.cu_q_lens),
    )
    pages_per_seq = page_indices.size // kv_lens.size
    decode_end, prefill_end, num_seqs = np.asarray(case.distribution)
    for seq in range(num_seqs):
        if decode_end <= seq < prefill_end:
            continue
        q_start, q_end = cu_q_lens[seq], cu_q_lens[seq + 1]
        first_pos = kv_lens[seq] - (q_end - q_start)
        for offset, token in enumerate(range(q_start, q_end)):
            pos = first_pos + offset
            page = page_indices[seq * pages_per_seq + pos // PAGE_SIZE]
            cache[page, pos % PAGE_SIZE, ..., : merged.shape[-1]] = merged[token]
    return cache


def _bits(array):
    return np.asarray(array).view(np.uint8)


@pytest.mark.parametrize("batch", list(BATCHES))
@pytest.mark.parametrize("num_kv_heads,head_dim,dtype", CACHES)
def test_write_kv_matches_reference(batch, num_kv_heads, head_dim, dtype):
    case = _make_case(batch, num_kv_heads, head_dim, dtype)
    written = write_kv(case.kv_cache, case.keys, case.values, *case.rpa_args())
    np.testing.assert_array_equal(_bits(written), _bits(_reference_write(case)))


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", list(BATCHES))
@pytest.mark.parametrize("num_kv_heads,head_dim,dtype", CACHES)
def test_write_kv_matches_rpa_v3_kernel(batch, num_kv_heads, head_dim, dtype):
    case = _make_case(batch, num_kv_heads, head_dim, dtype)
    quantized = dtype == jnp.float8_e4m3fn
    _, kernel_cache = rpa_v3.ragged_paged_attention(
        jnp.zeros((case.keys.shape[0], 2 * num_kv_heads, head_dim), jnp.bfloat16),
        case.keys,
        case.values,
        jnp.copy(case.kv_cache),  # The kernel donates its cache.
        *case.rpa_args(),
        use_causal_mask=False,
        k_scale=1.0 if quantized else None,
        v_scale=1.0 if quantized else None,
    )
    written = write_kv(case.kv_cache, case.keys, case.values, *case.rpa_args())
    np.testing.assert_array_equal(_bits(written), _bits(kernel_cache))


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_kv_heads,head_dim,dtype", CACHES)
def test_write_kv_updates_the_donated_cache_in_place(num_kv_heads, head_dim, dtype):
    case = _make_case("verify_windows", num_kv_heads, head_dim, dtype)
    compiled = (
        jax.jit(write_kv, donate_argnums=0)
        .lower(case.kv_cache, case.keys, case.values, *case.rpa_args())
        .compile()
    )
    cache_bytes = case.kv_cache.size * case.kv_cache.dtype.itemsize
    memory = compiled.memory_analysis()
    assert memory.alias_size_in_bytes >= cache_bytes
    assert memory.temp_size_in_bytes < cache_bytes
    num_pages, page_size = case.kv_cache.shape[:2]
    cache_shapes = (f"[{num_pages},{page_size},", f"[{num_pages * page_size},")
    copies = [
        line
        for line in compiled.as_text().splitlines()
        if re.search(r"\b(copy|copy-start|transpose)\(", line)
        and any(shape in line for shape in cache_shapes)
    ]
    assert not copies, copies

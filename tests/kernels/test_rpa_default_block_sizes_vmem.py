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
"""RPA v3 default block sizes must fit the kernel's VMEM budget.

The heuristic in ``get_default_block_sizes`` derives the query/KV blocks from
the query and KV lengths only. For shards with a large per-token footprint
(``num_kv_heads * head_dim`` and ``num_q_heads * head_dim``) the resulting
scratch (``get_vmem_estimate_bytes``) can exceed the chip's VMEM and the kernel
then fails to compile. ``fit_block_sizes_to_vmem`` shrinks the blocks until the
estimate fits ``vmem_limit_bytes``.

Properties covered (CPU only; the TPU generation is patched):
* identity: if the heuristic blocks already fit, they are returned unchanged;
* fit: otherwise the returned blocks fit, or are at the smallest legal size;
* invariants the kernel validates (``bq_sz % bq_csz == 0``,
  ``bkv_sz % bkv_csz == 0``, ``bkv_* % page_size == 0``, ``bq_csz >= 1``);
* shrink order: KV-fetch block first, then query block, then KV down to one
  page;
* ``vmem_limit_bytes`` is honoured by ``get_default_block_sizes``.
"""

import itertools
from unittest import mock

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa

MIB = 1024 * 1024
HUGE = 1 << 40

# Per-shard shapes: (num_q_heads, num_kv_heads, head_dim). Covers MHA, GQA
# ratios 2..16, head dims 128/256/512 and single-KV-head shards.
SHARD_SHAPES = [
    (8, 8, 128),
    (32, 8, 128),
    (32, 4, 128),
    (64, 8, 128),
    (16, 1, 128),
    (16, 8, 256),
    (8, 4, 256),
    (16, 2, 512),
    (4, 1, 512),
]
DTYPES = [(jnp.bfloat16, jnp.bfloat16), (jnp.bfloat16, jnp.float8_e4m3fn)]
PAGE_SIZES = [16, 128, 256]
MAX_NUM_TOKENS = [256, 512, 1024, 2048, 4096, 8192]
VMEM_CAPACITIES = [64 * MIB, 128 * MIB]
GENERATIONS = [5, 6, 7]


def _estimate(bs, *, q_heads, kv_heads, head_dim, q_dtype, kv_dtype):
    return rpa.get_vmem_estimate_bytes(
        kv_heads,
        q_heads // kv_heads,
        head_dim,
        bs["bq_sz"],
        bs["bkv_sz"],
        q_dtype,
        kv_dtype,
    )


def _defaults(shape, dtypes, page_size, max_num_tokens, case, vmem_limit_bytes):
    q_heads, kv_heads, head_dim = shape
    q_dtype, kv_dtype = dtypes
    return rpa.get_default_block_sizes(
        q_dtype,
        kv_dtype,
        q_heads,
        kv_heads,
        head_dim,
        page_size=page_size,
        max_num_tokens=max_num_tokens,
        max_num_seqs=256,
        pages_per_seq=max(2, 32768 // page_size),
        case=case,
        vmem_limit_bytes=vmem_limit_bytes,
    )


def _check_invariants(bs, page_size):
    assert bs["bq_sz"] >= 1 and bs["bq_csz"] >= 1
    assert bs["bq_sz"] % bs["bq_csz"] == 0
    assert bs["bkv_sz"] % page_size == 0 and bs["bkv_csz"] % page_size == 0
    assert bs["bkv_sz"] % bs["bkv_csz"] == 0


def _grid():
    return itertools.product(
        GENERATIONS,
        SHARD_SHAPES,
        DTYPES,
        PAGE_SIZES,
        MAX_NUM_TOKENS,
        [rpa.RpaCase.MIXED, rpa.RpaCase.DECODE],
        VMEM_CAPACITIES,
    )


@pytest.mark.parametrize("generation", GENERATIONS)
def test_fit_is_identity_when_heuristic_fits(generation):
    """Configurations whose estimate fits must produce the identical blocks."""
    with mock.patch.object(rpa, "get_tpu_version", return_value=generation):
        checked = 0
        for gen, shape, dtypes, page, tokens, case, cap in _grid():
            if gen != generation:
                continue
            unfitted = _defaults(shape, dtypes, page, tokens, case, HUGE)
            est = _estimate(
                unfitted,
                q_heads=shape[0],
                kv_heads=shape[1],
                head_dim=shape[2],
                q_dtype=dtypes[0],
                kv_dtype=dtypes[1],
            )
            if est > cap:
                continue
            fitted = _defaults(shape, dtypes, page, tokens, case, cap)
            assert fitted == unfitted, (shape, dtypes, page, tokens, case, cap)
            checked += 1
        assert checked > 0


@pytest.mark.parametrize("generation", GENERATIONS)
def test_fit_result_fits_or_is_minimal(generation):
    """When the heuristic overflows, the result fits (or nothing is left to shrink)."""
    with mock.patch.object(rpa, "get_tpu_version", return_value=generation):
        shrunk = 0
        for gen, shape, dtypes, page, tokens, case, cap in _grid():
            if gen != generation:
                continue
            kw = dict(
                q_heads=shape[0],
                kv_heads=shape[1],
                head_dim=shape[2],
                q_dtype=dtypes[0],
                kv_dtype=dtypes[1],
            )
            unfitted = _defaults(shape, dtypes, page, tokens, case, HUGE)
            if _estimate(unfitted, **kw) <= cap:
                continue
            fitted = _defaults(shape, dtypes, page, tokens, case, cap)
            _check_invariants(fitted, page)
            assert fitted["bq_sz"] <= unfitted["bq_sz"]
            assert fitted["bkv_sz"] <= unfitted["bkv_sz"]
            at_floor = fitted["bq_sz"] == 1 and fitted["bkv_sz"] == page
            assert _estimate(fitted, **kw) <= cap or at_floor, (
                shape,
                dtypes,
                page,
                tokens,
                case,
                cap,
            )
            shrunk += 1
        # The v5/v6 heuristics cap bq at 1024/q_per_kv and bkv at 1024 and do
        # not overflow 64 MiB for these shapes; the v7 heuristic does.
        if generation >= 7:
            assert shrunk > 0


@pytest.mark.parametrize("generation", GENERATIONS)
def test_invariants_hold_for_all_defaults(generation):
    with mock.patch.object(rpa, "get_tpu_version", return_value=generation):
        for gen, shape, dtypes, page, tokens, case, cap in _grid():
            if gen != generation:
                continue
            _check_invariants(_defaults(shape, dtypes, page, tokens, case, cap), page)


def test_wide_shard_overflows_without_fit_and_fits_with_it():
    """A 2:1 GQA shard with 8 KV heads x head_dim 256 on a 64 MiB VMEM part.

    With 1024 batched tokens the unconstrained heuristic picks bq=512/bkv=2048,
    whose scratch (68 MiB) cannot fit; halving the KV block (50 MiB) does.
    """
    shape, dtypes = (16, 8, 256), (jnp.bfloat16, jnp.bfloat16)
    kw = dict(q_heads=16, kv_heads=8, head_dim=256, q_dtype=dtypes[0], kv_dtype=dtypes[1])
    with mock.patch.object(rpa, "get_tpu_version", return_value=7):
        unfitted = _defaults(shape, dtypes, 16, 1024, rpa.RpaCase.MIXED, HUGE)
        assert (unfitted["bq_sz"], unfitted["bkv_sz"]) == (512, 2048)
        assert _estimate(unfitted, **kw) == 68 * MIB
        fitted = _defaults(shape, dtypes, 16, 1024, rpa.RpaCase.MIXED, 64 * MIB)
        assert (fitted["bq_sz"], fitted["bkv_sz"]) == (512, 1024)
        assert _estimate(fitted, **kw) == 50 * MIB
        # At 512 batched tokens the heuristic (bq=256) already fits: unchanged.
        small = _defaults(shape, dtypes, 16, 512, rpa.RpaCase.MIXED, 64 * MIB)
        assert small == _defaults(shape, dtypes, 16, 512, rpa.RpaCase.MIXED, HUGE)


def test_shrink_order_kv_then_query_then_kv_to_page():
    kw = dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=16,
        actual_num_kv_heads=8,
        head_dim=256,
        page_size=16,
    )
    start = (512, 2048, 512, 512)
    # Generous budget: identity.
    assert rpa.fit_block_sizes_to_vmem(*start, vmem_limit_bytes=HUGE, **kw) == start
    # First step shrinks only the KV-fetch block.
    bq, bkv, bq_c, bkv_c = rpa.fit_block_sizes_to_vmem(*start, vmem_limit_bytes=60 * MIB, **kw)
    assert (bq, bkv) == (512, 1024) and (bq_c, bkv_c) == (512, 512)
    # Tighter: KV reaches the 512 floor before the query block shrinks.
    bq, bkv, _, _ = rpa.fit_block_sizes_to_vmem(*start, vmem_limit_bytes=30 * MIB, **kw)
    assert bkv == 512 and bq < 512
    # Tightest: the query block hits 1, then KV shrinks (page-aligned) until the
    # estimate fits.
    bq, bkv, bq_c, bkv_c = rpa.fit_block_sizes_to_vmem(*start, vmem_limit_bytes=1 * MIB, **kw)
    assert bq == 1 and bq_c == 1 and 16 <= bkv < 512 and bkv % 16 == 0 and bkv_c <= bkv
    assert rpa.get_vmem_estimate_bytes(8, 2, 256, bq, bkv, jnp.bfloat16, jnp.bfloat16) <= 1 * MIB


@pytest.mark.parametrize("page_size", [16, 128, 256])
def test_compute_blocks_stay_divisors_after_shrinking(page_size):
    kw = dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=32,
        actual_num_kv_heads=8,
        head_dim=128,
        page_size=page_size,
    )
    for limit in (HUGE, 48 * MIB, 24 * MIB, 8 * MIB, 2 * MIB):
        bq, bkv, bq_c, bkv_c = rpa.fit_block_sizes_to_vmem(
            256, 3 * 1024, 256, 1024, vmem_limit_bytes=limit, **kw
        )
        _check_invariants(dict(bq_sz=bq, bkv_sz=bkv, bq_csz=bq_c, bkv_csz=bkv_c), page_size)


def test_largest_divisor_helper():
    assert rpa._largest_divisor_at_most(512, 512) == 512
    assert rpa._largest_divisor_at_most(512, 300) == 256
    assert rpa._largest_divisor_at_most(768, 512, multiple_of=256) == 256
    assert rpa._largest_divisor_at_most(768, 512, multiple_of=16) == 384
    assert rpa._largest_divisor_at_most(16, 512, multiple_of=16) == 16
    assert rpa._largest_divisor_at_most(1, 1) == 1

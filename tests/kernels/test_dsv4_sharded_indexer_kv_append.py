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
"""Specs for sequence-sharded DCP indexer K cache append."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_append,
    kv_cache_utils,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout


def _make_16dev_mesh() -> Mesh:
    devs = jax.devices()
    if len(devs) < 16:
        pytest.skip(f"needs 16 JAX devices, got {len(devs)}")
    return Mesh(np.array(devs[:16]), ("dcp",))


def _make_cache(
    rng: np.random.Generator,
    *,
    num_pages: int,
    page_size: int,
    layout: KVLayout,
    sentinel: int = 0xA5,
) -> np.ndarray:
    del rng
    has_cache = np.full((num_pages, page_size // 4, 4, 256), sentinel, dtype=np.uint8)
    if layout == KVLayout.SEQ_ALONG_LANE:
        return np.asarray(
            st_lib.convert_cache_to_seq_along_lane(
                jnp.asarray(has_cache), actual_head_dim=128
            )
        )
    return has_cache


def _random_batch(
    rng: np.random.Generator,
    *,
    num_tokens: int,
    num_seqs: int,
    live_seqs: int,
    page_size: int,
    dcp_size: int,
    pages_per_seq: int,
    prefix_offset: int = 0,
    include_oob_pages: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_lens = np.zeros(num_seqs, dtype=np.int32)
    live_total = num_tokens - int(rng.integers(0, min(64, num_tokens // 4 + 1)))
    if live_seqs == 1:
        q_lens[0] = live_total
    else:
        n_dec = live_seqs - 1
        dec_q = rng.integers(
            1, max(2, min(5, live_total // (2 * live_seqs) + 1)), size=n_dec
        )
        q_lens[:n_dec] = dec_q
        q_lens[n_dec] = max(1, live_total - int(dec_q.sum()))
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    max_pos = page_size * dcp_size * pages_per_seq
    seq_lens = np.zeros(num_seqs, dtype=np.int32)
    for s in range(live_seqs):
        max_ctx = max(0, max_pos - int(q_lens[s]) - prefix_offset)
        ctx = prefix_offset + (int(rng.integers(0, max_ctx + 1)) if max_ctx > 0 else 0)
        seq_lens[s] = min(max_pos, ctx + int(q_lens[s]))
    total_local_pages = num_seqs * pages_per_seq
    local_pages = rng.permutation(total_local_pages)
    virt_pages = (
        local_pages
        + total_local_pages * rng.integers(0, dcp_size, size=total_local_pages)
    ).astype(np.int32)
    block_tables = virt_pages.reshape(num_seqs, pages_per_seq)
    if include_oob_pages and live_seqs >= 2:
        block_tables[0, -1] = -1
        block_tables[1, 0] = total_local_pages * dcp_size + 7
    return seq_lens, cu, block_tables


@pytest.mark.parametrize("width", [128, 129, 130, 131, 132])
def test_should_roundtrip_band_pack_for_aligned_and_unaligned_widths(
    width: int,
) -> None:
    rng = np.random.default_rng(700 + width)
    src = jnp.asarray(rng.integers(0, 256, size=(64, width), dtype=np.uint8))
    packed, orig_width = kv_cache_append._pack_u8_to_u32_band(src)  # noqa: SLF001 -- unit spec for band packing invariant
    assert packed.dtype == jnp.uint32
    assert orig_width == width
    got = kv_cache_append._unpack_u32_band_to_u8(packed, orig_width)  # noqa: SLF001 -- unit spec for band unpacking invariant
    assert got.dtype == jnp.uint8
    np.testing.assert_array_equal(np.asarray(got), np.asarray(src))


@pytest.mark.parametrize("rows_local", [64, 128, 1024])
def test_should_never_exceed_sharded_a2a_cap_across_random_batches(
    rows_local: int,
) -> None:
    dcp, il, num_seqs, page_size, pps = 16, 4, 64, 256, 8
    rng = np.random.default_rng(900 + rows_local)
    cap_full = kv_cache_utils.sharded_a2a_cap(
        rows_local, num_seqs=num_seqs, dcp_size=dcp, interleave_size=il
    )
    fast_limit = kv_cache_utils.sharded_a2a_fast_seq_limit(
        rows_local, dcp_size=dcp, interleave_size=il
    )
    for live_seqs in (1, 4, 17, 32, 64):
        sl, cu, bt = _random_batch(
            rng,
            num_tokens=dcp * rows_local,
            num_seqs=num_seqs,
            live_seqs=live_seqs,
            page_size=page_size,
            dcp_size=dcp,
            pages_per_seq=pps,
        )
        for src_rank in range(dcp):
            lo, hi = src_rank * rows_local, (src_rank + 1) * rows_local
            for owner in range(dcp):
                p, _ = kv_cache_utils.get_dcp_page_and_slot(
                    dcp * rows_local,
                    jnp.asarray(sl),
                    jnp.asarray(bt),
                    jnp.asarray(cu),
                    owner,
                    page_size=page_size,
                    num_pages=num_seqs * pps,
                    dcp_size=dcp,
                    interleave_size=il,
                )
                cnt = int((np.asarray(p[lo:hi]) < kv_cache_utils.OOB_PAGE).sum())
                assert cnt <= cap_full
                if (
                    live_seqs <= fast_limit
                    and cap_full > kv_cache_utils.SHARDED_A2A_FAST_CAP
                ):
                    assert cnt <= kv_cache_utils.SHARDED_A2A_FAST_CAP


def test_should_match_replicated_append_for_single_rank_dcp1() -> None:
    rng = np.random.default_rng(101)
    num_tokens, num_seqs, page_size, pps = 64, 4, 128, 2
    num_pages = num_seqs * pps
    sl, cu, bt = _random_batch(
        rng,
        num_tokens=num_tokens,
        num_seqs=num_seqs,
        live_seqs=3,
        page_size=page_size,
        dcp_size=1,
        pages_per_seq=pps,
    )
    k = jnp.asarray(rng.integers(0, 256, size=(num_tokens, 129), dtype=np.uint8))
    cache = jnp.asarray(
        _make_cache(
            rng,
            num_pages=num_pages,
            page_size=page_size,
            layout=KVLayout.SEQ_ALONG_LANE,
        )
    )
    got = kv_cache_append.sharded_kv_append_dcp(
        cache,
        k,
        jnp.asarray(sl),
        jnp.asarray(bt),
        jnp.asarray(cu),
        jnp.int32(0),
        dcp_size=1,
        interleave_size=4,
        axis_name="dcp",
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        use_pallas_scatter=False,
    )
    p, s = kv_cache_utils.get_dcp_page_and_slot(
        num_tokens,
        jnp.asarray(sl),
        jnp.asarray(bt),
        jnp.asarray(cu),
        0,
        page_size=page_size,
        num_pages=num_pages,
        dcp_size=1,
        interleave_size=4,
    )
    want = kv_cache_append.scatter_rows_reference(
        cache, k, kv_cache_utils.get_dst_rows(p, s, page_size), KVLayout.SEQ_ALONG_LANE
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize(
    ("rows_local", "live_seqs", "prefix_offset", "include_oob", "layout"),
    [
        (1024, 4, 0, False, KVLayout.SEQ_ALONG_LANE),
        (1024, 24, 0, False, KVLayout.SEQ_ALONG_LANE),
        (256, 6, 4096, True, KVLayout.SEQ_ALONG_LANE),
        (64, 3, 256, True, KVLayout.HEAD_ALONG_SUBLANE),
    ],
)
def test_should_match_replicated_append_bit_for_bit_on_16_devices(
    rows_local: int,
    live_seqs: int,
    prefix_offset: int,
    include_oob: bool,
    layout: KVLayout,
) -> None:
    mesh = _make_16dev_mesh()
    dcp, il, num_seqs, page_size, pps = 16, 4, 32, 128, 4
    num_pages = num_seqs * pps
    num_tokens = dcp * rows_local
    rng = np.random.default_rng(2026 + rows_local + live_seqs + prefix_offset)
    sl, cu, bt = _random_batch(
        rng,
        num_tokens=num_tokens,
        num_seqs=num_seqs,
        live_seqs=live_seqs,
        page_size=page_size,
        dcp_size=dcp,
        pages_per_seq=pps,
        prefix_offset=prefix_offset,
        include_oob_pages=include_oob,
    )
    k_np = rng.integers(0, 256, size=(num_tokens, 129), dtype=np.uint8)
    # Non-zero sentinel on padded tail rows so out-of-batch writes fail loudly.
    live_tokens = int(cu[-1])
    if live_tokens < num_tokens:
        k_np[live_tokens:] = 0xE7
    base_cache = _make_cache(
        rng, num_pages=num_pages, page_size=page_size, layout=layout
    )
    caches_np = np.stack([base_cache for _ in range(dcp)], axis=0)
    caches_sh = jax.device_put(
        jnp.asarray(caches_np.reshape(dcp * num_pages, *base_cache.shape[1:])),
        NamedSharding(mesh, P("dcp")),
    )
    k_sh = jax.device_put(jnp.asarray(k_np), NamedSharding(mesh, P("dcp")))
    k_rep = jax.device_put(jnp.asarray(k_np), NamedSharding(mesh, P()))
    sl_rep = jax.device_put(jnp.asarray(sl), NamedSharding(mesh, P()))
    bt_rep = jax.device_put(jnp.asarray(bt), NamedSharding(mesh, P()))
    cu_rep = jax.device_put(jnp.asarray(cu), NamedSharding(mesh, P()))
    ranks_sh = jax.device_put(
        jnp.arange(dcp, dtype=jnp.int32), NamedSharding(mesh, P("dcp"))
    )

    def _run_ref(cache_kv, k_full, seq_lens, block_tables, cu_q_lens, rk):
        p, s = kv_cache_utils.get_dcp_page_and_slot(
            num_tokens,
            seq_lens,
            block_tables,
            cu_q_lens,
            rk[0],
            page_size=page_size,
            num_pages=num_pages,
            dcp_size=dcp,
            interleave_size=il,
        )
        return kv_cache_append.scatter_rows_reference(
            cache_kv, k_full, kv_cache_utils.get_dst_rows(p, s, page_size), layout
        )

    def _run_sharded(cache_kv, k_local, seq_lens, block_tables, cu_q_lens, rk):
        return kv_cache_append.sharded_kv_append_dcp(
            cache_kv,
            k_local,
            seq_lens,
            block_tables,
            cu_q_lens,
            rk[0],
            dcp_size=dcp,
            interleave_size=il,
            axis_name="dcp",
            kv_layout=layout,
            use_pallas_scatter=False,
        )

    ref_fn = jax.jit(
        jax.shard_map(
            _run_ref,
            mesh=mesh,
            in_specs=(P("dcp"), P(), P(), P(), P(), P("dcp")),
            out_specs=P("dcp"),
            check_vma=False,
        )
    )
    sh_fn = jax.jit(
        jax.shard_map(
            _run_sharded,
            mesh=mesh,
            in_specs=(P("dcp"), P("dcp"), P(), P(), P(), P("dcp")),
            out_specs=P("dcp"),
            check_vma=False,
        )
    )
    want = np.asarray(ref_fn(caches_sh, k_rep, sl_rep, bt_rep, cu_rep, ranks_sh))
    got = np.asarray(sh_fn(caches_sh, k_sh, sl_rep, bt_rep, cu_rep, ranks_sh))
    # Anti-cancellation check: confirm writes actually mutated cache from sentinel.
    assert int(np.count_nonzero(want != caches_np.reshape(want.shape))) > 0
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    ("dcp_size", "interleave_size", "page_size"),
    [(0, 4, 16), (4, 0, 16), (4, 6, 16)],
)
def test_should_reject_invalid_sharded_dcp_geometry(
    dcp_size: int, interleave_size: int, page_size: int
) -> None:
    with pytest.raises(ValueError):
        kv_cache_utils.build_sharded_dcp_a2a_plan(
            16,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2, 2), dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.int32(0),
            page_size=page_size,
            num_pages=4,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
            cap=16,
            axis_name="dcp",
        )

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
"""Behavioral specs for step-level zero-ppermute K-band dual-zigzag DCP layout."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_append,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_utils as idx_kvu,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla import shift_quant_split_3d as sq3d
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp

AXIS = "dcp"


def _make_mesh(dcp: int = 4) -> Mesh:
    devs = jax.devices()
    if len(devs) < dcp:
        pytest.skip(f"needs {dcp} JAX devices, got {len(devs)}")
    return Mesh(np.array(devs[:dcp]), (AXIS,))


def _interpret_flag():
    if jax.default_backend() == "tpu":
        return False
    return pltpu.InterpretParams(detect_races=True)


def _zigzag_pack_np(arr: np.ndarray, dcp: int, num_bands: int) -> np.ndarray:
    """Arrange global `[T, ...]` into rank-major `[dcp * R, ...]` dual-zigzag order."""
    num_tokens = arr.shape[0]
    band = num_tokens // (dcp * num_bands)
    bands = arr.reshape(num_bands, dcp, band, *arr.shape[1:])
    chunks = [bands[k, ::-1] if (k & 1) else bands[k] for k in range(num_bands)]
    return np.stack(chunks, axis=1).reshape(num_tokens, *arr.shape[1:])


def _zigzag_unpack_np(sharded: np.ndarray, dcp: int, num_bands: int) -> np.ndarray:
    """Invert `_zigzag_pack_np` back into global `[T, ...]` order."""
    num_tokens = sharded.shape[0]
    band = num_tokens // (dcp * num_bands)
    g = sharded.reshape(dcp, num_bands, band, *sharded.shape[1:])
    parts = [g[::-1, k] if (k & 1) else g[:, k] for k in range(num_bands)]
    return np.concatenate(parts, axis=0).reshape(num_tokens, *sharded.shape[1:])


def _make_batch(
    q_lens: list[int], prefixes: list[int], max_reqs: int
) -> tuple[np.ndarray, np.ndarray, int]:
    seq_lens = np.zeros((max_reqs,), dtype=np.int32)
    qsl = np.zeros((max_reqs + 1,), dtype=np.int32)
    cur = 0
    for i, (ql, pre) in enumerate(zip(q_lens, prefixes, strict=True)):
        cur += ql
        qsl[i + 1] = cur
        seq_lens[i] = pre + ql
    for i in range(len(q_lens), max_reqs):
        qsl[i + 1] = cur
    return seq_lens, qsl, len(q_lens)


def test_should_reject_invalid_zigzag_bands_and_resolve_effective_bounds() -> None:
    with pytest.raises(ValueError, match="zigzag_bands"):
        tp.TokenParallelLayout(
            dcp_size=4,
            interleave_size=4,
            page_size=16,
            split_rows=4,
            max_pages=4,
            zigzag_bands=0,
        )
    with pytest.raises(ValueError, match="zigzag_bands"):
        tp.TokenParallelLayout(
            dcp_size=4,
            interleave_size=4,
            page_size=16,
            split_rows=4,
            max_pages=4,
            zigzag_bands=3,
        )

    lay = tp.TokenParallelLayout(
        dcp_size=16,
        interleave_size=4,
        page_size=256,
        split_rows=64,
        max_pages=44,
        zigzag_bands=4,
    )
    assert lay.effective_zigzag_bands(2048) == 1
    assert lay.effective_zigzag_bands(4096) == 4
    assert lay.effective_zigzag_bands(8192) == 4
    assert lay.effective_zigzag_bands(16384) == 4
    assert lay.effective_zigzag_bands(4096 + 16) == 1

    contig = tp.TokenParallelLayout(
        dcp_size=16,
        interleave_size=4,
        page_size=256,
        split_rows=64,
        max_pages=44,
        zigzag_bands=1,
    )
    assert contig.effective_zigzag_bands(16384) == 1


@pytest.mark.parametrize("num_bands", [2, 4])
def test_should_match_global_token_metadata_bijection_across_adversarial_batches(
    num_bands: int,
) -> None:
    dcp, num_tokens, split_rows, max_reqs = 4, 256, 8, 24
    lay = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=16,
        zigzag_bands=num_bands,
    )
    rows = lay.rows_per_rank(num_tokens)
    cases = [
        ([], []),
        ([1], [10]),
        ([1, 1, 1, 1, 1, 1, 1, 1], [3, 5, 7, 11, 13, 17, 19, 23]),
        ([split_rows], [12]),
        ([split_rows + 1], [12]),
        ([256], [40]),
        ([173], [29]),
        ([64], [100]),
        ([1, 1, 130, 41, 12], [20, 45, 50, 12, 0]),
        ([0, 37, 0, 115, 0, 79], [0, 10, 0, 200, 0, 5]),
        ([10] * 24, [i * 3 for i in range(24)]),
    ]
    for q_lens, prefixes in cases:
        seq_lens_np, qsl_np, n_seqs = _make_batch(q_lens, prefixes, max_reqs)
        total_valid = int(qsl_np[n_seqs])
        seq_lens_j = jnp.asarray(seq_lens_np)
        qsl_j = jnp.asarray(qsl_np)
        n_seqs_j = jnp.int32(n_seqs)
        pages, offsets = tp.step_pages(lay, seq_lens_j, qsl_j, n_seqs_j)
        seen_global: list[int] = []

        for r in range(dcp):
            rank_j = jnp.int32(r)
            rr = tp.rank_rows(lay, rank_j, num_tokens, seq_lens_j, qsl_j, n_seqs_j)
            toks = np.asarray(tp.zigzag_rank_global_tokens(r, rows, dcp, num_bands))
            tail_toks = toks[split_rows:] if r == 0 else toks
            r_reqs = np.asarray(tp.row_requests(rr, rows))
            cu = np.asarray(rr.cu_q_lens)
            sl = np.asarray(rr.seq_lens)
            n_loc = int(rr.num_seqs)
            n_rows = int(rr.num_rows)
            assert cu[0] == 0
            assert cu[n_loc] == n_rows
            assert rr.req_ids is not None
            rids = np.asarray(rr.req_ids)
            assert n_loc <= max_reqs + num_bands - 1

            expected_valid = [g for g in tail_toks if g < total_valid]
            assert n_rows == len(expected_valid)
            for loc in range(n_rows):
                g = int(tail_toks[loc])
                assert g == expected_valid[loc]
                k = int(np.searchsorted(cu[1 : n_loc + 1], loc, side="right"))
                req_g = int(rids[k])
                assert r_reqs[loc] == req_g
                assert qsl_np[req_g] <= g < qsl_np[req_g + 1]
                q_len_span = int(cu[k + 1] - cu[k])
                local_pos = int(sl[k]) - q_len_span + (loc - int(cu[k]))
                true_pos = (
                    int(seq_lens_np[req_g])
                    - (int(qsl_np[req_g + 1]) - int(qsl_np[req_g]))
                    + (g - int(qsl_np[req_g]))
                )
                assert local_pos == true_pos
                seen_global.append(g)

            tbl = np.asarray(
                tp.natural_page_table(lay, rr, offsets, vpages_per_seq=4, cap=8)
            )
            offsets_np = np.asarray(offsets)
            assert tbl.shape[0] == (max_reqs + num_bands) * 4 * dcp
            tbl_2d = tbl.reshape(max_reqs + num_bands, 4 * dcp)
            for k in range(n_loc):
                req_g = int(rids[k])
                exp_row = np.minimum(
                    offsets_np[req_g] * dcp + np.arange(4 * dcp, dtype=np.int32),
                    8 * dcp - 1,
                )
                np.testing.assert_array_equal(tbl_2d[k], exp_row)

        exp_all = (
            list(range(split_rows, total_valid)) if total_valid > split_rows else []
        )
        assert sorted(seen_global) == exp_all
        assert int(jnp.sum(pages)) >= 0


def test_should_support_exact_split_rows_equals_band_size() -> None:
    dcp, num_tokens, split_rows, max_reqs = 4, 128, 8, 8
    lay = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=8,
        zigzag_bands=4,
    )
    assert lay.effective_zigzag_bands(num_tokens) == 4
    seq_lens, qsl, n_seqs = _make_batch([99], [17], max_reqs)
    seen: list[int] = []
    for r in range(dcp):
        rr = tp.rank_rows(
            lay,
            jnp.int32(r),
            num_tokens,
            jnp.asarray(seq_lens),
            jnp.asarray(qsl),
            jnp.int32(n_seqs),
        )
        toks = np.asarray(tp.zigzag_rank_global_tokens(r, num_tokens // dcp, dcp, 4))
        tail = toks[split_rows:] if r == 0 else toks
        n_rows = int(rr.num_rows)
        seen.extend(int(t) for t in tail[:n_rows])
    assert sorted(seen) == list(range(split_rows, 99))


def test_should_match_local_shift_quant_split_3d_kernel_to_reference() -> None:
    rng = np.random.default_rng(7)
    num_heads, num_rows, split_rows, head_dim = 4, 128, 32, 128
    lead = jnp.asarray(
        rng.standard_normal((num_heads, split_rows, head_dim)), dtype=jnp.bfloat16
    )
    own = jnp.asarray(
        rng.standard_normal((num_heads, num_rows, head_dim)), dtype=jnp.bfloat16
    )
    # Zero row sentinel verifies zero-scale clamp on both rank 0 and nonzero rank.
    lead = lead.at[:, 3, :].set(jnp.bfloat16(0.0))
    own = own.at[:, 5, :].set(jnp.bfloat16(0.0))
    interpret = _interpret_flag()

    for r in range(4):
        rank_arr = jnp.asarray([r], dtype=jnp.int32)
        m_ref, p_ref, s_ref = sq3d.shift_quant_split_3d_local_ref(lead, own, rank_arr)
        m_pal, p_pal, s_pal = sq3d.shift_quant_split_3d_fp8_local(
            lead, own, rank_arr, interpret=interpret
        )
        np.testing.assert_array_equal(
            np.asarray(m_pal.view(jnp.uint8)), np.asarray(m_ref.view(jnp.uint8))
        )
        np.testing.assert_array_equal(
            np.asarray(p_pal.view(jnp.uint8)), np.asarray(p_ref.view(jnp.uint8))
        )
        np.testing.assert_array_equal(np.asarray(s_pal), np.asarray(s_ref))

    # Positive control: shifting nonzero rank as if it were rank 0 must differ.
    m_r0, _, _ = sq3d.shift_quant_split_3d_local_ref(
        lead, own, jnp.asarray([0], dtype=jnp.int32)
    )
    m_r2, _, _ = sq3d.shift_quant_split_3d_local_ref(
        lead, own, jnp.asarray([2], dtype=jnp.int32)
    )
    assert not np.array_equal(
        np.asarray(m_r0.view(jnp.uint8)), np.asarray(m_r2.view(jnp.uint8))
    )


def test_should_match_gather_qc_head_rows_and_torch_q_rows() -> None:
    dcp, num_tokens, split_rows, cols = 4, 256, 8, 16
    mesh = _make_mesh(dcp)
    rng = np.random.default_rng(19)
    global_qc = jnp.asarray(rng.standard_normal((num_tokens, cols)), dtype=jnp.bfloat16)
    packed_qc = _zigzag_pack_np(np.asarray(global_qc), dcp, num_bands=4)
    shd = NamedSharding(mesh, P(AXIS))
    qc_shd = jax.device_put(jnp.asarray(packed_qc), shd)
    ranks = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)

    out = jax.shard_map(
        lambda q, rk: tp.gather_qc_head_rows(
            q,
            rk,
            dcp_size=dcp,
            split_rows=split_rows,
            axis_name=AXIS,
            zigzag_local=True,
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(qc_shd, ranks)

    rows = num_tokens // dcp
    lay = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=8,
        zigzag_bands=4,
    )
    out_np = np.asarray(out).reshape(dcp, split_rows + rows, cols)
    global_u16 = np.asarray(global_qc.view(jnp.uint16))
    for r in range(dcp):
        stub = SimpleNamespace(layout=lay, rank=torch.tensor([r], dtype=torch.int32))
        q_idx = tp_op_mod.q_rows(stub, num_tokens).numpy()
        np.testing.assert_array_equal(
            np.asarray(jnp.asarray(out_np[r], dtype=jnp.bfloat16).view(jnp.uint16)),
            global_u16[q_idx],
        )


def test_should_roundtrip_heads_to_tokens_and_tokens_to_heads_under_zigzag() -> None:
    dcp, num_tokens, split_rows, h_local, dim = 4, 128, 8, 2, 4
    mesh = _make_mesh(dcp)
    lay = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=4,
        zigzag_bands=4,
    )
    rng = np.random.default_rng(31)
    x_global = rng.standard_normal((num_tokens, dcp * h_local, dim)).astype(np.float32)
    x_head_shd = jax.device_put(
        jnp.asarray(x_global), NamedSharding(mesh, P(None, AXIS, None))
    )

    def _roundtrip(x_h):
        tok = tp.heads_to_tokens(lay, x_h, AXIS)
        return tp.tokens_to_heads(tok, AXIS, layout=lay)

    back = jax.shard_map(
        _roundtrip,
        mesh=mesh,
        in_specs=P(None, AXIS, None),
        out_specs=P(None, AXIS, None),
        check_vma=False,
    )(x_head_shd)
    back_np = np.asarray(back)
    np.testing.assert_array_equal(
        back_np[: num_tokens - split_rows], x_global[split_rows:]
    )
    np.testing.assert_array_equal(back_np[num_tokens - split_rows :], 0.0)


@pytest.mark.parametrize("num_bands", [2, 4])
def test_should_write_identical_mla_and_indexer_kv_caches_in_zigzag_order(
    num_bands: int,
) -> None:
    dcp, rows_local, page_size, interleave = 4, 64, 32, 4
    num_tokens = dcp * rows_local
    num_pages = 24
    mesh = _make_mesh(dcp)
    rng = np.random.default_rng(42 + num_bands)

    seq_lens, qsl, _ = _make_batch([1, 19, 105, 43, 17], [15, 7, 32, 0, 61], max_reqs=8)
    # Every request gets distinct local physical pages (`req * 2 + p < num_pages`).
    bt = np.arange(8 * 3, dtype=np.int32).reshape(8, 3)

    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, num_pages, page_size, 64, 4
    )
    rope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE,
        KVCacheLayout.SPARSECORE,
        num_pages,
        page_size,
        16,
        4,
        packed_rope=True,
    )
    nope_one = np.full(nope_spec.shape, 0x5A, dtype=nope_spec.jax_dtype)
    rope_one = np.full(rope_spec.shape, 0x3C, dtype=rope_spec.jax_dtype)
    nope_init = jnp.asarray(np.concatenate([nope_one] * dcp, axis=0))
    rope_init = jnp.asarray(np.concatenate([rope_one] * dcp, axis=0))
    kv_c = (rng.standard_normal((num_tokens, 64)) * 0.5).astype(np.float32)
    k_pe = (rng.standard_normal((num_tokens, 16)) * 0.5).astype(np.float32)
    kv_c[int(qsl[-1]) :] = 7.5
    k_pe[int(qsl[-1]) :] = -7.5
    kv_c_fp8 = np.asarray(jnp.asarray(kv_c, dtype=jnp.float8_e4m3fn))
    k_pe_fp8 = np.asarray(jnp.asarray(k_pe, dtype=jnp.float8_e4m3fn))

    shd, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    ranks = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)
    sl_j = jax.device_put(jnp.asarray(seq_lens), rep)
    bt_j = jax.device_put(jnp.asarray(bt.reshape(-1)), rep)
    qsl_j = jax.device_put(jnp.asarray(qsl), rep)

    n_contig, r_contig = jax.shard_map(
        lambda nc, rc, kvc, kpe, sl, b, q, rk: (
            kv_cache_utils.update_sparse_mla_kv_cache_dcp_sharded_jax(
                nc,
                rc,
                kvc,
                kpe,
                sl,
                b,
                q,
                rk[0],
                axis_name=AXIS,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp,
                interleave_size=interleave,
                zigzag_bands=1,
            )
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(AXIS), P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=(P(AXIS), P(AXIS)),
        check_vma=False,
    )(
        jax.device_put(nope_init, shd),
        jax.device_put(rope_init, shd),
        jax.device_put(jnp.asarray(kv_c_fp8), shd),
        jax.device_put(jnp.asarray(k_pe_fp8), shd),
        sl_j,
        bt_j,
        qsl_j,
        ranks,
    )

    kv_c_zz = _zigzag_pack_np(kv_c_fp8, dcp, num_bands)
    k_pe_zz = _zigzag_pack_np(k_pe_fp8, dcp, num_bands)
    n_zz, r_zz = jax.shard_map(
        lambda nc, rc, kvc, kpe, sl, b, q, rk: (
            kv_cache_utils.update_sparse_mla_kv_cache_dcp_sharded_jax(
                nc,
                rc,
                kvc,
                kpe,
                sl,
                b,
                q,
                rk[0],
                axis_name=AXIS,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp,
                interleave_size=interleave,
                zigzag_bands=num_bands,
            )
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(AXIS), P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=(P(AXIS), P(AXIS)),
        check_vma=False,
    )(
        jax.device_put(nope_init, shd),
        jax.device_put(rope_init, shd),
        jax.device_put(jnp.asarray(kv_c_zz), shd),
        jax.device_put(jnp.asarray(k_pe_zz), shd),
        sl_j,
        bt_j,
        qsl_j,
        ranks,
    )
    np.testing.assert_array_equal(np.asarray(n_zz), np.asarray(n_contig))
    np.testing.assert_array_equal(np.asarray(r_zz), np.asarray(r_contig))

    # Positive control: writing zigzag-ordered latents with contig plan must differ.
    n_bad, _ = jax.shard_map(
        lambda nc, rc, kvc, kpe, sl, b, q, rk: (
            kv_cache_utils.update_sparse_mla_kv_cache_dcp_sharded_jax(
                nc,
                rc,
                kvc,
                kpe,
                sl,
                b,
                q,
                rk[0],
                axis_name=AXIS,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp,
                interleave_size=interleave,
                zigzag_bands=1,
            )
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(AXIS), P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=(P(AXIS), P(AXIS)),
        check_vma=False,
    )(
        jax.device_put(nope_init, shd),
        jax.device_put(rope_init, shd),
        jax.device_put(jnp.asarray(kv_c_zz), shd),
        jax.device_put(jnp.asarray(k_pe_zz), shd),
        sl_j,
        bt_j,
        qsl_j,
        ranks,
    )
    assert not np.array_equal(np.asarray(n_bad), np.asarray(n_contig))

    # Indexer sharded K-cache append (`SEQ_ALONG_LANE`) bit-exact parity.
    # Non-aliasing physical page table (`max_reqs * 2` <= `idx_num_pages`) so no
    # two requests write the same local cache page via `% idx_num_pages`.
    idx_page_size = 128
    idx_num_pages = 16
    bt_idx = np.arange(8 * 2, dtype=np.int32).reshape(8, 2)
    bt_idx_j = jax.device_put(jnp.asarray(bt_idx.reshape(-1)), rep)
    has_one = np.full((idx_num_pages, idx_page_size // 4, 4, 256), 0xA5, dtype=np.uint8)
    sal_one = np.asarray(
        st_lib.convert_cache_to_seq_along_lane(
            jnp.asarray(has_one), actual_head_dim=128
        )
    )
    idx_init = jnp.asarray(np.tile(sal_one, (dcp, 1, 1, 1)))
    k_packed = rng.integers(0, 255, size=(num_tokens, 132), dtype=np.uint8)
    k_packed_zz = _zigzag_pack_np(k_packed, dcp, num_bands)

    idx_contig = jax.shard_map(
        lambda c, kp, sl, b, q, rk: kv_cache_append.sharded_kv_append_dcp(
            c,
            kp,
            sl,
            b,
            q,
            rk[0],
            dcp_size=dcp,
            interleave_size=interleave,
            axis_name=AXIS,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            use_pallas_scatter=False,
            zigzag_bands=1,
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(
        jax.device_put(idx_init, shd),
        jax.device_put(jnp.asarray(k_packed), shd),
        sl_j,
        bt_idx_j,
        qsl_j,
        ranks,
    )

    idx_zz = jax.shard_map(
        lambda c, kp, sl, b, q, rk: kv_cache_append.sharded_kv_append_dcp(
            c,
            kp,
            sl,
            b,
            q,
            rk[0],
            dcp_size=dcp,
            interleave_size=interleave,
            axis_name=AXIS,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            use_pallas_scatter=False,
            zigzag_bands=num_bands,
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(
        jax.device_put(idx_init, shd),
        jax.device_put(jnp.asarray(k_packed_zz), shd),
        sl_j,
        bt_idx_j,
        qsl_j,
        ranks,
    )
    np.testing.assert_array_equal(np.asarray(idx_zz), np.asarray(idx_contig))

    # Positive control: writing zigzag rows with contig plan corrupts cache.
    idx_bad = jax.shard_map(
        lambda c, kp, sl, b, q, rk: kv_cache_append.sharded_kv_append_dcp(
            c,
            kp,
            sl,
            b,
            q,
            rk[0],
            dcp_size=dcp,
            interleave_size=interleave,
            axis_name=AXIS,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            use_pallas_scatter=False,
            zigzag_bands=1,
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(
        jax.device_put(idx_init, shd),
        jax.device_put(jnp.asarray(k_packed_zz), shd),
        sl_j,
        bt_idx_j,
        qsl_j,
        ranks,
    )
    assert not np.array_equal(np.asarray(idx_bad), np.asarray(idx_contig))


def test_should_match_end_to_end_contig_and_zigzag_sequence_rows_with_zero_diff() -> (
    None
):
    dcp, num_tokens, split_rows, head_dim = 4, 256, 8, 128
    num_heads = 4
    rows = num_tokens // dcp
    mesh = _make_mesh(dcp)
    rng = np.random.default_rng(77)

    full_global = rng.standard_normal((num_heads, num_tokens, head_dim)).astype(
        np.float32
    )
    full_bf16 = np.asarray(jnp.asarray(full_global, dtype=jnp.bfloat16))
    lead_bf16 = full_bf16[:, :split_rows, :]

    # Contig layout: own rows on rank r hold global [split_rows + r*R : ...].
    own_contig = np.zeros((dcp, num_heads, rows, head_dim), dtype=full_bf16.dtype)
    for r in range(dcp):
        lo = split_rows + r * rows
        hi = min(lo + rows, num_tokens)
        own_contig[r, :, : hi - lo, :] = full_bf16[:, lo:hi, :]
    own_contig_shd = own_contig.transpose(0, 2, 1, 3).reshape(
        num_tokens, num_heads * head_dim
    )

    # Dual-zigzag: nonzero ranks hold all R zigzag rows; rank 0 holds [s:R] + pad.
    packed_t = _zigzag_pack_np(full_bf16.transpose(1, 0, 2), dcp, num_bands=4)
    own_zz = np.zeros((dcp, rows, num_heads, head_dim), dtype=full_bf16.dtype)
    for r in range(dcp):
        chunk = packed_t[r * rows : (r + 1) * rows]
        if r == 0:
            own_zz[r, : rows - split_rows] = chunk[split_rows:]
        else:
            own_zz[r] = chunk
    own_zz_shd = own_zz.reshape(num_tokens, num_heads * head_dim)

    lay_c = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=4,
        zigzag_bands=1,
    )
    lay_z = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=4,
        zigzag_bands=4,
    )
    shd, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    lead_2d = jax.device_put(
        jnp.asarray(lead_bf16.transpose(1, 0, 2).reshape(split_rows, -1)), rep
    )
    ranks = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)

    seq_c = jax.shard_map(
        lambda own_r, lead_r, rk: tp.token_rows_to_sequence_rows(
            lay_c, own_r, lead_r, rk[0], AXIS
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(jax.device_put(jnp.asarray(own_contig_shd), shd), lead_2d, ranks)

    seq_z = jax.shard_map(
        lambda own_r, lead_r, rk: tp.token_rows_to_sequence_rows(
            lay_z, own_r, lead_r, rk[0], AXIS
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(jax.device_put(jnp.asarray(own_zz_shd), shd), lead_2d, ranks)

    unzig_z = _zigzag_unpack_np(np.asarray(seq_z), dcp, num_bands=4)
    np.testing.assert_array_equal(
        np.asarray(jnp.asarray(unzig_z, dtype=jnp.bfloat16).view(jnp.uint16)),
        np.asarray(jnp.asarray(seq_c, dtype=jnp.bfloat16).view(jnp.uint16)),
    )


def test_should_roundtrip_torch_moe_prefill_sp_own_rows_zigzag() -> None:
    tp_size, num_bands, num_tokens, hidden = 4, 4, 128, 12
    x = torch.arange(num_tokens * hidden, dtype=torch.int32).reshape(num_tokens, hidden)
    shards = [
        sp.own_rows_zigzag(
            x, torch.tensor([[r]], dtype=torch.int32), tp_size, num_bands
        )
        for r in range(tp_size)
    ]
    gathered = torch.cat(shards, dim=0)
    expected = torch.from_numpy(_zigzag_pack_np(x.numpy(), tp_size, num_bands))
    torch.testing.assert_close(gathered, expected)

    # Inverse recovers exact original global order on both 1D and 2D tensors.
    back = torch.from_numpy(_zigzag_unpack_np(gathered.numpy(), tp_size, num_bands))
    torch.testing.assert_close(back, x)

    mask = torch.arange(num_tokens) >= 73
    mask_shards = [
        sp.own_rows_zigzag(
            mask, torch.tensor([[r]], dtype=torch.int32), tp_size, num_bands
        )
        for r in range(tp_size)
    ]
    back_mask = torch.from_numpy(
        _zigzag_unpack_np(torch.cat(mask_shards, dim=0).numpy(), tp_size, num_bands)
    )
    torch.testing.assert_close(back_mask, mask)


def test_should_match_unified_indexer_and_nonidx_mla_oproj_zigzag_vs_contig() -> None:
    """4-device 2-layer CPU step: indexer (unified_qb) -> flat_topk -> non-indexer."""
    dcp, num_tokens, split_rows, page_size, il = 4, 256, 8, 16, 4
    h_heads, q_lora, d_n, d_r, d_v, k_top, d_idx, hidden = 4, 16, 8, 4, 8, 6, 4, 12
    num_local_pages, max_reqs = 24, 6
    mesh = _make_mesh(dcp)
    rng = np.random.default_rng(2026)

    lay_c = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=il,
        page_size=page_size,
        split_rows=split_rows,
        max_pages=10,
        gather_caps=(4, 10),
        zigzag_bands=1,
    )
    lay_z = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=il,
        page_size=page_size,
        split_rows=split_rows,
        max_pages=10,
        gather_caps=(4, 10),
        zigzag_bands=4,
    )

    # Mix decode + short/long prefill + partial tail (total 219 < 256).
    q_lens = [1, 1, 125, 61, 31]
    prefixes = [30, 80, 140, 20, 45]
    seq_lens_np, qsl_np, n_seqs = _make_batch(q_lens, prefixes, max_reqs)
    dist_np = np.array([0, 0, n_seqs], dtype=np.int32)
    vpages_per_seq = 6
    idx_bt_np = rng.permutation(num_local_pages * dcp)[
        : max_reqs * vpages_per_seq
    ].astype(np.int32)
    mla_bt_np = rng.permutation(num_local_pages * dcp)[
        : max_reqs * vpages_per_seq
    ].astype(np.int32)

    idx_cache_np = rng.integers(
        1, 200, size=(dcp * num_local_pages, d_idx, 4, page_size), dtype=np.uint8
    )
    nope_cache_np = (
        rng.standard_normal((dcp * num_local_pages, page_size, d_n)) * 0.25
    ).astype(np.float32)
    rope_cache_np = (
        rng.standard_normal((dcp * num_local_pages, page_size, d_r)) * 0.25
    ).astype(np.float32)
    # Global step tensors (replicated inputs sliced to contig vs zigzag per rank).
    qc_global = np.asarray(
        jnp.asarray(rng.standard_normal((num_tokens, q_lora)) * 0.5, dtype=jnp.bfloat16)
    )
    cs_global = np.asarray(
        jnp.asarray(rng.standard_normal((num_tokens, d_r)) * 0.5, dtype=jnp.bfloat16)
    )
    q_idx_global = rng.standard_normal((num_tokens, 2, d_idx)).astype(np.float32)
    w_idx_global = rng.standard_normal((num_tokens, 2)).astype(np.float32)
    w_qb_np = np.asarray(
        jnp.asarray(
            rng.standard_normal((q_lora, h_heads, d_n + d_r)) * 0.2, dtype=jnp.bfloat16
        )
    )
    w_uv_np = np.asarray(
        jnp.asarray(rng.standard_normal((h_heads, d_n, d_v)) * 0.2, dtype=jnp.bfloat16)
    )
    w_o_pair = np.asarray(
        jnp.asarray(
            rng.standard_normal((2, (h_heads * d_v) // 2, hidden)) * 0.2,
            dtype=jnp.float8_e4m3fn,
        )
    )
    # Replicate the pair halves `[h0, h1]` across `(r=0,1)` and `(r=2,3)`.
    w_o_np = np.tile(w_o_pair, (dcp // 2, 1, 1)).reshape(
        dcp * ((h_heads * d_v) // 2), hidden
    )
    s_o_np = np.tile(np.ones((1, hidden), dtype=np.float32) * 0.5, (dcp, 1))
    head_topk_np = rng.integers(0, 16, size=(dcp * split_rows, k_top), dtype=np.int32)

    def _ref_fused_qb(qc, w_qb, _s_qb, _w_uk, cs, s):
        proj = jnp.einsum(
            "rq,qhd->rhd", qc.astype(jnp.float32), w_qb.astype(jnp.float32)
        )
        ql = proj[..., :d_n].astype(jnp.bfloat16)
        qpe = (proj[..., d_n:] + cs[:, None, :].astype(jnp.float32) * 0.1).astype(
            jnp.bfloat16
        )
        return ql[:s], ql[s:], qpe[:s], qpe[s:]

    def _ref_select_topk(q_r, w_r, nat, rr, _table):
        n_rows = q_r.shape[0]
        i = jnp.arange(n_rows, dtype=jnp.int32)
        loc = jnp.minimum(
            jnp.searchsorted(rr.cu_q_lens[1:], i, side="right"),
            jnp.maximum(rr.num_seqs - 1, 0),
        )
        q_len_req = rr.cu_q_lens[loc + 1] - rr.cu_q_lens[loc]
        pos_max = rr.seq_lens[loc] - q_len_req + (i - rr.cu_q_lens[loc])
        # Deterministic top-k strictly from the row's causal position & query value.
        key = jnp.sum(q_r[:, 0, :], axis=-1) + w_r[:, 0]
        offset = jnp.floor(jnp.abs(key) * 3.0).astype(jnp.int32)
        picks = (
            pos_max[:, None]
            - jnp.arange(k_top, dtype=jnp.int32)[None, :]
            - (offset[:, None] % 3)
        )
        valid = (i[:, None] < rr.num_rows) & (picks >= 0)
        return jnp.where(valid, picks, jnp.int32(-1))

    def _ref_lead(ql_h, qpe_h, nope, rope, h_tk, bt, sl, qsl, dist):
        del rope, h_tk, bt, sl, qsl, dist
        scale = jnp.mean(nope.astype(jnp.float32)) * 0.01
        part = (
            ql_h.astype(jnp.float32)
            + jnp.sum(qpe_h.astype(jnp.float32), axis=-1, keepdims=True) * 0.05
            + scale
        )
        lse = jnp.ones((split_rows, h_heads), dtype=jnp.float32) * 0.3
        return part.astype(jnp.float32), lse

    def _ref_tail(ql_t, g_nope, g_rope, flat, num_rows, qpe_t):
        n_rows = ql_t.shape[0]
        fn = g_nope.reshape(-1, d_n).astype(jnp.float32)
        fr = g_rope.reshape(-1, d_r).astype(jnp.float32)
        safe = jnp.maximum(flat, 0)
        kn, kr = fn[safe], fr[safe]
        logits = jnp.einsum("rhd,rkd->rhk", ql_t.astype(jnp.float32), kn) + jnp.einsum(
            "rhd,rkd->rhk", qpe_t.astype(jnp.float32), kr
        )
        valid = (flat >= 0)[:, None, :] & (jnp.arange(n_rows)[:, None, None] < num_rows)
        w = jnp.where(
            valid, jax.nn.softmax(jnp.where(valid, logits, -1e9), axis=-1), 0.0
        )
        return jnp.einsum("rhk,rkd->rhd", w, kn).astype(ql_t.dtype)

    shd, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    ranks = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)
    idx_cache = jax.device_put(jnp.asarray(idx_cache_np), shd)
    nope_cache = jax.device_put(jnp.asarray(nope_cache_np, dtype=jnp.bfloat16), shd)
    rope_cache = jax.device_put(jnp.asarray(rope_cache_np, dtype=jnp.bfloat16), shd)
    sl_j = jax.device_put(jnp.asarray(seq_lens_np), rep)
    qsl_j = jax.device_put(jnp.asarray(qsl_np), rep)
    dist_j = jax.device_put(jnp.asarray(dist_np), rep)
    idx_bt_j = jax.device_put(jnp.asarray(idx_bt_np), rep)
    mla_bt_j = jax.device_put(jnp.asarray(mla_bt_np), rep)
    w_qb_j = jax.device_put(jnp.asarray(w_qb_np), rep)
    s_qb_j = jax.device_put(jnp.ones((1,), dtype=jnp.float32), rep)
    w_uk_j = jax.device_put(jnp.ones((1,), dtype=jnp.bfloat16), rep)
    w_uv_j = jax.device_put(jnp.asarray(w_uv_np), rep)
    w_o_j = jax.device_put(jnp.asarray(w_o_np), shd)
    s_o_j = jax.device_put(jnp.asarray(s_o_np), shd)
    head_topk_j = jax.device_put(jnp.asarray(head_topk_np), shd)

    def _run_two_layers(
        layout: tp.TokenParallelLayout,
    ) -> tuple[np.ndarray, np.ndarray]:
        nb = layout.zigzag_bands
        if nb > 1:
            qc_shd_raw = _zigzag_pack_np(qc_global, dcp, nb)
            cs_shd_raw = _zigzag_pack_np(cs_global, dcp, nb)
        else:
            qc_shd_raw = qc_global
            cs_shd_raw = cs_global
        qc_qrows = jax.shard_map(
            lambda q, rk: tp.gather_qc_head_rows(
                q,
                rk,
                dcp_size=dcp,
                split_rows=split_rows,
                axis_name=AXIS,
                zigzag_local=nb > 1,
            ),
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )(jax.device_put(jnp.asarray(qc_shd_raw), shd), ranks)
        cs_qrows = jax.shard_map(
            lambda q, rk: tp.gather_qc_head_rows(
                q,
                rk,
                dcp_size=dcp,
                split_rows=split_rows,
                axis_name=AXIS,
                zigzag_local=nb > 1,
            ),
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )(jax.device_put(jnp.asarray(cs_shd_raw), shd), ranks)

        q_idx_tail = jax.shard_map(
            lambda x, rk: tp.own_rows(layout, x, rk[0]),
            mesh=mesh,
            in_specs=(P(), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )(jax.device_put(jnp.asarray(q_idx_global), rep), ranks)
        w_idx_tail = jax.shard_map(
            lambda x, rk: tp.own_rows(layout, x, rk[0]),
            mesh=mesh,
            in_specs=(P(), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )(jax.device_put(jnp.asarray(w_idx_global), rep), ranks)

        out_idx, flat_topk = tp.unified_indexer_in_branch_qb_mla_oproj(
            layout,
            mesh,
            AXIS,
            idx_cache,
            q_idx_tail,
            w_idx_tail,
            idx_bt_j,
            nope_cache,
            rope_cache,
            qc_qrows,
            w_qb_j,
            s_qb_j,
            w_uk_j,
            cs_qrows,
            head_topk_j,
            mla_bt_j,
            sl_j,
            qsl_j,
            dist_j,
            select_topk=_ref_select_topk,
            fused_qb=_ref_fused_qb,
            lead_attend=_ref_lead,
            tail_attend=_ref_tail,
            w_uv=w_uv_j,
            w_o=w_o_j,
            w_o_scale=s_o_j,
            num_tokens=num_tokens,
            w_uv_replicated=True,
            rank_ids=ranks,
            epilogue_3d=False,
        )
        out_nonidx = tp.sparse_mla_attention_with_in_branch_qb_oproj(
            layout,
            mesh,
            AXIS,
            nope_cache,
            rope_cache,
            qc_qrows,
            w_qb_j,
            s_qb_j,
            w_uk_j,
            cs_qrows,
            head_topk_j,
            flat_topk,
            sl_j,
            mla_bt_j,
            qsl_j,
            dist_j,
            fused_qb=_ref_fused_qb,
            lead_attend=_ref_lead,
            tail_attend=_ref_tail,
            w_uv=w_uv_j,
            w_o=w_o_j,
            w_o_scale=s_o_j,
            num_tokens=num_tokens,
            flat_topk_cap_strided=True,
            w_uv_replicated=True,
            rank_ids=ranks,
            epilogue_3d=False,
        )
        y1 = np.asarray(out_idx)
        y2 = np.asarray(out_nonidx)
        if nb > 1:
            y1 = _zigzag_unpack_np(y1, dcp, nb)
            y2 = _zigzag_unpack_np(y2, dcp, nb)
        return y1, y2

    y1_c, y2_c = _run_two_layers(lay_c)
    y1_z, y2_z = _run_two_layers(lay_z)
    valid = int(qsl_np[n_seqs])
    np.testing.assert_array_equal(
        np.asarray(jnp.asarray(y1_z[:valid], dtype=jnp.bfloat16).view(jnp.uint16)),
        np.asarray(jnp.asarray(y1_c[:valid], dtype=jnp.bfloat16).view(jnp.uint16)),
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.asarray(y2_z[:valid], dtype=jnp.bfloat16).view(jnp.uint16)),
        np.asarray(jnp.asarray(y2_c[:valid], dtype=jnp.bfloat16).view(jnp.uint16)),
    )


def _duplicate_granule_runs(dst_rows: np.ndarray, num_rows: int, granule: int) -> int:
    seen: set[int] = set()
    dups = 0
    next_row = -1
    for raw in dst_rows.tolist():
        if raw < 0 or raw >= num_rows:
            continue
        if raw != next_row or (raw % granule) == 0:
            gid = raw // granule
            if gid in seen:
                dups += 1
            seen.add(gid)
        next_row = raw + 1
    return dups


def test_should_eliminate_duplicate_granule_runs_when_zigzag_recv_is_sorted() -> None:
    dcp, rows_local, page_size, num_pages, interleave = 4, 128, 256, 8, 4
    num_tokens = dcp * rows_local
    num_bands = 4
    mesh = _make_mesh(dcp)
    ranks = jax.device_put(
        jnp.arange(dcp, dtype=jnp.int32), NamedSharding(mesh, P(AXIS))
    )
    num_seqs = 3
    qsl_np = np.array([0, 320, 412, 495], dtype=np.int32)
    sl_np = np.array([320 + 64, 92 + 512, 83 + 12], dtype=np.int32)
    bt_np = np.array(
        [
            [5, 2, 11, 0],
            [7, 1, 9, 14],
            [3, 12, 6, 4],
        ],
        dtype=np.int32,
    )
    cap = idx_kvu.sharded_a2a_cap(
        rows_local,
        num_seqs=num_seqs,
        dcp_size=dcp,
        interleave_size=interleave,
        zigzag_bands=num_bands,
    )
    recv_dst = jax.shard_map(
        lambda sl, bt, qsl, rk: idx_kvu.build_sharded_dcp_a2a_plan(
            rows_local,
            sl,
            bt,
            qsl,
            rk[0],
            page_size=page_size,
            num_pages=num_pages,
            dcp_size=dcp,
            interleave_size=interleave,
            cap=cap,
            axis_name=AXIS,
            zigzag_bands=num_bands,
        )[1].reshape(1, -1),
        mesh=mesh,
        in_specs=(P(), P(), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(
        jnp.asarray(sl_np),
        jnp.asarray(bt_np),
        jnp.asarray(qsl_np),
        ranks,
    )
    num_rows = num_pages * page_size
    recv_dst_np = np.asarray(recv_dst)
    # Positive control: raw zigzag_bands=4 receive order splits 128-lane
    # SEQ_ALONG_LANE granules across multiple runs on rank 0.
    raw_dups = sum(
        _duplicate_granule_runs(recv_dst_np[r], num_rows, 128) for r in range(dcp)
    )
    assert raw_dups > 0

    max_valid = kv_cache_append.owned_rows_bound(
        num_tokens,
        num_seqs=num_seqs,
        dcp_size=dcp,
        interleave_size=interleave,
    )
    for r in range(dcp):
        row_ids = jnp.arange(recv_dst_np.shape[1], dtype=jnp.uint32)[:, None]
        sorted_vals, sorted_dst = kv_cache_append.sort_recv_by_dst_row(
            row_ids,
            jnp.asarray(recv_dst_np[r]),
            num_rows=num_rows,
            max_valid_rows=max_valid,
        )
        sd_np = np.asarray(sorted_dst)
        assert _duplicate_granule_runs(sd_np, num_rows, 128) == 0
        assert _duplicate_granule_runs(sd_np, num_rows, 4) == 0
        expected_valid = sorted(
            int(v) for v in recv_dst_np[r].tolist() if 0 <= v < num_rows
        )
        actual_valid = [int(v) for v in sd_np.tolist() if v >= 0]
        assert actual_valid == expected_valid
        picked = np.asarray(sorted_vals)[:, 0].astype(np.int64)
        for idx, dst in enumerate(sd_np.tolist()):
            if dst >= 0:
                assert int(recv_dst_np[r, picked[idx]]) == dst

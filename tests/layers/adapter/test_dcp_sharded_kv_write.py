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
"""Specs for sequence-sharded DCP KV prep and owner-routed all_to_all writes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla import dcp_token_parallel, kv_cache_utils
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op


def _make_16dev_mesh() -> Mesh:
    devs = jax.devices()
    if len(devs) < 16:
        pytest.skip(f"needs 16 JAX devices, got {len(devs)}")
    return Mesh(np.array(devs[:16]), ("dcp",))


def _sharded_array(arr: np.ndarray, mesh: Mesh) -> jax.Array:
    return jax.device_put(jnp.asarray(arr), NamedSharding(mesh, P("dcp")))


def _replicated_array(arr: np.ndarray, mesh: Mesh) -> jax.Array:
    return jax.device_put(jnp.asarray(arr), NamedSharding(mesh, P()))


def _run_sharded_vs_replicated_kv_write(
    *,
    mesh: Mesh,
    dcp: int,
    rows_local: int,
    seq_lens: np.ndarray,
    block_tables: np.ndarray,
    query_start_loc: np.ndarray,
    num_pages: int = 32,
    page_size: int = 32,
    interleave: int = 4,
    layout: kv_cache_utils.KVCacheLayout = kv_cache_utils.KVCacheLayout.SPARSECORE,
) -> None:
    num_tokens = dcp * rows_local
    kv_dim = 512 if layout is kv_cache_utils.KVCacheLayout.TENSORCORE else 64
    rope_dim = 64 if layout is kv_cache_utils.KVCacheLayout.TENSORCORE else 16
    nope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        layout,
        num_pages,
        page_size,
        kv_dim,
        4,
    )
    rope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        layout,
        num_pages,
        page_size,
        rope_dim,
        4,
        packed_rope=layout is kv_cache_utils.KVCacheLayout.SPARSECORE,
    )

    rng = np.random.default_rng(1234 + rows_local + int(seq_lens[0]))
    kv_c_np = (rng.standard_normal((num_tokens, kv_dim)) * 0.5).astype(np.float32)
    k_pe_np = (rng.standard_normal((num_tokens, rope_dim)) * 0.5).astype(np.float32)
    # Anti-cancellation sentinel on padded tail rows so out-of-batch writes fail loudly.
    live_tokens = int(query_start_loc[-1])
    if live_tokens < num_tokens:
        kv_c_np[live_tokens:] = 7.5
        k_pe_np[live_tokens:] = -7.5

    kv_c_f8 = jnp.asarray(kv_c_np, dtype=jnp.float8_e4m3fn)
    k_pe_f8 = jnp.asarray(k_pe_np, dtype=jnp.float8_e4m3fn)

    # Non-zero sentinel initial caches verify untouched pages/slots stay intact.
    init_nope_one = np.full(nope_spec.shape, 0x5A, dtype=nope_spec.jax_dtype)
    init_rope_one = np.full(rope_spec.shape, 0x3C, dtype=rope_spec.jax_dtype)
    init_nope = _sharded_array(np.concatenate([init_nope_one] * dcp, axis=0), mesh)
    init_rope = _sharded_array(np.concatenate([init_rope_one] * dcp, axis=0), mesh)

    sl_dev = _replicated_array(seq_lens, mesh)
    bt_dev = _replicated_array(block_tables, mesh)
    qsl_dev = _replicated_array(query_start_loc, mesh)
    ranks_dev = _sharded_array(np.arange(dcp, dtype=np.int32), mesh)

    kv_c_rep = jax.device_put(kv_c_f8, NamedSharding(mesh, P()))
    k_pe_rep = jax.device_put(k_pe_f8, NamedSharding(mesh, P()))
    kv_c_shd = jax.device_put(kv_c_f8, NamedSharding(mesh, P("dcp")))
    k_pe_shd = jax.device_put(k_pe_f8, NamedSharding(mesh, P("dcp")))

    def _ref_fn(nc, rc, kc, kp, sl, bt, qsl, rk):
        return kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
            nc,
            rc,
            kc,
            kp,
            sl,
            bt,
            qsl,
            rk[0],
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp,
            interleave_size=interleave,
        )

    def _got_fn(nc, rc, kc, kp, sl, bt, qsl, rk):
        return kv_cache_utils.update_sparse_mla_kv_cache_dcp_sharded_jax(
            nc,
            rc,
            kc,
            kp,
            sl,
            bt,
            qsl,
            rk[0],
            axis_name="dcp",
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp,
            interleave_size=interleave,
        )

    shd, rp = P("dcp"), P()
    want_n, want_r = jax.jit(
        jax.shard_map(
            _ref_fn,
            mesh=mesh,
            in_specs=(shd, shd, rp, rp, rp, rp, rp, shd),
            out_specs=(shd, shd),
            check_vma=False,
        )
    )(init_nope, init_rope, kv_c_rep, k_pe_rep, sl_dev, bt_dev, qsl_dev, ranks_dev)

    got_n, got_r = jax.jit(
        jax.shard_map(
            _got_fn,
            mesh=mesh,
            in_specs=(shd, shd, shd, shd, rp, rp, rp, shd),
            out_specs=(shd, shd),
            check_vma=False,
        )
    )(init_nope, init_rope, kv_c_shd, k_pe_shd, sl_dev, bt_dev, qsl_dev, ranks_dev)

    np.testing.assert_array_equal(np.asarray(got_n), np.asarray(want_n))
    np.testing.assert_array_equal(np.asarray(got_r), np.asarray(want_r))


def test_should_match_replicated_dcp_kv_write_on_chunked_prefill_continuation():
    mesh = _make_16dev_mesh()
    dcp, rows_local, max_seqs, pages_per_seq = 16, 64, 8, 8
    num_tokens = dcp * rows_local
    qsl = np.zeros((max_seqs + 1,), dtype=np.int32)
    qsl[1:] = num_tokens
    sl = np.zeros((max_seqs,), dtype=np.int32)
    # Non-zero prefix offset (e.g. chunked prefill or prefix-cache continuation).
    sl[0] = 512 + num_tokens
    bt = np.arange(max_seqs * pages_per_seq, dtype=np.int32) % 32
    _run_sharded_vs_replicated_kv_write(
        mesh=mesh,
        dcp=dcp,
        rows_local=rows_local,
        seq_lens=sl,
        block_tables=bt,
        query_start_loc=qsl,
    )


def test_should_match_replicated_dcp_kv_write_on_mixed_padded_batch():
    mesh = _make_16dev_mesh()
    dcp, rows_local, max_seqs, pages_per_seq = 16, 128, 12, 8
    # 4 decode (1 token each) + 3 prefill chunks + trailing padded tokens (< 2048).
    lengths = [1, 1, 1, 1, 317, 509, 777]
    offsets = [91, 17, 255, 64, 128, 3, 400]
    qsl = np.zeros((max_seqs + 1,), dtype=np.int32)
    sl = np.zeros((max_seqs,), dtype=np.int32)
    cur = 0
    for i, (ln, off) in enumerate(zip(lengths, offsets)):
        qsl[i] = cur
        cur += ln
        sl[i] = off + ln
    qsl[len(lengths) :] = cur
    bt = np.arange(max_seqs * pages_per_seq, dtype=np.int32) % 32
    _run_sharded_vs_replicated_kv_write(
        mesh=mesh,
        dcp=dcp,
        rows_local=rows_local,
        seq_lens=sl,
        block_tables=bt,
        query_start_loc=qsl,
    )


@pytest.mark.parametrize("live_seqs", [17, 18, 64])
def test_should_match_replicated_dcp_kv_write_across_fast_and_supremum_tiers(
    live_seqs: int,
):
    mesh = _make_16dev_mesh()
    # rows_local=1024, max_seqs=64 hits the exact 16-way DCP tier boundary:
    # live_seqs=17 -> fast tier (cap=128); live_seqs=18 & 64 -> supremum tier (cap=304).
    dcp, rows_local, max_seqs, pages_per_seq = 16, 1024, 64, 4
    num_tokens = dcp * rows_local
    qsl = np.zeros((max_seqs + 1,), dtype=np.int32)
    sl = np.zeros((max_seqs,), dtype=np.int32)
    cur = 0
    # Pack (live_seqs - 1) length-4 sequences at pos=0..3 (all owner 0) plus remainder.
    for s in range(live_seqs - 1):
        qsl[s] = cur
        sl[s] = 4
        cur += 4
    qsl[live_seqs - 1] = cur
    qsl[live_seqs:] = num_tokens
    sl[live_seqs - 1] = num_tokens - cur

    bt_2d = np.zeros((max_seqs, pages_per_seq), dtype=np.int32)
    p = 0
    for s in range(live_seqs):
        for k in range(pages_per_seq):
            bt_2d[s, k] = p % 128
            p += 1
    _run_sharded_vs_replicated_kv_write(
        mesh=mesh,
        dcp=dcp,
        rows_local=rows_local,
        seq_lens=sl,
        block_tables=bt_2d.reshape(-1),
        query_start_loc=qsl,
        num_pages=128,
        page_size=256,
    )


def test_should_match_replicated_dcp_kv_write_on_tensorcore_layout():
    mesh = _make_16dev_mesh()
    dcp, rows_local, max_seqs, pages_per_seq = 16, 32, 4, 8
    num_tokens = dcp * rows_local
    qsl = np.array([0, 200, num_tokens, num_tokens, num_tokens], dtype=np.int32)
    sl = np.array([212, num_tokens - 200 + 64, 0, 0], dtype=np.int32)
    bt = np.arange(max_seqs * pages_per_seq, dtype=np.int32) % 32
    _run_sharded_vs_replicated_kv_write(
        mesh=mesh,
        dcp=dcp,
        rows_local=rows_local,
        seq_lens=sl,
        block_tables=bt,
        query_start_loc=qsl,
        layout=kv_cache_utils.KVCacheLayout.TENSORCORE,
    )


def test_should_pass_through_when_dcp_size_is_one():
    mesh = Mesh(np.array(jax.devices()[:1]), ("dcp",))
    qsl = np.array([0, 32, 32], dtype=np.int32)
    sl = np.array([48, 0], dtype=np.int32)
    bt = np.arange(2 * 4, dtype=np.int32) % 8
    _run_sharded_vs_replicated_kv_write(
        mesh=mesh,
        dcp=1,
        rows_local=32,
        seq_lens=sl,
        block_tables=bt,
        query_start_loc=qsl,
        num_pages=8,
        page_size=32,
    )


@pytest.mark.parametrize("with_indexer", [False, True])
def test_should_omit_kv_lora_from_all_gather_when_sharded_kv_armed(
    with_indexer: bool,
):
    tp, num_tokens, s = 4, 32, 4
    rows_local = num_tokens // tp
    hidden, q_lora, kv_dim, kw_dim = 16, 12, 10, 6
    cols_qc = (s * q_lora) // rows_local

    torch.manual_seed(7)
    full_x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    w_qkv = torch.randn(hidden, q_lora + kv_dim, dtype=torch.bfloat16)
    w_kw = torch.randn(hidden, kw_dim, dtype=torch.bfloat16)
    full_qkv = full_x @ w_qkv
    full_qc, full_kv = full_qkv.split([q_lora, kv_dim], dim=-1)
    full_kw = full_x @ w_kw
    positions = torch.arange(100, 100 + num_tokens, dtype=torch.int32)

    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=tp,
        split_rows=s,
        page_size=16,
        interleave_size=4,
        max_pages=4,
    )

    per_rank_gathered = []
    for r in range(tp):
        lo, hi = r * rows_local, (r + 1) * rows_local
        head_packed = full_qc[lo : lo + s].reshape(rows_local, cols_qc)
        parts = [head_packed, full_kw[lo:hi]] if with_indexer else [head_packed]
        per_rank_gathered.append(torch.cat(parts, dim=-1))
    expected_ag = torch.cat(per_rank_gathered, dim=0)

    for r in range(tp):
        lo, hi = r * rows_local, (r + 1) * rows_local
        rank_t = torch.tensor([r], dtype=torch.int32)
        tp_op = SimpleNamespace(
            layout=layout,
            q_replicated=True,
            rank=rank_t,
            serves_sharded_kv=True,
        )
        indexer = (
            SimpleNamespace(
                wk_weights_proj=lambda z: (z @ w_kw, None),
                indexer_op=SimpleNamespace(topk_op_tp=SimpleNamespace(q_own_rows=True)),
            )
            if with_indexer
            else None
        )
        mla = SimpleNamespace(
            fused_qkv_a_proj=lambda z: (z @ w_qkv, None),
            indexer=indexer,
            skip_topk=not with_indexer,
            is_sparse=True,
            g_proj=None,
            o_proj=SimpleNamespace(),
            mla_attn=SimpleNamespace(sparse_mla_tp_op=tp_op),
            gate_is_fused=False,
            kimi_preprocess=None,
            q_lora_rank=q_lora,
        )
        seen_ag_cols: list[int] = []

        def _fake_ag(
            tensor: torch.Tensor, dim: int, _seen: list[int] = seen_ag_cols
        ) -> torch.Tensor:
            assert dim == 0
            _seen.append(tensor.shape[-1])
            return expected_ag

        pos_loc = positions[lo:hi]
        group = sp._ShardedKVGatherProjections(positions, tp)  # noqa: SLF001 -- exercises per-step gatherer contract directly
        with (
            patch.object(envs, "TPU_DCP_TP_SHARDED_KV_WRITE", True),
            patch.object(envs, "TPU_DCP_TP_PREFILL_LOCAL_OPROJ", True),
            patch.object(tp_op_mod, "applies", return_value=True),
            patch.object(tp_op_mod, "serves_sequence_row_oproj", return_value=True),
            patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
            patch.object(
                sp.fused_moe_ep, "rank_operand", return_value=rank_t.view(1, 1)
            ),
            patch.object(sp, "tensor_model_parallel_all_gather", side_effect=_fake_ag),
        ):
            pre_qkv, pre_kw = group.project_gather(mla, full_x[lo:hi], num_tokens)

        expected_cols = cols_qc + (kw_dim if with_indexer else 0)
        assert seen_ag_cols == [expected_cols]
        assert isinstance(pre_qkv, tuple) and len(pre_qkv) == 3
        got_qc, got_kv_local, got_pos_local = pre_qkv
        q_idx = tp_op_mod.q_rows(tp_op, num_tokens)
        torch.testing.assert_close(
            got_qc, full_qc.index_select(0, q_idx), rtol=0, atol=0
        )
        torch.testing.assert_close(got_kv_local, full_kv[lo:hi], rtol=0, atol=0)
        torch.testing.assert_close(got_pos_local, pos_loc, rtol=0, atol=0)
        if with_indexer:
            assert pre_kw is not None
            torch.testing.assert_close(pre_kw, full_kw, rtol=0, atol=0)
        else:
            assert pre_kw is None


def test_should_preserve_empty_cache_probe_rows_when_kv_sharded():
    impl = object.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = 16
    layer = SimpleNamespace(num_heads=4, v_head_dim=256)
    kv_local = torch.randn(64, 512, dtype=torch.bfloat16)
    k_pe_local = torch.randn(64, 1, 64, dtype=torch.bfloat16)
    q_dummy = (
        torch.randn(80, 64, 192, dtype=torch.bfloat16),
        torch.randn(80, 64, 64, dtype=torch.bfloat16),
    )
    w_o = (torch.empty(512, 6144, dtype=torch.bfloat16), torch.empty(6144))

    out_sharded = impl.forward(
        layer=layer,
        q=q_dummy,
        kv_c_normed=kv_local,
        k_pe=k_pe_local,
        kv_cache=torch.empty(0),
        attn_metadata=None,
        sp_o_proj=w_o,
        kv_sharded=True,
    )
    assert tuple(out_sharded.shape) == (64, 6144)

    out_replicated = impl.forward(
        layer=layer,
        q=q_dummy,
        kv_c_normed=torch.randn(1024, 512, dtype=torch.bfloat16),
        k_pe=torch.randn(1024, 1, 64, dtype=torch.bfloat16),
        kv_cache=torch.empty(0),
        attn_metadata=None,
        sp_o_proj=w_o,
        kv_sharded=False,
    )
    assert tuple(out_replicated.shape) == (64, 6144)


def test_should_allow_none_kv_latents_in_sparse_mla_attention_when_write_kv_is_none():
    mesh = Mesh(np.array(jax.devices()[:4]), ("dcp",))
    dcp, num_tokens, split_rows, heads, lkv, rope = 4, 32, 4, 4, 16, 8
    rows = num_tokens // dcp
    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        split_rows=split_rows,
        page_size=16,
        interleave_size=4,
        max_pages=4,
    )
    shd, rp = NamedSharding(mesh, P("dcp")), NamedSharding(mesh, P())
    nope = jax.device_put(jnp.zeros((dcp * 4, 16, 4), jnp.uint32), shd)
    rope_c = jax.device_put(jnp.zeros((dcp * 4, 16, 2), jnp.uint32), shd)
    q = jax.device_put(jnp.ones((dcp * rows, heads, lkv + rope), jnp.bfloat16), shd)
    topk = jax.device_put(jnp.full((dcp * rows, 4), -1, jnp.int32), shd)
    sl = jax.device_put(jnp.array([num_tokens, 0], jnp.int32), rp)
    bt = jax.device_put(jnp.arange(8, dtype=jnp.int32) % 4, rp)
    qsl = jax.device_put(jnp.array([0, num_tokens, num_tokens], jnp.int32), rp)
    dist = jax.device_put(jnp.array([0, 0, 1], jnp.int32), rp)
    ranks = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)

    def _attend(q_tok, g_nope, g_rope, flat, num_rows):
        del g_nope, g_rope, flat
        return jnp.broadcast_to(
            num_rows.astype(jnp.bfloat16), (q_tok.shape[0], q_tok.shape[1], lkv)
        )

    kv_dummy = jax.device_put(jnp.zeros((num_tokens, lkv), jnp.float8_e4m3fn), rp)
    kpe_dummy = jax.device_put(jnp.zeros((num_tokens, rope), jnp.float8_e4m3fn), rp)
    _, _, out_with_kv = dcp_token_parallel.sparse_mla_attention(
        layout,
        mesh,
        "dcp",
        nope,
        rope_c,
        q,
        kv_dummy,
        kpe_dummy,
        topk,
        sl,
        bt,
        qsl,
        dist,
        write_kv=None,
        attend=_attend,
        rank_ids=ranks,
        q_own_rows=True,
        token_rows=True,
    )
    _, _, out_none_kv = dcp_token_parallel.sparse_mla_attention(
        layout,
        mesh,
        "dcp",
        nope,
        rope_c,
        q,
        None,
        None,
        topk,
        sl,
        bt,
        qsl,
        dist,
        write_kv=None,
        attend=_attend,
        rank_ids=ranks,
        q_own_rows=True,
        token_rows=True,
        num_tokens=num_tokens,
    )
    np.testing.assert_array_equal(np.asarray(out_none_kv), np.asarray(out_with_kv))


@pytest.mark.parametrize("with_indexer", [False, True])
def test_should_route_sharded_kv_through_mla_wrapper_with_local_rope_and_oproj(
    with_indexer: bool,
):
    tp, num_tokens, s = 4, 32, 4
    rows_local = num_tokens // tp
    hidden, q_lora, kv_lora, rope_dim, nope_dim, v_dim = 16, 12, 8, 4, 6, 8
    heads, kw_dim = 2, 6
    cols_qc = (s * q_lora) // rows_local

    torch.manual_seed(19)
    full_x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    w_qkv = torch.randn(hidden, q_lora + kv_lora + rope_dim, dtype=torch.bfloat16)
    w_qb = torch.randn(q_lora, heads * tp * (nope_dim + rope_dim), dtype=torch.bfloat16)
    w_kw = torch.randn(hidden, kw_dim, dtype=torch.bfloat16)
    w_o = torch.randn(heads * v_dim, hidden, dtype=torch.bfloat16)
    s_o = torch.ones(hidden, dtype=torch.float32)
    positions = torch.arange(20, 20 + num_tokens, dtype=torch.int32)
    topk_buf = torch.arange(num_tokens * 4, dtype=torch.int32).reshape(num_tokens, 4)

    full_qkv = full_x @ w_qkv
    full_qc, full_kv = full_qkv.split([q_lora, kv_lora + rope_dim], dim=-1)
    full_kw = full_x @ w_kw

    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=tp, split_rows=s, page_size=16, interleave_size=4, max_pages=4
    )

    def _fake_rope(pos: torch.Tensor, q_: torch.Tensor, k_: torch.Tensor):
        # Position-dependent rotation that fails loudly if positions mismatch.
        scale = (pos.to(torch.float32) * 0.01 + 1.0).to(q_.dtype)
        return q_ * scale[:, None, None], k_ * scale[:, None, None]

    def _build_wrapper(rank_idx: int, serves_sharded: bool):
        captured: dict[str, object] = {}
        rank_t = torch.tensor([rank_idx], dtype=torch.int32)
        tp_op = SimpleNamespace(
            layout=layout,
            q_replicated=True,
            rank=rank_t,
            serves_oproj=True,
            serves_sharded_kv=serves_sharded,
        )

        def _fake_mla_attn(
            q_pair,
            kv_c_normed,
            k_pe,
            *,
            output_shape,
            topk_indices,
            q_dcp_replicated,
            sp_o_proj,
            kv_sharded=False,
        ):
            captured.update(
                q_pair=q_pair,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output_shape=output_shape,
                topk_indices=topk_indices,
                q_dcp_replicated=q_dcp_replicated,
                sp_o_proj=sp_o_proj,
                kv_sharded=kv_sharded,
            )
            return torch.zeros(rows_local, hidden, dtype=torch.bfloat16)

        _fake_mla_attn.sparse_mla_tp_op = tp_op  # type: ignore[attr-defined]
        _fake_mla_attn.W_UV = torch.zeros(1)  # type: ignore[attr-defined]

        def _fake_indexer(h_in, qr_in, pos_in, rope_in, kw=None, q_rows=None):
            del qr_in, pos_in, rope_in, q_rows
            assert h_in.shape[0] == num_tokens
            assert kw is not None and kw.shape == (num_tokens, kw_dim)
            return topk_buf

        def _proj_kw(z: torch.Tensor):
            return z @ w_kw, None

        def _proj_qkv(z: torch.Tensor):
            return z @ w_qkv, None

        def _norm_qa(z: torch.Tensor):
            return z * 1.25

        def _norm_kv(z: torch.Tensor):
            return z * 0.75

        def _q_proj_layer_fn():
            return SimpleNamespace(group_size=tp)

        def _proj_qb(z: torch.Tensor):
            return z @ w_qb, None

        _fake_indexer.wk_weights_proj = _proj_kw  # type: ignore[attr-defined]
        _fake_indexer.indexer_op = SimpleNamespace(  # type: ignore[attr-defined]
            topk_op_tp=SimpleNamespace(q_own_rows=True)
        )

        wrapper = object.__new__(
            mla_attention_op.VllmTPUMultiHeadLatentAttentionWrapper
        )
        torch.nn.Module.__init__(wrapper)
        wrapper.hidden_size = hidden
        wrapper.num_heads = heads
        wrapper.qk_nope_head_dim = nope_dim
        wrapper.qk_rope_head_dim = rope_dim
        wrapper.qk_head_dim = nope_dim + rope_dim
        wrapper.v_head_dim = v_dim
        wrapper.q_lora_rank = q_lora
        wrapper.kv_lora_rank = kv_lora
        wrapper.dcp_q_replicate = True
        wrapper.dcp_group_size = tp
        wrapper.is_sparse = True
        wrapper.skip_topk = not with_indexer
        wrapper.gate_is_fused = False
        wrapper.g_proj = None
        wrapper.kimi_preprocess = None
        wrapper.qb_overlap = False
        wrapper.decode_prologue = False
        wrapper.topk_indices_buffer = topk_buf
        wrapper.indexer = _fake_indexer if with_indexer else None  # type: ignore[assignment]
        wrapper.indexer_rope_emb = None
        wrapper.fused_qkv_a_proj = _proj_qkv  # type: ignore[assignment]
        wrapper.q_a_layernorm = _norm_qa  # type: ignore[assignment]
        wrapper.kv_a_layernorm = _norm_kv  # type: ignore[assignment]
        wrapper._q_proj_layer = _q_proj_layer_fn  # type: ignore[assignment]  # noqa: SLF001 -- stubs parent q_proj accessor on isolated wrapper unit
        wrapper.q_b_proj = _proj_qb  # type: ignore[assignment]
        wrapper.rotary_emb = _fake_rope  # type: ignore[assignment]
        wrapper.o_proj = SimpleNamespace(weight=w_o, weight_scale=s_o)  # type: ignore[assignment]
        wrapper.mla_attn = _fake_mla_attn  # type: ignore[assignment]
        return wrapper, captured

    def _ag_for(sharded_kv: bool) -> torch.Tensor:
        chunks = []
        for r in range(tp):
            lo, hi = r * rows_local, (r + 1) * rows_local
            hc = full_qc[lo : lo + s].reshape(rows_local, cols_qc)
            parts = [hc] if sharded_kv else [full_kv[lo:hi], hc]
            if with_indexer:
                parts.append(full_kw[lo:hi])
            chunks.append(torch.cat(parts, dim=-1))
        return torch.cat(chunks, dim=0)

    rank_idx = 2
    lo, hi = rank_idx * rows_local, (rank_idx + 1) * rows_local
    w_ref, cap_ref = _build_wrapper(rank_idx, serves_sharded=False)
    w_shd, cap_shd = _build_wrapper(rank_idx, serves_sharded=True)

    with (
        patch.object(envs, "TPU_DCP_TP_PREFILL_LOCAL_OPROJ", True),
        patch.object(envs, "TPU_DCP_TP_SHARDED_KV_WRITE", True),
        patch.object(tp_op_mod, "applies", return_value=True),
        patch.object(tp_op_mod, "serves_sequence_row_oproj", return_value=True),
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
    ):
        with patch.object(
            sp, "tensor_model_parallel_all_gather", return_value=_ag_for(False)
        ):
            w_ref.forward(
                positions,
                full_x[lo:hi],
                prefill_group=sp._GATHER_PROJECTIONS,  # noqa: SLF001 -- exercises replicated gather baseline
            )
        with (
            patch.object(
                sp, "tensor_model_parallel_all_gather", return_value=_ag_for(True)
            ),
            patch.object(
                sp.fused_moe_ep,
                "rank_operand",
                return_value=torch.tensor([[rank_idx]], dtype=torch.int32),
            ),
        ):
            w_shd.forward(
                positions,
                full_x[lo:hi],
                prefill_group=sp._ShardedKVGatherProjections(positions, tp),  # noqa: SLF001 -- exercises sharded gather path
            )

    assert cap_ref["kv_sharded"] is False
    assert cap_shd["kv_sharded"] is True
    assert (
        cap_shd["output_shape"]
        == cap_ref["output_shape"]
        == (num_tokens, heads * v_dim)
    )
    assert cap_shd["sp_o_proj"] is not None
    torch.testing.assert_close(
        cap_shd["kv_c_normed"], cap_ref["kv_c_normed"][lo:hi], rtol=0, atol=0
    )
    torch.testing.assert_close(cap_shd["k_pe"], cap_ref["k_pe"][lo:hi], rtol=0, atol=0)
    torch.testing.assert_close(
        cap_shd["q_dcp_replicated"][0], cap_ref["q_dcp_replicated"][0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        cap_shd["q_dcp_replicated"][1], cap_ref["q_dcp_replicated"][1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        cap_shd["topk_indices"], cap_ref["topk_indices"], rtol=0, atol=0
    )


def test_should_scale_num_tokens_and_forward_kv_sharded_in_tp_attention():
    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=4, split_rows=4, page_size=16, interleave_size=4, max_pages=4
    )
    calls: list[dict[str, object]] = []

    def _fake_tp_op(*args, kv_sharded: bool = False):
        calls.append({"args": args, "kv_sharded": kv_sharded})
        return torch.zeros(8, 16, dtype=torch.bfloat16)

    _fake_tp_op.layout = layout  # type: ignore[attr-defined]
    _fake_tp_op.q_replicated = True  # type: ignore[attr-defined]
    _fake_tp_op.fused_subbatch = True  # type: ignore[attr-defined]
    _fake_tp_op.uv_in_tp = True  # type: ignore[attr-defined]

    layer = SimpleNamespace(
        sparse_mla_tp_op=_fake_tp_op,
        W_UV=torch.zeros(1),
        W_UV_tp_all=torch.zeros(4, 8, 8, dtype=torch.bfloat16),
    )
    md = SimpleNamespace(
        seq_lens=torch.zeros(2, dtype=torch.int32),
        block_tables=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    ql_split = (
        torch.zeros(4, 4, 8, dtype=torch.bfloat16),
        torch.zeros(8, 4, 8, dtype=torch.bfloat16),
    )
    q_pe = torch.zeros(12, 4, 4, dtype=torch.bfloat16)
    kv_local = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
    k_pe_local = torch.zeros(8, 4, dtype=torch.float8_e4m3fn)
    topk = torch.zeros(32, 4, dtype=torch.int32)
    w_o = (torch.zeros(8, 16), torch.ones(16))

    with pytest.raises(ValueError, match="kv_sharded requires"):
        tp_op_mod.attention(
            layer,
            (torch.zeros(1), torch.zeros(1)),
            ql_split,
            q_pe,
            kv_local,
            k_pe_local,
            topk,
            md,
            q_replicated=True,
            w_o=None,
            kv_sharded=True,
        )

    out = tp_op_mod.attention(
        layer,
        (torch.zeros(1), torch.zeros(1)),
        ql_split,
        q_pe,
        kv_local,
        k_pe_local,
        topk,
        md,
        q_replicated=True,
        w_o=w_o,
        kv_sharded=True,
    )
    assert tuple(out.shape) == (8, 16)
    assert len(calls) == 1 and calls[0]["kv_sharded"] is True


def test_should_lift_rank_buffer_through_layer_keep_before_positions_slice(
    monkeypatch,
):
    rank_buf = torch.tensor([[1]], dtype=torch.int32)
    monkeypatch.setattr(sp.envs, "TPU_DCP_TP_SHARDED_KV_WRITE", True)
    monkeypatch.setattr(sp.envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(sp.fused_moe_ep, "_RANK_BUFFER", rank_buf)
    monkeypatch.setattr(sp.small_all_reduce_op, "retain", lambda x: x)

    events: list[str] = []
    sliced_positions: list[torch.Tensor] = []

    class _FakeLayer(torch.nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.idx = idx
            setattr(self, sp.KEEP_ATTR, [rank_buf])

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
            llama_4_scaling: torch.Tensor | None = None,
            *,
            gather_group=None,
        ):
            events.append(f"keep_L{self.idx}")
            hidden_states = sp._retain_buffers(  # noqa: SLF001 -- mirrors patched_decoder_layer_forward retain sequence
                hidden_states, *getattr(self, sp.KEEP_ATTR)
            )
            if gather_group is not None:
                events.append(f"gather_L{self.idx}")
                hidden_states, _ = gather_group.project_gather(
                    None, hidden_states, positions.shape[0]
                )
            else:
                hidden_states = (
                    hidden_states + positions.sum().to(hidden_states.dtype) * 0
                )
            return hidden_states, residual

    orig_own_rows = sp.own_rows

    def _spy_own_rows(x: torch.Tensor, rank: torch.Tensor, tp: int) -> torch.Tensor:
        events.append("own_rows")
        out = orig_own_rows(x, rank, tp)
        sliced_positions.append(out)
        return out

    monkeypatch.setattr(
        sp,
        "_retain_buffers",
        lambda x, *keep: x + sum(k.sum() * 0 for k in keep).to(x.dtype),
    )
    monkeypatch.setattr(sp, "own_rows", _spy_own_rows)
    monkeypatch.setattr(
        sp._GatherProjections,  # noqa: SLF001 -- stubs downstream projection gather for ordering check
        "project_gather",
        staticmethod(
            lambda mla, x, n, *, positions_local=None: (
                x + positions_local.sum().to(x.dtype) * 0,
                None,
            )
        ),
    )
    monkeypatch.setattr(sp, "applies", lambda n: n >= 4)

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.start_layer = 0
            self.end_layer = 2
            self.layers = torch.nn.ModuleList([_FakeLayer(0), _FakeLayer(1)])
            setattr(self, sp.KEEP_ATTR, True)
            setattr(self, sp.EMBED_RS_ATTR, False)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            n = input_ids.shape[0]
            return torch.zeros(n // 2 if n >= 4 else n, 4, dtype=torch.bfloat16)

        def norm(
            self, x: torch.Tensor, r: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            return x, r

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            if not sp.applies(positions.shape[0]):
                x = self.embed_input_ids(input_ids)
                res = None
                for layer in self.layers:
                    x, res = layer(positions, x, res)
                return x
            return sp.patched_model_forward(self, input_ids, positions)

    monkeypatch.setattr(sp, "_all_rows", lambda x, n: torch.cat([x, x], dim=0)[:n])

    model = _FakeModel()
    positions = torch.arange(8, dtype=torch.int32)
    input_ids = torch.zeros(8, dtype=torch.int32)
    out = sp.patched_model_forward(model, input_ids, positions)
    assert tuple(out.shape) == (8, 4)
    assert events == ["keep_L0", "gather_L0", "own_rows", "keep_L1", "gather_L1"]
    assert len(sliced_positions) == 1
    torch.testing.assert_close(
        sliced_positions[0], torch.tensor([4, 5, 6, 7], dtype=torch.int32)
    )

    # Full-graph Dynamo parity: decode (n=2) and prefill (n=8) must bind rank_buf
    # to the exact same FX placeholder name with zero graph breaks.
    monkeypatch.setattr(sp, "own_rows", orig_own_rows)
    placeholders_by_mode: dict[str, list[str]] = {}
    for label, n in (("decode", 2), ("prefill", 8)):
        captured: list[list[str]] = []

        def _capture_backend(gm, _example_inputs, _cap=captured):
            _cap.append(
                [node.name for node in gm.graph.nodes if node.op == "placeholder"]
            )
            return gm.forward

        torch._dynamo.reset()  # noqa: SLF001 -- isolates per-bucket Dynamo compile trace
        compiled = torch.compile(model, backend=_capture_backend, fullgraph=True)
        compiled(torch.zeros(n, dtype=torch.int32), torch.arange(n, dtype=torch.int32))
        assert len(captured) == 1
        placeholders_by_mode[label] = captured[0]

    assert set(placeholders_by_mode["prefill"]) == set(placeholders_by_mode["decode"])
    assert (
        "l_self_modules_layers_modules_0_tpu_sp_keep_0_"
        in placeholders_by_mode["prefill"]
    )
    assert not any("fused_moe_ep" in name for name in placeholders_by_mode["prefill"])

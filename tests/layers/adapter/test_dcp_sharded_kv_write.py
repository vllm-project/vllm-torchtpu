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

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import CompilationConfig, CompilationMode
from vllm.model_executor import custom_op as custom_op_mod
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from vllm_torchtpu.distributed import dcp as dcp_mod
from vllm_torchtpu.kernels.mla import dcp_token_parallel, kv_cache_utils
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op
from vllm_torchtpu.models.vllm import glm_dsa_dispatch


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
            patch.object(sp, "glm_dsa_v7x_active", return_value=True),
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
            unified_indexer_inputs=None,
            in_branch_qb_inputs=None,
        ):
            del unified_indexer_inputs, in_branch_qb_inputs
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
        patch.object(sp, "glm_dsa_v7x_active", return_value=True),
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
    monkeypatch.setattr(sp, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    monkeypatch.setattr(sp, "_SP_ENABLED", True)
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
            lambda mla, x, n, *, positions_local=None, **_kw: (
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


def test_should_match_all_gather_bit_exact_when_gathering_qc_head_rows():
    mesh = _make_16dev_mesh()
    dcp, rows_local, split_rows, q_lora = 16, 128, 16, 64
    rng = np.random.default_rng(7711)
    full_qc = jnp.asarray(
        (rng.standard_normal((dcp * rows_local, q_lora)) * 3.5).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    # Include exact zero and sentinel rows at rank boundaries and tail.
    full_qc = full_qc.at[0, :].set(jnp.bfloat16(0.0))
    full_qc = full_qc.at[-1, :].set(jnp.bfloat16(-123.5))
    shd = P("dcp")
    qc_shd = jax.device_put(full_qc, NamedSharding(mesh, shd))
    ranks_shd = _sharded_array(np.arange(dcp, dtype=np.int32), mesh)

    def _ref_ag(qc_l, rk):
        all_heads = jax.lax.all_gather(qc_l[:split_rows], "dcp", axis=0, tiled=False)
        r = rk.reshape(())
        next_r = jnp.minimum(r + 1, dcp - 1)
        tail_s = jnp.where(
            r < dcp - 1,
            all_heads[next_r],
            jnp.broadcast_to(qc_l[-1:], (split_rows, q_lora)),
        )
        return jnp.concatenate([all_heads[0], qc_l[split_rows:], tail_s], axis=0)

    def _got_pperm(qc_l, rk):
        return dcp_token_parallel.gather_qc_head_rows(
            qc_l, rk, dcp_size=dcp, split_rows=split_rows, axis_name="dcp"
        )

    want = jax.jit(
        jax.shard_map(
            _ref_ag, mesh=mesh, in_specs=(shd, shd), out_specs=shd, check_vma=False
        )
    )(qc_shd, ranks_shd)
    got = jax.jit(
        jax.shard_map(
            _got_pperm, mesh=mesh, in_specs=(shd, shd), out_specs=shd, check_vma=False
        )
    )(qc_shd, ranks_shd)

    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(want, jnp.uint16)),
    )


@pytest.mark.parametrize("with_indexer", [False, True], ids=["non_indexer", "indexer"])
def test_should_use_qc_head_shift_op_only_on_non_indexer_layers(with_indexer: bool):
    tp, rows_local, s, q_lora, kv_dim, kw_dim = 4, 32, 8, 16, 24, 12
    num_tokens = tp * rows_local
    torch.manual_seed(919)
    full_qc = torch.randn(num_tokens, q_lora, dtype=torch.bfloat16)
    full_kv = torch.randn(num_tokens, kv_dim, dtype=torch.bfloat16)
    full_kw = torch.randn(num_tokens, kw_dim, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.int32)
    full_x = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(-1)
    cols_qc = (s * q_lora) // rows_local

    for r in range(tp):
        lo, hi = r * rows_local, (r + 1) * rows_local
        rank_t = torch.tensor([r], dtype=torch.int32)
        shift_calls: list[int] = []
        ag_calls: list[int] = []

        def _fake_shift(
            qc_local: torch.Tensor, rk: torch.Tensor, _calls=shift_calls
        ) -> torch.Tensor:
            _calls.append(int(rk.item()))
            ri = int(rk.item())
            head0 = full_qc[:s]
            tail = (
                full_qc[(ri + 1) * rows_local : (ri + 1) * rows_local + s]
                if ri < tp - 1
                else qc_local[-1:].expand(s, -1)
            )
            return torch.cat([head0, qc_local[s:], tail], dim=0)

        tp_op = SimpleNamespace(
            layout=dcp_token_parallel.TokenParallelLayout(
                dcp_size=tp,
                page_size=16,
                interleave_size=4,
                max_pages=8,
                split_rows=s,
            ),
            q_replicated=True,
            rank=rank_t,
            serves_sharded_kv=True,
            qc_head_shift_op=_fake_shift,
        )

        def _qkv_proj(x: torch.Tensor) -> tuple[torch.Tensor, None]:
            idx = x.squeeze(-1).long()
            return torch.cat([full_qc[idx], full_kv[idx]], dim=-1), None

        def _wk_proj(x: torch.Tensor) -> tuple[torch.Tensor, None]:
            idx = x.squeeze(-1).long()
            return full_kw[idx], None

        expected_ag = torch.cat(
            [
                full_qc.view(tp, rows_local, q_lora)[:, :s, :].reshape(
                    num_tokens, cols_qc
                ),
                full_kw,
            ],
            dim=-1,
        )

        def _fake_ag(
            t: torch.Tensor,
            dim: int,
            _calls=ag_calls,
            _expected=expected_ag,
        ) -> torch.Tensor:
            assert dim == 0
            _calls.append(t.shape[-1])
            return _expected

        mla = SimpleNamespace(
            fused_qkv_a_proj=_qkv_proj,
            indexer=SimpleNamespace(
                wk_weights_proj=_wk_proj,
                indexer_op=SimpleNamespace(topk_op_tp=SimpleNamespace(q_own_rows=True)),
            )
            if with_indexer
            else None,
            skip_topk=not with_indexer,
            is_sparse=True,
            g_proj=None,
            o_proj=SimpleNamespace(),
            mla_attn=SimpleNamespace(sparse_mla_tp_op=tp_op),
            gate_is_fused=False,
            kimi_preprocess=None,
            q_lora_rank=q_lora,
        )
        group = sp._ShardedKVGatherProjections(positions, tp)  # noqa: SLF001 -- exercises per-step gatherer contract directly
        with (
            patch.object(sp, "glm_dsa_v7x_active", return_value=True),
            patch.object(tp_op_mod, "applies", return_value=True),
            patch.object(tp_op_mod, "serves_sequence_row_oproj", return_value=True),
            patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
            patch.object(
                sp.fused_moe_ep, "rank_operand", return_value=rank_t.view(1, 1)
            ),
            patch.object(sp, "tensor_model_parallel_all_gather", side_effect=_fake_ag),
        ):
            pre_qkv, pre_kw = group.project_gather(mla, full_x[lo:hi], num_tokens)

        assert isinstance(pre_qkv, tuple) and len(pre_qkv) == 3
        got_qc, got_kv_local, got_pos_local = pre_qkv
        q_idx = tp_op_mod.q_rows(tp_op, num_tokens)
        torch.testing.assert_close(
            got_qc, full_qc.index_select(0, q_idx), rtol=0, atol=0
        )
        torch.testing.assert_close(got_kv_local, full_kv[lo:hi], rtol=0, atol=0)
        torch.testing.assert_close(got_pos_local, positions[lo:hi], rtol=0, atol=0)
        if with_indexer:
            assert shift_calls == []
            assert ag_calls == [cols_qc + kw_dim]
            assert pre_kw is not None
            torch.testing.assert_close(pre_kw, full_kw, rtol=0, atol=0)
        else:
            assert shift_calls == [r]
            assert ag_calls == []
            assert pre_kw is None


def test_should_keep_indexer_kw_all_gather_under_zigzag_when_sharded_k_is_off() -> None:
    tp, nb, s, rows_local, q_lora, kv_dim, k_dim, w_dim = 4, 4, 2, 16, 16, 12, 8, 4
    num_tokens = tp * rows_local
    torch.manual_seed(78)
    full_qc = torch.randn(num_tokens, q_lora, dtype=torch.bfloat16)
    full_kv = torch.randn(num_tokens, kv_dim, dtype=torch.bfloat16)
    full_kw = torch.randn(num_tokens, k_dim + w_dim, dtype=torch.bfloat16)
    full_x = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(-1)
    positions = 4096 + torch.arange(num_tokens, dtype=torch.int32)
    cols_qc = (s * q_lora) // rows_local

    # Build the real per-rank all-gather table in zigzag rank-major order.
    rank_inputs = [
        sp.own_rows_zigzag(full_x, torch.tensor(rk, dtype=torch.int32), tp, nb)
        for rk in range(tp)
    ]
    rank_parts = []
    for x_r in rank_inputs:
        idx = x_r.squeeze(-1).long()
        qc_head = full_qc[idx][:s].reshape(rows_local, cols_qc)
        rank_parts.append(torch.cat([qc_head, full_kw[idx]], dim=-1))
    gathered_all = torch.cat(rank_parts, dim=0)

    for r in range(tp):
        rank_t = torch.tensor(r, dtype=torch.int32)
        ag_calls: list[int] = []
        tp_op = SimpleNamespace(
            layout=dcp_token_parallel.TokenParallelLayout(
                dcp_size=tp,
                page_size=16,
                interleave_size=4,
                max_pages=8,
                split_rows=s,
                zigzag_bands=nb,
            ),
            q_replicated=True,
            rank=rank_t,
            serves_sharded_kv=True,
            qc_head_shift_op=lambda *_a: (_ for _ in ()).throw(
                AssertionError("unexpected qc_head_shift_op call")
            ),
        )
        mla = SimpleNamespace(
            fused_qkv_a_proj=lambda x: (
                torch.cat(
                    [full_qc[x.squeeze(-1).long()], full_kv[x.squeeze(-1).long()]],
                    dim=-1,
                ),
                None,
            ),
            indexer=SimpleNamespace(
                head_dim=k_dim,
                wk_weights_proj=lambda x: (full_kw[x.squeeze(-1).long()], None),
                indexer_op=SimpleNamespace(
                    topk_op_tp=SimpleNamespace(q_own_rows=True, serves_sharded_k=True)
                ),
            ),
            skip_topk=False,
            is_sparse=True,
            g_proj=None,
            o_proj=SimpleNamespace(),
            mla_attn=SimpleNamespace(sparse_mla_tp_op=tp_op),
            gate_is_fused=False,
            kimi_preprocess=None,
            q_lora_rank=q_lora,
        )
        group = sp._ShardedKVGatherProjections(positions, tp, zigzag_bands=nb)  # noqa: SLF001 -- verifies unsharded zigzag fallback contract
        with (
            patch.object(sp, "glm_dsa_v7x_active", return_value=True),
            patch.object(tp_op_mod, "applies", return_value=True),
            patch.object(tp_op_mod, "serves_sequence_row_oproj", return_value=True),
            patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
            patch.object(
                sp.fused_moe_ep, "rank_operand", return_value=rank_t.view(1, 1)
            ),
            patch.object(
                sp,
                "tensor_model_parallel_all_gather",
                side_effect=lambda t, dim, _calls=ag_calls: (
                    _calls.append(t.shape[-1]),
                    gathered_all,
                )[1],
            ),
        ):
            pre_qkv, pre_kw = group.project_gather(mla, rank_inputs[r], num_tokens)

        assert ag_calls == [cols_qc + k_dim + w_dim]
        assert isinstance(pre_qkv, tuple) and len(pre_qkv) == 3
        assert isinstance(pre_kw, torch.Tensor)
        got_qc, got_kv_local, got_pos_local = pre_qkv
        q_idx = tp_op_mod.q_rows(tp_op, num_tokens)
        own_idx = rank_inputs[r].squeeze(-1).long()
        torch.testing.assert_close(
            got_qc, full_qc.index_select(0, q_idx), rtol=0, atol=0
        )
        torch.testing.assert_close(got_kv_local, full_kv[own_idx], rtol=0, atol=0)
        torch.testing.assert_close(got_pos_local, positions[own_idx], rtol=0, atol=0)
        torch.testing.assert_close(pre_kw, full_kw, rtol=0, atol=0)


@pytest.mark.parametrize("permuted", [True, False])
def test_should_match_replicated_indexer_forward_with_sharded_k(
    permuted: bool,
) -> None:
    tp, s, rows_local = 4, 2, 8
    num_tokens = tp * rows_local
    n_head, head_dim, rope_dim, q_lora = 4, 32, 16, 16
    torch.manual_seed(2026 + int(permuted))

    ind = mla_attention_op.VllmTPUIndexer.__new__(mla_attention_op.VllmTPUIndexer)
    torch.nn.Module.__init__(ind)
    ind.n_head = n_head
    ind.head_dim = head_dim
    ind.rope_dim = rope_dim
    ind.softmax_scale = head_dim**-0.5
    ind._rope_neox_permuted = permuted  # noqa: SLF001 -- exercises permuted vs fallback RoPE branch
    w_qb = torch.nn.Linear(q_lora, n_head * head_dim, bias=False).to(torch.bfloat16)
    ind.wq_b = lambda x: (w_qb(x), None)  # type: ignore[assignment]
    ind.k_norm = torch.nn.LayerNorm(head_dim).to(torch.bfloat16)

    captured: list[tuple[tuple, dict]] = []

    def _capture(*args, **kwargs):
        captured.append((args, kwargs))
        return torch.zeros((args[1].shape[0], 4), dtype=torch.int32)

    ind.indexer_op = _capture  # type: ignore[assignment]
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        rotary_emb = RotaryEmbedding(
            head_size=rope_dim,
            rotary_dim=rope_dim,
            max_position_embeddings=8192,
            base=10000,
            is_neox_style=False,
            dtype=torch.bfloat16,
        )

    positions = 512 + torch.arange(num_tokens, dtype=torch.int64)
    full_kw = torch.randn(num_tokens, head_dim + n_head, dtype=torch.bfloat16)
    rank = 2
    lo, hi = rank * rows_local, (rank + 1) * rows_local
    tp_op = SimpleNamespace(
        layout=dcp_token_parallel.TokenParallelLayout(
            dcp_size=tp, page_size=16, interleave_size=4, max_pages=8, split_rows=s
        ),
        rank=torch.tensor(rank, dtype=torch.int32),
    )
    q_rows = tp_op_mod.q_rows(tp_op, num_tokens)
    qr = torch.randn(q_rows.shape[0], q_lora, dtype=torch.bfloat16)
    dummy_hidden = torch.zeros((num_tokens, 4), dtype=torch.bfloat16)

    ind.forward(dummy_hidden, qr, positions, rotary_emb, kw=full_kw, q_rows=q_rows)
    ind.forward(
        dummy_hidden[:rows_local],
        qr,
        positions,
        rotary_emb,
        kw=(
            full_kw[lo:hi, :head_dim],
            full_kw[:, head_dim:].index_select(0, q_rows),
            positions[lo:hi],
        ),
        q_rows=q_rows,
    )

    (rep_args, rep_kw), (sh_args, sh_kw) = captured
    assert rep_kw == {}
    assert sh_kw == {"k_sharded": True}
    torch.testing.assert_close(
        sh_args[1].view(torch.uint8), rep_args[1].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(sh_args[2], rep_args[2][lo:hi], rtol=0, atol=0)
    torch.testing.assert_close(sh_args[3], rep_args[3], rtol=0, atol=0)


def test_should_build_and_dispatch_sharded_k_indexer_op(monkeypatch) -> None:
    built: list[tuple[str, tuple]] = []

    def _fake_pcp_op(name, fn, *, input_partition_specs, **kwargs):
        del fn, kwargs
        built.append((name, tuple(input_partition_specs)))

        class _Op:
            def __init__(self, op_name: str):
                self.op_name = op_name

            def register_fake(self, _f):
                return None

            def __call__(self, *args):
                # Return (new_cache, flat_topk) tagged with which op ran.
                tag = 1 if "sharded_k" in self.op_name else 0
                return args[2], torch.full(
                    (args[0].shape[0], 4), tag, dtype=torch.int32
                )

        return _Op(name)

    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=4, page_size=16, interleave_size=4, max_pages=8, split_rows=2
    )
    monkeypatch.setattr(tp_op_mod, "pcp_streaming_jax_op", _fake_pcp_op)
    monkeypatch.setattr(
        tp_op_mod, "_sharded_rank_ids", lambda: torch.tensor([0], dtype=torch.int32)
    )
    monkeypatch.setattr(
        dcp_mod, "dcp_lane_operand", lambda: torch.tensor([0], dtype=torch.int32)
    )
    monkeypatch.setattr(
        tp_op_mod, "get_or_create_dcp_mesh", lambda axis_name: "idx_sharded_mesh"
    )
    idx_stub = SimpleNamespace(
        topk_tokens=4,
        k_cache=SimpleNamespace(prefix="model.layers.0.self_attn.indexer"),
        _shared_meta=True,
    )
    op = tp_op_mod.build_indexer_op(idx_stub, layout, q_own_rows=True)
    assert getattr(op, "serves_sharded_k", False) is True
    assert len(built) == 2
    assert "sharded_k" not in built[0][0]
    assert built[0][1][3] == P()
    assert "sharded_k" in built[1][0]
    assert built[1][1][3] == P("dcp")

    cache = torch.zeros((2, 16), dtype=torch.uint8)
    args = (
        torch.zeros((10, 4, 16), dtype=torch.uint8),
        torch.zeros((10, 4), dtype=torch.bfloat16),
        cache,
        torch.zeros((8, 17), dtype=torch.uint8),
        torch.zeros((2,), dtype=torch.int32),
        torch.zeros((2, 2), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int32),
    )
    idx_holder = SimpleNamespace(topk_op_tp=op)
    out_rep = tp_op_mod.indexer_rows(idx_holder, *args)
    out_sh = tp_op_mod.indexer_rows(idx_holder, *args, k_sharded=True)
    assert int(out_rep[0, 0].item()) == 0
    assert int(out_sh[0, 0].item()) == 1


@pytest.mark.parametrize("zigzag_bands", [1, 2, 4])
def test_should_bound_sharded_kv_a2a_send_counts_across_adversarial_batches(
    zigzag_bands: int,
) -> None:
    dcp, interleave, rows_local = 16, 4, 1024
    total_tokens = dcp * rows_local
    band = rows_local // zigzag_bands
    cycle = dcp * interleave
    num_blocks = dcp * zigzag_bands
    rng = np.random.default_rng(77 + zigzag_bands)

    # Known tight serving profiles under dcp=16, rows_local=1024, zigzag_bands=4:
    exact_checks = [
        ([0, total_tokens], 64),
        (list(range(8)) + [total_tokens], 71),
        (list(range(16)) + [total_tokens], 79),
        (list(range(32)) + [total_tokens], 95),
        (list(range(64)) + [total_tokens], 124),
        ([0, 4096, 8192, 12288, total_tokens], 64),
        ([0, 4001, 8002, 12003, total_tokens], 68),
    ]
    for raw_qsl, exp_ub in exact_checks:
        qsl_pad = raw_qsl + [raw_qsl[-1]] * (65 - len(raw_qsl))
        got = int(
            kv_cache_utils._sharded_a2a_max_send_bound(  # noqa: SLF001 -- verifying exact tier bound formula
                jnp.asarray(qsl_pad, dtype=jnp.int32),
                rows_local=rows_local,
                dcp_size=dcp,
                interleave_size=interleave,
                zigzag_bands=zigzag_bands,
            )
        )
        assert got == exp_ub if zigzag_bands == 4 else got >= 64

    tiers = kv_cache_utils._sharded_a2a_cap_tiers(  # noqa: SLF001 -- verifying tier lattice endpoints
        rows_local, cap_full=320, dcp_size=dcp
    )
    assert tiers == (64, 128, 320)

    def _pref(x: int) -> np.ndarray:
        return (x // cycle) * interleave + np.clip(
            (x % cycle) - np.arange(dcp) * interleave, 0, interleave
        )

    r_idx = np.arange(dcp)[:, None]
    k_idx = np.arange(zigzag_bands)[None, :]
    blk_id = np.where(
        (k_idx & 1) == 0, k_idx * dcp + r_idx, (k_idx + 1) * dcp - 1 - r_idx
    )
    for step_idx in range(100):
        nseq = int(rng.integers(1, 65))
        total = int(rng.integers(1, total_tokens + 1))
        if step_idx % 2 == 0:
            nd = min(nseq - 1, total)
            qsl = np.array(list(range(nd + 1)) + [total] * (64 - nd), dtype=np.int32)
        else:
            cuts = sorted(rng.choice(total + 1, size=nseq - 1, replace=True).tolist())
            tail = [total] * (65 - (nseq + 1))
            qsl = np.array([0] + cuts + [total] + tail, dtype=np.int32)
        ql = qsl[1:] - qsl[:-1]
        # Worst-case alignment shift placing every sequence head on owner 0.
        sl = ql + (0 if step_idx % 3 == 0 else rng.integers(0, 5000, size=64))
        cnt_blk = np.zeros((num_blocks, dcp), dtype=np.int32)
        for s in range(64):
            qs, qe = int(qsl[s]), int(qsl[s + 1])
            if qe <= qs:
                continue
            p_base = int(sl[s]) - (qe - qs)
            for blk in range(qs // band, min((qe - 1) // band + 1, num_blocks)):
                lo, hi = max(qs, blk * band), min(qe, (blk + 1) * band)
                cnt_blk[blk] += _pref(p_base + hi - qs) - _pref(p_base + lo - qs)
        true_max = int(np.max(np.sum(cnt_blk[blk_id], axis=1)))
        bound = int(
            kv_cache_utils._sharded_a2a_max_send_bound(  # noqa: SLF001 -- comparing bound against exact count
                jnp.asarray(qsl, dtype=jnp.int32),
                rows_local=rows_local,
                dcp_size=dcp,
                interleave_size=interleave,
                zigzag_bands=zigzag_bands,
            )
        )
        assert bound >= true_max, (
            f"step_idx={step_idx}: bound={bound} < true_max={true_max}"
        )


def test_should_match_replicated_kv_write_across_tight_tier_switches() -> None:
    dcp = 16
    mesh = _make_16dev_mesh()
    rows_local = 64  # total 1024 tokens, base tier = 4 -> rounded to 16
    cases = [
        # Single aligned prefill -> tier 16
        (
            np.array([2048] + [0] * 15, dtype=np.int32),
            np.array([0] + [1024] * 16, dtype=np.int32),
        ),
        # 15 one-token decodes + 1 prefill -> middle tier
        (
            np.array([100] * 15 + [1009 + 200], dtype=np.int32),
            np.array(list(range(16)) + [1024], dtype=np.int32),
        ),
        # Multi-prefill unaligned crossing seams
        (
            np.array([500, 700, 900, 1200] + [0] * 12, dtype=np.int32),
            np.array([0, 17, 303, 661] + [1024] * 13, dtype=np.int32),
        ),
    ]
    for sl, qsl in cases:
        bt = np.arange(16 * 16, dtype=np.int32).reshape(16, 16) % 32
        _run_sharded_vs_replicated_kv_write(
            mesh=mesh,
            dcp=dcp,
            rows_local=rows_local,
            seq_lens=sl,
            block_tables=bt,
            query_start_loc=qsl,
        )


def _make_dcp_mesh(dcp: int) -> Mesh:
    devs = jax.devices()
    if len(devs) < dcp:
        pytest.skip(f"needs {dcp} JAX devices, got {len(devs)}")
    return Mesh(np.array(devs[:dcp]), ("dcp",))


def _make_unique_bf16_rows(
    num_tokens: int,
    cols: int,
    split_rows: int,
) -> jax.Array:
    assert cols >= 16 and num_tokens <= 16384
    rows = np.arange(num_tokens, dtype=np.int32)
    arr = np.empty((num_tokens, cols), dtype=np.float32)
    # Exact bf16 base-128 coordinates stay collision-free up to 16,384 rows.
    arr[:, 0] = (rows // 128).astype(np.float32)
    arr[:, 1] = (rows % 128).astype(np.float32)
    arr[:, 2] = -((rows % 64) + 1).astype(np.float32)
    col_off = np.arange(cols - 3, dtype=np.float32)[None, :] * 0.125
    arr[:, 3:] = (rows[:, None] % 31).astype(np.float32) * 0.25 + col_off
    for row_idx, sentinel in (
        (0, 101.0),
        (split_rows - 1, -103.0),
        (split_rows, 107.0),
        (num_tokens - 1, -109.0),
    ):
        arr[row_idx, -1] = sentinel
    bf16 = jnp.asarray(arr, dtype=jnp.bfloat16)
    u16 = np.asarray(jax.lax.bitcast_convert_type(bf16, jnp.uint16))
    assert np.unique(u16, axis=0).shape[0] == num_tokens
    return bf16


def _capture_token_parallel_ops(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mesh: Mesh,
    layout: dcp_token_parallel.TokenParallelLayout,
    mla_kv_spec: tuple[object, object] | None = None,
) -> tuple[object, list[dict]]:
    captured: list[dict] = []

    def _capture_op(name, fn, **kwargs):
        record = {"name": name, "fn": fn, "kwargs": kwargs, "fake": None}

        def _op(*args):
            return record["fake"](*args)

        def _register_fake(fake_fn):
            record["fake"] = fake_fn

        _op.register_fake = _register_fake
        captured.append(record)
        return _op

    monkeypatch.setattr(glm_dsa_dispatch, "_shared_jax_ops", {})
    monkeypatch.setattr(tp_op_mod, "pcp_streaming_jax_op", _capture_op)
    monkeypatch.setattr(
        tp_op_mod, "_sharded_rank_ids", lambda: torch.zeros((1,), dtype=torch.int32)
    )
    monkeypatch.setattr(
        tp_op_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size",
        lambda: layout.dcp_size,
    )

    class _LayerStub:
        kv_cache_quantized_dtype = None
        scale = 0.125
        num_heads = 4
        v_head_dim = 16
        kv_lora_rank = 32

    layer = _LayerStub()
    if mla_kv_spec is not None:
        layer.mla_kv_spec = mla_kv_spec  # type: ignore[attr-defined]
    impl = SimpleNamespace(
        _dcp_mesh=mesh,
        dcp_size=layout.dcp_size,
        dcp_interleave_size=layout.interleave_size,
    )
    tp_op_mod.build_attention_op(layer, impl, layout, q_replicated=True)
    # The built ops hold the layer by weak reference; return it to keep it alive.
    return layer, captured


def _capture_qc_head_shift_fn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mesh: Mesh,
    layout: dcp_token_parallel.TokenParallelLayout,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    _, captured = _capture_token_parallel_ops(monkeypatch, mesh=mesh, layout=layout)
    matches = [r for r in captured if r["name"].startswith("pallas::dcp_qc_head_shift")]
    assert len(matches) == 1
    record = matches[0]
    assert record["kwargs"]["mesh"] is mesh
    assert record["kwargs"]["input_partition_specs"] == (P("dcp"), P("dcp"))
    assert record["kwargs"]["output_partition_specs"] == (P("dcp"),)
    return record["fn"]


def _assert_qc_head_shift_matches_q_rows(
    *,
    shift_fn: Callable[[jax.Array, jax.Array], jax.Array],
    mesh: Mesh,
    layout: dcp_token_parallel.TokenParallelLayout,
    cases: list[tuple[int, int]],
) -> None:
    dcp = layout.dcp_size
    split_rows = layout.split_rows
    shd = NamedSharding(mesh, P("dcp"))
    ranks_shd = _sharded_array(np.arange(dcp, dtype=np.int32), mesh)
    mismatches: list[tuple[int, int, int, list[int]]] = []

    for num_tokens, cols in cases:
        rows_local = layout.rows_per_rank(num_tokens)
        nb = layout.effective_zigzag_bands(num_tokens)
        full_qc = _make_unique_bf16_rows(num_tokens, cols, split_rows)
        full_u16 = np.asarray(jax.lax.bitcast_convert_type(full_qc, jnp.uint16))
        full_np = np.asarray(full_qc)

        rank_chunks = [
            full_np[
                np.asarray(
                    dcp_token_parallel.zigzag_rank_global_tokens(r, rows_local, dcp, nb)
                )
            ]
            for r in range(dcp)
        ]
        qc_shd = jax.device_put(
            jnp.asarray(np.concatenate(rank_chunks, axis=0), dtype=jnp.bfloat16),
            shd,
        )
        got = jax.jit(shift_fn)(qc_shd, ranks_shd)
        got_u16 = np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint16)).reshape(
            dcp, split_rows + rows_local, cols
        )

        bad_ranks: list[int] = []
        for r in range(dcp):
            want_idx = tp_op_mod.q_rows(
                SimpleNamespace(
                    layout=layout, rank=torch.tensor([r], dtype=torch.int32)
                ),
                num_tokens,
            ).numpy()
            if not np.array_equal(got_u16[r], full_u16[want_idx]):
                bad_ranks.append(r)
        if bad_ranks:
            mismatches.append((num_tokens, cols, nb, bad_ranks))

    assert not mismatches, (
        f"qc_head_shift mismatches (T, cols, nb, bad_ranks): {mismatches}"
    )


def test_should_align_qc_head_shift_with_global_token_count_on_4dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dcp = 4
    mesh = _make_dcp_mesh(dcp)
    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        page_size=16,
        interleave_size=4,
        split_rows=16,
        max_pages=8,
        zigzag_bands=4,
    )
    shift_fn = _capture_qc_head_shift_fn(monkeypatch, mesh=mesh, layout=layout)
    _assert_qc_head_shift_matches_q_rows(
        shift_fn=shift_fn,
        mesh=mesh,
        layout=layout,
        cases=[
            (64, 16),  # R == split_rows, band-size contiguous fallback (nb=1)
            (128, 16),  # band-size contiguous fallback (nb=1)
            (256, 16),  # active 4-band zigzag (nb=4)
            (260, 24),  # indivisible by dcp*zb (nb=1) and wider [T, Q+W] width
        ],
    )


def test_should_align_qc_head_shift_with_global_token_count_on_16dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dcp = 16
    mesh = _make_dcp_mesh(dcp)
    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        page_size=64,
        interleave_size=4,
        split_rows=64,
        max_pages=8,
        zigzag_bands=4,
    )
    shift_fn = _capture_qc_head_shift_fn(monkeypatch, mesh=mesh, layout=layout)
    _assert_qc_head_shift_matches_q_rows(
        shift_fn=shift_fn,
        mesh=mesh,
        layout=layout,
        cases=[
            (1024, 16),  # R == split_rows, contiguous fallback (nb=1)
            (2048, 24),  # contiguous fallback (nb=1) and wider [T, Q+W] width
            (4096, 16),  # active 4-band zigzag (nb=4)
        ],
    )


class _StopAfterKvWrite(Exception):
    pass


@pytest.mark.parametrize(
    ("dcp", "split_rows", "num_tokens"),
    [
        (4, 16, 64),  # contiguous fallback
        (4, 16, 256),  # active zigzag
        (16, 64, 1024),  # contiguous fallback
        (16, 64, 4096),  # active zigzag
    ],
)
def test_should_use_global_token_count_in_oproj_op_when_kv_is_sharded(
    monkeypatch: pytest.MonkeyPatch,
    dcp: int,
    split_rows: int,
    num_tokens: int,
) -> None:
    mesh = _make_dcp_mesh(dcp)
    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        page_size=split_rows,
        interleave_size=4,
        split_rows=split_rows,
        max_pages=8,
        zigzag_bands=4,
    )
    spec = SimpleNamespace(
        page_size=layout.page_size,
        layout=SimpleNamespace(value="stub"),
        packed_rope=False,
        token_bytes=64,
    )
    layer, captured = _capture_token_parallel_ops(
        monkeypatch, mesh=mesh, layout=layout, mla_kv_spec=(spec, spec)
    )
    assert layer is not None
    monkeypatch.setattr(tp_op_mod, "make_attend", lambda **_kw: None)

    # Spy on the token count the op body derives outside shard_map and on the
    # per-rank rows its KV write sees inside it; stop before any kernel runs.
    op_token_counts: list[int] = []
    kv_writes: list[tuple[int, int | None]] = []
    orig_rows_per_rank = dcp_token_parallel.TokenParallelLayout.rows_per_rank

    def _spy_rows_per_rank(self, n: int) -> int:
        op_token_counts.append(n)
        return orig_rows_per_rank(self, n)

    def _spy_kv_write(_nope, _rope, kv_c, *_args, **kwargs):
        kv_writes.append((kv_c.shape[0], kwargs.get("zigzag_bands")))
        raise _StopAfterKvWrite

    monkeypatch.setattr(
        dcp_token_parallel.TokenParallelLayout, "rows_per_rank", _spy_rows_per_rank
    )
    monkeypatch.setattr(
        tp_op_mod, "update_sparse_mla_kv_cache_dcp_sharded", _spy_kv_write
    )
    monkeypatch.setattr(tp_op_mod, "update_sparse_mla_kv_cache_dcp", _spy_kv_write)

    unused_shd = _sharded_array(np.zeros((dcp, 1), np.float32), mesh)
    unused_rep = _replicated_array(np.zeros((1,), np.int32), mesh)
    cache = _sharded_array(np.zeros((dcp * 2, layout.page_size, 8), np.uint32), mesh)
    ranks = _sharded_array(np.arange(dcp, dtype=np.int32), mesh)

    for kv_sharded in (False, True):
        matches = [
            r
            for r in captured
            if r["name"].startswith("pallas::sparse_mla_tp_oproj")
            and ("sharded_kv" in r["name"]) == kv_sharded
        ]
        assert len(matches) == 1
        place = _sharded_array if kv_sharded else _replicated_array
        kv_c = place(np.zeros((num_tokens, 32), np.float32), mesh)
        k_pe = place(np.zeros((num_tokens, 8), np.float32), mesh)
        op_token_counts.clear()
        kv_writes.clear()
        with pytest.raises(_StopAfterKvWrite):
            matches[0]["fn"](
                cache,  # nope cache
                cache,  # rope cache
                unused_shd,  # q head
                unused_shd,  # q tail
                unused_shd,  # q_pe
                kv_c,
                k_pe,
                unused_shd,  # top-k rows
                unused_rep,  # seq_lens
                unused_rep,  # block_tables
                unused_rep,  # query_start_loc
                unused_rep,  # request_distribution
                ranks,
                unused_rep,  # w_uv
                unused_shd,  # w_o
                unused_shd,  # w_o scale
            )
        assert op_token_counts[0] == num_tokens
        if kv_sharded:
            assert kv_writes[0] == (
                num_tokens // dcp,
                layout.effective_zigzag_bands(num_tokens),
            )
        else:
            assert kv_writes[0] == (num_tokens, None)

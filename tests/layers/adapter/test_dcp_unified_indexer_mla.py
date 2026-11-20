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

from __future__ import annotations

import inspect
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.layers.adapter import dcp_unified_indexer_mla_op as unified_op_mod
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

AXIS = "dcp"
DN, DR, DV = 8, 4, 6
ROPE_PER_ROW = 4
IDX_ROWS, IDX_SUB = 3, 4
HEADS = 4
TOPK = 12


def _mesh(dcp: int) -> Mesh:
    if jax.device_count() < dcp:
        pytest.skip(f"requires {dcp} devices; CPU virtual devices are sufficient")
    return Mesh(np.array(jax.devices()[:dcp]), (AXIS,))


def _make_step_fixture(
    layout: tp.TokenParallelLayout,
    num_tokens: int,
    reqs: list[tuple[int, int]],
    *,
    max_reqs: int = 8,
    seed: int = 196,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vps, ps, dcp = layout.virtual_page_size, layout.page_size, layout.dcp_size
    num_seqs = len(reqs)
    seq_lens = np.zeros(max_reqs, np.int32)
    qsl = [0]
    for i, (prefix, chunk) in enumerate(reqs):
        seq_lens[i] = prefix + chunk
        qsl.append(qsl[-1] + chunk)
    total = qsl[-1]
    assert total <= num_tokens
    qsl_arr = np.full(max_reqs + 1, total, np.int32)
    qsl_arr[: len(qsl)] = qsl
    dist = np.array([0, 0, num_seqs], np.int32)

    vpages_per_seq = -(-int(seq_lens.max()) // vps) + 1
    num_pages = 4 * max_reqs * vpages_per_seq
    # Anti-cancellation: assign distinct physical block permutations for
    # indexer K-cache vs MLA latent cache so accidental block-table reuse fails.
    idx_perm = rng.permutation(num_pages)
    mla_perm = rng.permutation(num_pages)
    idx_bt = np.zeros((max_reqs, vpages_per_seq), np.int32)
    mla_bt = np.zeros((max_reqs, vpages_per_seq), np.int32)
    cursor = 0
    for i, s in enumerate(seq_lens[:num_seqs]):
        k = -(-int(s) // vps)
        idx_bt[i, :k] = idx_perm[cursor : cursor + k] + num_pages * (i % 2)
        mla_bt[i, :k] = mla_perm[cursor : cursor + k] + num_pages * ((i + 1) % 2)
        cursor += k

    nope_c = np.zeros((dcp, num_pages, ps, DN), np.float32)
    rope_c = np.zeros(
        (dcp, num_pages, ps // ROPE_PER_ROW, ROPE_PER_ROW, DR), np.float32
    )
    idx_c = np.zeros((dcp, num_pages, IDX_ROWS, IDX_SUB, ps), np.uint8)
    for i, (prefix, chunk) in enumerate(reqs):
        s_len = prefix + chunk
        nope_seq = rng.standard_normal((s_len, DN)).astype(np.float32)
        rope_seq = rng.standard_normal((s_len, DR)).astype(np.float32)
        idx_seq = rng.integers(1, 255, (s_len, IDX_ROWS * IDX_SUB), dtype=np.uint8)
        for p in range(s_len):
            owner, slot = (int(v) for v in tp.owner_and_slot(layout, np.int32(p)))
            idx_pg = idx_bt[i, p // vps] % num_pages
            mla_pg = mla_bt[i, p // vps] % num_pages
            idx_c[owner, idx_pg, :, :, slot] = idx_seq[p].reshape(IDX_ROWS, IDX_SUB)
            nope_c[owner, mla_pg, slot] = nope_seq[p]
            rope_c[owner, mla_pg, slot // ROPE_PER_ROW, slot % ROPE_PER_ROW] = rope_seq[
                p
            ]

    rows_per_rank = layout.rows_per_rank(num_tokens)
    return {
        "idx_cache": idx_c.reshape(dcp * num_pages, IDX_ROWS, IDX_SUB, ps),
        "nope_cache": nope_c.reshape(dcp * num_pages, ps, DN),
        "rope_cache": rope_c.reshape(dcp * num_pages, ps // ROPE_PER_ROW, -1),
        "idx_bt": idx_bt.reshape(-1),
        "mla_bt": mla_bt.reshape(-1),
        "seq_lens": seq_lens,
        "qsl": qsl_arr,
        "dist": dist,
        "q_idx": rng.standard_normal(
            (dcp * rows_per_rank, 2, IDX_ROWS * IDX_SUB)
        ).astype(np.float32),
        "w_idx": rng.uniform(0.1, 1.0, (dcp * rows_per_rank, 2)).astype(np.float32),
        "q": rng.standard_normal((dcp * rows_per_rank, HEADS, DN)).astype(np.float32),
        "q_pe": rng.standard_normal((dcp * rows_per_rank, HEADS, DR)).astype(
            np.float32
        ),
        "w_uv": rng.standard_normal((HEADS, DN, DV)).astype(np.float32),
        "lead_partial": rng.standard_normal(
            (dcp * layout.split_rows, HEADS, DN)
        ).astype(np.float32),
        "lead_lse": rng.standard_normal((dcp * layout.split_rows, HEADS)).astype(
            np.float32
        ),
        "rank_ids": np.arange(dcp, dtype=np.int32),
    }


def _ref_select_topk(q_r, w_r, natural, rr, table):
    # Score every query row against every natural-order key slot in its own
    # sequence via table[req, vpage], then select top-TOPK causal positions.
    n_rows = q_r.shape[0]
    nat_tok = natural.reshape(-1, natural.shape[1] * natural.shape[2], natural.shape[3])
    nat_tok = jnp.swapaxes(nat_tok, 1, 2).astype(jnp.float32)  # [cap*dcp, ps, D]
    ps = natural.shape[3]
    table_2d = table.reshape(rr.seq_lens.shape[0], -1)
    vpages = table_2d.shape[1]
    max_pos = vpages * ps
    req_ids = jnp.minimum(
        jnp.searchsorted(rr.cu_q_lens[1:], jnp.arange(n_rows), side="right"),
        rr.seq_lens.shape[0] - 1,
    )
    req_pages = table_2d[req_ids]  # [n_rows, vpages]
    gathered_k = nat_tok[req_pages].reshape(n_rows, max_pos, -1)  # [n_rows, max_pos, D]
    q_proj = jnp.einsum("rhd,rh->rd", q_r, w_r)
    scores = jnp.einsum("rd,rkd->rk", q_proj, gathered_k)
    q_len_req = rr.cu_q_lens[req_ids + 1] - rr.cu_q_lens[req_ids]
    row_pos = (
        rr.seq_lens[req_ids] - q_len_req + (jnp.arange(n_rows) - rr.cu_q_lens[req_ids])
    )
    pos_grid = jnp.arange(max_pos)[None, :]
    valid = (pos_grid <= row_pos[:, None]) & (pos_grid < rr.seq_lens[req_ids][:, None])
    masked = jnp.where(valid, scores, -1e30)
    _, top_pos = jax.lax.top_k(masked, TOPK)
    return jnp.where(
        (top_pos <= row_pos[:, None]) & (top_pos < rr.seq_lens[req_ids][:, None]),
        top_pos.astype(jnp.int32),
        jnp.int32(-1),
    )


def _ref_attend(q_tok, g_nope, g_rope, flat_topk, num_rows, q_pe=None):
    n_rows = q_tok.shape[0]
    flat_n = g_nope.reshape(-1, DN)
    flat_r = g_rope.reshape(-1, DR)
    safe_idx = jnp.maximum(flat_topk, 0)
    k_n = flat_n[safe_idx]  # [R, K, DN]
    k_r = flat_r[safe_idx]  # [R, K, DR]
    logits = jnp.einsum("rhd,rkd->rhk", q_tok, k_n)
    if q_pe is not None:
        logits = logits + jnp.einsum("rhd,rkd->rhk", q_pe, k_r)
    valid = (flat_topk >= 0)[:, None, :] & (
        jnp.arange(n_rows)[:, None, None] < num_rows
    )
    logits = jnp.where(valid, logits, -1e9)
    probs = jax.nn.softmax(logits, axis=-1)
    probs = jnp.where(valid, probs, 0.0)
    return jnp.einsum("rhk,rkd->rhd", probs, k_n)


@pytest.mark.parametrize("tie_after_idx_max_cap", [0, 4, 8])
def test_should_match_separate_indexer_and_mla_with_distinct_block_tables(
    tie_after_idx_max_cap: int,
):
    mesh = _mesh(4)
    layout = tp.TokenParallelLayout(
        4, 4, 32, split_rows=16, max_pages=8, gather_caps=(2, 4, 6, 8)
    )
    num_tokens = 256
    # Mix decode rows + two prefill chunks crossing split_rows and rank bounds.
    reqs = [(700, 1), (90, 1), (260, 1), (300, 150), (0, 40)]
    fx = _make_step_fixture(layout, num_tokens, reqs)

    shd, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    idx_cache = jax.device_put(jnp.asarray(fx["idx_cache"]), shd)
    nope_cache = jax.device_put(jnp.asarray(fx["nope_cache"]), shd)
    rope_cache = jax.device_put(jnp.asarray(fx["rope_cache"]), shd)
    idx_bt = jax.device_put(jnp.asarray(fx["idx_bt"]), rep)
    mla_bt = jax.device_put(jnp.asarray(fx["mla_bt"]), rep)
    seq_lens = jax.device_put(jnp.asarray(fx["seq_lens"]), rep)
    qsl = jax.device_put(jnp.asarray(fx["qsl"]), rep)
    dist = jax.device_put(jnp.asarray(fx["dist"]), rep)
    q_idx = jax.device_put(jnp.asarray(fx["q_idx"]), shd)
    w_idx = jax.device_put(jnp.asarray(fx["w_idx"]), shd)
    q = jax.device_put(jnp.asarray(fx["q"]), shd)
    q_pe = jax.device_put(jnp.asarray(fx["q_pe"]), shd)
    w_uv = jax.device_put(jnp.asarray(fx["w_uv"]), rep)
    lead_partial = jax.device_put(jnp.asarray(fx["lead_partial"]), shd)
    lead_lse = jax.device_put(jnp.asarray(fx["lead_lse"]), shd)
    rank_ids = jax.device_put(jnp.asarray(fx["rank_ids"]), shd)

    # Independent reference: separate indexer_topk -> sparse_mla_attention_with_lead.
    ref_flat = tp.indexer_topk(
        layout,
        mesh,
        AXIS,
        idx_cache,
        q_idx,
        w_idx,
        seq_lens,
        idx_bt,
        qsl,
        dist,
        select_topk=_ref_select_topk,
        rank_ids=rank_ids,
        q_own_rows=True,
        flat_topk_cap_strided=True,
    )
    _, _, ref_own_v, ref_lead_v = tp.sparse_mla_attention_with_lead(
        layout,
        mesh,
        AXIS,
        nope_cache,
        rope_cache,
        q,
        None,
        None,
        ref_flat,
        seq_lens,
        mla_bt,
        qsl,
        dist,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        write_kv=None,
        attend=_ref_attend,
        w_uv=w_uv,
        rank_ids=rank_ids,
        q_own_rows=True,
        q_pe=q_pe,
        w_uv_replicated=True,
        num_tokens=num_tokens,
        flat_topk_cap_strided=True,
    )

    # Act: unified single cap-switch kernel.
    got_own_v, got_lead_v, got_flat = tp.unified_indexer_mla_with_lead(
        layout,
        mesh,
        AXIS,
        idx_cache,
        q_idx,
        w_idx,
        idx_bt,
        nope_cache,
        rope_cache,
        q,
        q_pe,
        mla_bt,
        seq_lens,
        qsl,
        dist,
        select_topk=_ref_select_topk,
        attend=_ref_attend,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        w_uv_replicated=True,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
    )

    np.testing.assert_array_equal(np.asarray(got_flat), np.asarray(ref_flat))
    np.testing.assert_array_equal(np.asarray(got_own_v), np.asarray(ref_own_v))
    np.testing.assert_array_equal(np.asarray(got_lead_v), np.asarray(ref_lead_v))

    # Fused single-shard_map o_proj path matches project_sequence_rows_from_v bit-exact.
    if tie_after_idx_max_cap == 4:
        rng_o = np.random.default_rng(910)
        k_full = ref_own_v.shape[1] * ref_own_v.shape[2]
        half_k = k_full // 2
        hidden = 16
        w_o_fp8 = jnp.asarray(
            rng_o.standard_normal((k_full, hidden)) * 0.5, jnp.float8_e4m3fn
        )
        w_o_shd = jax.device_put(
            jnp.concatenate(
                [
                    w_o_fp8[:half_k] if r % 2 == 0 else w_o_fp8[half_k:]
                    for r in range(4)
                ],
                axis=0,
            ),
            shd,
        )
        s_o_shd = jax.device_put(
            jnp.tile(
                jnp.asarray(rng_o.uniform(0.01, 0.05, (hidden,)), jnp.float32), (4,)
            ),
            shd,
        )
        ref_proj = tp.project_sequence_rows_from_v(
            layout, mesh, AXIS, ref_lead_v, ref_own_v, rank_ids, w_o_shd, s_o_shd
        )
        got_proj, got_flat_o = tp.unified_indexer_mla_with_lead_oproj(
            layout,
            mesh,
            AXIS,
            idx_cache,
            q_idx,
            w_idx,
            idx_bt,
            nope_cache,
            rope_cache,
            q,
            q_pe,
            mla_bt,
            seq_lens,
            qsl,
            dist,
            select_topk=_ref_select_topk,
            attend=_ref_attend,
            lead_partial=lead_partial,
            lead_lse=lead_lse,
            w_uv=w_uv,
            w_o=w_o_shd,
            w_o_scale=s_o_shd,
            num_tokens=num_tokens,
            rank_ids=rank_ids,
            w_uv_replicated=True,
            tie_after_idx_max_cap=tie_after_idx_max_cap,
            epilogue_3d=True,
        )
        np.testing.assert_array_equal(np.asarray(got_flat_o), np.asarray(ref_flat))
        np.testing.assert_array_equal(
            np.asarray(got_proj).view(np.uint16), np.asarray(ref_proj).view(np.uint16)
        )
    # Anti-cancellation check: reusing mla_bt for idx_bt MUST diverge.
    wrong_own_v, _, wrong_flat = tp.unified_indexer_mla_with_lead(
        layout,
        mesh,
        AXIS,
        idx_cache,
        q_idx,
        w_idx,
        mla_bt,
        nope_cache,
        rope_cache,
        q,
        q_pe,
        mla_bt,
        seq_lens,
        qsl,
        dist,
        select_topk=_ref_select_topk,
        attend=_ref_attend,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        w_uv_replicated=True,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
    )
    assert not np.array_equal(np.asarray(wrong_flat), np.asarray(ref_flat))
    assert not np.allclose(np.asarray(wrong_own_v), np.asarray(ref_own_v))


def test_should_return_none_from_builder_when_unified_env_is_disabled(monkeypatch):
    monkeypatch.setenv("TPU_DCP_TP_UNIFIED_INDEXER_MLA", "0")
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8)
    dummy_cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", None),  # type: ignore[arg-type]
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((1, 1, 1), (1, 1, 1)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda prefix: 0,
    )
    built = unified_op_mod.build_unified_indexer_attention_op(
        layer=types.SimpleNamespace(),
        impl=types.SimpleNamespace(),
        indexer_op=types.SimpleNamespace(),
        layout=layout,
        q_replicated=True,
        callbacks=dummy_cb,
    )
    assert built is None


def test_should_dispatch_unified_attention_and_update_topk_buffer_in_place():
    written: list[tuple[int, torch.Tensor]] = []
    captured_calls: list[dict[str, Any]] = []

    class _FakeIndexerOp:
        def __init__(self) -> None:
            self.k_cache = types.SimpleNamespace(
                kv_cache=torch.ones((2, 3, 4, 8), dtype=torch.uint8)
            )

        def _write_topk_rows(self, num_rows: int, rows: torch.Tensor) -> torch.Tensor:
            written.append((num_rows, rows.clone()))
            return rows

    expected_topk = torch.arange(20 * 4, dtype=torch.int32).reshape(20, 4)
    expected_out = torch.full((16, 6), 3.5, dtype=torch.bfloat16)

    def _fake_unified_op(
        index_cache,
        kv_cache,
        q_idx_bytes,
        w_idx,
        k_packed,
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        seq_lens,
        idx_bt,
        mla_bt,
        qsl,
        idx_dist,
        mla_dist,
        w_uv,
        w_o,
        *,
        kv_sharded: bool = False,
        k_sharded: bool = False,
    ):
        captured_calls.append(
            {
                "idx_bt": idx_bt.clone(),
                "mla_bt": mla_bt.clone(),
                "idx_dist": idx_dist.clone(),
                "mla_dist": mla_dist.clone(),
                "kv_sharded": kv_sharded,
                "k_sharded": k_sharded,
                "w_uv_dtype": w_uv.dtype,
            }
        )
        return expected_topk, expected_out

    fake_layer = types.SimpleNamespace(
        sparse_mla_unified_tp_op=_fake_unified_op,
        W_UV_tp_all=torch.ones((4, 8, 6), dtype=torch.float32),
    )
    fake_idx_op = _FakeIndexerOp()
    idx_md = types.SimpleNamespace(
        block_tables=torch.tensor([11, 13, 17, 19], dtype=torch.int32),
        request_distribution=torch.tensor([2, 3, 5], dtype=torch.int32),
    )
    mla_md = types.SimpleNamespace(
        seq_lens=torch.tensor([64, 32], dtype=torch.int32),
        block_tables=torch.tensor([2, 4, 6, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 16, 20], dtype=torch.int32),
        request_distribution=torch.tensor([2, 3, 5], dtype=torch.int32),
    )
    unified_in = unified_op_mod.UnifiedIndexerInputs(
        indexer_op=fake_idx_op,
        q_bytes=torch.zeros((20, 2, 12), dtype=torch.uint8),
        weights=torch.ones((20, 2), dtype=torch.bfloat16),
        k_packed=torch.zeros((64, 13), dtype=torch.uint8),
        idx_metadata=idx_md,
    )
    q_split = (
        torch.zeros((4, 4, 8), dtype=torch.bfloat16),
        torch.zeros((16, 4, 8), dtype=torch.bfloat16),
    )
    out = unified_op_mod.unified_indexer_attention(
        fake_layer,
        (torch.zeros((2, 8, 8)), torch.zeros((2, 2, 16))),
        q_split,
        torch.zeros((20, 4, 4), dtype=torch.bfloat16),
        torch.zeros((16, 8), dtype=torch.bfloat16),
        torch.zeros((16, 4), dtype=torch.bfloat16),
        unified_in,
        mla_md,
        w_o=(torch.zeros((8, 6)), torch.ones((6,))),
        kv_sharded=True,
    )

    assert torch.equal(out, expected_out)
    assert len(captured_calls) == 1
    assert torch.equal(captured_calls[0]["idx_bt"], idx_md.block_tables)
    assert torch.equal(captured_calls[0]["mla_bt"], mla_md.block_tables)
    assert torch.equal(
        captured_calls[0]["idx_dist"], torch.tensor([2, 2, 5], dtype=torch.int32)
    )
    assert torch.equal(captured_calls[0]["mla_dist"], mla_md.request_distribution)
    assert captured_calls[0]["kv_sharded"] is True
    assert captured_calls[0]["w_uv_dtype"] == torch.bfloat16
    assert len(written) == 1
    assert written[0][0] == 20
    assert torch.equal(written[0][1], expected_topk)


def test_should_reject_missing_unified_op_on_layer():
    fake_layer = types.SimpleNamespace(sparse_mla_unified_tp_op=None)
    dummy_in = unified_op_mod.UnifiedIndexerInputs(
        indexer_op=types.SimpleNamespace(),
        q_bytes=torch.zeros((4, 2, 12), dtype=torch.uint8),
        weights=torch.ones((4, 2), dtype=torch.bfloat16),
        k_packed=torch.zeros((4, 13), dtype=torch.uint8),
        idx_metadata=types.SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="sparse_mla_unified_tp_op was not built"):
        unified_op_mod.unified_indexer_attention(
            fake_layer,
            (torch.zeros((1,)), torch.zeros((1,))),
            torch.zeros((4, 4, 8), dtype=torch.bfloat16),
            torch.zeros((4, 4, 4), dtype=torch.bfloat16),
            torch.zeros((4, 8), dtype=torch.bfloat16),
            torch.zeros((4, 4), dtype=torch.bfloat16),
            dummy_in,
            types.SimpleNamespace(),
            w_o=(torch.zeros((4, 4)), torch.ones((4,))),
        )


def test_should_gate_live_unified_cache_and_pack_indexer_k(monkeypatch):
    indexer_stub = mla_op.VllmTPUSparseAttnIndexer.__new__(
        mla_op.VllmTPUSparseAttnIndexer
    )
    indexer_stub.k_cache = types.SimpleNamespace(
        prefix="layer0.idx", kv_cache=torch.empty(0, dtype=torch.uint8)
    )
    meta = mla_op.AttentionMetadata.__new__(mla_op.AttentionMetadata)
    monkeypatch.setattr(
        mla_op,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"layer0.idx": meta}),
    )
    q_fp8 = torch.zeros((4, 2, 16), dtype=torch.float8_e4m3fn)
    k_bf16 = torch.ones((8, 16), dtype=torch.bfloat16)
    w_bf16 = torch.ones((4, 2), dtype=torch.bfloat16)

    # Empty cache (determine_available_memory probe): must report False / None.
    assert not indexer_stub.has_live_unified_cache()
    assert indexer_stub.prepare_unified_inputs(q_fp8, k_bf16, w_bf16) is None

    # Non-empty cache: must pack fp8 + 1-byte e8m0 scale ([8, 17] uint8).
    indexer_stub.k_cache.kv_cache = torch.ones((2, 3, 4, 8), dtype=torch.uint8)
    assert indexer_stub.has_live_unified_cache()
    prepared = indexer_stub.prepare_unified_inputs(q_fp8, k_bf16, w_bf16)
    assert prepared is not None
    assert prepared.q_bytes.shape == (4, 2, 16)
    assert prepared.q_bytes.dtype == torch.uint8
    assert prepared.k_packed.shape == (8, 17)
    assert prepared.k_packed.dtype == torch.uint8
    assert prepared.idx_metadata is meta


@pytest.mark.parametrize("has_lane", [False, True])
def test_should_pass_torch_tpu_pallas_verify_signature_for_unified_op_wrappers(
    monkeypatch, has_lane: bool
):
    mesh = _mesh(4)
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8)
    monkeypatch.setenv("TPU_DCP_TP_UNIFIED_INDEXER_MLA", "1")
    monkeypatch.setenv("TPU_DCP_TP_PREFILL_LOCAL_OPROJ", "1")
    monkeypatch.setenv("TPU_DCP_RANK_OPERAND", "1")
    monkeypatch.setenv("TPU_DCP_TP_SHARDED_KV_WRITE", "1")
    monkeypatch.setattr(
        unified_op_mod, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        unified_op_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )
    monkeypatch.setattr(
        unified_op_mod,
        "dcp_lane_operand",
        lambda: torch.zeros((1,), dtype=torch.int32) if has_lane else None,
    )

    registered: list[tuple[str, Any, tuple[Any, ...], tuple[Any, ...]]] = []

    class _FakeBuiltOp:
        def __init__(self) -> None:
            self.fake_fn: Any = None

        def register_fake(self, fn: Any) -> Any:
            self.fake_fn = fn
            return fn

    def _capture_jax_op(
        name: str,
        fn: Any,
        *,
        donate_argnums: tuple[int, ...],
        mesh: Any,
        input_partition_specs: tuple[Any, ...],
        output_partition_specs: tuple[Any, ...],
    ) -> _FakeBuiltOp:
        del donate_argnums, mesh
        sig = inspect.signature(fn)
        pallas_impl._verify_signature(sig)  # noqa: SLF001 -- real torch_tpu op-registration signature check
        assert pallas_impl._infer_static_argnums(sig) == ()  # noqa: SLF001 -- every operand must bind as jax.Array
        pallas_impl._get_torch_signature(sig)  # noqa: SLF001 -- verify Torch schema translation succeeds
        assert len(sig.parameters) == len(input_partition_specs)
        registered.append((name, fn, input_partition_specs, output_partition_specs))
        return _FakeBuiltOp()

    monkeypatch.setattr(unified_op_mod, "pcp_streaming_jax_op", _capture_jax_op)
    monkeypatch.setattr(
        unified_op_mod,
        "get_shared_jax_op",
        lambda kind, key, build: build(f"p::{kind}"),
    )

    class _FakeLayer:
        kv_cache_quantized_dtype = None
        scale = 0.135
        num_heads = 4
        v_head_dim = 6
        use_sparse = True

    layer = _FakeLayer()
    impl = types.SimpleNamespace(
        _dcp_mesh=mesh,
        dcp_size=4,
        dcp_interleave_size=4,
    )
    indexer_op = types.SimpleNamespace(
        topk_tokens=12,
        k_cache=types.SimpleNamespace(prefix="layer0.idx"),
    )
    cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", None),  # type: ignore[arg-type]
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((1, 1, 1), (1, 1, 1)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda prefix: 0,
    )

    op = unified_op_mod.build_unified_indexer_attention_op(
        layer, impl, indexer_op, layout, q_replicated=True, callbacks=cb
    )
    assert op is not None
    assert getattr(op, "serves_sharded_k", False) is True
    assert len(registered) == 3
    expected_arity = 22 if has_lane else 21
    assert all(len(specs) == expected_arity for _, _, specs, _ in registered)

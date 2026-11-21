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

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.layers.adapter import attention as attn_mod
from vllm_torchtpu.layers.adapter import dcp_nonidx_qb_mla_op as in_branch_mod
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter import dcp_unified_idx_qb_mla_op as unified_qb_mod
from vllm_torchtpu.layers.adapter import dcp_unified_indexer_mla_op as unified_op_mod
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

AXIS = "dcp"
Q_LORA = 12
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
    seed: int = 1101,
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
    idx_perm = rng.permutation(num_pages)
    mla_perm = rng.permutation(num_pages)
    idx_bt = np.zeros((max_reqs, vpages_per_seq), np.int32)
    mla_bt = np.zeros((max_reqs, vpages_per_seq), np.int32)
    cursor = 0
    for i, s_len in enumerate(seq_lens[:num_seqs]):
        k = -(-int(s_len) // vps)
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
    own_rows = layout.split_rows + rows_per_rank
    # Shared lead rows [0..split_rows) must match across ranks; tail rows vary per rank.
    q_c_head = rng.standard_normal((layout.split_rows, Q_LORA)).astype(np.float32)
    q_c_tail = rng.standard_normal((dcp, rows_per_rank, Q_LORA)).astype(np.float32)
    q_c_own = np.concatenate(
        [np.broadcast_to(q_c_head[None], (dcp, layout.split_rows, Q_LORA)), q_c_tail],
        axis=1,
    ).reshape(dcp * own_rows, Q_LORA)
    cs_head = rng.standard_normal((layout.split_rows, DR)).astype(np.float32)
    cs_tail = rng.standard_normal((dcp, rows_per_rank, DR)).astype(np.float32)
    cos_sin = np.concatenate(
        [np.broadcast_to(cs_head[None], (dcp, layout.split_rows, DR)), cs_tail],
        axis=1,
    ).reshape(dcp * own_rows, DR)

    head_topk = rng.integers(
        -1, 64, size=(dcp * layout.split_rows, TOPK), dtype=np.int32
    )
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
        "q_c_own": q_c_own,
        "w_qb": rng.standard_normal((Q_LORA, HEADS * (DN + DR))).astype(np.float32),
        "s_qb": rng.uniform(0.5, 1.5, (HEADS * (DN + DR),)).astype(np.float32),
        "w_uk_t": rng.standard_normal((HEADS, DN, DN)).astype(np.float32),
        "cos_sin": cos_sin,
        "head_topk": head_topk,
        "w_uv": rng.standard_normal((HEADS, DN, DV)).astype(np.float32),
        "rank_ids": np.arange(dcp, dtype=np.int32),
    }


def _ref_fused_qb(q_c, w_qb, s_qb, w_uk_t, cos_sin, split_rows: int):
    q_proj = (q_c.astype(jnp.float32) @ w_qb.astype(jnp.float32)) * s_qb
    q_proj = q_proj.reshape(q_c.shape[0], HEADS, DN + DR)
    q_nope = q_proj[:, :, :DN].astype(q_c.dtype)
    q_pe = (
        q_proj[:, :, DN:] * (1.0 + 0.1 * cos_sin.astype(jnp.float32)[:, None, :])
    ).astype(q_c.dtype)
    ql = jnp.einsum(
        "rhd,hdk->rhk",
        q_nope.astype(jnp.float32),
        w_uk_t.astype(jnp.float32),
    ).astype(q_c.dtype)
    return ql[:split_rows], ql[split_rows:], q_pe[:split_rows], q_pe[split_rows:]


def _ref_lead_attend(ql_head, qpe_head, nope, rope, h_topk, bt, sl, qsl, dist):
    del bt, sl, qsl, dist
    nope_s = jnp.mean(nope.astype(jnp.float32), axis=(0, 1))
    rope_s = jnp.mean(rope.astype(jnp.float32).reshape(-1, DR), axis=0)
    k_bias = jnp.mean(jnp.maximum(h_topk, 0).astype(jnp.float32), axis=-1)
    partial = (
        ql_head.astype(jnp.float32) * (1.0 + 0.05 * nope_s[None, None, :])
        + 0.01 * k_bias[:, None, None]
    ).astype(ql_head.dtype)
    lse = (
        jnp.sum(qpe_head.astype(jnp.float32) * rope_s[None, None, :], axis=-1)
        + 0.02 * k_bias[:, None]
    )
    return partial, lse


def _ref_select_topk(q_r, w_r, natural, rr, table):
    n_rows = q_r.shape[0]
    nat_tok = natural.reshape(-1, natural.shape[1] * natural.shape[2], natural.shape[3])
    nat_tok = jnp.swapaxes(nat_tok, 1, 2).astype(jnp.float32)
    ps = natural.shape[3]
    table_2d = table.reshape(rr.seq_lens.shape[0], -1)
    vpages = table_2d.shape[1]
    max_pos = vpages * ps
    req_ids = jnp.minimum(
        jnp.searchsorted(rr.cu_q_lens[1:], jnp.arange(n_rows), side="right"),
        rr.seq_lens.shape[0] - 1,
    )
    req_pages = table_2d[req_ids]
    gathered_k = nat_tok[req_pages].reshape(n_rows, max_pos, -1)
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
    flat_n = g_nope.reshape(-1, DN).astype(jnp.float32)
    flat_r = g_rope.reshape(-1, DR).astype(jnp.float32)
    safe_idx = jnp.maximum(flat_topk, 0)
    k_n = flat_n[safe_idx]
    k_r = flat_r[safe_idx]
    logits = jnp.einsum("rhd,rkd->rhk", q_tok.astype(jnp.float32), k_n)
    if q_pe is not None:
        logits = logits + jnp.einsum("rhd,rkd->rhk", q_pe.astype(jnp.float32), k_r)
    valid = (flat_topk >= 0)[:, None, :] & (
        jnp.arange(n_rows)[:, None, None] < num_rows
    )
    logits = jnp.where(valid, logits, -1e9)
    probs = jax.nn.softmax(logits, axis=-1)
    probs = jnp.where(valid, probs, 0.0)
    return jnp.einsum("rhk,rkd->rhd", probs, k_n).astype(q_tok.dtype)


@pytest.mark.parametrize(
    ("tie_after_idx_max_cap", "tie_lead_min_cap"),
    [(0, 0), (4, 0), (8, 4)],
)
def test_should_match_separate_qb_indexer_lead_and_tail_mla_bit_exact(
    tie_after_idx_max_cap: int,
    tie_lead_min_cap: int,
):
    mesh = _mesh(4)
    layout = tp.TokenParallelLayout(
        4, 4, 32, split_rows=16, max_pages=8, gather_caps=(2, 4, 6, 8)
    )
    num_tokens = 256
    reqs = [(700, 1), (90, 1), (260, 1), (300, 150), (0, 40)]
    fx = _make_step_fixture(layout, num_tokens, reqs)

    shd, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    idx_cache = jax.device_put(jnp.asarray(fx["idx_cache"]), shd)
    nope_cache = jax.device_put(jnp.asarray(fx["nope_cache"], dtype=jnp.bfloat16), shd)
    rope_cache = jax.device_put(jnp.asarray(fx["rope_cache"], dtype=jnp.bfloat16), shd)
    idx_bt = jax.device_put(jnp.asarray(fx["idx_bt"]), rep)
    mla_bt = jax.device_put(jnp.asarray(fx["mla_bt"]), rep)
    seq_lens = jax.device_put(jnp.asarray(fx["seq_lens"]), rep)
    qsl = jax.device_put(jnp.asarray(fx["qsl"]), rep)
    dist = jax.device_put(jnp.asarray(fx["dist"]), rep)
    q_idx = jax.device_put(jnp.asarray(fx["q_idx"]), shd)
    w_idx = jax.device_put(jnp.asarray(fx["w_idx"]), shd)
    q_c_own = jax.device_put(jnp.asarray(fx["q_c_own"], dtype=jnp.bfloat16), shd)
    w_qb = jax.device_put(jnp.asarray(fx["w_qb"], dtype=jnp.float8_e4m3fn), rep)
    s_qb = jax.device_put(jnp.asarray(fx["s_qb"], dtype=jnp.float32), rep)
    w_uk_t = jax.device_put(jnp.asarray(fx["w_uk_t"], dtype=jnp.bfloat16), rep)
    cos_sin = jax.device_put(jnp.asarray(fx["cos_sin"], dtype=jnp.bfloat16), shd)
    head_topk = jax.device_put(jnp.asarray(fx["head_topk"]), shd)
    w_uv = jax.device_put(jnp.asarray(fx["w_uv"], dtype=jnp.bfloat16), rep)
    rank_ids = jax.device_put(jnp.asarray(fx["rank_ids"]), shd)

    # Independent reference: eager fused_qb + lead_attend -> separate indexer -> MLA.
    def _ref_qb_and_lead(
        qc_r, wqb_r, sqb_r, wuk_r, cs_r, nope_r, rope_r, ht_r, bt, sl, qsl_r, d_r
    ):
        ql_h, ql_t, qpe_h, qpe_t = _ref_fused_qb(
            qc_r, wqb_r, sqb_r, wuk_r, cs_r, split_rows=layout.split_rows
        )
        lp, ll = _ref_lead_attend(ql_h, qpe_h, nope_r, rope_r, ht_r, bt, sl, qsl_r, d_r)
        return ql_t, qpe_t, lp, ll

    p_shd, p_rep = P(AXIS), P()
    ref_ql_tail, ref_qpe_tail, ref_lp, ref_ll = jax.shard_map(
        _ref_qb_and_lead,
        mesh=mesh,
        in_specs=(
            p_shd,
            p_rep,
            p_rep,
            p_rep,
            p_shd,
            p_shd,
            p_shd,
            p_shd,
            p_rep,
            p_rep,
            p_rep,
            p_rep,
        ),
        out_specs=(p_shd, p_shd, p_shd, p_shd),
        check_vma=False,
    )(
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        nope_cache,
        rope_cache,
        head_topk,
        mla_bt,
        seq_lens,
        qsl,
        dist,
    )
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
        ref_ql_tail,
        None,
        None,
        ref_flat,
        seq_lens,
        mla_bt,
        qsl,
        dist,
        lead_partial=ref_lp,
        lead_lse=ref_ll,
        write_kv=None,
        attend=_ref_attend,
        w_uv=w_uv,
        rank_ids=rank_ids,
        q_own_rows=True,
        q_pe=ref_qpe_tail,
        w_uv_replicated=True,
        num_tokens=num_tokens,
        flat_topk_cap_strided=True,
    )

    got_own_v, got_lead_v, got_flat = tp.unified_indexer_in_branch_qb_mla(
        layout,
        mesh,
        AXIS,
        idx_cache,
        q_idx,
        w_idx,
        idx_bt,
        nope_cache,
        rope_cache,
        q_c_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        mla_bt,
        seq_lens,
        qsl,
        dist,
        select_topk=_ref_select_topk,
        fused_qb=_ref_fused_qb,
        lead_attend=_ref_lead_attend,
        tail_attend=_ref_attend,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        w_uv_replicated=True,
        tie_after_idx_max_cap=tie_after_idx_max_cap,
        tie_lead_min_cap=tie_lead_min_cap,
    )

    np.testing.assert_array_equal(np.asarray(got_flat), np.asarray(ref_flat))
    np.testing.assert_array_equal(
        np.asarray(got_own_v).view(np.uint16), np.asarray(ref_own_v).view(np.uint16)
    )
    np.testing.assert_array_equal(
        np.asarray(got_lead_v).view(np.uint16), np.asarray(ref_lead_v).view(np.uint16)
    )

    if tie_after_idx_max_cap == 4:
        rng_o = np.random.default_rng(1102)
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
        for epilogue_3d in (False, True):
            got_proj, got_flat_o = tp.unified_indexer_in_branch_qb_mla_oproj(
                layout,
                mesh,
                AXIS,
                idx_cache,
                q_idx,
                w_idx,
                idx_bt,
                nope_cache,
                rope_cache,
                q_c_own,
                w_qb,
                s_qb,
                w_uk_t,
                cos_sin,
                head_topk,
                mla_bt,
                seq_lens,
                qsl,
                dist,
                select_topk=_ref_select_topk,
                fused_qb=_ref_fused_qb,
                lead_attend=_ref_lead_attend,
                tail_attend=_ref_attend,
                w_uv=w_uv,
                w_o=w_o_shd,
                w_o_scale=s_o_shd,
                num_tokens=num_tokens,
                rank_ids=rank_ids,
                w_uv_replicated=True,
                tie_after_idx_max_cap=tie_after_idx_max_cap,
                tie_lead_min_cap=tie_lead_min_cap,
                epilogue_3d=epilogue_3d,
            )
            np.testing.assert_array_equal(np.asarray(got_flat_o), np.asarray(ref_flat))
            np.testing.assert_array_equal(
                np.asarray(got_proj).view(np.uint16),
                np.asarray(ref_proj).view(np.uint16),
            )

        # Anti-cancellation: swapping idx_bt -> mla_bt or flipping q_c must diverge.
        wrong_bt_v, _, wrong_bt_flat = tp.unified_indexer_in_branch_qb_mla(
            layout,
            mesh,
            AXIS,
            idx_cache,
            q_idx,
            w_idx,
            mla_bt,
            nope_cache,
            rope_cache,
            q_c_own,
            w_qb,
            s_qb,
            w_uk_t,
            cos_sin,
            head_topk,
            mla_bt,
            seq_lens,
            qsl,
            dist,
            select_topk=_ref_select_topk,
            fused_qb=_ref_fused_qb,
            lead_attend=_ref_lead_attend,
            tail_attend=_ref_attend,
            w_uv=w_uv,
            num_tokens=num_tokens,
            rank_ids=rank_ids,
            w_uv_replicated=True,
            tie_after_idx_max_cap=tie_after_idx_max_cap,
            tie_lead_min_cap=tie_lead_min_cap,
        )
        assert not np.array_equal(np.asarray(wrong_bt_flat), np.asarray(ref_flat))
        assert not np.allclose(np.asarray(wrong_bt_v), np.asarray(ref_own_v))

        neg_own_v, neg_lead_v, _ = tp.unified_indexer_in_branch_qb_mla(
            layout,
            mesh,
            AXIS,
            idx_cache,
            q_idx,
            w_idx,
            idx_bt,
            nope_cache,
            rope_cache,
            -q_c_own,
            w_qb,
            s_qb,
            w_uk_t,
            cos_sin,
            head_topk,
            mla_bt,
            seq_lens,
            qsl,
            dist,
            select_topk=_ref_select_topk,
            fused_qb=_ref_fused_qb,
            lead_attend=_ref_lead_attend,
            tail_attend=_ref_attend,
            w_uv=w_uv,
            num_tokens=num_tokens,
            rank_ids=rank_ids,
            w_uv_replicated=True,
            tie_after_idx_max_cap=tie_after_idx_max_cap,
            tie_lead_min_cap=tie_lead_min_cap,
        )
        assert not np.allclose(np.asarray(neg_own_v), np.asarray(ref_own_v))
        assert not np.allclose(np.asarray(neg_lead_v), np.asarray(ref_lead_v))


@pytest.mark.parametrize(
    "disabled_env",
    [
        "TPU_DCP_TP_UNIFIED_INDEXER_IN_BRANCH_QB",
        "TPU_DCP_TP_UNIFIED_INDEXER_MLA",
    ],
)
def test_should_return_none_from_builder_when_gate_env_disabled(
    monkeypatch, disabled_env: str
):
    monkeypatch.setenv(disabled_env, "0")
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8)
    dummy_cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", None),  # type: ignore[arg-type]
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((1, 1, 1), (1, 1, 1)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda prefix: 0,
    )
    built = unified_qb_mod.build_unified_indexer_in_branch_qb_op(
        layer=types.SimpleNamespace(),
        impl=types.SimpleNamespace(),
        indexer_op=types.SimpleNamespace(),
        layout=layout,
        q_replicated=True,
        callbacks=dummy_cb,
    )
    assert built is None


@pytest.mark.parametrize("has_lane", [False, True])
def test_should_pass_torch_tpu_pallas_verify_signature_for_unified_idx_qb_wrappers(
    monkeypatch, has_lane: bool
):
    mesh = _mesh(4)
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8)
    monkeypatch.setenv("TPU_DCP_TP_UNIFIED_INDEXER_MLA", "1")
    monkeypatch.setenv("TPU_DCP_TP_UNIFIED_INDEXER_IN_BRANCH_QB", "1")
    monkeypatch.setenv("TPU_DCP_TP_PREFILL_LOCAL_OPROJ", "1")
    monkeypatch.setenv("TPU_DCP_RANK_OPERAND", "1")
    monkeypatch.setenv("TPU_DCP_TP_SHARDED_KV_WRITE", "1")
    monkeypatch.setattr(
        unified_qb_mod, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        unified_qb_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )
    monkeypatch.setattr(
        unified_qb_mod,
        "dcp_lane_operand",
        lambda: torch.zeros((1,), dtype=torch.int32) if has_lane else None,
    )

    registered: list[tuple[str, Any, Any, tuple[Any, ...], tuple[Any, ...]]] = []

    class _FakeBuiltOp:
        def __init__(self, name: str) -> None:
            self.name = name
            self.fake_fn: Any = None
            self.calls: int = 0

        def register_fake(self, fn: Any) -> Any:
            self.fake_fn = fn
            return fn

        def __call__(self, *args: Any) -> tuple[torch.Tensor, ...]:
            self.calls += 1
            return self.fake_fn(*args)

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
        built_op = _FakeBuiltOp(name)
        registered.append(
            (name, fn, built_op, input_partition_specs, output_partition_specs)
        )
        return built_op

    monkeypatch.setattr(unified_qb_mod, "pcp_streaming_jax_op", _capture_jax_op)
    monkeypatch.setattr(
        unified_qb_mod,
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
    impl = types.SimpleNamespace(_dcp_mesh=mesh, dcp_size=4, dcp_interleave_size=4)
    indexer_op = types.SimpleNamespace(
        topk_tokens=TOPK, k_cache=types.SimpleNamespace(prefix="layer0.idx")
    )
    cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", None),  # type: ignore[arg-type]
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((1, 1, 1), (1, 1, 1)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda prefix: 0,
    )

    op = unified_qb_mod.build_unified_indexer_in_branch_qb_op(
        layer, impl, indexer_op, layout, q_replicated=True, callbacks=cb
    )
    assert op is not None
    assert getattr(op, "serves_sharded_kv", False) is True
    assert getattr(op, "serves_sharded_k", False) is True
    assert len(registered) == 3
    expected_arity = 24 if has_lane else 23
    for name, _, built_op, in_specs, out_specs in registered:
        assert len(in_specs) == expected_arity
        assert len(out_specs) == 5
        is_sharded_k = "sharded_kv_k" in name
        assert in_specs[5] == (P(AXIS) if is_sharded_k else P())
        is_sharded = "sharded_kv" in name
        kv_rows = 16 if is_sharded else 64
        k_rows = 16 if is_sharded_k else 64
        lane_args = (torch.zeros((1,), dtype=torch.int32),) if has_lane else ()
        fake_outs = built_op.fake_fn(
            torch.zeros((4, 3, 4, 32), dtype=torch.uint8),
            torch.zeros((4, 32, 8), dtype=torch.bfloat16),
            torch.zeros((4, 8, 16), dtype=torch.bfloat16),
            torch.zeros((32, 2, 12), dtype=torch.uint8),
            torch.ones((32, 2), dtype=torch.bfloat16),
            torch.zeros((k_rows, 13), dtype=torch.uint8),
            torch.zeros((32, 12), dtype=torch.bfloat16),
            torch.zeros((12, 48), dtype=torch.float8_e4m3fn),
            torch.ones((48,), dtype=torch.float32),
            torch.ones((4, 8, 8), dtype=torch.bfloat16),
            torch.zeros((32, 4), dtype=torch.bfloat16),
            torch.zeros((kv_rows, 8), dtype=torch.bfloat16),
            torch.zeros((kv_rows, 4), dtype=torch.bfloat16),
            torch.zeros((8,), dtype=torch.int32),
            torch.zeros((16,), dtype=torch.int32),
            torch.zeros((16,), dtype=torch.int32),
            torch.zeros((9,), dtype=torch.int32),
            torch.zeros((3,), dtype=torch.int32),
            torch.zeros((3,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            torch.ones((4, 8, 6), dtype=torch.bfloat16),
            torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
            torch.ones((32,), dtype=torch.float32),
            *lane_args,
        )
        assert len(fake_outs) == 5
        assert fake_outs[3].shape == (32, TOPK)
        assert fake_outs[3].dtype == torch.int32
        assert fake_outs[4].shape == (16, 32)
        assert fake_outs[4].dtype == torch.bfloat16

    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((32, 12), dtype=torch.bfloat16),
        w_qb=torch.zeros((12, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.zeros((32, 4), dtype=torch.bfloat16),
    )
    with pytest.raises(RuntimeError, match="requires kv_sharded=True"):
        op(
            torch.zeros((4, 3, 4, 32), dtype=torch.uint8),
            (
                torch.zeros((4, 32, 8), dtype=torch.bfloat16),
                torch.zeros((4, 8, 16), dtype=torch.bfloat16),
            ),
            torch.zeros((32, 2, 12), dtype=torch.uint8),
            torch.ones((32, 2), dtype=torch.bfloat16),
            torch.zeros((16, 13), dtype=torch.uint8),
            qb_in,
            torch.zeros((64, 8), dtype=torch.bfloat16),
            torch.zeros((64, 4), dtype=torch.bfloat16),
            torch.zeros((8,), dtype=torch.int32),
            torch.zeros((16,), dtype=torch.int32),
            torch.zeros((16,), dtype=torch.int32),
            torch.zeros((9,), dtype=torch.int32),
            torch.zeros((3,), dtype=torch.int32),
            torch.zeros((3,), dtype=torch.int32),
            torch.ones((4, 8, 6), dtype=torch.bfloat16),
            (
                torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
                torch.ones((32,), dtype=torch.float32),
            ),
            kv_sharded=False,
            k_sharded=True,
        )
    topk_out, attn_out = op(
        torch.zeros((4, 3, 4, 32), dtype=torch.uint8),
        (
            torch.zeros((4, 32, 8), dtype=torch.bfloat16),
            torch.zeros((4, 8, 16), dtype=torch.bfloat16),
        ),
        torch.zeros((32, 2, 12), dtype=torch.uint8),
        torch.ones((32, 2), dtype=torch.bfloat16),
        torch.zeros((16, 13), dtype=torch.uint8),
        qb_in,
        torch.zeros((16, 8), dtype=torch.bfloat16),
        torch.zeros((16, 4), dtype=torch.bfloat16),
        torch.zeros((8,), dtype=torch.int32),
        torch.zeros((16,), dtype=torch.int32),
        torch.zeros((16,), dtype=torch.int32),
        torch.zeros((9,), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int32),
        torch.ones((4, 8, 6), dtype=torch.bfloat16),
        (
            torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
            torch.ones((32,), dtype=torch.float32),
        ),
        kv_sharded=True,
        k_sharded=True,
    )
    assert topk_out.shape == (32, TOPK)
    assert attn_out.shape == (16, 32)
    assert registered[2][2].calls == 1


def test_should_dispatch_unified_idx_qb_attention_and_update_topk_buffer_in_place():
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
    expected_out = torch.full((16, 6), 4.5, dtype=torch.bfloat16)

    def _fake_unified_qb_op(
        index_cache,
        kv_cache,
        q_idx_bytes,
        w_idx,
        k_packed,
        qb_inputs,
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
        del (
            index_cache,
            kv_cache,
            q_idx_bytes,
            w_idx,
            k_packed,
            kv_c_normed,
            k_pe,
            seq_lens,
            qsl,
            w_o,
        )
        captured_calls.append(
            {
                "qb_inputs": qb_inputs,
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
        sparse_mla_unified_idx_qb_op=_fake_unified_qb_op,
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
    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((20, 12), dtype=torch.bfloat16),
        w_qb=torch.zeros((12, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.zeros((20, 4), dtype=torch.bfloat16),
    )
    out = unified_qb_mod.unified_indexer_in_branch_qb_attention(
        fake_layer,
        (torch.zeros((2, 8, 8)), torch.zeros((2, 2, 16))),
        qb_in,
        torch.zeros((16, 8), dtype=torch.bfloat16),
        torch.zeros((16, 4), dtype=torch.bfloat16),
        unified_in,
        mla_md,
        w_o=(torch.zeros((8, 6)), torch.ones((6,))),
        kv_sharded=True,
    )
    assert torch.equal(out, expected_out)
    assert len(captured_calls) == 1
    assert captured_calls[0]["qb_inputs"] is qb_in
    assert torch.equal(captured_calls[0]["idx_bt"], idx_md.block_tables)
    assert torch.equal(captured_calls[0]["mla_bt"], mla_md.block_tables)
    assert torch.equal(
        captured_calls[0]["idx_dist"], torch.tensor([2, 2, 5], dtype=torch.int32)
    )
    assert captured_calls[0]["kv_sharded"] is True
    assert captured_calls[0]["w_uv_dtype"] == torch.bfloat16
    assert len(written) == 1
    assert written[0][0] == 20
    assert torch.equal(written[0][1], expected_topk)


def test_should_reject_missing_unified_idx_qb_op_on_layer():
    fake_layer = types.SimpleNamespace(sparse_mla_unified_idx_qb_op=None)
    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((8, 12), dtype=torch.bfloat16),
        w_qb=torch.zeros((12, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.zeros((8, 4), dtype=torch.bfloat16),
    )
    dummy_in = unified_op_mod.UnifiedIndexerInputs(
        indexer_op=types.SimpleNamespace(),
        q_bytes=torch.zeros((8, 2, 12), dtype=torch.uint8),
        weights=torch.ones((8, 2), dtype=torch.bfloat16),
        k_packed=torch.zeros((8, 13), dtype=torch.uint8),
        idx_metadata=types.SimpleNamespace(),
    )
    with pytest.raises(
        RuntimeError, match="sparse_mla_unified_idx_qb_op was not built"
    ):
        unified_qb_mod.unified_indexer_in_branch_qb_attention(
            fake_layer,
            (torch.zeros((1,)), torch.zeros((1,))),
            qb_in,
            torch.zeros((8, 8), dtype=torch.bfloat16),
            torch.zeros((8, 4), dtype=torch.bfloat16),
            dummy_in,
            types.SimpleNamespace(),
            w_o=(torch.zeros((4, 4)), torch.ones((4,))),
        )


@pytest.mark.parametrize(
    ("has_unified_qb_op", "live_cache", "expect_deferred_unified"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_should_defer_fused_qb_on_active_indexer_only_when_unified_qb_op_and_cache_live(
    monkeypatch,
    has_unified_qb_op: bool,
    live_cache: bool,
    expect_deferred_unified: bool,
):
    wrapper = mla_op.VllmTPUMultiHeadLatentAttentionWrapper.__new__(
        mla_op.VllmTPUMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.is_sparse = True
    wrapper.skip_topk = False

    sentinel_unified = unified_op_mod.UnifiedIndexerInputs(
        indexer_op=types.SimpleNamespace(),
        q_bytes=torch.zeros((24, 2, 12), dtype=torch.uint8),
        weights=torch.ones((24, 2), dtype=torch.bfloat16),
        k_packed=torch.zeros((64, 13), dtype=torch.uint8),
        idx_metadata=types.SimpleNamespace(),
    )

    class _FakeIndexer:
        indexer_op = types.SimpleNamespace(
            topk_op_tp=types.SimpleNamespace(q_own_rows=True),
            has_live_unified_cache=lambda: live_cache,
            prepare_unified_inputs=lambda q, k, w, *, k_sharded=False: (
                sentinel_unified if live_cache else None
            ),
        )

        def prepare_inputs(self, *args, **kwargs):
            del args, kwargs
            return (
                torch.zeros((24, 2, 12), dtype=torch.float8_e4m3fn),
                torch.zeros((64, 12), dtype=torch.bfloat16),
                torch.ones((24, 2), dtype=torch.bfloat16),
            )

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return torch.zeros((64, TOPK), dtype=torch.int32)

    wrapper.indexer = _FakeIndexer()
    wrapper.indexer_rope_emb = None
    wrapper.kimi_preprocess = None
    wrapper.q_lora_rank = 16
    wrapper.kv_lora_rank = 8
    wrapper.qk_nope_head_dim = 8
    wrapper.qk_rope_head_dim = 4
    wrapper.qk_head_dim = 12
    wrapper.num_heads = 4
    wrapper.v_head_dim = 6
    wrapper.dcp_group_size = 4
    wrapper.dcp_q_replicate = True
    wrapper.gate_is_fused = False
    wrapper.g_proj = None
    wrapper.rotary_emb = object()
    wrapper.topk_indices_buffer = torch.zeros((64, TOPK), dtype=torch.int32)
    wrapper.q_a_layernorm = lambda x: x
    wrapper.kv_a_layernorm = lambda x: x
    wrapper.fused_qkv_a_proj = object()
    wrapper.q_b_proj = object()
    wrapper.o_proj = types.SimpleNamespace(
        weight=torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones((32,), dtype=torch.float32),
    )

    captured_mla: list[dict[str, Any]] = []

    class _FakeMlaAttn:
        sparse_mla_tp_op = types.SimpleNamespace(
            q_replicated=True,
            serves_oproj=True,
            rank=torch.zeros((1,), dtype=torch.int32),
        )
        sparse_mla_unified_tp_op = types.SimpleNamespace(serves_sharded_kv=True)
        sparse_mla_unified_idx_qb_op = (
            types.SimpleNamespace(serves_sharded_kv=True) if has_unified_qb_op else None
        )
        sparse_mla_in_branch_qb_op = None
        W_UV = torch.zeros((4, 8, 6), dtype=torch.bfloat16)
        W_O_tp_half = None

        def __call__(self, q, kv_c, k_pe, **kw):
            del q, kv_c, k_pe
            captured_mla.append(kw)
            return torch.zeros((16, 32), dtype=torch.bfloat16)

    wrapper.mla_attn = _FakeMlaAttn()

    eager_calls: list[int] = []

    def _fake_eager_qb(q_c, w_qb, s_qb, w_uk_t, cos_sin, split_rows):
        del w_qb, s_qb, w_uk_t, cos_sin
        eager_calls.append(split_rows)
        return (
            torch.zeros((split_rows, 4, 8), dtype=q_c.dtype),
            torch.zeros((q_c.shape[0] - split_rows, 4, 8), dtype=q_c.dtype),
            torch.zeros((q_c.shape[0], 4, 4), dtype=q_c.dtype),
        )

    w_qb = torch.zeros((16, 48), dtype=torch.float8_e4m3fn)
    s_qb = torch.ones((48,), dtype=torch.float32)
    w_uk = torch.ones((4, 8, 8), dtype=torch.bfloat16)
    cos_sin_cache = torch.ones((64, 4), dtype=torch.bfloat16)

    monkeypatch.setenv("TPU_DCP_TP_PREFILL_LOCAL_OPROJ", "1")
    monkeypatch.setattr(mla_op, "_FUSED_QB_ROPE_ABSORB_SPLIT_OP", _fake_eager_qb)
    monkeypatch.setattr(mla_op, "rope_half_split", lambda _r, _p, q_, k_: (q_, k_))
    monkeypatch.setattr(mla_op, "_retain_buffers", lambda x, *_a: x)
    monkeypatch.setattr(tp_op_mod, "applies", lambda n: n >= 64)
    monkeypatch.setattr(tp_op_mod, "serves_sequence_row_oproj", lambda _o: True)
    monkeypatch.setattr(
        tp_op_mod, "q_rows", lambda _op, _n: torch.arange(24, dtype=torch.long)
    )
    monkeypatch.setattr(
        wrapper,
        "_fused_qb_rope_absorb_operands",
        lambda *_a, **_kw: (w_qb, s_qb, w_uk, 8, cos_sin_cache),
    )

    prefill_group = types.SimpleNamespace(
        project_gather=lambda _w, _h, _n: (
            (
                torch.zeros((24, 16), dtype=torch.bfloat16),
                torch.zeros((16, 12), dtype=torch.bfloat16),
                torch.arange(16, dtype=torch.long),
            ),
            None,
        )
    )
    positions = torch.arange(64, dtype=torch.long)
    hidden = torch.zeros((16, 32), dtype=torch.bfloat16)

    out = wrapper.forward(positions, hidden, prefill_group=prefill_group)
    assert out.shape == (16, 32)
    assert len(captured_mla) == 1
    if expect_deferred_unified:
        assert eager_calls == []
        assert captured_mla[0]["in_branch_qb_inputs"] is not None
        assert captured_mla[0]["unified_indexer_inputs"] is sentinel_unified
        assert captured_mla[0]["topk_indices"] is None
    else:
        assert eager_calls == [8]
        assert captured_mla[0]["in_branch_qb_inputs"] is None
        if live_cache:
            assert captured_mla[0]["unified_indexer_inputs"] is sentinel_unified
        else:
            assert captured_mla[0]["unified_indexer_inputs"] is None
            assert captured_mla[0]["topk_indices"] is not None


def test_should_route_backend_forward_to_unified_idx_qb_attention(monkeypatch):
    impl = attn_mod.PallasMLAttentionBackendImpl.__new__(
        attn_mod.PallasMLAttentionBackendImpl
    )
    impl.dcp_size = 4
    impl.dcp_interleave_size = 4

    called_unified_qb: list[dict[str, Any]] = []
    called_nonidx: list[int] = []

    def _fake_unified_qb(
        layer,
        kv_cache,
        qb_inputs,
        kv_c_normed,
        k_pe,
        unified_inputs,
        attn_metadata,
        w_o,
        *,
        kv_sharded: bool = False,
    ):
        del layer, attn_metadata
        called_unified_qb.append(
            {
                "qb_inputs": qb_inputs,
                "unified_inputs": unified_inputs,
                "kv_sharded": kv_sharded,
                "w_o": w_o,
                "kv_c_shape": tuple(kv_c_normed.shape),
                "k_pe_shape": tuple(k_pe.shape),
            }
        )
        kv_cache[0].fill_(2.0)
        kv_cache[1].fill_(4.0)
        return torch.full((16, 32), 9.0, dtype=qb_inputs.q_c.dtype)

    monkeypatch.setattr(
        unified_qb_mod, "unified_indexer_in_branch_qb_attention", _fake_unified_qb
    )
    monkeypatch.setattr(
        in_branch_mod,
        "in_branch_qb_attention",
        lambda *a, **kw: called_nonidx.append(1),
    )
    monkeypatch.setattr(tp_op_mod, "applies", lambda n: n >= 64)
    monkeypatch.setattr(
        impl, "_get_kv_scales", lambda _layer: (None, None, None), raising=False
    )

    layer = types.SimpleNamespace(
        num_heads=4,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        kv_cache_quantized_dtype=None,
        sparse_mla_tp_op=types.SimpleNamespace(serves_oproj=True),
        W_UK_T=torch.ones((4, 8, 8), dtype=torch.bfloat16),
    )
    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((24, 16), dtype=torch.bfloat16),
        w_qb=torch.zeros((16, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.zeros((24, 4), dtype=torch.bfloat16),
    )
    unified_in = unified_op_mod.UnifiedIndexerInputs(
        indexer_op=types.SimpleNamespace(),
        q_bytes=torch.zeros((24, 2, 12), dtype=torch.uint8),
        weights=torch.ones((24, 2), dtype=torch.bfloat16),
        k_packed=torch.zeros((64, 13), dtype=torch.uint8),
        idx_metadata=types.SimpleNamespace(),
    )
    q_dead = (
        torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        torch.zeros((1, 4, 4), dtype=torch.bfloat16),
    )
    sp_o_proj = (
        torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
        torch.ones((32,), dtype=torch.float32),
    )
    kv_c = torch.ones((16, 8), dtype=torch.bfloat16)
    k_pe = torch.ones((16, 1, 4), dtype=torch.bfloat16)
    nope_cache = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    rope_cache = torch.zeros((4, 4, 16), dtype=torch.bfloat16)
    meta = types.SimpleNamespace()

    out = impl.forward(
        layer=layer,
        q=q_dead,
        kv_c_normed=kv_c,
        k_pe=k_pe,
        kv_cache=(nope_cache, rope_cache),
        attn_metadata=meta,
        topk_indices=None,
        unified_indexer_inputs=unified_in,
        sp_o_proj=sp_o_proj,
        kv_sharded=True,
        in_branch_qb_inputs=qb_in,
    )
    assert isinstance(out, torch.Tensor)
    assert torch.all(out == 9.0)
    assert torch.all(nope_cache == 2.0)
    assert torch.all(rope_cache == 4.0)
    assert called_nonidx == []
    assert len(called_unified_qb) == 1
    assert called_unified_qb[0]["qb_inputs"] is qb_in
    assert called_unified_qb[0]["unified_inputs"] is unified_in
    assert called_unified_qb[0]["kv_sharded"] is True

    # Missing sp_o_proj must fail fast on the in_branch_qb_inputs guard.
    with pytest.raises(ValueError, match="in_branch_qb_inputs requires"):
        impl.forward(
            layer=layer,
            q=q_dead,
            kv_c_normed=kv_c,
            k_pe=k_pe,
            kv_cache=(nope_cache, rope_cache),
            attn_metadata=meta,
            topk_indices=None,
            unified_indexer_inputs=unified_in,
            sp_o_proj=None,
            kv_sharded=False,
            in_branch_qb_inputs=qb_in,
        )

    # Missing both topk_indices and unified_indexer_inputs must fail fast.
    with pytest.raises(ValueError, match="sp_o_proj is only served"):
        impl.forward(
            layer=layer,
            q=q_dead,
            kv_c_normed=kv_c,
            k_pe=k_pe,
            kv_cache=(nope_cache, rope_cache),
            attn_metadata=meta,
            topk_indices=None,
            unified_indexer_inputs=None,
            sp_o_proj=sp_o_proj,
            kv_sharded=True,
            in_branch_qb_inputs=qb_in,
        )


def test_should_forward_k_sharded_flag_through_unified_idx_qb_and_unified_mla():
    seen_qb: list[dict[str, Any]] = []
    seen_mla: list[dict[str, Any]] = []

    def _rec_qb(*args, kv_sharded: bool = False, k_sharded: bool = False):
        del args
        seen_qb.append({"kv_sharded": kv_sharded, "k_sharded": k_sharded})
        return torch.zeros((20, 4), dtype=torch.int32), torch.zeros((16, 6))

    def _rec_mla(*args, kv_sharded: bool = False, k_sharded: bool = False):
        del args
        seen_mla.append({"kv_sharded": kv_sharded, "k_sharded": k_sharded})
        return torch.zeros((20, 4), dtype=torch.int32), torch.zeros((16, 6))

    idx_op = types.SimpleNamespace(
        k_cache=types.SimpleNamespace(kv_cache=torch.zeros((4, 8), dtype=torch.uint8)),
        _write_topk_rows=lambda n, rows: rows,
    )
    meta = types.SimpleNamespace(
        seq_lens=torch.tensor([64, 32], dtype=torch.int32),
        block_tables=torch.tensor([2, 4, 6, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 16, 20], dtype=torch.int32),
        request_distribution=torch.tensor([2, 3, 5], dtype=torch.int32),
    )
    unified_in = unified_op_mod.UnifiedIndexerInputs(
        indexer_op=idx_op,
        q_bytes=torch.zeros((20, 2, 12), dtype=torch.uint8),
        weights=torch.ones((20, 2), dtype=torch.bfloat16),
        k_packed=torch.zeros((16, 13), dtype=torch.uint8),
        idx_metadata=meta,
        k_sharded=True,
    )
    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((20, 16), dtype=torch.bfloat16),
        w_qb=torch.zeros((16, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.ones((20, 4), dtype=torch.bfloat16),
    )
    layer = types.SimpleNamespace(
        sparse_mla_unified_idx_qb_op=_rec_qb,
        sparse_mla_unified_tp_op=_rec_mla,
        W_UV_tp_all=torch.ones((4, 8, 6), dtype=torch.bfloat16),
    )
    kv_cache = (torch.zeros((4, 16, 8)), torch.zeros((4, 4, 16)))
    w_o = (torch.zeros((24, 6)), torch.ones((6,)))

    unified_qb_mod.unified_indexer_in_branch_qb_attention(
        layer,
        kv_cache,
        qb_in,
        torch.zeros((16, 8), dtype=torch.bfloat16),
        torch.zeros((16, 1, 4), dtype=torch.bfloat16),
        unified_in,
        meta,
        w_o,
        kv_sharded=True,
    )
    unified_op_mod.unified_indexer_attention(
        layer,
        kv_cache,
        torch.zeros((20, 4, 8), dtype=torch.bfloat16),
        torch.zeros((20, 4, 4), dtype=torch.bfloat16),
        torch.zeros((16, 8), dtype=torch.bfloat16),
        torch.zeros((16, 1, 4), dtype=torch.bfloat16),
        unified_in,
        meta,
        w_o,
        kv_sharded=True,
    )
    assert seen_qb == [{"kv_sharded": True, "k_sharded": True}]
    assert seen_mla == [{"kv_sharded": True, "k_sharded": True}]


@pytest.mark.parametrize(
    ("split_rows", "bkvp", "bq", "want_bq", "want_dam"),
    [
        (64, (2, 2, 32), (1, 1, 4), (1, 1, 16), True),
        (16, (2, 2, 32), (1, 1, 4), (1, 1, 16), True),
        (8, (2, 2, 32), (1, 1, 4), (1, 1, 4), False),
        (64, (2, 2, 32), (1, 1, 16), (1, 1, 16), False),
        (24, (2, 2, 32), (1, 1, 4), (1, 1, 4), False),
    ],
)
def test_should_select_lead_streamidx_decode_as_mixed_when_split_rows_aligned(
    split_rows: int,
    bkvp: tuple[int, int, int],
    bq: tuple[int, int, int],
    want_bq: tuple[int, int, int],
    want_dam: bool,
) -> None:
    got_bkvp, got_bq, got_dam = tp_op_mod.lead_streamidx_block_sizes(
        split_rows, bkvp, bq
    )
    assert got_bkvp == bkvp
    assert got_bq == want_bq
    assert got_dam is want_dam


@pytest.mark.parametrize(
    ("nat_shape", "topk", "want_early_exit"),
    [
        ((16, 33, 4, 256), 2048, True),
        ((32, 33, 4, 256), 2048, False),
        ((128, 33, 256), 2048, False),
    ],
)
def test_should_disable_streamidx_early_exit_on_multi_page_cap_branches(
    monkeypatch: pytest.MonkeyPatch,
    nat_shape: tuple[int, ...],
    topk: int,
    want_early_exit: bool,
) -> None:
    seen: list[bool] = []

    def _fake_streamindex_topk(*args, **kwargs):
        del args
        seen.append(bool(kwargs["enable_early_exit"]))
        return jnp.zeros((16, topk), dtype=jnp.int32)

    monkeypatch.setattr(tp_op_mod, "streamindex_topk", _fake_streamindex_topk)
    select = tp_op_mod.make_select_topk(
        topk,
        KVLayout.SEQ_ALONG_LANE,
        chunk_tokens=256,
        small_bucket=(4, 32, 64),
        skip_empty_decode=False,
        skip_empty_decode_chunked=False,
        mixed_bkvp=8,
        decode_pack=True,
        shared_meta=False,
    )
    rr = types.SimpleNamespace(
        seq_lens=jnp.ones((4,), dtype=jnp.int32),
        cu_q_lens=jnp.arange(5, dtype=jnp.int32),
        num_seqs=jnp.int32(1),
    )
    nat_dtype = jnp.uint32 if len(nat_shape) == 3 else jnp.uint8
    select(
        jnp.zeros((1024, 32, 128), dtype=jnp.float8_e4m3fn),
        jnp.zeros((1024, 32), dtype=jnp.bfloat16),
        jnp.zeros(nat_shape, dtype=nat_dtype),
        rr,
        jnp.zeros((4, 16), dtype=jnp.int32),
    )
    assert seen == [want_early_exit]


@pytest.mark.parametrize(
    ("split_rows", "want_bq", "want_dam"),
    [
        (16, (1, 1, 16), True),
        (8, (1, 1, 4), False),
    ],
)
def test_should_forward_lead_decode_as_mixed_in_unified_idx_qb_op(
    monkeypatch: pytest.MonkeyPatch,
    split_rows: int,
    want_bq: tuple[int, int, int],
    want_dam: bool,
) -> None:
    mesh = _mesh(4)
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=split_rows, max_pages=8)
    monkeypatch.setenv("TPU_DCP_TP_UNIFIED_INDEXER_MLA", "1")
    monkeypatch.setenv("TPU_DCP_TP_UNIFIED_INDEXER_IN_BRANCH_QB", "1")
    monkeypatch.setenv("TPU_DCP_TP_PREFILL_LOCAL_OPROJ", "1")
    monkeypatch.setenv("TPU_DCP_RANK_OPERAND", "1")
    monkeypatch.setattr(
        unified_qb_mod, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        unified_qb_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )
    monkeypatch.setattr(unified_qb_mod, "dcp_lane_operand", lambda: None)

    captured_fn: list[Any] = []

    class _FakeOp:
        def register_fake(self, fn: Any) -> Any:
            return fn

    def _capture_op(name, fn, **kwargs):
        del name, kwargs
        captured_fn.append(fn)
        return _FakeOp()

    monkeypatch.setattr(unified_qb_mod, "pcp_streaming_jax_op", _capture_op)
    monkeypatch.setattr(
        unified_qb_mod,
        "get_shared_jax_op",
        lambda kind, key, build: build(f"p::{kind}"),
    )
    monkeypatch.setattr(
        unified_qb_mod,
        "update_sparse_mla_kv_cache_dcp",
        lambda n, r, *a, **kw: (n, r),
    )

    class _StopAfterLead(Exception):
        pass

    seen: list[tuple[tuple[int, int, int], tuple[int, int, int], bool]] = []

    def _capture_lead(*args, **kwargs):
        del args
        seen.append(
            (
                kwargs["num_kv_pages_per_block"],
                kwargs["num_queries_per_block"],
                kwargs["decode_as_mixed"],
            )
        )
        raise _StopAfterLead()

    monkeypatch.setattr(unified_qb_mod, "streamindex_topk_dcp", _capture_lead)
    spec_n = types.SimpleNamespace(
        page_size=32, layout=types.SimpleNamespace(value="xla")
    )
    spec_r = types.SimpleNamespace(
        page_size=32, layout=spec_n.layout, packed_rope=False
    )

    class _FakeLayer:
        kv_cache_quantized_dtype = None
        scale = 0.135
        num_heads = 4
        v_head_dim = 6
        use_sparse = True
        mla_kv_spec = (spec_n, spec_r)

    layer = _FakeLayer()
    impl = types.SimpleNamespace(_dcp_mesh=mesh, dcp_size=4, dcp_interleave_size=4)
    indexer_op = types.SimpleNamespace(
        topk_tokens=TOPK, k_cache=types.SimpleNamespace(prefix="layer0.idx")
    )
    cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", KVLayout.SEQ_ALONG_LANE),
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((2, 2, 32), (1, 1, 4)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda prefix: 0,
    )
    built = unified_qb_mod.build_unified_indexer_in_branch_qb_op(
        layer, impl, indexer_op, layout, q_replicated=True, callbacks=cb
    )
    assert built is not None
    shd = NamedSharding(mesh, P(AXIS))
    rep = NamedSharding(mesh, P())
    with pytest.raises(_StopAfterLead):
        captured_fn[0](
            jax.device_put(jnp.zeros((16, 3, 4, 32), jnp.uint8), shd),
            jax.device_put(jnp.zeros((16, 32, 8), jnp.bfloat16), shd),
            jax.device_put(jnp.zeros((16, 8, 16), jnp.bfloat16), shd),
            jax.device_put(jnp.zeros((4 * (16 + split_rows), 2, 12), jnp.uint8), shd),
            jax.device_put(jnp.ones((4 * (16 + split_rows), 2), jnp.bfloat16), shd),
            jax.device_put(jnp.zeros((64, 13), jnp.uint8), rep),
            jax.device_put(jnp.zeros((4 * (16 + split_rows), 12), jnp.bfloat16), shd),
            jax.device_put(jnp.zeros((12, 48), jnp.float8_e4m3fn), rep),
            jax.device_put(jnp.ones((48,), jnp.float32), rep),
            jax.device_put(jnp.ones((4, 8, 8), jnp.bfloat16), rep),
            jax.device_put(jnp.zeros((4 * (16 + split_rows), 4), jnp.bfloat16), shd),
            jax.device_put(jnp.zeros((64, 8), jnp.bfloat16), rep),
            jax.device_put(jnp.zeros((64, 4), jnp.bfloat16), rep),
            jax.device_put(jnp.zeros((8,), jnp.int32), rep),
            jax.device_put(jnp.zeros((16,), jnp.int32), rep),
            jax.device_put(jnp.zeros((16,), jnp.int32), rep),
            jax.device_put(jnp.zeros((9,), jnp.int32), rep),
            jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep),
            jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep),
            jax.device_put(jnp.arange(4, dtype=jnp.int32), shd),
            jax.device_put(jnp.ones((4, 8, 6), jnp.bfloat16), rep),
            jax.device_put(jnp.zeros((24, 32), jnp.float8_e4m3fn), shd),
            jax.device_put(jnp.ones((32,), jnp.float32), shd),
        )
    assert seen == [((2, 2, 32), want_bq, want_dam)]

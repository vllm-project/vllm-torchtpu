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
"""Contract tests for in-branch lead indexer deferral in unified MLA ops."""

from __future__ import annotations

import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.layers.adapter import (
    dcp_unified_idx_qb_mla_op as unified_qb_mod,
)
from vllm_torchtpu.layers.adapter import (
    dcp_unified_indexer_mla_op as unified_op_mod,
)

AXIS = "dcp"
TOPK = 4


def _mesh(n: int = 4) -> Mesh:
    devs = jax.devices()
    if len(devs) < n:
        pytest.skip(f"needs {n} devices, got {len(devs)}")
    return Mesh(np.array(devs[:n]), (AXIS,))


def _u16_bits(arr: jax.Array) -> np.ndarray:
    return np.asarray(
        jax.lax.bitcast_convert_type(jnp.asarray(arr), jnp.uint16), dtype=np.uint16
    )


def _dummy_unified_tensors(mesh: Mesh, dcp: int, s: int, rows: int):
    shd = NamedSharding(mesh, P(AXIS))
    rep = NamedSharding(mesh, P())
    rng = np.random.default_rng(7)
    num_heads, nope_d, rope_d, r_dim, v_dim = 4, 4, 2, 4, 4
    q_lora, pages_per_rank, page_size = 6, 4, 8
    idx_cache = jax.device_put(
        jnp.asarray(
            rng.integers(0, 64, (dcp * pages_per_rank, 2, 4, page_size)), jnp.uint8
        ),
        shd,
    )
    q_idx = jax.device_put(
        jnp.asarray(rng.normal(0, 0.2, (dcp * rows, 2, 4)), jnp.float32), shd
    )
    w_idx = jax.device_put(
        jnp.asarray(rng.uniform(0.2, 1.0, (dcp * rows, 2)), jnp.bfloat16), shd
    )
    lead_q = jax.device_put(
        jnp.asarray(rng.normal(0, 0.2, (s, 2, 4)), jnp.float32), rep
    )
    lead_w = jax.device_put(
        jnp.asarray(rng.uniform(0.2, 1.0, (s, 2)), jnp.bfloat16), rep
    )
    nope = jax.device_put(
        jnp.asarray(
            rng.normal(0, 0.2, (dcp * pages_per_rank, page_size, r_dim)), jnp.bfloat16
        ),
        shd,
    )
    rope = jax.device_put(
        jnp.asarray(
            rng.normal(0, 0.2, (dcp * pages_per_rank, page_size, rope_d)), jnp.bfloat16
        ),
        shd,
    )
    qc_own = jax.device_put(
        jnp.asarray(rng.normal(0, 0.3, (dcp * (s + rows), q_lora)), jnp.bfloat16), shd
    )
    w_qb = jax.device_put(
        jnp.asarray(
            rng.normal(0, 0.2, (q_lora, num_heads * (nope_d + rope_d))), jnp.bfloat16
        ),
        rep,
    )
    s_qb = jax.device_put(jnp.ones((num_heads * (nope_d + rope_d),), jnp.float32), rep)
    w_uk_t = jax.device_put(
        jnp.asarray(rng.normal(0, 0.2, (num_heads, nope_d, r_dim)), jnp.bfloat16), rep
    )
    cos_sin = jax.device_put(
        jnp.asarray(rng.normal(0, 0.2, (dcp * (s + rows), rope_d)), jnp.bfloat16), shd
    )
    w_uv = jax.device_put(
        jnp.asarray(rng.normal(0, 0.2, (num_heads, r_dim, v_dim)), jnp.bfloat16), rep
    )
    bt = jax.device_put(jnp.arange(16, dtype=jnp.int32), rep)
    seq_lens = jax.device_put(jnp.array([18, 0, 0, 0], dtype=jnp.int32), rep)
    qsl = jax.device_put(
        jnp.array([0, dcp * rows, dcp * rows, dcp * rows, dcp * rows], jnp.int32), rep
    )
    dist = jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep)
    lead_sl = jax.device_put(jnp.array([10, 0, 0, 0], dtype=jnp.int32), rep)
    lead_qsl = jax.device_put(jnp.array([0, s, s, s, s], dtype=jnp.int32), rep)
    lead_dist = jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep)
    rank_ids = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)
    w_o = jax.device_put(
        jnp.asarray(
            rng.normal(0, 0.2, (dcp * num_heads * v_dim // dcp, 8)), jnp.bfloat16
        ),
        shd,
    )
    s_o = jax.device_put(jnp.ones((dcp, 8), jnp.float32), shd)
    return dict(
        idx_cache=idx_cache,
        q_idx=q_idx,
        w_idx=w_idx,
        lead_q=lead_q,
        lead_w=lead_w,
        nope=nope,
        rope=rope,
        qc_own=qc_own,
        w_qb=w_qb,
        s_qb=s_qb,
        w_uk_t=w_uk_t,
        cos_sin=cos_sin,
        w_uv=w_uv,
        bt=bt,
        seq_lens=seq_lens,
        qsl=qsl,
        dist=dist,
        lead_sl=lead_sl,
        lead_qsl=lead_qsl,
        lead_dist=lead_dist,
        rank_ids=rank_ids,
        w_o=w_o,
        s_o=s_o,
        num_heads=num_heads,
        nope_d=nope_d,
        rope_d=rope_d,
        r_dim=r_dim,
        v_dim=v_dim,
    )


def _deterministic_lead_topk(
    ql_r: jax.Array,
    wl_r: jax.Array,
    cache_r: jax.Array,
    sl_r: jax.Array,
    bt_r: jax.Array,
    qsl_r: jax.Array,
    dist_r: jax.Array,
    cp_rank_r: jax.Array,
    *,
    corrupt_first: bool = False,
) -> jax.Array:
    del wl_r, qsl_r, dist_r
    s = ql_r.shape[0]
    base = jnp.arange(TOPK, dtype=jnp.int32)[None, :] + (
        jnp.arange(s, dtype=jnp.int32)[:, None]
        + cp_rank_r.astype(jnp.int32)
        + cache_r[0, 0, 0, 0].astype(jnp.int32)
        + bt_r[0]
        + sl_r[0]
    )
    out = jnp.mod(base, jnp.int32(32))
    if corrupt_first:
        out = out.at[0, 0].set(out[0, 0] + jnp.int32(1))
    return out


def _stub_select_tail(q_r, w_r, nat, rr, table):
    del q_r, w_r, nat, table
    rows = rr.seq_lens.shape[0] if rr.seq_lens.ndim == 2 else 8
    rows = 8
    return jnp.broadcast_to(jnp.arange(TOPK, dtype=jnp.int32)[None, :], (rows, TOPK))


def _stub_fused_qb(qc, wqb, sqb, wuk, cs, split_rows):
    del sqb, cs
    nh, _, rd = wuk.shape
    h = qc @ wqb
    q_h = h[:split_rows, : nh * rd].reshape(split_rows, nh, rd).astype(jnp.bfloat16)
    q_t = (
        h[split_rows:, : nh * rd]
        .reshape(qc.shape[0] - split_rows, nh, rd)
        .astype(jnp.bfloat16)
    )
    p_h = jnp.zeros((split_rows, nh, 2), jnp.bfloat16)
    p_t = jnp.zeros((qc.shape[0] - split_rows, nh, 2), jnp.bfloat16)
    return q_h, q_t, p_h, p_t


def _stub_lead_attend(ql_head, qpe_head, nope, rope, h_topk, bt, sl, qsl, dist):
    del qpe_head, rope, bt, sl, qsl, dist
    s, nh, rd = ql_head.shape
    tag = (h_topk[:, :1].astype(jnp.float32) * 0.05)[:, None, :]
    out = (
        ql_head.astype(jnp.float32) + tag + nope[0, 0, 0].astype(jnp.float32) * 0.01
    ).astype(jnp.bfloat16)
    lse = jnp.full((s, nh), 1.0, dtype=jnp.float32) + (
        h_topk[:, :1].astype(jnp.float32) * 0.02
    )
    return out, lse


def _stub_tail_attend(ql_tail, g_nope, g_rope, flat, num_rows, qpe_tail):
    del g_nope, g_rope, num_rows, qpe_tail
    return (
        ql_tail.astype(jnp.float32) + flat[:, :1, None].astype(jnp.float32) * 0.01
    ).astype(jnp.bfloat16)


def test_should_match_outside_lead_topk_bit_exact_when_deferred_in_branch() -> None:
    dcp, s, rows = 4, 2, 8
    mesh = _mesh(dcp)
    t = _dummy_unified_tensors(mesh, dcp, s, rows)
    layout = tp.TokenParallelLayout(
        dcp_size=dcp, interleave_size=2, page_size=8, split_rows=s, max_pages=4
    )
    shd = P(AXIS)
    rep = P()

    def _ref_head_call(ic, ql, wl, lsl, bt, lqsl, ldist, r):
        return _deterministic_lead_topk(ql, wl, ic, lsl, bt, lqsl, ldist, r[0])

    head_topk_ref = jax.shard_map(
        _ref_head_call,
        mesh=mesh,
        in_specs=(shd, rep, rep, rep, rep, rep, rep, shd),
        out_specs=shd,
        check_vma=False,
    )(
        t["idx_cache"],
        t["lead_q"],
        t["lead_w"],
        t["lead_sl"],
        t["bt"],
        t["lead_qsl"],
        t["lead_dist"],
        t["rank_ids"],
    )

    ref_own, ref_lead, ref_tail_tk = tp.unified_indexer_in_branch_qb_mla(
        layout,
        mesh,
        AXIS,
        t["idx_cache"],
        t["q_idx"],
        t["w_idx"],
        t["bt"],
        t["nope"],
        t["rope"],
        t["qc_own"],
        t["w_qb"],
        t["s_qb"],
        t["w_uk_t"],
        t["cos_sin"],
        head_topk_ref,
        t["bt"],
        t["seq_lens"],
        t["qsl"],
        t["dist"],
        select_topk=_stub_select_tail,
        fused_qb=_stub_fused_qb,
        lead_attend=_stub_lead_attend,
        tail_attend=_stub_tail_attend,
        w_uv=t["w_uv"],
        num_tokens=dcp * rows,
        rank_ids=t["rank_ids"],
        w_uv_replicated=True,
    )
    ref_full_tk = jax.shard_map(
        lambda h, tl: jnp.concatenate([h, tl], axis=0),
        mesh=mesh,
        in_specs=(shd, shd),
        out_specs=shd,
        check_vma=False,
    )(head_topk_ref, ref_tail_tk)

    got_own, got_lead, got_full_tk = tp.unified_indexer_in_branch_qb_mla(
        layout,
        mesh,
        AXIS,
        t["idx_cache"],
        t["q_idx"],
        t["w_idx"],
        t["bt"],
        t["nope"],
        t["rope"],
        t["qc_own"],
        t["w_qb"],
        t["s_qb"],
        t["w_uk_t"],
        t["cos_sin"],
        None,
        t["bt"],
        t["seq_lens"],
        t["qsl"],
        t["dist"],
        select_topk=_stub_select_tail,
        fused_qb=_stub_fused_qb,
        lead_attend=_stub_lead_attend,
        tail_attend=_stub_tail_attend,
        w_uv=t["w_uv"],
        num_tokens=dcp * rows,
        rank_ids=t["rank_ids"],
        w_uv_replicated=True,
        select_lead_topk=_deterministic_lead_topk,
        lead_topk_inputs=(
            t["lead_q"],
            t["lead_w"],
            t["lead_sl"],
            t["lead_qsl"],
            t["lead_dist"],
        ),
    )

    assert np.array_equal(_u16_bits(got_own), _u16_bits(ref_own))
    assert np.array_equal(_u16_bits(got_lead), _u16_bits(ref_lead))
    assert np.array_equal(np.asarray(got_full_tk), np.asarray(ref_full_tk))

    # Positive control: flipping one head slot must alter both full_topk and lead_v.
    bad_own, bad_lead, bad_full_tk = tp.unified_indexer_in_branch_qb_mla(
        layout,
        mesh,
        AXIS,
        t["idx_cache"],
        t["q_idx"],
        t["w_idx"],
        t["bt"],
        t["nope"],
        t["rope"],
        t["qc_own"],
        t["w_qb"],
        t["s_qb"],
        t["w_uk_t"],
        t["cos_sin"],
        None,
        t["bt"],
        t["seq_lens"],
        t["qsl"],
        t["dist"],
        select_topk=_stub_select_tail,
        fused_qb=_stub_fused_qb,
        lead_attend=_stub_lead_attend,
        tail_attend=_stub_tail_attend,
        w_uv=t["w_uv"],
        num_tokens=dcp * rows,
        rank_ids=t["rank_ids"],
        w_uv_replicated=True,
        select_lead_topk=lambda *a: _deterministic_lead_topk(*a, corrupt_first=True),
        lead_topk_inputs=(
            t["lead_q"],
            t["lead_w"],
            t["lead_sl"],
            t["lead_qsl"],
            t["lead_dist"],
        ),
    )
    del bad_own
    assert not np.array_equal(np.asarray(bad_full_tk), np.asarray(ref_full_tk))
    assert not np.array_equal(_u16_bits(bad_lead), _u16_bits(ref_lead))


def test_should_reject_invalid_lead_select_combinations() -> None:
    dcp, s, rows = 4, 2, 8
    mesh = _mesh(dcp)
    t = _dummy_unified_tensors(mesh, dcp, s, rows)
    layout = tp.TokenParallelLayout(
        dcp_size=dcp, interleave_size=2, page_size=8, split_rows=s, max_pages=4
    )
    head_dummy = jax.device_put(
        jnp.zeros((dcp * s, TOPK), jnp.int32), NamedSharding(mesh, P(AXIS))
    )
    lead_in = (
        t["lead_q"],
        t["lead_w"],
        t["lead_sl"],
        t["lead_qsl"],
        t["lead_dist"],
    )
    base_kw: dict[str, Any] = dict(
        select_topk=_stub_select_tail,
        fused_qb=_stub_fused_qb,
        lead_attend=_stub_lead_attend,
        tail_attend=_stub_tail_attend,
        w_uv=t["w_uv"],
        num_tokens=dcp * rows,
        rank_ids=t["rank_ids"],
        w_uv_replicated=True,
    )
    common = (
        layout,
        mesh,
        AXIS,
        t["idx_cache"],
        t["q_idx"],
        t["w_idx"],
        t["bt"],
        t["nope"],
        t["rope"],
        t["qc_own"],
        t["w_qb"],
        t["s_qb"],
        t["w_uk_t"],
        t["cos_sin"],
    )
    tail = (t["bt"], t["seq_lens"], t["qsl"], t["dist"])

    with pytest.raises(ValueError, match="passed together"):
        tp.unified_indexer_in_branch_qb_mla(
            *common, None, *tail, select_lead_topk=_deterministic_lead_topk, **base_kw
        )
    with pytest.raises(ValueError, match="not both"):
        tp.unified_indexer_in_branch_qb_mla(
            *common,
            head_dummy,
            *tail,
            select_lead_topk=_deterministic_lead_topk,
            lead_topk_inputs=lead_in,
            **base_kw,
        )
    with pytest.raises(ValueError, match="head_topk is required"):
        tp.unified_indexer_in_branch_qb_mla(*common, None, *tail, **base_kw)


@pytest.mark.parametrize(
    ("enable_flag", "large_bucket", "hist_merge", "expect_deferred"),
    [
        ("1", True, "1", True),
        ("0", True, "1", False),
        ("1", False, "1", False),
    ],
)
def test_should_route_in_branch_lead_indexer_by_gate_and_hist_merge(
    monkeypatch: pytest.MonkeyPatch,
    enable_flag: str,
    large_bucket: bool,
    hist_merge: str,
    expect_deferred: bool,
) -> None:
    dcp, split_rows = 4, 16
    mesh = _mesh(dcp)
    layout = tp.TokenParallelLayout(dcp, 4, 32, split_rows=split_rows, max_pages=8)
    monkeypatch.setattr(unified_qb_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    del hist_merge
    monkeypatch.setattr(
        unified_qb_mod, "get_tensor_model_parallel_world_size", lambda: dcp
    )
    monkeypatch.setattr(
        unified_qb_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )
    monkeypatch.setattr(
        unified_qb_mod, "dcp_lane_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )

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

    shd = NamedSharding(mesh, P(AXIS))
    rep = NamedSharding(mesh, P())
    outside_calls: list[int] = []
    unified_calls: list[dict[str, Any]] = []
    lead_cb_kwargs: list[dict[str, Any]] = []

    def _capture_outside(*args, **kwargs):
        del args, kwargs
        outside_calls.append(1)
        return jax.device_put(jnp.zeros((dcp * split_rows, TOPK), jnp.int32), shd)

    def _capture_streamidx(*args, **kwargs):
        del args
        lead_cb_kwargs.append(dict(kwargs))
        return jnp.zeros((split_rows, TOPK), jnp.int32)

    def _capture_unified(*args, **kwargs):
        # Invoke deferred callback if supplied so streamindex_topk kwargs are verified.
        unified_calls.append(
            {
                "head_topk_is_none": args[14] is None,
                "has_select_lead": "select_lead_topk" in kwargs,
                "has_lead_inputs": "lead_topk_inputs" in kwargs,
            }
        )
        if "select_lead_topk" in kwargs:
            sel = kwargs["select_lead_topk"]
            ql, wl, sl_s, qsl_s, dist_s = kwargs["lead_topk_inputs"]
            idx_c = args[3]
            idx_bt = args[6]
            sel(ql, wl, idx_c, sl_s, idx_bt, qsl_s, dist_s, jnp.int32(0))
            return (
                jax.device_put(jnp.zeros((64, 32), jnp.bfloat16), shd),
                jax.device_put(
                    jnp.zeros((dcp * (split_rows + 16), TOPK), jnp.int32), shd
                ),
            )
        return (
            jax.device_put(jnp.zeros((64, 32), jnp.bfloat16), shd),
            jax.device_put(jnp.zeros((64, TOPK), jnp.int32), shd),
        )

    monkeypatch.setattr(unified_qb_mod, "streamindex_topk_dcp", _capture_outside)
    monkeypatch.setattr(unified_qb_mod, "streamindex_topk", _capture_streamidx)
    monkeypatch.setattr(tp, "unified_indexer_in_branch_qb_mla_oproj", _capture_unified)

    spec_n = types.SimpleNamespace(
        page_size=32, num_pages=4, layout=types.SimpleNamespace(value="xla")
    )
    spec_r = types.SimpleNamespace(
        page_size=32,
        num_pages=4,
        token_bytes=16,
        layout=spec_n.layout,
        packed_rope=False,
    )

    class _FakeLayer:
        kv_cache_quantized_dtype = None
        scale = 0.135
        num_heads = 4
        v_head_dim = 6
        use_sparse = True
        mla_kv_spec = (spec_n, spec_r)

    layer = _FakeLayer()
    impl = types.SimpleNamespace(_dcp_mesh=mesh, dcp_size=dcp, dcp_interleave_size=4)
    indexer_op = types.SimpleNamespace(
        topk_tokens=TOPK, k_cache=types.SimpleNamespace(prefix="layer0.idx")
    )
    cb = unified_op_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", KVLayout.SEQ_ALONG_LANE),
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((2, 2, 32), (1, 1, 4)),
        dcp_indexer_a2a_for_bucket=lambda *a: large_bucket,
        register_indexer_layer=lambda prefix: 0,
    )
    built = unified_qb_mod.build_unified_indexer_in_branch_qb_op(
        layer,
        impl,
        indexer_op,
        layout,
        q_replicated=True,
        callbacks=cb,
        in_branch_lead=(enable_flag == "1"),
    )
    assert built is not None

    out = captured_fn[0](
        jax.device_put(jnp.zeros((16, 3, 4, 32), jnp.uint8), shd),
        jax.device_put(jnp.zeros((16, 32, 8), jnp.bfloat16), shd),
        jax.device_put(jnp.zeros((16, 8, 16), jnp.bfloat16), shd),
        jax.device_put(jnp.zeros((dcp * (16 + split_rows), 2, 12), jnp.uint8), shd),
        jax.device_put(jnp.ones((dcp * (16 + split_rows), 2), jnp.bfloat16), shd),
        jax.device_put(jnp.zeros((64, 13), jnp.uint8), rep),
        jax.device_put(jnp.zeros((dcp * (16 + split_rows), 12), jnp.bfloat16), shd),
        jax.device_put(jnp.zeros((12, 48), jnp.float8_e4m3fn), rep),
        jax.device_put(jnp.ones((48,), jnp.float32), rep),
        jax.device_put(jnp.ones((4, 8, 8), jnp.bfloat16), rep),
        jax.device_put(jnp.zeros((dcp * (16 + split_rows), 4), jnp.bfloat16), shd),
        jax.device_put(jnp.zeros((64, 8), jnp.bfloat16), rep),
        jax.device_put(jnp.zeros((64, 4), jnp.bfloat16), rep),
        jax.device_put(jnp.zeros((8,), jnp.int32), rep),
        jax.device_put(jnp.zeros((16,), jnp.int32), rep),
        jax.device_put(jnp.zeros((16,), jnp.int32), rep),
        jax.device_put(jnp.zeros((9,), jnp.int32), rep),
        jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep),
        jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep),
        jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd),
        jax.device_put(jnp.ones((4, 8, 6), jnp.bfloat16), rep),
        jax.device_put(jnp.zeros((24, 32), jnp.float8_e4m3fn), shd),
        jax.device_put(jnp.ones((32,), jnp.float32), shd),
        jax.device_put(jnp.zeros((dcp,), jnp.int32), shd),
    )
    assert out[3].shape == (dcp * (16 + split_rows), TOPK)
    if expect_deferred:
        assert outside_calls == []
        assert unified_calls == [
            {
                "head_topk_is_none": True,
                "has_select_lead": True,
                "has_lead_inputs": True,
            }
        ]
        assert len(lead_cb_kwargs) == 1
        kw = lead_cb_kwargs[0]
        assert kw["dcp_axis_name"] == AXIS
        assert kw["enable_early_exit"] is False
        assert kw["return_scores"] is False
        assert kw["decode_as_mixed"] is True
        assert kw["cp_size"] == dcp
    else:
        assert len(outside_calls) == 1
        assert unified_calls == [
            {
                "head_topk_is_none": False,
                "has_select_lead": False,
                "has_lead_inputs": False,
            }
        ]

# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): L47-FOLD-S1's layer-47 sink export (VLLM_TPU_MK_L47_FOLD) x H3's draft attention
kind (VLLM_TPU_MK_DRAFT_ATTN), at main-KV DCP 1 / 2 / 4, CPU.

Data path (C): with the fold on, the whole step's export is expanded by `qwen38_l47fold.wrap_call` / `sink_rows` and
written into layer 47's `_selection_sink`; the MTP draft's step 0 copies each request's last accepted row of that sink
into its `_selection_rows` (qsa.py, the stock bind_shared_selection path); H3's `DraftAttn` attends
`attn._selection_rows[:n]` in every fused-loop pass. With the fold off, the stock layer-47 attend writes the stock
request-shard path's `selected` into the same sink.

Rules (stated before data), per DCP size: (1) the fold's sink (`sink_rows` of the export rows, and `wrap_call`'s output on
every rank of the CPU mesh, gathered in rank order) equals the stock path's `selected` for the same per-row selection,
column for column, every row incl. idle slots and padded rows; (2) the draft's carried rows taken at each request's
accepted row are equal with the fold on and off, and for live rows equal `selected_tokens` of the draft rows' selection
at their own context (the layout H3's call reads); (3) H3's draft attention call (Pallas interpret, as H3's own
call-vs-today test) on the fold-on rows and on the fold-off rows: attention and every rank's pool bytes BITWISE equal;
(4) against today's draft path on the fold-on rows: within H3's own rule (2 BF16 ulps or 1e-6, live rows; idle rows 0).
H3's case builder, kernel and today's path are H3's own (test_l1h3_draft_attn_call.py, imported); the stock `selected`
rule is L47-FOLD-S1's `_stock_selected` (tests/megakernel/test_l1l47f.py, copied here with credit).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import test_l1h3_draft_attn_call as H3
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens

T = 4  # target rows per request (MTP k = 3)


def _stock_selected(ids, ctx, starts, B, K):
    """L47-FOLD-S1's `_stock_selected` (tests/megakernel/test_l1l47f.py), copied: the stock request-shard path's `selected`
    for a per-row selection (rows past the live tokens -1 blocks, lengths context + t + 1, `visible_blocks` when every row
    fits, else the selection, then `selected_tokens`)."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import visible_blocks

    lens = np.zeros(B, np.int32)
    for q in range(len(ctx)):
        for t in range(starts[q], starts[q + 1]):
            lens[t] = ctx[q] + t - starts[q] + 1
    blocks = np.where(np.arange(K)[None] < np.minimum(lens // 4, K)[:, None], ids, -1)
    blocks = np.where((np.arange(B) < starts[-1])[:, None], blocks, -1)
    if (lens // 4).max() <= K:
        blocks = np.asarray(visible_blocks(jnp.asarray(lens), 4, K))
    return np.asarray(selected_tokens(jnp.asarray(blocks), jnp.asarray(lens), 4 * K, 4))


def _target_step(seed, c, K):
    """A target step whose accepted rows carry the draft rows' selections: request q (one draft row at context ctx_d[q])
    accepted target row a_q, so its target context is ctx_d[q] - a_q - 1 and that row sees ctx_d[q] // 4 groups; its
    export row is the draft row's selection (ascending, the export's form). Other rows: ascending random selections.
    Returns (B, ctx_t, starts, ids, accepted rows, the draft rows' ascending blocks)."""
    rng = np.random.default_rng(seed)
    n, live, ctx_d = c["n"], H3.LIVE, np.asarray(c["ctx"])
    acc = rng.integers(0, T, n)
    acc[:live] = np.minimum(acc[:live], ctx_d[:live] - 1)
    ctx_t = np.zeros(n, np.int32)
    ctx_t[:live] = ctx_d[:live] - acc[:live] - 1
    assert (ctx_t >= 0).all()
    starts = np.zeros(n + 1, np.int32)
    starts[1 : live + 1] = np.arange(1, live + 1) * T
    starts[live + 1 :] = live * T
    B = n * T
    ids = np.zeros((B, K), np.int32)
    dblocks = np.full((n, K), -1, np.int32)
    for q in range(live):
        for t in range(T):
            r = starts[q] + t
            nv = (int(ctx_t[q]) + t + 1) // 4
            sel = np.arange(nv) if nv <= K else np.sort(rng.choice(nv, size=K, replace=False))
            ids[r, : len(sel)] = sel
            if t == acc[q]:
                assert nv == int(ctx_d[q]) // 4
                dblocks[q, : len(sel)] = sel
    rows = np.array([starts[q] + acc[q] if q < live else 0 for q in range(n)], np.int64)  # idle seats: row 0 (clamp_min(0))
    return B, ctx_t, starts, ids, rows, dblocks


@pytest.mark.parametrize("D", [1, 2, 4])
def test_l47f_sink_is_what_draft_attn_consumes(rpa_stand_in, monkeypatch, D):
    from vllm_torchtpu.megakernel_runtime.qwen38_l47fold import sink_rows, sink_width, wrap_call

    H3._interpret_keep_params(monkeypatch)
    H3._route_any_dcp(monkeypatch)
    R = max(D, 2)
    if D > 1:
        from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(D))
    c = H3._case(31 + D, R, D, 1)
    K = H3.TOPK // H3.RATIO
    B, ctx_t, starts, ids, rows, dblocks = _target_step(97 + D, c, K)
    # (1) the fold's sink vs the stock path's selected (the sink with the fold off)
    on = np.asarray(sink_rows(jnp.asarray(ids), jnp.asarray(starts), jnp.asarray(ctx_t), B=B, budget=K))
    off = _stock_selected(ids, ctx_t, starts, B, K)
    assert on.shape == off.shape == (B, sink_width(K)), (on.shape, off.shape)
    np.testing.assert_array_equal(on, off, err_msg="fold sink vs stock selected")
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp, per = P("tp"), B // R
    assert per * R == B
    export = jax.device_put(jnp.asarray(ids.reshape(R * per, 1, K)), jax.sharding.NamedSharding(mesh, tp))
    tail = [jnp.arange(B, dtype=jnp.int32), jnp.asarray(starts), jnp.asarray(ctx_t), jnp.zeros((8, 128), jnp.float32)]
    tail += [jnp.zeros((256,), jnp.float32)] * (2 * 12) + ([jnp.arange(R, dtype=jnp.int32) % D] if D > 1 else [])
    fn, _, _, _ = wrap_call(lambda *a: (a[0], a[0], export), (), (tp, tp, tp), (), mesh=mesh, B=B, n_att=12, dcp=D,
                            gather=True, budget=K)
    ranks = np.asarray(fn(jnp.zeros((R * B, 1), jnp.float32), *tail)[2]).reshape(R, B, -1)
    for rk in range(R):
        np.testing.assert_array_equal(ranks[rk], off, err_msg=f"wrap_call sink rank {rk}")
    # (2) the draft's carried rows (step 0: each request's accepted row of the sink)
    car_on, car_off = ranks[0][rows], off[rows]
    np.testing.assert_array_equal(car_on, car_off, err_msg="carried rows fold on vs off")
    live = H3.LIVE
    want_layout = np.asarray(selected_tokens(jnp.asarray(dblocks), jnp.asarray(c["ctx"]), H3.TOPK, H3.RATIO))
    np.testing.assert_array_equal(car_on[:live], want_layout[:live], err_msg="carried rows vs the draft rows' layout")
    # (3) H3's draft attention call on the fold-on rows vs the fold-off rows: bitwise
    c_on, c_off = dict(c, shared=car_on), dict(c, shared=car_off)
    got_on, pools_on = H3._kernel(c_on, R, D)
    got_off, pools_off = H3._kernel(c_off, R, D)
    assert np.isfinite(got_on).all()
    np.testing.assert_array_equal(got_on, got_off, err_msg="draft attention fold on vs off")
    assert [int((a != b).sum()) for a, b in zip(pools_on, pools_off)] == [0] * R
    # (4) against today's draft path on the fold-on rows (H3's rule)
    want, _ = H3._stock(c_on, R, D)
    lv = np.arange(c["n"]) < live
    err = np.abs(got_on[:, lv] - want[:, lv])
    assert np.abs(want[:, lv]).mean() > 1e-3
    bad = err > 2 * H3._ulp(want[:, lv]) + 1e-6
    selecting = int(sum(1 for q in range(live) if int(c["ctx"][q]) // 4 > K))
    print(f"V5 L47FxH3 D={D} R={R}: sink {B} rows x {on.shape[1]} cols equal on {R} ranks; carried {live} live rows "
          f"({selecting} selecting); draft attention on vs off bitwise; vs today max ulps "
          f"{float((err / H3._ulp(want[:, lv])).max()):.1f}", flush=True)
    assert not bad.any(), ("attention beyond 2 ulps", int(bad.sum()))
    assert (got_on[:, ~lv] == 0).all()
    assert selecting >= 1, "the case needs a selecting request (ctx // 4 > K)"

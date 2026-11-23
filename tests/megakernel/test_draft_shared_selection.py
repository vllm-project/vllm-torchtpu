# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_DRAFT_SHARED_SELECTION: the MK draft kernel attends the target's exported
selection (`QSAFullIndexedBlock.shared_sel`) instead of running its own indexer (CPU).

`shared_selection_inputs` turns the target's shared-selection rows (the sink layout,
`export_selection`) into the kernel's `qi_id` rows and `kept * 2 + g_a flag`. For draft
step 0 the draft rows are the target's rows, so it must give back the target kernel's own
ids; fed those, the draft kernel must produce exactly what its own indexer produces when
that indexer selects the same groups (here: the same kernel, same inputs, run twice).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.indexer import topk_reference
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import (
    export_selection,
    shared_selection_inputs,
)

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY
BUDGET, PAGES, MAX_GROUPS = 64, 8, 1024


def _kernel_rows(contexts, live, K, T, seed):
    """Per verify row the target kernel's exported ids (pool path below g_a, ascending,
    slots past the count 0), its kept count and g_a flag, and the context/starts."""
    rng = np.random.default_rng(seed)
    M = len(contexts) * T
    ids = np.full((M, 1, K), 7, np.int32)  # idle rows: stale
    kept = np.zeros(M, np.int32)
    flag = np.zeros(M, np.int32)
    for q, c in enumerate(contexts):
        if not live[q]:
            continue
        scores = rng.standard_normal(c // 4 + 2).astype(np.float32)
        if seed % 2 == 0:
            scores[c // 4] = 10.0  # g_a selected: the kernel's slot K-1 case
        for t in range(T):
            r, nv, ga = q * T + t, (c + t + 1) // 4, c // 4
            sel = topk_reference(scores, nv, K)
            pool = sorted(g for g in sel if g < ga)
            ids[r, 0] = 0
            ids[r, 0, : len(pool)] = pool
            kept[r], flag[r] = len(pool), int(ga in sel)
    q_len = np.where(live, T, 0)
    starts = np.concatenate([[0], np.cumsum(q_len)]).astype(np.int32)
    return ids, kept, flag, np.asarray(contexts, np.int32), starts


@pytest.mark.parametrize("contexts,K,live", [
    ([3001, 100, 4096, 5], 64, [True] * 4),
    ([255, 256, 257, 258, 259, 260], 64, [True] * 6),
    ([8191, 0, 1023, 200777 % 9000], 512, [True, False, True, True]),
])
def test_step0_rows_give_back_the_kernel_selection(contexts, K, live):
    """Step 0 (draft rows = target rows): the target's ids, kept count and g_a flag."""
    T = 4
    for seed in range(4):
        ids, kept, flag, ctx, starts = _kernel_rows(contexts, live, K, T, seed)
        M = ids.shape[0]
        rows = export_selection(jnp.asarray(ids), jnp.asarray(ctx), jnp.asarray(starts),
                                T=T, K=K, rows=M)
        got_ids, meta = shared_selection_inputs(rows, jnp.asarray(ctx), jnp.asarray(starts),
                                                T=T, K=K, M=M + 8)
        got_ids, meta = np.asarray(got_ids), np.asarray(meta)
        lv = np.repeat(np.asarray(live), T)
        np.testing.assert_array_equal(got_ids[:M][lv], ids[lv])
        np.testing.assert_array_equal(meta[:M][lv] >> 1, kept[lv])
        np.testing.assert_array_equal(meta[:M][lv] & 1, flag[lv])
        assert (meta[:M][~lv] == 0).all() and (meta[M:] == 0).all()  # idle, past B


def test_loop_rows_keep_the_stock_shared_rule():
    """A loop step (T = 1, context k tokens past the target row's): a row that sees at
    most K complete groups attends all of them; else the target row's groups below its
    own g_a (a selected target g_a below it joins the pool path), its g_a only if the
    target selected that same group."""
    K, T0 = 64, 4
    contexts, live = [3001, 100, 1022, 3003], [True] * 4
    for seed in range(4):
        ids, kept, flag, ctx, starts = _kernel_rows(contexts, live, K, T0, seed)
        rows = np.asarray(export_selection(jnp.asarray(ids), jnp.asarray(ctx),
                                           jnp.asarray(starts), T=T0, K=K, rows=16))
        for k in (1, 2, 3):
            pick = np.arange(4) * T0 + 3  # each request's last verified row
            c1 = ctx + 3 + k
            st1 = np.arange(5, dtype=np.int32)
            got_ids, meta = shared_selection_inputs(
                jnp.asarray(rows[pick]), jnp.asarray(c1), jnp.asarray(st1), T=1, K=K, M=4
            )
            got_ids, meta = np.asarray(got_ids)[:, 0], np.asarray(meta)
            for q in range(4):
                nv, ga = (c1[q] + 1) // 4, c1[q] // 4
                grp = rows[pick[q], 0 : 4 * K : 4]
                grp = np.where(grp >= 0, grp // 4, -1)
                if nv <= K:
                    want = list(range(ga))
                    want_flag = int(ga < nv)
                else:
                    want = [g for g in grp if 0 <= g < ga]
                    want_flag = int(ga in grp and ga < nv)
                assert meta[q] >> 1 == len(want) and meta[q] & 1 == want_flag, (q, k)
                np.testing.assert_array_equal(got_ids[q, : len(want)], want)
                assert (got_ids[q, len(want):] == 0).all()


def _draft_run(T, shared, ids_meta=None):
    """The draft-step kernel (one QSA layer first, the served request loop / sharded
    selection / union attention / skip_idle, MoE, tail) on 2 ranks; two live requests,
    the tile's other request slots idle. Returns (residual, tokens, exported ids, d)."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import SEL_IDS, SH_IDS, SH_META
    from vllm_torchtpu.megakernel.wholestep.tail import TOP, greedy_tokens

    R, V, Lt = 2, 384, 1
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        41, G, ex, Lt, budget=BUDGET, n_pages=PAGES, identical_experts=True,
        first="qsa", vocab_shard=V,
    )
    nreq, live = ex.M // T, ex.B // T
    ctx = [3001, 100][:live] + [0] * (nreq - live)
    d = wi.add_indexer_inputs(41, G, ex, d, Lt, max_groups=MAX_GROUPS, ctx=ctx)
    d = wi.add_step_inputs(41, G, ex, d, Lt, ctx=ctx)
    d = wi.add_per_layer(41, G, ex, d, Lt)
    d = wi.add_carrier(41, d)
    d["hbm_tables"] = True
    wi.wholestep_reference(G, ex, d, Lt)
    d.pop("_step_out")
    idx = dict(
        max_groups=MAX_GROUPS, pages_per_seq=d["mtab"].shape[-1], carrier="bf16",
        chunk_pages=8, n_layers=1, hbm_tables=True, per_request_scores=True,
        bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True,
        skip_idle=True, async_dma=True, xmerge=True, req_async=2, xreq_prefetch=3,
        **({"shared_sel": True} if shared else {"export_ids": True}),
    )
    spec = wholestep_spec(
        G, R, L=Lt, qsa=dict(budget=BUDGET), step=True, indexed=idx, first="qsa",
        tail=dict(vocab_shard=V),
    )
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, Lt)
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    extras = list(wi.extras_c(d, mesh, R))
    names = [x.name for x in kernel.extra_inputs(spec, ex)]
    if shared:
        assert names[-2:] == [SH_IDS, SH_META], names
        ids, meta = ids_meta
        tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
        extras += [
            jax.device_put(kernel.fold(jnp.broadcast_to(ids[None], (R,) + ids.shape)), tp),
            jax.device_put(meta, rep),
        ]
    assert len(extras) == len(names)
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *extras)
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
    tok = np.asarray(greedy_tokens(outs[TOP], R, ex.M))[: ex.B]
    sel = None if shared else np.asarray(outs[SEL_IDS]).reshape(R, ex.M, -1)
    return res, tok, sel, d, ex


@pytest.mark.parametrize("T", [1, 4])
def test_draft_kernel_on_the_shared_selection_matches_its_own(T):
    """Fed the selection its own indexer makes (exported, then through the sink layout
    and `shared_selection_inputs`), the shared-selection draft kernel gives byte-identical
    residuals and greedy tokens for the live rows; T = 4 builds each request's union from
    the rows' codes, T = 1 writes the one-row union directly."""
    res_a, tok_a, sel, d, ex = _draft_run(T, shared=False)
    ctx = jnp.asarray(d["ctxq"], jnp.int32)
    live = np.asarray(d["ctxq"]) > 0
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(np.where(live, T, 0))]), jnp.int32)
    rows = export_selection(jnp.asarray(sel[0]), ctx, starts, T=T, K=BUDGET, rows=ex.B)
    ids, meta = shared_selection_inputs(rows, ctx, starts, T=T, K=BUDGET, M=ex.M)
    np.testing.assert_array_equal(np.asarray(ids)[: ex.B, 0], sel[0, : ex.B])
    assert (np.asarray(meta)[ex.B :] == 0).all()
    res_b, tok_b, _, _, _ = _draft_run(T, shared=True, ids_meta=(ids, meta))
    assert np.isfinite(res_b).all()
    np.testing.assert_array_equal(res_b, res_a)
    np.testing.assert_array_equal(tok_b, tok_a)


def test_shared_selection_off_leaves_the_draft_spec_unchanged(monkeypatch):
    """Flag off: no `shared_sel`, no extra inputs (the spec, hence the kernel, as before)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime import qwen38_draft as qd
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_SHARED_SELECTION", False)
    monkeypatch.setattr(wsc, "SHARD_INDEX", True)
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "LOOP_REQUESTS", True)
    assert not qd.shared_selection_on()
    kw = dict(budget=512, max_groups=65536, vocab_shard=31040, B=16)
    ex = ExecSpec(R=8, B=16, T=4, ctx=128, tiles=_TILES)
    off = qd.draft_spec(qwen38.FULL, 8, **kw)
    on = qd.draft_spec(qwen38.FULL, 8, shared=True, **kw)
    names_off = [x.name for x in kernel.extra_inputs(off, ex)]
    names_on = [x.name for x in kernel.extra_inputs(on, ex)]
    assert "qsa_sh_ids" not in names_off and names_on == names_off + ["qsa_sh_ids",
                                                                     "qsa_sh_meta"]
    blk_off = wsc._find_qsa_block(off)
    assert blk_off.shared_sel is False and wsc._find_qsa_block(on).shared_sel is True


def test_shared_selection_needs_the_export(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime import qwen38_draft as qd

    for name in ("VLLM_TPU_MK_WHOLE_EXPORT_SELECTION", "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION",
                 "VLLM_TPU_QSA_DRAFT_SELECTION_SLICE"):
        monkeypatch.setattr(envs, name, True)
    qd.check_shared_selection()
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DRAFT_SELECTION_SLICE", False)
    with pytest.raises(ValueError, match="SELECTION_SLICE"):
        qd.check_shared_selection()

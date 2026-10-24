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
"""Whole-step prototype: two `[GDN, GDN, GDN, QSA]` groups chained in one kernel (CPU).

Every QSA layer is a full in-kernel layer (HC mix, projection, norms, RoPE,
sparse attention, gate, o_proj). The reference chains the library's block
references with `project_reference` + `reference_rows` for the QSA layers.
"""

from __future__ import annotations

from typing import Any

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
from vllm_torchtpu.megakernel.ops import BF16
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY
BUDGET, PAGES, L = 64, 8, 8
KV_LEN = [0, 1, 9, 100, 255, 257, 130, 64]  # empty row, single token, tile edges


def _run(
    key: int = 11, T: int = 4, zero: tuple[str, ...] = (), identical_experts: bool = False
):
    R = 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    # Router gain: eight chained layers drift enough for a near-tied top-k router logit
    # to flip one row's expert set; the library tests use the same knob for the same reason.
    d = wi.random_wholestep_inputs(
        key, G, ex, L, budget=BUDGET, n_pages=PAGES, kv_len=KV_LEN, zero=zero,
        identical_experts=identical_experts,
    )
    ref, kv_ref = wi.wholestep_reference(G, ex, d, L)
    spec = wholestep_spec(G, R, L=L, qsa=dict(budget=BUDGET))
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == wi.EXTRA_NAMES
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras(d, mesh, R))
    B = ex.B
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    kv = np.asarray(outs[KV_NEW], np.float32).reshape(R, L // G.att_every, ex.M, -1)[:, :, :B]
    x = np.asarray(d["inp"].x[:B], np.float32)
    return res, np.asarray(ref, np.float32), x, kv, np.asarray(kv_ref, np.float32)


def _assert_within_budget(res: np.ndarray, ref: np.ndarray, layers: int = L) -> None:
    """The segment test's budget (mean 1e-2, q999 5e-2, max 1e-1 of mean |ref|, set for
    L = 4: eight BF16-rounded sublayers), scaled by sqrt(sublayers / 8): BF16 rounding of the
    residual stream accumulates as a random walk over the sublayers. Measured with identical
    experts (no router flips, no outlier rows): L = 4 mean 0.0046 / q999 0.0475 / max 0.098;
    L = 8 mean 0.0084 / q999 0.0590 / max 0.0755 (library segment L = 4: 0.0047 / 0.0446 /
    0.047)."""
    k = np.sqrt(2 * layers / 8)
    scale = float(np.abs(ref).mean())
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * k * scale, (r, d.mean(), scale)
        assert np.quantile(d, 0.999) < 5e-2 * k * scale, (r, np.quantile(d, 0.999), scale)
        assert d.max() < 1e-1 * k * scale, (r, d.max(), scale)


def test_two_groups_identical_experts_match_reference():
    """Strict: with routing flips made harmless, every row is inside the budget."""
    res, ref, x, kv, kv_ref = _run(identical_experts=True)
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)
    # The step's K/V rows (cache-write payload): the first QSA layer's input has seen three
    # layers of drift, the second's seven; both inside the residual's depth-scaled budget.
    np.testing.assert_allclose(kv[:, 0], kv_ref[:, 0], rtol=2e-2, atol=2e-2)
    for o in range(kv.shape[1]):
        for r in range(kv.shape[0]):  # per-rank projections: per-rank references
            _assert_within_budget(kv[r : r + 1, o], kv_ref[r, o], layers=4 * (o + 1))


def test_two_groups_random_experts_statistical():
    """Random experts: a 1-ulp difference upstream of a router may flip one row's top-k
    (seen as a single-row outlier, absent with MoE silenced); bound the mean and the
    number of outlier rows instead of every element."""
    res, ref, x, _, _ = _run()
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * L / 8)
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * k * scale, (r, d.mean(), scale)
        assert int((d.max(axis=1) > 1e-1 * k * scale).sum()) <= 2, (r, d.max(axis=1), scale)


def test_two_groups_budget_catches_a_dropped_qsa_layer(monkeypatch):
    """Positive control: the strict budget rejects a kernel whose QSA layers add nothing."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full import QSAFullBlock

    orig = QSAFullBlock.emit
    monkeypatch.setattr(QSAFullBlock, "emit", lambda self, ctx, l, slot: 0.0 * orig(self, ctx, l, slot))
    res, ref, _, _, _ = _run(identical_experts=True)
    with pytest.raises(AssertionError):
        _assert_within_budget(res, ref)


# --------------------------------------------------------------------- stage B
MAX_GROUPS = 1024
CTX = [3000, 100]  # request 0: n_vis 750 > K (top-K path); request 1: n_vis 25 <= K (select all)


def _run_b(key: int = 13, T: int = 4, identical_experts: bool = True):
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ

    R = 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        key, G, ex, L, budget=BUDGET, n_pages=PAGES, identical_experts=identical_experts
    )
    d = wi.add_indexer_inputs(key, G, ex, d, L, max_groups=MAX_GROUPS, ctx=CTX)
    ref, _ = wi.wholestep_reference(G, ex, d, L)
    pps = d["mtab"].shape[1]
    spec = wholestep_spec(
        G, R, L=L, qsa=dict(budget=BUDGET),
        indexed=dict(max_groups=MAX_GROUPS, pages_per_seq=pps, debug=True),
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == wi.EXTRA_NAMES_B
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_b(d, mesh, R))
    B, n_att = ex.B, L // G.att_every
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    iq = np.asarray(outs[DBG_IQ], np.float32).reshape(R, n_att, ex.M, -1)[0, :, :B]
    ids = np.asarray(outs[DBG_IDS]).reshape(R, n_att, ex.M, -1)[0, :, :B]
    return res, np.asarray(ref, np.float32), np.asarray(d["inp"].x[:B], np.float32), iq, ids, d, ex


def test_in_kernel_selection_exact_on_its_own_queries():
    """Strict: the kernel's selected group ids equal the stock rule (top-K, ties -> lower id;
    all visible groups when n_vis <= K) applied to the kernel's own index queries, every row
    and QSA layer. Separates selection logic (must be exact) from upstream rounding."""
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference

    _, _, _, iq, ids, d, ex = _run_b()
    T = ex.T
    for o in range(iq.shape[0]):
        for r in range(ex.B):
            qr = jnp.asarray(iq[o, r].reshape(4, -1), BF16)
            st = np.asarray(score_rows(qr, d["ck"][r // T]))[0]
            n = int(d["nvis"][r])
            want = set(topk_reference(st, n, BUDGET))
            got = set(ids[o, r, : min(n, BUDGET)].tolist())
            assert got == want, (o, r, len(got ^ want))


def test_two_groups_in_kernel_indexer_match_reference():
    """End to end: select-all rows (request 1) strictly inside the depth-scaled budget;
    top-K rows (request 0) may see a boundary flip from 1-ulp drift upstream of the indexer
    (as MoE routing does), so they are bounded statistically."""
    res, ref, x, _, _, _, ex = _run_b()
    assert float(np.std(ref - x)) > 0.05
    T = ex.T
    _assert_within_budget(res[:, T : 2 * T], ref[T : 2 * T])
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * L / 8)
    for r in range(res.shape[0]):
        d = np.abs(res[r, :T] - ref[:T])
        assert int((d.max(axis=1) > 1e-1 * k * scale).sum()) <= 2, (r, d.max(axis=1), scale)


def test_selection_check_catches_an_off_by_one_id(monkeypatch):
    """Positive control for the strict selection check: ids shifted by one group fail it."""
    from vllm_torchtpu.megakernel.wholestep import indexer as ix

    orig = ix.compact_row
    monkeypatch.setattr(ix, "compact_row", lambda *a, **k: orig(*a, **k) + 1)
    with pytest.raises(AssertionError):
        test_in_kernel_selection_exact_on_its_own_queries()


# --------------------------------------------------------------------- stage C
CTX_C = [3001, 100]  # request 0: g_a holds one old token; request 1: g_a starts at the step


def _run_c(key: int = 17, T: int = 4):
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, RING_NEW

    R = 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(key, G, ex, L, budget=BUDGET, n_pages=PAGES, identical_experts=True)
    d = wi.add_indexer_inputs(key, G, ex, d, L, max_groups=MAX_GROUPS, ctx=CTX_C)
    d = wi.add_step_inputs(key, G, ex, d, L, ctx=CTX_C)
    ref, _ = wi.wholestep_reference(G, ex, d, L)
    steps = d.pop("_step_out")
    pps = d["mtab"].shape[1]
    spec = wholestep_spec(
        G, R, L=L, qsa=dict(budget=BUDGET), step=True,
        indexed=dict(max_groups=MAX_GROUPS, pages_per_seq=pps, debug=True),
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == wi.EXTRA_NAMES_C
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    B, n_att, nreq = ex.B, L // G.att_every, ex.M // T
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    ck = np.asarray(outs[CK_NEW], np.float32).reshape(R, n_att, nreq, -1)[0]
    ring = np.asarray(outs[RING_NEW], np.float32).reshape(R, n_att, nreq, 8, -1)[0]
    iq = np.asarray(outs[DBG_IQ], np.float32).reshape(R, n_att, ex.M, -1)[0, :, :B]
    ids = np.asarray(outs[DBG_IDS]).reshape(R, n_att, ex.M, -1)[0, :, :B]
    x0 = np.asarray(d["inp"].x[:B], np.float32)
    return res, np.asarray(ref, np.float32), x0, ck, ring, steps, iq, ids, d


def test_step_selection_exact_on_its_own_queries():
    """Strict: the pool-path ids are the stock selection (from the kernel's own queries and
    its own completed-group key) restricted to groups older than the step, every row/layer."""
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference

    _, _, _, ck, _, _, iq, ids, d = _run_c()
    T = 4
    for o in range(iq.shape[0]):
        for r in range(iq.shape[1]):
            q = r // T
            c0 = int(d["ctxq"][q])
            p3 = c0 + (3 - c0 % 4) % 4
            qr = jnp.asarray(iq[o, r].reshape(4, -1), BF16)
            st = np.array(score_rows(qr, d["ck"][q]))[0]
            st[p3 // 4] = np.asarray(
                score_rows(qr, jnp.broadcast_to(jnp.asarray(ck[o, q], BF16), (8, ck.shape[-1])))
            )[0, 0]
            n = int(d["nvis"][r])
            want = {i for i in topk_reference(st, n, BUDGET) if i < c0 // 4}
            got = set(ids[o, r, : len(want)].tolist())
            assert got == want, (o, r, len(got ^ want))


def test_step_tokens_match_stock_semantics():
    """Stage C: the completed group's key and the new ring rows equal the stock write for
    every request and QSA layer (bf16-close: their inputs carry the residual's drift); the
    residual, with every row's own tokens attended from VMEM, is inside the depth-scaled budget
    on select-all rows (request 1). Top-K rows (request 0, 750 groups -> 64) are not held to
    the per-row budget: a 1-ulp drift of the index query flips boundary groups there (the
    selection itself is exact on the kernel's own queries, the strict test above); only their
    mean error is bounded, at 3x the depth-scaled mean budget."""
    res, ref, x, ck, ring, steps, _, _, _ = _run_c()
    assert float(np.std(ref - x)) > 0.05
    for o, s in enumerate(steps):
        for q in range(len(CTX_C)):
            np.testing.assert_allclose(ck[o, q], np.asarray(s["ck"][q], np.float32)[0], atol=0.1, rtol=0.05)
            np.testing.assert_allclose(ring[o, q], np.asarray(s["ring"][q], np.float32), atol=0.1, rtol=0.05)
    T = 4
    _assert_within_budget(res[:, T : 2 * T], ref[T : 2 * T])
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * L / 8)
    for r in range(res.shape[0]):
        dd = np.abs(res[r, :T] - ref[:T])
        assert dd.mean() < 3e-2 * k * scale, (r, dd.mean() / scale, dd.max(axis=1) / scale)

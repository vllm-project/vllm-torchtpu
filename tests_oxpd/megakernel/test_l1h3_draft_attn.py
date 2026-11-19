# SPDX-License-Identifier: Apache-2.0
"""q38 L1-H3 (H3-DRAFT-ATTN-D14): the draft attention kind (`wholestep.qsa_draft`) on CPU (Pallas interpret).

The kind attends, per request, one row over a host-given selection (the stock draft's carried target
selection in `pooled_qsa.shared_blocks` layout) plus the row's recent tile, on D14's union and DCP
exchange blocks, and returns the gated attention before o_proj. Reference = `wholestep.inputs.
step_attention`, the stock semantics the whole-step tests hold D14 to (selected complete groups from the
pool, the step's own tokens FP8 write-then-read, the partial group causal), per rank on the unsplit pool.

Rule (stated before data): every attended row's output is within 2 BF16 ulps of the reference elementwise
(m1's draft-attention bar, d1eea8ba2), at main-KV DCP 1 / 2 / 4, for the fused loop's passes (T = 1) and
draft step 0 (T = 4, one draft row per request at different row slots), and every value is finite. The
DCP runs hold each rank a 1 / D share of the main KV (`_dcp_split`, the rest NaN-poisoned). The cases
cover: a select-all row (`n_vis <= K`), top-K rows, g_a listed and complete for the row (ctx % 4 == 3),
a fresh group (ctx % 4 == 0), page / DCP-owner boundaries, idle padding requests.
"""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.test_wholestep import _TILES, G, _dcp_split
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models.qwen38_inputs import _rep, _rows
from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights, shard_step_inputs
from vllm_torchtpu.megakernel.ops import BF16, F32
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep import qsa_draft as qd
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW, project_reference, qkvg_width
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import KR_NEW

MG, K = 1024, 64
# T = 1 (loop pass): ctx = tokens before the pass. 3071: g_a 767 complete for the row and listed; 1024: a fresh
# group on a new page (DCP-4 owner 0); 37: select-all; 255: g_a 63 complete, last group of page 1 (owner 1).
CTX_LOOP = [3071, 37, 1024, 255]
# T = 4 (step 0): the draft row per request at slot 3, 0, 2, 1 (ctx 3070: rows cross page 23 -> 24).
CTX_STEP0 = [3070, 37, 1021, 254]
ATT_STEP0 = [3, 0, 2, 1]


def _blocks(rng, lens: np.ndarray, attended: np.ndarray, ctxq: np.ndarray, T: int) -> np.ndarray:
    """Stock `shared_blocks` rows: every visible group where they fit, else K distinct groups (always
    including g_a when it is complete for the row, so the recent tile's listed-g_a path runs)."""
    M = len(lens)
    out = np.full((M, K), -1, np.int32)
    for r in range(M):
        if not attended[r]:
            continue
        n = int(lens[r]) // 4
        if n <= K:
            out[r, :n] = np.arange(n)
            continue
        ga = int(ctxq[r // T]) // 4
        pick = rng.permutation(n)[:K]
        if ga < n and ga not in pick:
            pick[0] = ga
        out[r] = rng.permutation(pick)
    return out


def _qsa_inputs(key: int, g, ex: ExecSpec) -> dict:
    """The QSA layer's weights and RoPE rows of `random_wholestep_inputs` (one layer, no GDN / MoE)."""
    R, nh, hd = ex.R, g.NH // ex.R, g.HD
    k = jax.random.split(jax.random.PRNGKey(key + 1), 8)
    qkvg = (jax.random.normal(k[1], (R, 1, g.H, qkvg_width(nh, hd)), F32) * 0.06).astype(BF16)
    qn = jax.random.normal(k[3], (1, hd), F32) * 0.05
    kn = jax.random.normal(k[4], (1, hd), F32) * 0.05
    pos = np.random.default_rng(key).integers(0, 1 << 17, size=(ex.M,))
    inv = 1.0 / (g.theta ** (np.arange(0, g.ROT, 2) / g.ROT))
    ang = pos[:, None] * inv[None, :]
    rope = np.zeros((ex.M, 128), np.float32)
    rope[:, : g.ROT // 2] = np.cos(ang).astype(ml_dtypes.bfloat16).astype(np.float32)
    rope[:, g.ROT // 2 : g.ROT] = np.sin(ang).astype(ml_dtypes.bfloat16).astype(np.float32)
    return dict(qkvg=qkvg, qn=qn, kn=kn, rope=jnp.asarray(rope), off=1)


def _ulp(x: np.ndarray) -> np.ndarray:
    a = np.abs(x).astype(np.float32)
    e = np.floor(np.log2(np.where(a > 0, a, 1e-30)))
    return np.where(a > 0, 2.0 ** (e - 7), 2.0**-133)


def run_draft(key: int = 5, T: int = 1, R: int = 2, dcp: int = 1, ctx=None, att=None, split: bool = True,
              phys: bool = False, H: int = 0):
    """One draft-kind call: `(out [R, B, nh*HD], ref [R, B, nh*HD], attended [B], kv [R, B, 512], kv_ref)`."""
    ctx = list(CTX_LOOP if ctx is None else ctx)
    nreq = len(ctx)
    Gx = dataclasses.replace(G, NH=3 * R)
    if R > 2:
        Gx = dataclasses.replace(Gx, KH=R, VH=3 * R)
    if H:  # a block width the core does not chunk (Qwen3.8's 2560 class): input zero-padded to `qd.core_width`
        Gx = dataclasses.replace(Gx, H=H)
    ex = ExecSpec(R=R, B=nreq * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    M, B, nh, HD = ex.M, ex.B, 3, Gx.HD
    d = _qsa_inputs(key, Gx, ex)
    d = wi.add_indexer_inputs(key, Gx, ex, d, 1, max_groups=MG, ctx=ctx)
    d = wi.add_step_inputs(key, Gx, ex, d, 1, ctx=ctx)
    # One DCP group of R ranks shares one KV head: same K / V columns, same pool (as `_run_c(shared_kv=True)`).
    kvw = 2 * HD
    d["qkvg"] = d["qkvg"].at[1:, ..., -kvw:].set(jnp.broadcast_to(d["qkvg"][:1, ..., -kvw:], d["qkvg"][1:, ..., -kvw:].shape))
    d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
    rng = np.random.default_rng(key)
    hidden = jnp.asarray(rng.standard_normal((M, Gx.H)) * 0.5, BF16).at[B:].set(0)
    lens = np.asarray(d["nvis"]) * 4 + np.asarray(d["tail"])
    ctxq = np.asarray(d["ctxq"])
    attended = np.zeros((M,), bool)
    for q in range(nreq):
        attended[q * T + (0 if att is None else att[q])] = True
    blocks = _blocks(rng, lens, attended, ctxq, T)
    sel = qd.selection_sets(blocks, lens, attended, 4)
    # Reference per rank on the unsplit pool (q heads differ per rank, K / V shared).
    refs, kv_refs = [], []
    for r_ in range(R):
        q, gate, kk, vv = project_reference(hidden[:B], d["qkvg"][r_, 0], d["qn"][0], d["kn"][0], d["rope"][:B],
                                            nh=nh, hd=HD, rot=Gx.ROT, eps=Gx.eps)
        full = {r: sel.get(r, set()) for r in range(B)}
        refs.append(np.asarray(wi.step_attention(Gx, ex, d, full, q, gate, kk, vv, r_), np.float32).reshape(B, -1))
        kv_refs.append(np.concatenate([np.asarray(kk, np.float32), np.asarray(vv, np.float32)], axis=-1))
    if dcp > 1 and split:
        _dcp_split(d, dcp)
    page = int(d["pool"].shape[2])
    KU = -(-(T * K) // 128) * 128
    rows = [qd.shared_rows(blocks, lens, ctxq, attended, T=T, KU=KU, ratio=4, page_size=page, dcp=dcp,
                           dcp_rank=r_ % max(dcp, 1)) for r_ in range(R)]
    shared = jnp.stack([x[0] for x in rows])
    meta = rows[0][1]
    pps = int(d["mtab"].shape[1])
    spec = qd.draft_attn_spec(Gx, R, budget=K, indexed=dict(
        max_groups=MG, pages_per_seq=pps, hbm_tables=True, per_request_scores=True, union_attn=True,
        loop_requests=True, union_phys=phys, dcp=dcp))
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == qd.EXTRA_NAMES
    w = pack_weights(spec, ex, {"qkvg": d["qkvg"]},
                     {"qsa_qn": _rows(_rep(d["qn"], R), (0,), 1), "qsa_kn": _rows(_rep(d["kn"], R), (0,), 1)})
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    hid_core = jnp.pad(hidden, ((0, 0), (0, qd.core_width(Gx.H) - Gx.H)))
    pos, hid, w, st = shard_step_inputs(mesh, ex, jnp.zeros((M,), jnp.int32), hid_core, w, {})
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    extras = (
        jax.device_put(kernel.fold(d["pool"]), tp),
        jax.device_put(d["nvis"], rep),
        jax.device_put(d["tail"], rep),
        jax.device_put(kernel.fold(_rep(jnp.asarray(d["mtab"])[None], R)), tp),
        jax.device_put(jnp.asarray([d["off"]], jnp.int32), rep),
        jax.device_put(jnp.asarray([1.0, 1.0], F32), rep),
        jax.device_put(kernel.fold(_rep(d["rope"], R)), tp),
        jax.device_put(d["ctxq"], rep),
        jax.device_put(kernel.fold(shared), tp),
        jax.device_put(meta, rep),
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos, hid, w, st, *extras)
    out = np.asarray(outs[qd.ATTN_OUT], np.float32).reshape(R, M, -1)[:, :B]
    kv = np.asarray(outs[KV_NEW], np.float32).reshape(R, M, -1)[:, :B]
    kr = np.asarray(outs[KR_NEW], np.float32).reshape(R, M, -1)[:, :B]
    return out, np.stack(refs), attended[:B], kv, np.stack(kv_refs), kr


def check(out, ref, attended, ulps: int = 2) -> dict:
    a = out[:, attended]
    b = ref[:, attended]
    assert np.isfinite(out).all()
    err = np.abs(a - b)
    lim = ulps * _ulp(b) + 1e-6
    bad = err > lim
    stats = dict(rows=int(attended.sum()), exact=float((err == 0).mean()), max_ulps=float((err / _ulp(b)).max()),
                 scale=float(np.abs(b).mean()))
    print(f"L1-H3 draft attention vs reference: {stats}", flush=True)
    assert float(np.abs(b).mean()) > 1e-3, "degenerate reference"
    assert not bad.any(), (int(bad.sum()), stats)
    return stats


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_loop_pass_matches_stock(monkeypatch, dcp):
    """T = 1 (fused-loop pass): 4 requests, R = dcp ranks of one DCP group (R = 2 at DCP 1)."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    out, ref, att, kv, kv_ref, _ = run_draft(T=1, R=max(dcp, 2), dcp=dcp)
    check(out, ref, att)


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_step0_matches_stock(monkeypatch, dcp):
    """T = 4 (draft step 0): every verify row projected (K / V out for the writer), one draft row per request
    attended at slots 3 / 0 / 2 / 1."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    out, ref, att, kv, kv_ref, kr = run_draft(T=4, R=max(dcp, 2), dcp=dcp, ctx=CTX_STEP0, att=ATT_STEP0)
    check(out, ref, att)
    # Every verify row's K / V leaves the kernel for the post-kernel writer (normed + roped K | V).
    np.testing.assert_allclose(kv[0], kv_ref[0], atol=0.05, rtol=0.02)
    assert np.isfinite(kr).all() and np.abs(kr).mean() > 1e-3


def test_core_width_pad_matches(monkeypatch):
    """H = 1280 (not a multiple of min(1024, H), as Qwen3.8's 2560): the core runs at 2048 columns, the input is
    zero-padded and the projection reads the true 1280 (DCP 4, T = 1)."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    assert qd.core_width(1280) == 2048 and qd.core_width(2560) == 3072 and qd.core_width(256) == 256
    assert qd.core_width(10240) == 10240
    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "4")
    out, ref, att, *_ = run_draft(T=1, R=4, dcp=4, H=1280)
    check(out, ref, att)


def test_union_phys_matches(monkeypatch):
    """`union_phys` (D14's served `wholephys`): physical micro-page ids per request, same addresses."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "4")
    out, ref, att, *_ = run_draft(T=1, R=4, dcp=4, phys=True)
    check(out, ref, att)


def test_dcp4_unsplit_pool_is_caught(monkeypatch):
    """Negative control: the DCP-4 kind on the unsplit pool (each rank reads rows it does not own as its own)
    must fail the rule, so the rule sees the ownership."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "4")
    out, ref, att, *_ = run_draft(T=1, R=4, dcp=4, split=False)
    with pytest.raises(AssertionError):
        check(out, ref, att)


def test_block_refuses_selection_options():
    """The kind has no indexer: the selection options and per-layer tables are refused; the DCP-4 variant is
    picked above DCP 2."""
    base = dict(max_groups=MG, pages_per_seq=32, hbm_tables=True, per_request_scores=True, union_attn=True,
                loop_requests=True)
    assert isinstance(qd.draft_attn_spec(G, 4, budget=K, indexed={**base, "dcp": 4}).layer_type.slots[0],
                      qd.QSADraftAttnGroupBlock)
    assert type(qd.draft_attn_spec(G, 2, budget=K, indexed={**base, "dcp": 2}).layer_type.slots[0]) is qd.QSADraftAttnBlock
    for bad in (dict(shard_index=True), dict(union_attn=False), dict(loop_requests=False), dict(n_layers=2),
                dict(hbm_tables=False)):
        with pytest.raises(AssertionError):
            qd.QSADraftAttnBlock(**{**base, **bad}, budget=K)
    blk = qd.QSADraftAttnBlock(**base, budget=K)
    assert blk.reduction == "none"

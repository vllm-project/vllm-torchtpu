# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_REPACK_L2 (q38 L2 D74, the "L2" rung) on CPU.

L2 lands a routed load with ONE descriptor-driven Pallas program: run descriptors (src row, dst row, rows, kind),
power-of-two DMAs, staging to pool directly, main KV through VMEM for its head select. Checked here:
  - the GDN template (whole 1 KiB conv rows) against make_repack's compact GDN rows;
  - the reference builder (run-length encoding of ``indices()``) writes exactly v2's rows outside the null page;
  - the production builder (O(pages), from page lists) equals the reference row for row, on random loads with
    prefix skips, draft gaps, every rank, at the served S12, O6 and a small geometry, and a gdn_pages=2 partial page;
  - the interpret-mode L2 kernel lands the pool byte-identical to v2's CPU path (G1, every page), both heads,
    pool kernel pages smaller than manager pages, uint8 and FP8 carriers;
  - plumbing: plan_repack / run_repack route through L2 and fall back to v2 per load.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

_DIR = pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu/models/vllm/qwen4_exp"
_spec = importlib.util.spec_from_file_location("pd_staging_l2_under_test", _DIR / "pd_staging.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)
rt = st.pd_route
ROWS, LANES = 4, 256


def gdn_pair(src, dst, conv_width=3):
    """(src, dst) GdnGeometry for (v, k) heads at head dims 128, f32 state (as the bench's gdn_geoms)."""
    tok = ROWS * LANES

    def rows_for(vh, kh):
        ssm = -(-vh * 128 * 128 * 4 // tok)
        conv_bytes, conv = conv_width * (2 * kh * 128 + vh * 128) * 4, 1
        while conv * tok < conv_bytes:
            conv *= 2
        return ssm, conv

    def mk(vh, kh):
        ssm, conv = rows_for(vh, kh)
        return st.GdnGeometry(ssm_tokens=ssm, conv_tokens=conv, v_heads=vh, k_heads=kh, head_k_dim=128,
                              head_v_dim=128, conv_width=conv_width, ssm_itemsize=4, conv_itemsize=4)

    return mk(*src), mk(*dst)


# name: (RouteGeometry, src heads, dst heads, FA layers, GDN layers)
GEOS = {
    "s12": (rt.RouteGeometry.fitted(tp=8, dcp=4, dst_block=512, src_block=3200, ratio=4, ssm_rows_src=3072,
                                    conv_rows_src=128, ssm_rows_dst=384), (48, 16), (6, 2), 12, 36),
    "o6": (rt.RouteGeometry.fitted(tp=4, dcp=2, dst_block=1024, src_block=3200, ratio=4, ssm_rows_src=3072,
                                   conv_rows_src=128, ssm_rows_dst=768), (48, 16), (12, 4), 4, 6),
    # dst_block 256: a GDN state page (128 SSM + 16 conv rows) must fit one pool page, as 400 of 512 rows do at S12
    "small": (rt.RouteGeometry.fitted(tp=4, dcp=2, dst_block=256, src_block=320, ratio=4, ssm_rows_src=512,
                                      conv_rows_src=64, ssm_rows_dst=128), (8, 8), (2, 2), 2, 3),
}


def rank_of(geo, r):
    return st.RankSlice(tp_size=geo.tp, tp_rank=r, num_kv_heads=2, main_dcp=geo.dcp)


def owners(fa, gdn, draft):
    names = []
    for i in range(fa):
        names += [f"l{i}.self_attn", f"l{i}.self_attn.compressed_k", f"l{i}.self_attn.raw_ring"]
    if draft:
        names += ["mtp.0.self_attn", "mtp.0.self_attn.compressed_k", "mtp.0.self_attn.raw_ring"]
    return names + [f"g{i}.linear_attn" for i in range(gdn)] + ["p.ple"]


class Load:
    """One routed load on a shell StagedConsumer (``_routed_indices``' inputs): random distinct staging pages per
    owner (``pd_route.staging_pages``) and random distinct destination pages per group."""

    def __init__(self, name, r, num_tokens, skip, draft_end=0, seed=0, draft=True, kernel_rows=None, fa=None,
                 gdn=None):
        geo, src_h, dst_h, fa0, gdn0 = GEOS[name]
        fa, gdn = fa or fa0, gdn if gdn is not None else gdn0
        self.geo, self.rank = geo, rank_of(geo, r)
        self.s_gdn, self.d_gdn = gdn_pair(src_h, dst_h)
        assert self.s_gdn.ssm_tokens == geo.ssm_rows_src and self.d_gdn.ssm_tokens == geo.ssm_rows_dst
        self.channels = st.conv_channels(self.s_gdn, self.d_gdn, self.rank)
        rng = np.random.default_rng(seed)
        self.names = owners(fa, gdn, draft)
        S, B = geo.stage_rows, geo.dst_block
        n_src = -(-num_tokens // geo.src_block)
        counts = {n: rt.staging_pages(st.owner_kind(n), num_tokens, n_src if st.owner_kind(n) in ("main", "compressed")
                                      else 1, skip, geo) for n in self.names}
        perm = list(rng.permutation(np.arange(1, sum(counts.values()) + 1 + 9)))
        self.pages = {n: [int(perm.pop()) for _ in range(counts[n])] for n in self.names}
        self.n_stage = max(max(p) for p in self.pages.values()) + 1
        entries = -(-num_tokens // (geo.dcp * B))
        pages_per = {"main": entries, "compressed": -(-num_tokens // B), "ring": 1, "ple": 1, "gdn": 1}
        need = sum(pages_per[st.owner_kind(n)] for n in self.names)
        dperm = list(rng.permutation(np.arange(1, need + 1 + 7)))
        self.tables = {n: [int(dperm.pop()) for _ in range(pages_per[st.owner_kind(n)])] for n in self.names}
        self.n_pool = need + 8
        self.kernel_rows = kernel_rows or B
        self.num_tokens, self.skip, self.draft_end = num_tokens, skip, draft_end
        c = object.__new__(st.StagedConsumer)
        c.route, c.rank, c.ratio, c.dst_block = geo, self.rank, geo.ratio, B
        c._dst_compressed = {n: (n, 0) for n in self.names if st.owner_kind(n) == "compressed"}
        c._dst_owner = {n: n for n in self.names}
        c._gdn_names = [n for n in self.names if st.owner_kind(n) == "gdn"]
        c._state_entry, c._zero_draft, c.shared, c.draft_kv = 0, {}, None, draft
        c.reservations, c.skips, c.draft_zero = {1: self.pages}, {1: (skip, None)}, {}
        c._l2_gdn = st.l2_gdn_template(self.s_gdn, self.d_gdn, self.channels)
        c.pool = SimpleNamespace(shape=(self.n_pool * B // self.kernel_rows, self.kernel_rows, 1, ROWS, LANES))
        self.c = c

    def indices(self):
        return self.c.indices(self.pages, self.tables, self.num_tokens, self.skip, None,
                              draft_zero_until=self.draft_end)

    def reference(self):
        idx = self.indices()
        runs = st.l2_reference_runs(idx, self.rank.kv_head, self.c._l2_gdn, self.geo.stage_rows, self.geo.dst_block)
        return st.l2_pieces(runs, self.kernel_rows), idx

    def production(self):
        c = self.c
        runs, landed, zeroed, rows = st.l2_routed_runs(
            g=self.geo, rank=self.rank, src_pages=self.pages, dst_tables=self.tables, num_tokens=self.num_tokens,
            skip=self.skip, draft_end=self.draft_end, dst_compressed=c._dst_compressed, dst_owner=c._dst_owner,
            gdn_names=c._gdn_names, state_entry=0, dst_block=self.geo.dst_block, ratio=self.geo.ratio,
            template=c._l2_gdn)
        return st.l2_pieces(runs, self.kernel_rows), (landed, zeroed), rows


def v2_rows(idx, d_gdn, dst_block, kv_head):
    """v2's written (dst row -> kind, src row) map outside the null page: main rows (HEAD), copy rows (FILL from
    staging row 0, else COPY), GDN dst rows (the template is checked separately: rows only)."""
    ms, md, cs, cd, gs, gd = idx
    head = st.L2_HEAD0 if kv_head == 0 else st.L2_HEAD1
    m = (np.asarray(md) >= 0)
    c = (np.asarray(cd) >= 0)
    dst = np.concatenate([np.asarray(md)[m], np.asarray(cd)[c]])
    src = np.concatenate([np.asarray(ms)[m], np.asarray(cs)[c]])
    kind = np.concatenate([np.full(m.sum(), head), np.where(np.asarray(cs)[c] == 0, st.L2_FILL, st.L2_COPY)])
    rows = {}
    for d, k, s in zip(dst.tolist(), kind.tolist(), src.tolist()):
        assert rows.get(d, (k, s)) == (k, s), "v2 writes a row twice with different sources"
        rows[d] = (k, s)
    gdn_dst = set()
    for page in np.asarray(gd).tolist():
        gdn_dst.update(range(page * dst_block, page * dst_block + d_gdn.ssm_tokens + d_gdn.conv_tokens))
    return rows, gdn_dst


# ---------------------------------------------------------------------------------------------------------------
# the GDN template


def test_s12_template_is_whole_rows():
    load = Load("s12", 5, 1000, 0)
    t = load.c._l2_gdn
    assert t is not None and t[0] == (0, 0, 384, st.L2_COPY)
    assert sum(n for _, _, n, _ in t) == 400 and t[-1] == (-1, 399, 1, st.L2_ZERO)
    assert len(t) == 11  # SSM, 3 x (q, k, v), zero pad


def test_non_whole_row_conv_is_refused():
    s = st.GdnGeometry(ssm_tokens=8, conv_tokens=8, v_heads=16, k_heads=8, head_k_dim=8, head_v_dim=16,
                       conv_width=3, ssm_itemsize=4, conv_itemsize=4)
    d = st.GdnGeometry(ssm_tokens=2, conv_tokens=2, v_heads=4, k_heads=2, head_k_dim=8, head_v_dim=16,
                       conv_width=3, ssm_itemsize=4, conv_itemsize=4)
    rank = st.RankSlice(tp_size=4, tp_rank=1, num_kv_heads=2, main_dcp=2)
    assert st.l2_gdn_template(s, d, st.conv_channels(s, d, rank)) is None


@pytest.mark.parametrize("name,r", [("s12", 0), ("s12", 5), ("o6", 3), ("small", 2)])
def test_template_matches_make_repack_gdn(name, r):
    load = Load(name, r, 500, 0)
    s, d, S = load.s_gdn, load.d_gdn, load.geo.stage_rows
    cols = load.geo.gdn_rows // S
    rng = np.random.default_rng(r)
    stage = rng.integers(0, 256, (cols + 1, S, ROWS, LANES), dtype=np.uint8)
    fn = st.make_repack(src_block=S, dst_block=load.geo.dst_block, src_gdn=s, dst_gdn=d, kv_head=load.rank.kv_head,
                        v_head_start=r * d.v_heads, channels=load.channels, compact_gdn=(S, cols),
                        gather=lambda st_, flat: st_[flat])
    gdn_src = jnp.asarray(np.arange(1, cols + 1).reshape(1, cols), jnp.int32)
    empty = jnp.zeros((0,), jnp.int32)
    _, _, want = fn.rows(jnp.asarray(stage), empty, empty, gdn_src)
    want = np.asarray(want)
    flat = stage.reshape(-1, ROWS, LANES)
    got = np.zeros_like(want)
    for c0, r0, n, kind in load.c._l2_gdn:
        for j in range(n):
            if kind == st.L2_ZERO:
                got[r0 + j] = 0
            else:
                cc = c0 + j
                got[r0 + j] = flat[(1 + cc // S) * S + cc % S]
    np.testing.assert_array_equal(got, want)


def test_partial_gdn_pages_t10_style():
    """gdn_pages = 2 with a partial second page (T10's 448-over-256 shape): 256 SSM + 128 conv = 384 rows."""
    s, d = gdn_pair((32, 16), (4, 2))
    rank = st.RankSlice(tp_size=8, tp_rank=3, num_kv_heads=2, main_dcp=4)
    t = st.l2_gdn_template(s, d, st.conv_channels(s, d, rank))
    assert t is not None and d.ssm_tokens + s.conv_tokens == 384
    runs = st.l2_gdn_runs([[7, 3]], [5], t, 256, 512)
    dst, kind, src = st.l2_expand(runs)
    assert dst.tolist() == list(range(5 * 512, 5 * 512 + d.ssm_tokens + d.conv_tokens))
    ssm = src[: d.ssm_tokens]
    assert ssm.tolist() == list(range(7 * 256, 8 * 256))  # SSM rows [0, 256) all on page 7
    conv = src[d.ssm_tokens:][kind[d.ssm_tokens:] == st.L2_COPY]
    assert ((conv >= 3 * 256) & (conv < 3 * 256 + 128)).all()  # conv rows on page 3, its first 128 rows


# ---------------------------------------------------------------------------------------------------------------
# builders


CASES = [  # (geo, num_tokens, skip, draft_end)
    ("s12", 2887, 0, 0), ("s12", 4800, 2048, 0), ("s12", 12859, 4096, 7001), ("s12", 700, 0, 333),
    ("s12", 20372, 18432, 19000), ("s12", 2049, 0, 2049), ("o6", 5000, 2048, 0), ("o6", 9001, 4096, 6000),
    ("small", 1000, 0, 0), ("small", 2049, 512, 900), ("small", 129, 0, 64), ("small", 3000, 1024, 3000),
]


@pytest.mark.parametrize("name,num_tokens,skip,draft_end", CASES)
def test_reference_writes_exactly_v2_rows(name, num_tokens, skip, draft_end):
    geo = GEOS[name][0]
    for r in range(geo.tp):
        load = Load(name, r, num_tokens, skip, draft_end, seed=num_tokens + r)
        assert load.d_gdn.ssm_tokens + load.d_gdn.conv_tokens <= geo.dst_block  # a valid geometry
        ref, idx = load.reference()
        rows, gdn_dst = v2_rows(idx, load.d_gdn, geo.dst_block, load.rank.kv_head)
        dst, kind, src = st.l2_expand(ref)
        assert len(np.unique(dst)) == len(dst) and st.l2_check_disjoint(ref)
        got = {d: (k, s) for d, k, s in zip(dst.tolist(), kind.tolist(), src.tolist())}
        nongdn = {d: v for d, v in got.items() if d not in gdn_dst}
        want = {d: v for d, v in rows.items() if d // geo.dst_block != 0}
        assert nongdn == want, f"rank {r}"
        assert set(got) - set(nongdn) == gdn_dst


@pytest.mark.parametrize("name,num_tokens,skip,draft_end", CASES)
@pytest.mark.parametrize("kr_div", [1, 4])
def test_builders_agree(name, num_tokens, skip, draft_end, kr_div):
    geo = GEOS[name][0]
    for r in range(geo.tp):
        load = Load(name, r, num_tokens, skip, draft_end, seed=7 * num_tokens + r,
                    kernel_rows=geo.dst_block // kr_div)
        ref, _ = load.reference()
        landed_zeroed = load.c.last_draft_rows
        prod, draft_rows, rows = load.production()
        for a, b in zip(st.l2_expand(ref), st.l2_expand(prod)):
            np.testing.assert_array_equal(a, b, err_msg=f"rank {r}")
        assert st.l2_check_disjoint(prod)
        assert draft_rows == tuple(landed_zeroed)
        assert (prod[:, 2] <= min(st.L2_MAX_RUN, load.kernel_rows)).all()
        page = prod[:, 1] // load.kernel_rows
        assert ((prod[:, 1] + prod[:, 2] - 1) // load.kernel_rows == page).all()


def test_tables_hold_any_count():
    desc = np.stack([np.arange(40000), np.arange(40000) * 2, np.ones(40000), np.zeros(40000)], 1)
    tables = st.l2_tables(desc, dcap=1 << 14)
    assert len(tables) == 3 and sum(int(t[0, 0]) for t in tables) == 40000
    assert all(t.shape == (1 << 14, 4) and t.dtype == np.int32 for t in tables)


# ---------------------------------------------------------------------------------------------------------------
# the kernel (Pallas TPU interpret mode) against v2's CPU path


def scatter(pool, rows, page, logical, valid):
    page = jnp.where(valid, page, pool.shape[0])
    return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")


def landed_pools(load, dtype=np.uint8, dcap=1024):
    geo, S, B = load.geo, load.geo.stage_rows, load.geo.dst_block
    rng = np.random.default_rng(load.num_tokens)
    staging = rng.integers(0, 256, (load.n_stage, S, 1, ROWS, LANES)).astype(np.uint8)
    pool0 = rng.integers(0, 256, (load.n_pool * B // load.kernel_rows, load.kernel_rows, 1, ROWS, LANES)).astype(
        np.uint8)
    if dtype != np.uint8:  # FP8 carriers: no NaN encodings (an XLA CPU scatter may canonicalize them)
        staging[(staging & 0x7F) == 0x7F] = 0x7E
        pool0[(pool0 & 0x7F) == 0x7F] = 0x7E
    staging[0] = 0
    pool0[0] = 0
    idx = load.indices()
    cols = geo.gdn_rows // S
    fn = st.make_repack(src_block=S, dst_block=B, src_gdn=load.s_gdn, dst_gdn=load.d_gdn,
                        kv_head=load.rank.kv_head, v_head_start=load.rank.tp_rank * load.d_gdn.v_heads,
                        channels=load.channels, write=scatter, compact_gdn=(S, cols),
                        gather=lambda st_, flat: st_[flat])
    view = (lambda a: jnp.asarray(a).view(jnp.float8_e4m3fn)) if dtype != np.uint8 else jnp.asarray
    want = view(pool0)
    for p in st.repack_programs_v2(*idx, 36):
        want = fn(want, view(staging), *(jnp.asarray(np.asarray(v), jnp.int32) for v in p))
    prod, _, _ = load.production()
    l2 = st.make_repack_l2(dcap=dcap, interpret=True)
    got = view(pool0)
    count = 0
    for t in st.l2_tables(prod, dcap):
        got, done = l2(got, view(staging), jnp.asarray(t))
        count += int(done)
    assert count == len(prod)
    as_u8 = lambda a: np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint8)) if dtype != np.uint8 else np.asarray(a)
    return as_u8(want), as_u8(got), pool0


KERNEL_CASES = [  # (geo, rank, tokens, skip, draft_end, kernel_rows divisor, dtype)
    ("small", 0, 1000, 0, 0, 1, np.uint8),
    ("small", 3, 2049, 512, 900, 1, np.uint8),
    ("small", 2, 700, 0, 300, 2, np.uint8),
    ("small", 1, 1500, 512, 0, 1, "fp8"),
    ("s12", 5, 1500, 0, 600, 1, np.uint8),
    ("s12", 0, 2600, 2048, 0, 4, np.uint8),
]


@pytest.mark.parametrize("name,r,num_tokens,skip,draft_end,kr_div,dtype", KERNEL_CASES)
def test_l2_kernel_matches_v2(name, r, num_tokens, skip, draft_end, kr_div, dtype):
    geo = GEOS[name][0]
    fa, gdn = (3, 4) if name == "s12" else (None, None)
    load = Load(name, r, num_tokens, skip, draft_end, seed=num_tokens, kernel_rows=geo.dst_block // kr_div,
                fa=fa, gdn=gdn)
    want, got, pool0 = landed_pools(load, dtype)
    np.testing.assert_array_equal(got, want)  # G1: every page
    assert (got != pool0).any()


def test_head_select_matches_make_repack_main():
    """HEAD0 / HEAD1 rows equal make_repack's [rows[:, 2h:2h+2], zeros] for random bytes."""
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 256, (64, 1, ROWS, LANES)).astype(np.uint8)
    for h in (0, 1):
        want = np.concatenate([rows[:, :, 2 * h : 2 * h + 2], np.zeros_like(rows[:, :, :2])], axis=2)
        staging = np.zeros((2, 64, 1, ROWS, LANES), np.uint8)
        staging[1] = rows
        pool = np.full((2, 64, 1, ROWS, LANES), 0xAB, np.uint8)
        desc = np.array([[64, 64, 64, st.L2_HEAD0 if h == 0 else st.L2_HEAD1]])
        out, done = st.make_repack_l2(dcap=256, interpret=True)(jnp.asarray(pool), jnp.asarray(staging),
                                                               jnp.asarray(st.l2_tables(desc, 256)[0]))
        out = np.asarray(out)
        np.testing.assert_array_equal(out[1], want)
        assert (out[0] == 0xAB).all() and int(done) == 1


# ---------------------------------------------------------------------------------------------------------------
# plumbing


def test_plan_and_run_route_through_l2_and_fall_back_to_v2(monkeypatch):
    load = Load("small", 1, 1000, 0)
    c = load.c
    c._repack_l2, c._v2_gdn_pages, c._pad_invalid = True, 36, False
    c.draft_zero_end = lambda uuid, n: 0
    plan = c.plan_repack(1, load.tables, load.num_tokens)
    assert plan.l2 and plan.passes == [] and len(plan.chunks) == 1 and plan.chunks[0].shape == (st.L2_DCAP, 4)
    ran = []
    monkeypatch.setattr(c, "_run_l2", lambda tables: ran.append(len(tables)), raising=False)
    monkeypatch.setattr(c, "_run_chunks", lambda chunks: ran.append(("v2", len(chunks))), raising=False)
    c.run_repack(plan)
    assert ran == [1]
    # a load the builder refuses lands with v2 (counted)
    monkeypatch.setattr(st, "l2_routed_runs", lambda **kw: (_ for _ in ()).throw(ValueError("refused")))
    plan2 = c.plan_repack(1, load.tables, load.num_tokens)
    # v2's own programs (prefer_fewer may pick repack_chunks' single program at this size): 6-array tuples
    assert not plan2.l2 and len(plan2.chunks) >= 1 and all(len(ch) == 6 for ch in plan2.chunks)
    assert c.l2_stats["fallbacks"] == 1
    c.run_repack(plan2)
    assert ran[-1][0] == "v2"


def test_flag_registration_and_off_by_default():
    import os

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert "QWEN4_EXP_PD_REPACK_L2" in _TPU_COMPILE_ENV_IGNORED
    os.environ.pop("QWEN4_EXP_PD_REPACK_L2", None)
    assert envs.QWEN4_EXP_PD_REPACK_L2 is False
    os.environ["QWEN4_EXP_PD_REPACK_L2"] = "1"
    try:
        assert envs.QWEN4_EXP_PD_REPACK_L2 is True
    finally:
        os.environ.pop("QWEN4_EXP_PD_REPACK_L2", None)


def test_unsupported_consumers_stay_off():
    load = Load("small", 0, 500, 0)
    c = load.c
    assert c._l2_unsupported() == []
    c.route = None
    assert c._l2_unsupported()
    c.route, c._l2_gdn = load.geo, None
    assert c._l2_unsupported()


def test_l2_jax_op_builds_with_real_torch_tpu():
    """``_build_op_l2`` builds its aliased jax op with the real torch_tpu: pallas.jax_op verifies the step function's
    annotations at build time (XFER-4 i1 died on that check at warm-up, j-4c6c1808), and the op's fake registers.
    The torch.compile wrapper is created, not compiled (no TPU here)."""
    pytest.importorskip("torch_tpu")
    c = object.__new__(st.StagedConsumer)
    c._l2_fn = st.make_repack_l2()
    c._l2_op_name = "pallas::qwen4_pd_staged_repack_l2_cputest"
    run = c._build_op_l2()
    assert callable(run)


# ---------------------------------------------------------------------------------------------------------------
# bind-time smoke of the landing op (q38 L2-X4KB, on the served branch): the real call chain down to the op


class _BindOps:
    """Stands in for ``torch_tpu._internal.pallas.jax_op`` while delegating the build to the real one, so torch_tpu's
    build-time signature check and fake registration run unchanged. Calling the op checks the call contract the TPU
    op would see (the arguments bind to the step function and to the registered fake; pool / staging share a row
    shape and dtype; the table is int32 [dcap, 4]), runs the step function itself (Pallas interpret mode, since the
    consumer's ``_l2_fn`` is built with interpret=True) and checks the fake's output contract against the real result."""

    def __init__(self, real_jax_op):
        self.real, self.built, self.calls = real_jax_op, [], []

    def __call__(self, name, fn, donate_argnums=()):
        import inspect

        import torch

        real = self.real(name, fn, donate_argnums=donate_argnums)
        ops = self
        ops.built.append(name)

        class Op:
            fake = None

            def register_fake(self, fake):
                real.register_fake(fake)
                self.fake = fake
                return fake

            def __call__(self, *args, **kwargs):
                assert not kwargs, kwargs
                inspect.signature(fn).bind(*args)
                inspect.signature(self.fake).bind(*args)
                pool, staging, desc = args
                assert pool.dtype == staging.dtype and tuple(pool.shape[2:]) == tuple(staging.shape[2:])
                assert desc.dtype == torch.int32 and desc.dim() == 2 and desc.shape[1] == 4, (desc.dtype, desc.shape)
                fake_pool, fake_marker = self.fake(*args)
                out, done = fn(*(jnp.asarray(a.numpy()) for a in args))
                new_pool = torch.from_numpy(np.array(out))
                marker = torch.tensor(int(done), dtype=torch.int32)
                assert (fake_pool.shape, fake_pool.dtype) == (new_pool.shape, new_pool.dtype)
                assert (fake_marker.shape, fake_marker.dtype) == (marker.shape, marker.dtype)
                assert int(marker) == int(desc[0, 0])  # the marker is the descriptor count, written last
                ops.calls.append((name, int(marker)))
                return new_pool, marker

        return Op()


def _bind_patch(monkeypatch):
    torch = pytest.importorskip("torch")
    pallas = pytest.importorskip("torch_tpu._internal.pallas")
    ops = _BindOps(pallas.jax_op)
    monkeypatch.setattr(pallas, "jax_op", ops)
    monkeypatch.setattr(torch, "compile", lambda *a, **k: (lambda f: f))  # no TPU backend here: run the wrapper eagerly
    return torch, ops


def test_l2_landing_binds_through_plan_and_run(monkeypatch):
    """plan_repack -> run_repack -> _run_l2 -> _build_op_l2 -> the op, on a routed shell with a prefix skip and a
    draft gap: the pool lands byte-identical to v2's CPU path (G1, every page) through the real dispatch code."""
    torch, ops = _bind_patch(monkeypatch)
    load = Load("small", 3, 2049, 512, 900, seed=2049)
    want, _, pool0_ref = landed_pools(load)
    S, B = load.geo.stage_rows, load.geo.dst_block
    rng = np.random.default_rng(load.num_tokens)  # landed_pools' own inputs, regenerated
    staging = rng.integers(0, 256, (load.n_stage, S, 1, ROWS, LANES)).astype(np.uint8)
    pool0 = rng.integers(0, 256, (load.n_pool * B // load.kernel_rows, load.kernel_rows, 1, ROWS, LANES)).astype(
        np.uint8)
    staging[0] = 0
    pool0[0] = 0
    assert np.array_equal(pool0, pool0_ref)
    c = load.c
    c._repack_l2, c._v2_gdn_pages = True, 36
    c.draft_zero = {1: load.draft_end}
    c._l2_fn = st.make_repack_l2(interpret=True)
    c._l2_op_name = "pallas::qwen4_pd_staged_repack_l2_bindtest_plan"
    c.pool, c.staging = torch.from_numpy(pool0.copy()), torch.from_numpy(staging.copy())
    assert c.draft_zero_end(1, load.num_tokens) == load.draft_end
    plan = c.plan_repack(1, load.tables, load.num_tokens)
    assert plan.l2 and plan.passes == [] and len(plan.chunks) >= 1
    c.run_repack(plan)
    assert ops.built == [c._l2_op_name] and len(ops.calls) == len(plan.chunks)
    got = c.pool.numpy()
    np.testing.assert_array_equal(got, want)
    assert (got != pool0).any()


def test_l2_warm_binds_through_the_connector(monkeypatch):
    """register_runner's _after_native_registration -> warm_repack_l2 -> _run_l2 -> _build_op_l2 -> the op with the
    null table: exactly pool row 0 takes staging row 0 (a served null page is zero on both sides), nothing else
    moves, and the connector logs the warm-up line. v2 is off on this consumer, so only L2 warms."""
    from types import SimpleNamespace as NS

    # Import before the jax_op recorder is installed: importing the connector builds other jax_ops at import time
    # (indexer_topk_rows_update, moe_router_topk, mamba_state_block_copy; i1 j-175b5b65 / diag j-775ce1c4).
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pc

    torch, ops = _bind_patch(monkeypatch)
    lines = []
    monkeypatch.setattr(pc, "logger", NS(info=lambda fmt, *a: lines.append(fmt % a if a else fmt),
                                         warning=lambda *a, **k: None))
    rng = np.random.default_rng(5)
    pool0 = rng.integers(0, 256, (4, 128, 1, ROWS, LANES)).astype(np.uint8)
    staging = rng.integers(0, 256, (3, 64, 1, ROWS, LANES)).astype(np.uint8)
    assert (pool0[0, 0] != staging[0, 0]).any()
    c = object.__new__(st.StagedConsumer)
    c._repack_l2, c._v2_gdn_pages = True, 0
    c._l2_fn = st.make_repack_l2(interpret=True)
    c._l2_op_name = "pallas::qwen4_pd_staged_repack_l2_bindtest_warm"
    c.pool, c.staging = torch.from_numpy(pool0.copy()), torch.from_numpy(staging.copy())
    w = object.__new__(pc.Qwen4ExpRaidenWorker)
    w.tp_rank, w._staging, w._qwen_shared = 0, c, None
    w._after_native_registration()
    assert ops.built == [c._l2_op_name] and ops.calls == [(c._l2_op_name, 1)]
    got = c.pool.numpy()
    np.testing.assert_array_equal(got[0, 0], staging[0, 0])
    want = pool0.copy()
    want[0, 0] = staging[0, 0]
    np.testing.assert_array_equal(got, want)
    assert any("descriptor program warmed" in x for x in lines), lines

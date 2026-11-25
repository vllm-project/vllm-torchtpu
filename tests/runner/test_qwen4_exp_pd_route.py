# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_ROUTED_DST on CPU: routed spans land byte-identical decode pages.

Today a TP1 pipeline producer sends every page to every decode rank and each
rank repacks its slice out of producer-geometry staging pages. Routed, each
rank receives only its slice into a dense compact staging space
(``pd_route``). For every rank, the decode pool after the routed landing must
equal the broadcast landing's, byte for byte.

The span applier below emulates Raiden's rules for the spans it applies and
asserts them (the native planner rejects a violation): per destination unit,
global-space (``dst_space_version=1``) coverage dense from 0 with only the
final destination page partial and no overlap; declared bytes equal to the
span total; ``dst_skip_bytes`` a whole number of destination pages.
"""

import importlib.util
import math
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

_DIR = pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu/models/vllm/qwen4_exp"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"route_test_{name}", _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load("pd_staging")
rt = st.pd_route
ROWS, LANES = 4, 256
RB = rt.ROW_BYTES


def scatter(pool, rows, page, logical, valid):
    page = jnp.where(valid, page, pool.shape[0])
    return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")


# Small Qwen-shaped geometry: TP4 decode, DCP2, 2 KV heads (ranks 0,2 hold
# slice 0; 1,3 slice 1), producer pages of 320 tokens (not a multiple of the
# 128-token decode block, as 3200 is not of 512: runs straddle pages).
SRC_BLOCK, DST_BLOCK, RATIO, TP, DCP, KV_HEADS = 320, 128, 4, 4, 2, 2
S_GDN = st.GdnGeometry(ssm_tokens=8, conv_tokens=8, v_heads=16, k_heads=8, head_k_dim=8,
                       head_v_dim=16, conv_width=3, ssm_itemsize=4, conv_itemsize=4)
D_GDN = st.GdnGeometry(ssm_tokens=2, conv_tokens=2, v_heads=4, k_heads=2, head_k_dim=8,
                       head_v_dim=16, conv_width=3, ssm_itemsize=4, conv_itemsize=4)
GEO = rt.RouteGeometry(tp=TP, dcp=DCP, dst_block=DST_BLOCK, src_block=SRC_BLOCK, ratio=RATIO,
                       ssm_rows_src=8, conv_rows_src=8, ssm_rows_dst=2, stage_rows=2)
FA_LAYERS, GDN_LAYERS = 2, 2


def owners(gdn_layers=GDN_LAYERS):
    names = []
    for i in range(FA_LAYERS):
        names += [f"l{i}.self_attn", f"l{i}.self_attn.compressed_k", f"l{i}.self_attn.raw_ring"]
    names += [f"g{i}.linear_attn" for i in range(gdn_layers)] + ["p.ple"]
    return names


def producer_pages(num_tokens, rng, names=None):
    """Per owner, [pages, SRC_BLOCK, 4, 256] uint8 of random bytes."""
    n_tok_pages = -(-num_tokens // SRC_BLOCK)
    out = {}
    for name in names or owners():
        kind = st.owner_kind(name)
        n = n_tok_pages if kind in ("main", "compressed") else 1
        out[name] = rng.integers(0, 256, (n, SRC_BLOCK, ROWS, LANES), dtype=np.uint8)
    return out


def rank_slice(r):
    return st.RankSlice(tp_size=TP, tp_rank=r, num_kv_heads=KV_HEADS, main_dcp=DCP)


def dst_tables(num_tokens, rng, names=None):
    n_main = -(-num_tokens // (DCP * DST_BLOCK))
    n_comp = -(-num_tokens // DST_BLOCK)
    ids = list(rng.permutation(np.arange(1, 400)))
    take = lambda k: [int(ids.pop()) for _ in range(k)]  # noqa: E731
    t = {}
    for name in names or owners():
        kind = st.owner_kind(name)
        t[name] = take(n_main if kind == "main" else n_comp if kind == "compressed" else 1)
    return t


def make_fn(rank, compact, gdn=True):
    return st.make_repack(
        src_block=SRC_BLOCK, dst_block=DST_BLOCK,
        src_gdn=S_GDN if gdn else None, dst_gdn=D_GDN if gdn else None,
        kv_head=rank.kv_head, v_head_start=rank.tp_rank * D_GDN.v_heads,
        channels=st.conv_channels(S_GDN, D_GDN, rank), write=scatter,
        compact_gdn=((GEO.stage_rows, GEO.gdn_rows // GEO.stage_rows) if compact else None),
    )


def run(fn, pool, staging, idx):
    size = st.bucket(max(len(idx[0]), len(idx[2]), 1), 16)
    out = jnp.asarray(pool)
    for plan in st.repack_chunks(*idx, size, max(1, len(idx[4]))):
        out = fn(out, jnp.asarray(staging), *(jnp.asarray(v, jnp.int32) for v in plan))
    return np.asarray(out)


# --- the reference: today's broadcast staging + repack ------------------------------


def broadcast_landing(num_tokens, skip, rank, prod, tables):
    names = owners()
    n_pages = sum(len(prod[n]) for n in names) + 1
    staging = np.zeros((n_pages, SRC_BLOCK, 1, ROWS, LANES), np.uint8)
    pages, cur = {}, 1
    for name in names:  # every page of every owner, whole (as broadcast lands them)
        pages[name] = list(range(cur, cur + len(prod[name])))
        staging[cur:cur + len(prod[name]), :, 0] = prod[name]
        cur += len(prod[name])
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = [], [], [], [], [], []
    for name in names:
        kind = st.owner_kind(name)
        if kind == "compressed":
            s, d = st.compressed_rows_np(num_tokens, RATIO, pages[name], SRC_BLOCK,
                                         tables[name], DST_BLOCK, start=skip)
            copy_s.append(s); copy_d.append(d)
    for name in names:
        kind = st.owner_kind(name)
        if kind == "main":
            s, d = st.main_rows_np(num_tokens, pages[name], SRC_BLOCK, tables[name],
                                   DST_BLOCK, rank, start=skip)
            main_s.append(s); main_d.append(d)
        elif kind in ("ring", "ple"):
            s, d = st.page_prefix_rows(pages[name][0], SRC_BLOCK, tables[name][0], DST_BLOCK,
                                       min(SRC_BLOCK, DST_BLOCK))
            copy_s.append(np.asarray(s)); copy_d.append(np.asarray(d))
        elif kind == "gdn":
            gdn_s.append(pages[name][0]); gdn_d.append(tables[name][0])
    def cat(parts):
        parts = [st.align_rows(p) for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    idx = (cat(main_s), cat(main_d), cat(copy_s), cat(copy_d),
           np.asarray(gdn_s, np.int64), np.asarray(gdn_d, np.int64))
    pool = np.zeros((400, DST_BLOCK, 1, ROWS, LANES), np.uint8)
    return run(make_fn(rank, False), pool, staging, idx)


# --- the routed path: producer spans -> applier -> compact staging -> repack ----------


def apply_spans(spans, declared, src_pages_bytes, unit, dest_pages, skip_bytes, staging_flat):
    """Raiden-rule emulation for one tag and one destination unit; asserts the rules."""
    assert declared == sum(n for _, _, _, n, _ in spans), "declared bytes != span total"
    page_bytes = GEO.stage_rows * RB
    assert skip_bytes % page_bytes == 0, "dst_skip_bytes is not a whole destination page"
    cover = []
    for src, so, do, n, u in spans:
        assert n > 0 and so % RB == 0 and do % RB == 0 and n % RB == 0
        if u is not None and u != unit:
            continue
        lo, hi = do, do + n
        if hi <= skip_bytes:
            continue
        if lo < skip_bytes:
            so, n, lo = so + skip_bytes - lo, n - (skip_bytes - lo), skip_bytes
        g0 = lo - skip_bytes
        cover.append((g0, g0 + n))
        data = src_pages_bytes[src].reshape(-1)[so:so + n]
        for off in range(0, n, RB):  # one row at a time, split at destination pages
            gpage, inpage = divmod(g0 + off, page_bytes)
            assert gpage < len(dest_pages), "span past the transfer's destination pages"
            row = dest_pages[gpage] * GEO.stage_rows + inpage // RB
            staging_flat[row] = data[off:off + RB].reshape(ROWS, LANES)
    cover.sort()
    end = 0
    for a, b in cover:
        assert a == end, f"coverage gap or overlap at byte {end} (span at {a})"
        end = b
    assert (len(dest_pages) - 1) * page_bytes < end <= len(dest_pages) * page_bytes, (
        "coverage must fill every destination page; only the final one may be partial"
    )


def consumer(rank, names=None, split=False, pairs=None):
    c = object.__new__(st.StagedConsumer)
    c.route, c.rank, c.ratio, c.dst_block = GEO, rank, RATIO, DST_BLOCK
    names = names or owners()
    c._dst_compressed = {n: (n, 0) for n in names if st.owner_kind(n) == "compressed"}
    # QWEN4_EXP_PACK_MAIN_KV: a paired member has no destination group of its own.
    away = {m for members in (pairs or {}).values() for m in members[1:]}
    c._dst_owner = {n: n for n in names if n not in away}
    if pairs:
        c._main_pairs = pairs
    c._gdn_names = [n for n in names if st.owner_kind(n) == "gdn"]
    c._state_entry = 0
    if split:  # VLLM_TPU_QSA_COMPRESSED_DCP: compressed keys are DCP-split owners
        c._dcp_owners = frozenset(c._dst_compressed)
    return c


def routed_staging(num_tokens, skip, rank, prod, names):
    """Compact staging after the producer's spans land, its page lists and the
    staging page count per owner."""
    counts = {n: rt.staging_pages(st.owner_kind(n), num_tokens, len(prod[n]), skip, GEO)
              for n in names}
    staging = np.zeros((sum(counts.values()) + 1, GEO.stage_rows, 1, ROWS, LANES), np.uint8)
    flat = staging.reshape(-1, ROWS, LANES)
    pages, cur = {}, 1
    for name in names:
        pages[name] = list(range(cur, cur + counts[name]))
        cur += counts[name]
        kind = st.owner_kind(name)
        spans = rt.producer_spans(kind, num_tokens, len(prod[name]), GEO)
        apply_spans(spans, sum(n for *_, n, _ in spans), prod[name], rank.tp_rank,
                    pages[name], rt.skip_rows(kind, skip, GEO) * RB, flat)
    return staging, pages, counts


def routed_landing(num_tokens, skip, rank, prod, tables, split=False, pairs=None):
    staging, pages, counts = routed_staging(num_tokens, skip, rank, prod, owners())
    idx = consumer(rank, split=split, pairs=pairs)._routed_indices(pages, tables, num_tokens, skip)
    pool = np.zeros((400, DST_BLOCK, 1, ROWS, LANES), np.uint8)
    return run(make_fn(rank, True), pool, staging, idx), counts


CASES = [1, 127, 128, 129, 255, 256, 319, 320, 321, 383, 640, 767, 1000, 2049]


@pytest.mark.parametrize("num_tokens", CASES)
def test_routed_landing_is_byte_identical_on_every_rank(num_tokens):
    rng = np.random.default_rng(num_tokens)
    prod, tables = producer_pages(num_tokens, rng), dst_tables(num_tokens, rng)
    for r in range(TP):
        rank = rank_slice(r)
        want = broadcast_landing(num_tokens, 0, rank, prod, tables)
        got, _ = routed_landing(num_tokens, 0, rank, prod, tables)
        np.testing.assert_array_equal(got, want, err_msg=f"rank {r}, {num_tokens} tokens")
        assert want.any()


@pytest.mark.parametrize("num_tokens,skip", [(600, 256), (2049, 512), (2049, 1792), (1000, 768)])
def test_routed_prefix_aware_landing_is_byte_identical(num_tokens, skip):
    rng = np.random.default_rng(num_tokens + skip)
    prod, tables = producer_pages(num_tokens, rng), dst_tables(num_tokens, rng)
    for r in range(TP):
        rank = rank_slice(r)
        want = broadcast_landing(num_tokens, skip, rank, prod, tables)
        got, _ = routed_landing(num_tokens, skip, rank, prod, tables)
        np.testing.assert_array_equal(got, want, err_msg=f"rank {r}, skip {skip}")


@pytest.mark.parametrize("num_tokens,skip", [(2049, 0), (2049, 1792), (321, 0)])
def test_repack_shape_with_every_gdn_owner_first_lands_identical_pools(
    num_tokens, skip
):
    """36 GDN owners, as Flash has (QWEN4_EXP_PD_REPACK_SHAPE): a first
    program that carries every GDN owner, later programs from the repack built
    without GDN state, and a copied-row size apart from main KV land the same
    pool, byte for byte, as the default 16384,16384,4 chunks."""
    names = owners(gdn_layers=36)
    rng = np.random.default_rng(num_tokens + skip + 36)
    prod = producer_pages(num_tokens, rng, names)
    tables = dst_tables(num_tokens, rng, names)

    def land(fns, staging, idx, shape, split, null_rows=None):
        main, copied, gdn = shape
        chunks = list(st.repack_chunks(
            *idx, main, gdn, copy_size=copied, split_gdn=split, null_rows=null_rows
        ))
        out = jnp.asarray(np.zeros((400, DST_BLOCK, 1, ROWS, LANES), np.uint8))
        for c in chunks:
            # As StagedConsumer._run_chunks: a chunk without GDN lists runs
            # the GDN-free repack, whose GDN arguments are never read.
            index = (jnp.asarray(0 if v is None else v, jnp.int32) for v in c)
            out = fns[c[4] is None](out, jnp.asarray(staging), *index)
        return np.asarray(out), chunks

    for r in range(TP):
        rank = rank_slice(r)
        staging, pages, _ = routed_staging(num_tokens, skip, rank, prod, names)
        idx = consumer(rank, names)._routed_indices(pages, tables, num_tokens, skip)
        assert len(idx[4]) == 36
        fns = (make_fn(rank, True), make_fn(rank, True, gdn=False))
        want, today = land(fns, staging, idx, (16384, 16384, 4), False)
        assert len(today) == 9 and all(c[4] is not None for c in today)
        # The served target shape (one program for these loads) and a small one
        # that cuts each load into several programs.
        # Spread padding (null_rows): staging page 0 is zero and the pool's
        # page 0 starts zero, so the whole pool, real rows included, matches.
        null_rows = (GEO.stage_rows, DST_BLOCK)
        got, _ = land(fns, staging, idx, (16384, 16384, 4), False, null_rows)
        np.testing.assert_array_equal(got, want, err_msg=f"rank {r}, spread padding")
        for shape in ((16384, 8192, 36), (256, 128, 36)):
            for rows in (None, null_rows):
                got, chunks = land(fns, staging, idx, shape, True, rows)
                assert len(chunks[0][4]) == 36 and all(c[4] is None for c in chunks[1:])
                assert len(chunks) == 1 if shape[0] == 16384 else len(chunks) > 1
                np.testing.assert_array_equal(
                    got, want, err_msg=f"rank {r}, shape {shape}, null rows {rows}"
                )
        assert want.any()


@pytest.mark.parametrize(
    "shape,split", [((16384, 16384, 4), False), ((16384, 8192, 36), True)]
)
def test_warm_repack_runs_null_chunks_of_the_serving_shapes(shape, split):
    c = consumer(rank_slice(0), owners(gdn_layers=36))
    c._repack_shape, c._split_gdn = shape, split
    ran = []
    c._run_chunks = ran.extend
    c.warm_repack()
    main, copied, gdn = shape
    first = ran[0]
    assert [len(v) for v in first[:4]] == [main, main, copied, copied]
    assert first[4].shape == (gdn, GEO.gdn_rows // GEO.stage_rows)
    assert first[5].shape == (gdn,)
    assert not any(np.asarray(v).any() for v in first)
    assert len(ran) == (2 if split else 1)
    if split:
        assert ran[1][4] is None and ran[1][5] is None
        assert all(a is b for a, b in zip(ran[1][:4], first[:4]))


def test_spread_padding_uses_distinct_null_page_rows_in_whole_tiles():
    main_s = np.arange(100, 228, dtype=np.int64)  # 128 real rows: two tiles
    main_d = np.arange(5000, 5128, dtype=np.int64)
    (chunk,) = st.repack_chunks(main_s, main_d, main_s[:64], main_d[:64], [7], [9],
                                size=1024, gdn_pages=1, null_rows=(256, 512))
    ms, md, cs, cd, gs, gd = chunk
    np.testing.assert_array_equal(ms[:128], main_s)
    np.testing.assert_array_equal(md[:128], main_d)
    np.testing.assert_array_equal(ms[128:], np.arange(1024 - 128) % 256)
    np.testing.assert_array_equal(md[128:], np.arange(1024 - 128) % 512)
    pad = md[128:].reshape(-1, st.REPACK_TILE)
    assert (np.diff(pad, axis=1) == 1).all()  # each padded tile is one contiguous run
    assert len(set(md[128:640].tolist())) == 512 and md[128:].max() < 512
    np.testing.assert_array_equal(cd[64:], np.arange(1024 - 64) % 512)
    assert list(gs) == [7] and list(gd) == [9]


def test_repack_shape_env_takes_commas_or_colons():
    assert st.parse_repack_shape("") == (16384, 16384, 4)
    assert st.parse_repack_shape("16384:8192:36") == (16384, 8192, 36)
    assert st.parse_repack_shape(" 8192,8192,36 ") == (8192, 8192, 36)
    for bad in ("16384,8192", "16384,8100,36", "0,8192,36", "16384,8192,0", "a,b,c"):
        with pytest.raises(ValueError):
            st.parse_repack_shape(bad)


def test_staging_counts_are_rank_independent_and_compact():
    # Counts never depend on the rank (one destination page list per tag).
    for n in CASES:
        for name in owners():
            kind = st.owner_kind(name)
            pages = -(-n // SRC_BLOCK) if kind in ("main", "compressed") else 1
            assert rt.staging_pages(kind, n, pages, 0, GEO) >= 1


def test_served_geometry_bytes_per_rank():
    """88K tokens at the served geometry: each rank receives ~436 MB, not 1,399 MB."""
    g = rt.RouteGeometry(tp=8, dcp=4, dst_block=512, src_block=3200, ratio=4,
                         ssm_rows_src=3072, conv_rows_src=128, ssm_rows_dst=384)
    n, pages = 88000, -(-88000 // 3200)
    per_rank = (12 * (rt.unit_bytes("main", n, pages, g) + rt.unit_bytes("compressed", n, pages, g)
                      + rt.unit_bytes("ring", n, 1, g))
                + 36 * rt.unit_bytes("gdn", n, 1, g) + rt.unit_bytes("ple", n, 1, g))
    broadcast = 12 * pages * (3_276_800 + 409_600 + 3_276_800 / pages) + 37 * 3_276_800
    assert 0.40e9 < per_rank < 0.46e9, per_rank
    assert broadcast / per_rank > 3.0
    # Every unit's coverage of every kind is dense (the native planner's rule).
    for kind, tag_pages in (("main", pages), ("compressed", pages), ("gdn", 1), ("ring", 1)):
        spans = rt.producer_spans(kind, n, tag_pages, g)
        for unit in range(8):
            cov = sorted((do, do + sz) for _, _, do, sz, u in spans if u in (None, unit))
            end = 0
            for a, b in cov:
                assert a == end
                end = b
            assert end == rt.extent_rows(kind, n, tag_pages, g) * RB


def test_served_geometry_every_compact_row_holds_its_token():
    """Served geometry, 88,001 tokens: in each rank's compact main-KV space,
    row j holds token entry * 512 + t % 512 of its slice (or padding past it)."""
    g = rt.RouteGeometry(tp=8, dcp=4, dst_block=512, src_block=3200, ratio=4,
                         ssm_rows_src=3072, conv_rows_src=128, ssm_rows_dst=384)
    n = 88001
    spans = rt.producer_spans("main", n, -(-n // 3200), g)
    for unit in range(8):
        tok = np.full(rt.main_rows(n, g), -1, np.int64)
        for p, so, do, sz, u in spans:
            if u != unit:
                continue
            rows = np.arange(sz // RB)
            tok[do // RB + rows] = p * 3200 + so // RB + rows
        t = np.arange(n)
        t = t[(t // 512) % 4 == unit % 4]
        np.testing.assert_array_equal(tok[: len(t)], t)  # padding past the slice is free


def test_routed_spans_and_layout_on_the_connector():
    """The producer's registrations and the routed fingerprint (pd_connector)."""
    pc = pytest.importorskip("vllm_torchtpu.models.vllm.qwen4_exp.pd_connector")
    w = object.__new__(pc.Qwen4ExpRaidenWorker)
    w._qwen_route = GEO
    names = owners()
    plan = SimpleNamespace(pool_groups=lambda: [
        (SimpleNamespace(tag=f"t.{n}", layer_name=n), i) for i, n in enumerate(names)
    ])
    n = 1000
    pages = tuple(tuple(range(10 * i, 10 * i + (3 if st.owner_kind(nm) in ("main", "compressed") else 1)))
                  for i, nm in enumerate(names))
    regs = w._routed_spans(plan, pages, n, None, GEO.pairs_per_src_page * RB)
    for reg, name in zip(regs, names):
        want = rt.producer_spans(st.owner_kind(name), n, len(pages[names.index(name)]), GEO)
        assert reg.dst_space_version == 1
        assert reg.declared_bytes == sum(s.size_bytes for s in reg.spans)
        assert [(s.src_block_ordinal, s.src_offset_bytes, s.dst_offset_bytes, s.size_bytes,
                 s.dst_unit_ordinal) for s in reg.spans] == want
    prod_layout = dict(torch_tpu="x", libtpu="y", schema="s",
                       carriers=[{"shape_per_kernel_page": [3200, 1, 4, 256], "tiles": [[4, 128], [4, 1]]}],
                       pages=[{"block_stride_bytes": 3276800, "dtype_tag": "qwen4-whole-fp8-page-v1"}])
    cons_layout = dict(prod_layout, carriers=[{"shape_per_kernel_page": [256, 1, 4, 256],
                                               "tiles": [[4, 128], [4, 1]]}],
                       pages=[{"block_stride_bytes": 262144, "dtype_tag": "qwen4-whole-fp8-page-v1"}])
    assert pc._routed_layout(prod_layout, GEO) == pc._routed_layout(cons_layout, GEO)
    other = rt.RouteGeometry(**{**GEO.__dict__, "dcp": 4})
    assert pc._routed_layout(prod_layout, GEO) != pc._routed_layout(cons_layout, other)


def test_route_env_takes_commas_or_colons():
    # Recipes pass server env as a comma-separated list, so launchers write "8:4:512".
    assert rt.parse_route_env("8,4,512") == rt.parse_route_env(" 8:4:512 ") == (8, 4, 512)
    assert rt.parse_route_env("") is None
    with pytest.raises(ValueError):
        rt.parse_route_env("8:4")


def test_a_consumer_built_without_init_takes_the_broadcast_path():
    """D-MK's admission tests assemble StagedConsumer with object.__new__ and
    drive reserve/release: nothing routed may be required (INT trial, 20:19Z)."""
    import types

    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import TransferExtent, TransferGroup

    s = object.__new__(pd_staging.StagedConsumer)
    assert s.route is None and s.shared is None and s.landing is None
    s.src_groups = [TransferGroup(0, ("fa",), 16, TransferExtent.TOKEN_PAGES),
                    TransferGroup(1, ("gdn",), 16, TransferExtent.REQUEST_STATE)]
    s.manifest = types.SimpleNamespace(pools=[types.SimpleNamespace(layer_name=n) for n in ("fa0", "gdn0")])
    s._src_owner = {"fa0": 0, "gdn0": 1}
    s.free = torch.tensor([0] + [1] * 15, dtype=torch.int32)
    s.reservations, s.skips = {}, {}
    assert s.reserve(7, 40)
    assert s.release(7) is not None
    # Released pages return after the staging quarantine (M1, lane_xfer §11.21).
    for _ in range(pd_staging.quarantine_steps()):
        s.advance_quarantine()
    assert int(s.free.sum()) == 15


_REFUSAL = "supports a TP1 producer only"


@pytest.mark.parametrize(
    "layout",
    [
        {"merged_compressed": True, "merged_key_span": 2},
        {"merged_compressed": True, "split_compressed": True},
        {"main_dcp": 2},
    ],
)
def test_routed_staging_refuses_a_spanned_or_split_key_or_pcp_producer(monkeypatch, layout):
    """pd_route models a TP1 producer with one main-KV page per ID. Merged keys
    spanning several columns per ID (PCP), split merged keys, or a PCP
    producer's split main KV (main_dcp > 1 on its layout) must fail when
    either leg builds the route, not be mis-routed."""
    monkeypatch.setenv("QWEN4_EXP_PD_ROUTED_DST", "8:4:512")
    row = ROWS * LANES
    dst = SimpleNamespace(token_bytes=row, block_size=512)
    rank = SimpleNamespace(tp_size=8, main_dcp=4)
    plain = SimpleNamespace(token_bytes=row, block_size=3200, merged_compressed=False, main_dcp=1)
    with pytest.raises(ValueError, match=_REFUSAL):
        st.route_geometry_for(None, SimpleNamespace(**{**plain.__dict__, **layout}), dst, rank)
    # Controls: the plain producer, and a TP1 pipeline producer with merged keys
    # of one column per ID, pass this check and fail later, on the missing model
    # config, for a different reason.
    merged = SimpleNamespace(**{**plain.__dict__, "merged_compressed": True, "merged_key_span": 1})
    for ok in (plain, merged):
        with pytest.raises(Exception) as other:
            st.route_geometry_for(None, ok, dst, rank)
        assert _REFUSAL not in str(other.value)


@pytest.mark.parametrize("num_tokens", [1, 320, 383, 1000, 2049])
def test_merged_producer_key_blocks_route_and_land_like_unmerged_pages(num_tokens):
    """QWEN4_EXP_MERGE_COMPRESSED_K on the producer: a QSA layer's keys for main-KV
    ID c are one key block (the live rows of an unmerged keys page, in order) in
    the carrier's region, listed as a pool on the main-KV IDs. The producer's
    routed spans read each block from offset 0 exactly as they read an unmerged
    page's live rows, so the decode lands the same bytes."""
    rng = np.random.default_rng(7 + num_tokens)
    prod, tables = producer_pages(num_tokens, rng), dst_tables(num_tokens, rng)
    live = GEO.pairs_per_src_page * RB
    merged = dict(prod)
    for name in owners():
        if st.owner_kind(name) == "compressed":
            # Only the key block exists on a merged producer: the page tail is gone.
            flat = prod[name].reshape(len(prod[name]), -1)[:, :live]
            merged[name] = np.ascontiguousarray(flat)
    for r in range(TP):
        want, _ = routed_landing(num_tokens, 0, rank_slice(r), prod, tables)
        got, _ = routed_landing(num_tokens, 0, rank_slice(r), merged, tables)
        np.testing.assert_array_equal(got, want)


# --- VLLM_TPU_QSA_COMPRESSED_DCP: split compressed keys, routed staging ---------
# The producer and the route are unchanged (every rank receives every pair);
# each rank lands the pairs of its own DST_BLOCK-token blocks, block k on rank
# k % DCP, at the same in-page rows of its page for entry k // DCP
# (pooled_qsa.compressed_pairs with dcp).


def split_tables(num_tokens, tables, rng):
    """``tables`` with each compressed owner's pages replaced by unused ones,
    one per DCP x DST_BLOCK entry."""
    used = {p for pages in tables.values() for p in pages}
    free = [p for p in rng.permutation(np.arange(1, 400)).tolist() if p not in used]
    entries = -(-num_tokens // (DCP * DST_BLOCK))
    out = dict(tables)
    for name in owners():
        if st.owner_kind(name) == "compressed":
            out[name] = [int(free.pop()) for _ in range(entries)]
    return out


def split_reference(num_tokens, skip, rank, prod, tables, split):
    """The slow way: today's unsplit routed landing (compressed keys on their
    per-block ``tables`` pages), then each block this rank owns copied to its
    split page, and the unsplit pages cleared (the split landing never
    writes them)."""
    unsplit, _ = routed_landing(num_tokens, skip, rank, prod, tables)
    want = unsplit.copy()
    rows = DST_BLOCK // (2 * RATIO)
    pairs = math.ceil(math.ceil(num_tokens / RATIO) / 2)
    for name in owners():
        if st.owner_kind(name) != "compressed":
            continue
        want[tables[name]] = 0
        for k in range(skip // DST_BLOCK, -(-num_tokens // DST_BLOCK)):
            if k % DCP != rank.tp_rank % DCP:
                continue
            n = min(rows, pairs - k * rows)
            want[split[name][k // DCP], :n] = unsplit[tables[name][k], :n]
    return want


SPLIT_CASES = [(n, 0) for n in (1, 127, 128, 129, 255, 256, 321, 1000, 2049)] + [
    (600, 256), (2049, 512), (2049, 1792), (1000, 768)
]


@pytest.mark.parametrize("num_tokens,skip", SPLIT_CASES)
def test_routed_split_compressed_landing_is_the_owned_part_of_the_unsplit_one(num_tokens, skip):
    rng = np.random.default_rng(num_tokens * 7 + skip)
    prod, tables = producer_pages(num_tokens, rng), dst_tables(num_tokens, rng)
    split = split_tables(num_tokens, tables, rng)
    for r in range(TP):
        rank = rank_slice(r)
        want = split_reference(num_tokens, skip, rank, prod, tables, split)
        got, counts = routed_landing(num_tokens, skip, rank, prod, split, split=True)
        # Page 0 is the null page (an empty plan rewrites its row 0).
        np.testing.assert_array_equal(got[1:], want[1:], err_msg=f"rank {r}, skip {skip}")
        # Staging is unchanged: every rank still stages every pair.
        _, unsplit_counts = routed_landing(num_tokens, skip, rank, prod, tables)
        assert counts == unsplit_counts


@pytest.mark.parametrize("num_tokens,skip", [(2049, 0), (2049, 512), (100, 0)])
def test_owned_compressed_src_rows_are_the_owned_subset(num_tokens, skip):
    pages = list(range(50, 50 + rt.staging_pages("compressed", num_tokens, 7, skip, GEO)))
    every = rt.compressed_src_rows(num_tokens, skip, pages, GEO)
    t = st.pair_tokens_np(num_tokens, RATIO, skip)
    for r in range(TP):
        own = rt.compressed_src_rows(num_tokens, skip, pages, GEO, rank=r)
        np.testing.assert_array_equal(own, every[(t // DST_BLOCK) % DCP == r % DCP])


# --- QWEN4_EXP_PACK_MAIN_KV: paired main KV, routed staging ----------------------
# The producer and the route are unchanged: every main-KV owner's slice lands in
# its own compact staging pages. The repack writes both members of a paired owner
# in one row of the owner's table, member s's head pair at head slots 2s, 2s + 1.
# The reference is the unpacked routed landing (each member on a table of its
# own), then each member's pair merged into the owner's rows by the decode's
# paired writer's rule (pooled_qsa._write_main_cache_paired); the members' own
# pages stay untouched. Three FA layers: l0 + l1 pair, l2 keeps whole rows.

PACK_PAIRS = {"l0.self_attn": ("l0.self_attn", "l1.self_attn")}


def merge_slot(old, new, slot):
    """``pooled_qsa._write_main_cache_paired`` on rows ``[n, 4, 256]``: FP8
    head slots of a lane are one 32-bit word, low byte first; slot ``s`` owns
    bytes 2s and 2s + 1, the keep mask preserves the rest."""
    pq = pytest.importorskip("vllm_torchtpu.kernels.qwen4_exp.pooled_qsa")
    moved = np.asarray(pq.main_slot_rows(jnp.asarray(new), slot))

    def words(rows):
        return np.ascontiguousarray(rows.transpose(0, 2, 1)).view(np.uint32)[..., 0]

    merged = (words(old) & np.uint32(pq._main_slot_keep_mask(slot))) | words(moved)
    return merged[..., None].view(np.uint8).transpose(0, 2, 1)


def owned_rows(num_tokens, skip, rank, table):
    t = np.arange(skip, num_tokens)
    t = t[(t // DST_BLOCK) % DCP == rank.tp_rank % DCP]
    return np.asarray(table, dtype=np.int64)[t // (DCP * DST_BLOCK)] * DST_BLOCK + t % DST_BLOCK


def packed_reference(num_tokens, skip, rank, prod, tables, split):
    unpacked, _ = routed_landing(num_tokens, skip, rank, prod, tables, split=split)
    ref = unpacked.copy()
    flat_u = unpacked.reshape(-1, ROWS, LANES)
    flat = ref.reshape(-1, ROWS, LANES)
    for owner, members in PACK_PAIRS.items():
        dst = owned_rows(num_tokens, skip, rank, tables[owner])
        for slot, member in enumerate(members):
            src = owned_rows(num_tokens, skip, rank, tables[member])
            flat[dst] = merge_slot(flat[dst], flat_u[src], slot)
        for member in members[1:]:
            ref[tables[member]] = 0  # its own pages: never written when packed
    return ref


PACK_CASES = [(n, 0) for n in (1, 128, 129, 321, 1000, 2049)] + [(600, 256), (2049, 512), (2049, 1792)]


@pytest.mark.parametrize("split", [False, True], ids=["unsplit_keys", "f1_split_keys"])
@pytest.mark.parametrize("num_tokens,skip", PACK_CASES)
def test_routed_packed_main_landing_is_the_slot_merge_of_the_unpacked_one(
    monkeypatch, num_tokens, skip, split
):
    monkeypatch.setitem(globals(), "FA_LAYERS", 3)
    rng = np.random.default_rng(num_tokens * 5 + skip + split)
    prod, tables = producer_pages(num_tokens, rng), dst_tables(num_tokens, rng)
    if split:
        tables = split_tables(num_tokens, tables, rng)
    for r in range(TP):
        rank = rank_slice(r)
        want = packed_reference(num_tokens, skip, rank, prod, tables, split)
        got, counts = routed_landing(num_tokens, skip, rank, prod, tables, split=split,
                                     pairs=PACK_PAIRS)
        # Page 0 is the null page (an empty plan rewrites its row 0).
        np.testing.assert_array_equal(got[1:], want[1:], err_msg=f"rank {r}, skip {skip}")
        # The odd layer keeps whole rows: slots 2-3 zero.
        rows = owned_rows(num_tokens, skip, rank, tables["l2.self_attn"])
        assert not got.reshape(-1, ROWS, LANES)[rows][:, 2:].any()
        # Staging is unchanged: every producer owner stages as before.
        _, unpacked_counts = routed_landing(num_tokens, skip, rank, prod, tables, split=split)
        assert counts == unpacked_counts


@pytest.mark.parametrize("num_tokens,skip", [(1000, 0), (2049, 512)])
def test_routed_packed_main_matches_the_broadcast_packed_landing(monkeypatch, num_tokens, skip):
    """The broadcast planner (``indices`` without a route) and the routed one
    plan the same paired rows: equal pools on every rank."""
    monkeypatch.setitem(globals(), "FA_LAYERS", 3)
    rng = np.random.default_rng(num_tokens + skip)
    prod, tables = producer_pages(num_tokens, rng), dst_tables(num_tokens, rng)
    names = owners()
    n_pages = sum(len(prod[n]) for n in names) + 1
    staging = np.zeros((n_pages, SRC_BLOCK, 1, ROWS, LANES), np.uint8)
    pages, cur = {}, 1
    for name in names:
        pages[name] = list(range(cur, cur + len(prod[name])))
        staging[cur : cur + len(prod[name]), :, 0] = prod[name]
        cur += len(prod[name])
    first = skip // SRC_BLOCK
    staged = {n: (p[first:] if st.owner_kind(n) in ("main", "compressed") else p)
              for n, p in pages.items()}
    for r in range(TP):
        rank = rank_slice(r)
        b = consumer(rank, pairs=PACK_PAIRS)
        b.route, b.src_block = None, SRC_BLOCK
        b._src_owner = {n: i for i, n in enumerate(names)}
        b.src_groups, b._zero_draft = [None] * len(names), {}
        groups = [first if st.owner_kind(n) in ("main", "compressed") else 0 for n in names]
        idx = b.indices(staged, tables, num_tokens, skip, groups)
        pool = np.zeros((400, DST_BLOCK, 1, ROWS, LANES), np.uint8)
        want = run(make_fn(rank, False), pool, staging, idx)
        got, _ = routed_landing(num_tokens, skip, rank, prod, tables, pairs=PACK_PAIRS)
        np.testing.assert_array_equal(got[1:], want[1:], err_msg=f"rank {r}")


def test_spread_padding_pads_both_paired_main_columns_with_distinct_rows():
    """Paired main-KV sources ([rows, 2], QWEN4_EXP_PACK_MAIN_KV): spread
    padding gives each column the distinct null-page rows a single column
    gets, so neither column's gather repeats one row; the shape stays
    [size, 2] for an empty list too."""
    main_s = np.stack([np.arange(100, 228), np.arange(900, 1028)], axis=1).astype(np.int64)
    main_d = np.arange(5000, 5128, dtype=np.int64)
    (chunk,) = st.repack_chunks(main_s, main_d, [], [], [7], [9], size=1024, gdn_pages=1,
                                null_rows=(256, 512))
    ms, md = chunk[0], chunk[1]
    assert ms.shape == (1024, 2)
    np.testing.assert_array_equal(ms[:128], main_s)
    for col in range(2):
        np.testing.assert_array_equal(ms[128:, col], np.arange(1024 - 128) % 256)
    np.testing.assert_array_equal(md[128:], np.arange(1024 - 128) % 512)
    (empty,) = st.repack_chunks(np.zeros((0, 2), np.int64), np.zeros(0, np.int64), [], [],
                                [7], [9], size=256, gdn_pages=1, null_rows=(256, 512))
    assert empty[0].shape == (256, 2) and len(set(empty[0][:, 1].tolist())) == 256


@pytest.mark.parametrize(
    "num_tokens,skip", [(2049, 0), (2049, 1792), (321, 0), (9000, 0), (10000, 512)]
)
def test_paired_main_lands_the_same_under_every_repack_shape(monkeypatch, num_tokens, skip):
    """Paired main KV with 36 GDN owners: QWEN4_EXP_PD_REPACK_SHAPE's split-GDN
    programs and spread padding land the pool the default 16384,16384,4
    chunks land (that landing is checked against the slot-merge reference
    above). 8192,8192,36 is the shape the stack runs with packing; the two
    largest loads give a rank more paired main rows (one pair and one odd
    owner) than that, without and with a local prefix, so they take two
    programs of it."""
    monkeypatch.setitem(globals(), "FA_LAYERS", 3)
    names = owners(gdn_layers=36)
    rng = np.random.default_rng(num_tokens + skip + 3)
    prod = producer_pages(num_tokens, rng, names)
    tables = dst_tables(num_tokens, rng, names)
    null_rows = (GEO.stage_rows, DST_BLOCK)
    for r in (0, 1, 3):
        rank = rank_slice(r)
        staging, pages, _ = routed_staging(num_tokens, skip, rank, prod, names)
        idx = consumer(rank, names, pairs=PACK_PAIRS)._routed_indices(
            pages, tables, num_tokens, skip
        )
        assert idx[0].ndim == 2
        fns = (make_fn(rank, True), make_fn(rank, True, gdn=False))
        results = []
        n_programs = {}
        for shape, split, rows in (((16384, 16384, 4), False, None),
                                   ((16384, 16384, 4), False, null_rows),
                                   ((8192, 8192, 36), True, null_rows),
                                   ((256, 128, 36), True, null_rows)):
            main, copied, gdn = shape
            out = jnp.asarray(np.zeros((400, DST_BLOCK, 1, ROWS, LANES), np.uint8))
            chunks = list(st.repack_chunks(*idx, main, gdn, copy_size=copied, split_gdn=split,
                                           null_rows=rows))
            for c in chunks:
                assert c[0].shape == (main, 2)
                index = (jnp.asarray(0 if v is None else v, jnp.int32) for v in c)
                out = fns[c[4] is None](out, jnp.asarray(staging), *index)
            results.append(np.asarray(out))
            n_programs[shape] = len(chunks)
        if num_tokens >= 9000:
            assert len(idx[1]) > 8192
            assert n_programs[(8192, 8192, 36)] == 2  # chunked at the packed stack's shape
        for got in results[1:]:
            np.testing.assert_array_equal(got, results[0], err_msg=f"rank {r}")
        assert results[0].any()


def test_warm_repack_runs_the_paired_main_shape(monkeypatch):
    monkeypatch.setitem(globals(), "FA_LAYERS", 3)
    c = consumer(rank_slice(0), owners(gdn_layers=36), pairs=PACK_PAIRS)
    c._repack_shape, c._split_gdn = (16384, 8192, 36), True
    c._null_rows = (GEO.stage_rows, DST_BLOCK)
    ran = []
    c._run_chunks = ran.extend
    c.warm_repack()
    assert ran[0][0].shape == (16384, 2) and ran[1][0].shape == (16384, 2)
    assert len(set(ran[0][0][:, 1].tolist())) == GEO.stage_rows  # spread over page-0 rows

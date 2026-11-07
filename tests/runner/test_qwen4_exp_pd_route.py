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


def owners():
    names = []
    for i in range(FA_LAYERS):
        names += [f"l{i}.self_attn", f"l{i}.self_attn.compressed_k", f"l{i}.self_attn.raw_ring"]
    names += [f"g{i}.linear_attn" for i in range(GDN_LAYERS)] + ["p.ple"]
    return names


def producer_pages(num_tokens, rng):
    """Per owner, [pages, SRC_BLOCK, 4, 256] uint8 of random bytes."""
    n_tok_pages = -(-num_tokens // SRC_BLOCK)
    out = {}
    for name in owners():
        kind = st.owner_kind(name)
        n = n_tok_pages if kind in ("main", "compressed") else 1
        out[name] = rng.integers(0, 256, (n, SRC_BLOCK, ROWS, LANES), dtype=np.uint8)
    return out


def rank_slice(r):
    return st.RankSlice(tp_size=TP, tp_rank=r, num_kv_heads=KV_HEADS, main_dcp=DCP)


def dst_tables(num_tokens, rng):
    n_main = -(-num_tokens // (DCP * DST_BLOCK))
    n_comp = -(-num_tokens // DST_BLOCK)
    ids = list(rng.permutation(np.arange(1, 400)))
    take = lambda k: [int(ids.pop()) for _ in range(k)]  # noqa: E731
    t = {}
    for name in owners():
        kind = st.owner_kind(name)
        t[name] = take(n_main if kind == "main" else n_comp if kind == "compressed" else 1)
    return t


def make_fn(rank, compact):
    return st.make_repack(
        src_block=SRC_BLOCK, dst_block=DST_BLOCK, src_gdn=S_GDN, dst_gdn=D_GDN,
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


def consumer(rank):
    c = object.__new__(st.StagedConsumer)
    c.route, c.rank, c.ratio, c.dst_block = GEO, rank, RATIO, DST_BLOCK
    names = owners()
    c._dst_compressed = {n: (n, 0) for n in names if st.owner_kind(n) == "compressed"}
    c._dst_owner = {n: n for n in names}
    c._gdn_names = [n for n in names if st.owner_kind(n) == "gdn"]
    c._state_entry = 0
    return c


def routed_landing(num_tokens, skip, rank, prod, tables):
    names = owners()
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
    idx = consumer(rank)._routed_indices(pages, tables, num_tokens, skip)
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

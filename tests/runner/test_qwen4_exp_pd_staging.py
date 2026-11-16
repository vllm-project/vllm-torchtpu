# SPDX-License-Identifier: Apache-2.0
"""Decode-side repack of TP1 pipeline-producer pages (qwen4_exp pd_staging).

Producer pages are built from logical values with an independent writer that
follows the pool layout (``pack_rows`` main-KV rows, compressed-key pairs, and
GDN state in Mosaic typed-view order: an F32 is the 4 sublanes at one lane, low
byte first); each decode rank's repacked pages must then hold exactly its
slice in its own geometry.
"""

import importlib.util
import math
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

_PATH = (
    pathlib.Path(__file__).parents[2]
    / "src/vllm_torchtpu/models/vllm/qwen4_exp/pd_staging.py"
)
_spec = importlib.util.spec_from_file_location("pd_staging", _PATH)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

ROWS, LANES = 4, 256


def carrier(pages: int, block: int) -> np.ndarray:
    # As the pools: one kernel page per manager page.
    return np.zeros((pages, block, 1, ROWS, LANES), np.uint8)


def flat_rows(c: np.ndarray) -> np.ndarray:
    return c.reshape(-1, ROWS, LANES)


def scatter(pool, rows, page, logical, valid):
    """The CPU stand-in for the serving path's aliased DMA scatter."""
    page = jnp.where(valid, page, pool.shape[0])  # out of range: dropped
    return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")


def write_typed(rows: np.ndarray, values: np.ndarray, itemsize: int) -> None:
    """Reference writer: element e -> row e // per_row, Mosaic word order."""
    raw = values.view(np.uint8).reshape(-1, itemsize)
    per_word = 4 // itemsize
    for e, b in enumerate(raw):
        word = e // per_word  # (row, sub-word, lane) streams lanes fastest
        if itemsize == 4:
            r, lane = divmod(word, LANES)
            rows[r, :, lane] = b
        else:
            r, rest = divmod(e, 2 * LANES)
            half, lane = divmod(rest, LANES)
            rows[r, 2 * half : 2 * half + 2, lane] = b


def read_typed(rows: np.ndarray, count: int, itemsize: int, dtype) -> np.ndarray:
    out = np.zeros(count * itemsize, np.uint8).reshape(count, itemsize)
    for e in range(count):
        if itemsize == 4:
            r, lane = divmod(e, LANES)
            out[e] = rows[r, :, lane]
        else:
            r, rest = divmod(e, 2 * LANES)
            half, lane = divmod(rest, LANES)
            out[e] = rows[r, 2 * half : 2 * half + 2, lane]
    return out.reshape(-1).view(dtype)


def test_typed_view_matches_reference_order():
    rng = np.random.default_rng(0)
    rows = np.zeros((3, ROWS, LANES), np.uint8)
    for itemsize, dtype in ((4, np.float32), (2, np.uint16)):
        values = rng.integers(0, 2**31, 600).astype(np.uint32).view(np.float32) if itemsize == 4 \
            else rng.integers(0, 2**16, 900).astype(np.uint16)
        rows[:] = 0
        write_typed(rows, values, itemsize)
        got = np.asarray(st._typed(jnp.asarray(rows), itemsize))[: len(values)]
        np.testing.assert_array_equal(got.view(np.uint8), values.view(np.uint8))
        back = np.asarray(st._untyped(jnp.asarray(got), 3, itemsize))
        np.testing.assert_array_equal(back, rows)


@pytest.mark.parametrize("tp,dcp,chunk", [(8, 2, None), (8, 1, None), (2, 1, None), (2, 1, 96)])
def test_repack_rank_slices(tp, dcp, chunk):
    rng = np.random.default_rng(tp * 10 + dcp)
    # Small stand-ins for the TP1 / TP-T geometries (B multiple of 128).
    src_block, dst_block = 384, 128
    n_tokens, ratio = 700, 4
    kv_heads = 2
    s = st.GdnGeometry(ssm_tokens=8, conv_tokens=8, v_heads=16, k_heads=8,
                       head_k_dim=8, head_v_dim=16, conv_width=3,
                       ssm_itemsize=4, conv_itemsize=4)
    v_local, k_local = 16 // tp, 8 // tp
    ssm_rows = -(-v_local * 8 * 16 // LANES)
    conv_rows = -(-3 * (2 * k_local * 8 + v_local * 16) // LANES)
    d = st.GdnGeometry(ssm_tokens=ssm_rows, conv_tokens=1 << (conv_rows - 1).bit_length(),
                       v_heads=v_local, k_heads=k_local, head_k_dim=8, head_v_dim=16,
                       conv_width=3, ssm_itemsize=4, conv_itemsize=4)
    # Producer: main KV [K0 V0 K1 V1] per token, compressed pairs, ring rows, GDN.
    staging = carrier(12, src_block)
    stage_rows = flat_rows(staging)
    src_main_pages = [3, 5]          # 700 tokens / 384
    src_comp_pages = [7, 2]
    src_ring_page, src_gdn_page = 9, 10
    kv = rng.integers(1, 255, (n_tokens, 4, LANES)).astype(np.uint8)
    for t in range(n_tokens):
        stage_rows[src_main_pages[t // src_block] * src_block + t % src_block] = kv[t]
    pairs = rng.integers(1, 255, (n_tokens // (2 * ratio) + 1, 4, LANES)).astype(np.uint8)
    for q in range(len(pairs)):
        t = q * 2 * ratio
        stage_rows[src_comp_pages[t // src_block] * src_block + (t % src_block) // (2 * ratio)] = pairs[q]
    ring = rng.integers(1, 255, (dst_block, 4, LANES)).astype(np.uint8)
    stage_rows[src_ring_page * src_block : src_ring_page * src_block + dst_block] = ring
    ssm = rng.standard_normal((s.v_heads, s.head_k_dim * s.head_v_dim)).astype(np.float32)
    conv = rng.standard_normal((s.conv_width, s.conv_dim)).astype(np.float32)
    base = src_gdn_page * src_block
    write_typed(stage_rows[base : base + s.ssm_tokens], ssm.reshape(-1), 4)
    write_typed(stage_rows[base + s.ssm_tokens : base + s.ssm_tokens + s.conv_tokens],
                conv.reshape(-1), 4)

    for r in range(tp):
        rank = st.RankSlice(tp_size=tp, tp_rank=r, num_kv_heads=kv_heads, main_dcp=dcp)
        pool = carrier(40, dst_block)
        n_main_pages = -(-n_tokens // (dcp * dst_block))
        dst_main = list(range(1, 1 + n_main_pages))
        dst_comp = list(range(20, 20 + -(-n_tokens // dst_block)))
        dst_ring, dst_gdn = 30, 31
        ms, md = st.main_rows(n_tokens, src_main_pages, src_block, dst_main, dst_block, rank)
        cs, cd = st.compressed_rows(n_tokens, ratio, src_comp_pages, src_block, dst_comp, dst_block)
        rs, rd = st.page_prefix_rows(src_ring_page, src_block, dst_ring, dst_block, dst_block)
        copy_s, copy_d = cs + rs, cd + rd
        fn = st.make_repack(
            src_block=src_block, dst_block=dst_block, src_gdn=s, dst_gdn=d,
            kv_head=rank.kv_head, v_head_start=r * d.v_heads,
            channels=st.conv_channels(s, d, rank), write=scatter,
        )
        # One program, or the serving path's fixed-size chunks (pool row 0
        # and staging row 0 are the null pages, as in serving).
        size = chunk or st.bucket(max(len(ms), len(copy_s)), 16)
        out = jnp.asarray(pool)
        plans = list(st.repack_chunks(ms, md, copy_s, copy_d, [src_gdn_page], [dst_gdn], size, 1))
        assert len(plans) == (-(-max(len(ms), len(copy_s)) // size) if chunk else 1)
        for plan in plans:
            out = fn(out, jnp.asarray(staging), *(jnp.asarray(v, jnp.int32) for v in plan))
        rows = flat_rows(np.asarray(out))
        h = rank.kv_head
        for t in range(n_tokens):
            if dcp > 1 and (t // dst_block) % dcp != rank.dcp_rank:
                continue
            row = rows[dst_main[t // (dcp * dst_block)] * dst_block + t % dst_block]
            np.testing.assert_array_equal(row[:2], kv[t, 2 * h : 2 * h + 2])
            assert not row[2:].any()
        for q in range(-(-(-(-n_tokens // ratio)) // 2)):
            t = q * 2 * ratio
            np.testing.assert_array_equal(
                rows[dst_comp[t // dst_block] * dst_block + (t % dst_block) // (2 * ratio)], pairs[q])
        np.testing.assert_array_equal(rows[dst_ring * dst_block : (dst_ring + 1) * dst_block], ring)
        gb = dst_gdn * dst_block
        got_ssm = read_typed(rows[gb : gb + d.ssm_tokens], d.v_heads * d.head_k_dim * d.head_v_dim, 4, np.float32)
        np.testing.assert_array_equal(got_ssm, ssm[r * d.v_heads : (r + 1) * d.v_heads].reshape(-1))
        got_conv = read_typed(rows[gb + d.ssm_tokens : gb + d.ssm_tokens + d.conv_tokens],
                              d.conv_width * d.conv_dim, 4, np.float32).reshape(d.conv_width, d.conv_dim)
        np.testing.assert_array_equal(got_conv, conv[:, st.conv_channels(s, d, rank)])


def test_pad_of_a_rank_without_rows_writes_the_null_row():
    # A DCP rank of a request shorter than one page owns no main-KV rows.
    rank = st.RankSlice(tp_size=8, tp_rank=6, num_kv_heads=2, main_dcp=4)
    src, dst = st.main_rows(24, [3], 384, [1], 128, rank)
    assert src == [] and dst == []
    assert st._pad(src, 4) == [0, 0, 0, 0]
    assert st._pad([5, 7], 4) == [5, 7, 7, 7]


def test_chunks_split_gdn_pages_and_pad_with_the_null_page():
    plans = list(st.repack_chunks([1, 2, 3], [4, 5, 6], [7], [8], [10, 11, 12], [20, 21, 22],
                                  size=2, gdn_pages=2))
    assert len(plans) == 2
    assert plans[0] == ([1, 2], [4, 5], [7, 7], [8, 8], [10, 11], [20, 21])
    assert plans[1] == ([3, 3], [6, 6], [0, 0], [0, 0], [12, 12], [22, 22])


def _rank_tokens(num_tokens, rank, dst_block, start=0):
    d = rank.main_dcp
    return [t for t in range(start, num_tokens)
            if d == 1 or (t // dst_block) % d == rank.dcp_rank]


@pytest.mark.parametrize("tp,rank_index,dcp", [(1, 0, 1), (8, 5, 2), (8, 7, 4)])
@pytest.mark.parametrize("num_tokens,skip", [(2000, 1024), (3072, 2048), (1025, 1024)])
def test_prefix_aware_main_rows_are_the_full_repack_tail(
    tp, rank_index, dcp, num_tokens, skip
):
    # A decode-local prefix of `skip` tokens: the staged suffix starts at the
    # first producer page not wholly inside it, and the repack writes exactly
    # the rows the full repack writes for tokens from `skip` on.
    rank = st.RankSlice(tp_size=tp, tp_rank=rank_index, num_kv_heads=2, main_dcp=dcp)
    src_block, dst_block = 384, 128
    src = list(range(10, 10 + -(-num_tokens // src_block)))
    dst = list(range(50, 50 + -(-num_tokens // (dcp * dst_block))))
    full_s, full_d = st.main_rows(num_tokens, src, src_block, dst, dst_block, rank)
    tokens = _rank_tokens(num_tokens, rank, dst_block)
    keep = [i for i, t in enumerate(tokens) if t >= skip]
    first = skip // src_block
    s, d = st.main_rows(num_tokens, src[first:], src_block, dst, dst_block, rank,
                        start=skip, src_first=first)
    # The staged suffix lists the same source pages from `first` on.
    assert s == [full_s[i] for i in keep]
    assert d == [full_d[i] for i in keep]
    assert all(row // dst_block in dst[skip // (dcp * dst_block):] for row in d)


@pytest.mark.parametrize("num_tokens,skip", [(2000, 1024), (1030, 1024), (4096, 256)])
def test_prefix_aware_compressed_rows_are_the_full_repack_tail(num_tokens, skip):
    ratio, src_block, dst_block = 4, 384, 128
    src = list(range(10, 10 + -(-num_tokens // src_block)))
    dst = list(range(50, 50 + -(-num_tokens // dst_block)))
    full_s, full_d = st.compressed_rows(
        num_tokens, ratio, src, src_block, dst, dst_block
    )
    first_pair = skip // (2 * ratio)
    first = skip // src_block
    s, d = st.compressed_rows(num_tokens, ratio, src[first:], src_block, dst, dst_block,
                              start=skip, src_first=first)
    assert s == full_s[first_pair:]
    assert d == full_d[first_pair:]


def test_prefix_aware_compressed_rows_refuse_a_split_pair():
    with pytest.raises(ValueError, match="compressed pair"):
        st.compressed_rows(100, 4, [1], 384, [1], 128, start=12)


def test_without_speculation_drops_only_the_draft_config():
    import dataclasses as dc

    @dc.dataclass
    class Cfg:
        speculative_config: object
        other: int = 3

    cfg = Cfg(speculative_config={"method": "mtp"})
    peer = st.without_speculation(cfg)
    assert peer.speculative_config is None and peer.other == 3
    assert cfg.speculative_config == {"method": "mtp"}
    plain = Cfg(speculative_config=None)
    assert st.without_speculation(plain) is plain

class _Layout:
    def __init__(self, owners=(), slots=1):
        self.compressed_owners = tuple(owners)
        self.compressed_slots = slots


def test_compressed_destinations_unpacked_owns_its_group_at_row_zero():
    dst_owner = {"model.layers.3.compressed_k": 5, "model.layers.3.self_attn": 4}
    out = st.compressed_destinations(_Layout(), dst_owner, 512, 4)
    assert out == {"model.layers.3.compressed_k": (5, 0)}


def test_compressed_destinations_packed_members_offset_by_slot():
    owners = [
        (f"model.layers.{i}", "model.layers.3.compressed_k", s)
        for s, i in enumerate((3, 7, 11))
    ]
    dst_owner = {"model.layers.3.compressed_k": 9}
    out = st.compressed_destinations(_Layout(owners, 8), dst_owner, 512, 4)
    # 512 tokens / (2 * 4) = 64 pair rows per member.
    assert out == {
        "model.layers.3.compressed_k": (9, 0),
        "model.layers.7.compressed_k": (9, 64),
        "model.layers.11.compressed_k": (9, 128),
    }


def test_compressed_destinations_refuse_overflow():
    owners = [("a", "a.compressed_k", 0)]
    with pytest.raises(ValueError):
        st.compressed_destinations(_Layout(owners, 9), {"a.compressed_k": 0}, 512, 4)


def test_packed_compressed_rows_match_kernel_pair_addressing():
    # pooled_qsa writes pair = page * block + slot * (block // (2r)) + (t % block) // (2r).
    block, ratio, slot = 512, 4, 3
    dst_pages = [7, 2]
    s, d = st.compressed_rows(
        1024, ratio, [1, 2, 3], 3200, dst_pages, block, dst_offset=slot * (block // (2 * ratio))
    )
    for q, row in enumerate(d):
        t = q * 2 * ratio
        assert row == dst_pages[t // block] * block + slot * (block // (2 * ratio)) + (t % block) // (2 * ratio)
    # The source rows are those of the unpacked repack.
    s0, d0 = st.compressed_rows(1024, ratio, [1, 2, 3], 3200, dst_pages, block)
    assert s == s0 and [x - slot * 64 for x in d] == d0


def _main_rows_loop(num_tokens, src_pages, src_block, dst_pages, dst_block, rank, start=0, src_first=0):
    src, dst = [], []
    d = rank.main_dcp
    for t in range(start, num_tokens):
        if d > 1 and (t // dst_block) % d != rank.dcp_rank:
            continue
        src.append(src_pages[t // src_block - src_first] * src_block + t % src_block)
        dst.append(dst_pages[t // (d * dst_block)] * dst_block + t % dst_block)
    return src, dst


def _compressed_rows_loop(num_tokens, ratio, src_pages, src_block, dst_pages, dst_block,
                          start=0, src_first=0, dst_offset=0):
    import math
    span = 2 * ratio
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    src, dst = [], []
    for q in range(start // span, pairs):
        t = q * span
        src.append(src_pages[t // src_block - src_first] * src_block + (t % src_block) // span)
        dst.append(dst_pages[t // dst_block] * dst_block + dst_offset + (t % dst_block) // span)
    return src, dst


@pytest.mark.parametrize("num_tokens,start", [(1, 0), (511, 0), (7521, 0), (196668, 0), (20000, 4096)])
@pytest.mark.parametrize("dcp,tp_rank", [(1, 0), (4, 0), (4, 3), (4, 6)])
def test_vectorized_rows_match_the_loop(num_tokens, start, dcp, tp_rank):
    rng = np.random.default_rng(num_tokens + dcp + tp_rank)
    src_block, dst_block, ratio = 3200, 512, 4
    first = start // src_block
    src_pages = rng.permutation(10_000)[: -(-num_tokens // src_block) - first + 1].tolist()
    dst_pages = rng.permutation(50_000)[: -(-num_tokens // dst_block) + 1].tolist()
    rank = st.RankSlice(tp_size=8, tp_rank=tp_rank, num_kv_heads=2, main_dcp=dcp)
    assert st.main_rows(num_tokens, src_pages, src_block, dst_pages, dst_block, rank,
                        start=start, src_first=first) == _main_rows_loop(
        num_tokens, src_pages, src_block, dst_pages, dst_block, rank, start, first)
    cstart = start - start % (2 * ratio)
    assert st.compressed_rows(num_tokens, ratio, src_pages, src_block, dst_pages, dst_block,
                              start=cstart, src_first=first, dst_offset=128) == _compressed_rows_loop(
        num_tokens, ratio, src_pages, src_block, dst_pages, dst_block, cstart, first, 128)


def test_vectorized_rows_are_fast():
    import time
    rank = st.RankSlice(tp_size=8, tp_rank=0, num_kv_heads=2, main_dcp=4)
    src_pages = list(range(1, 70))
    dst_pages = list(range(1, 400))
    t0 = time.perf_counter()
    for _ in range(12):  # one request's 12 main-KV layers
        st.main_rows_np(200_000, src_pages, 3200, dst_pages, 512, rank)
    assert time.perf_counter() - t0 < 0.1


def test_pad_keeps_list_and_array_forms():
    assert st._pad([3, 4], 4) == [3, 4, 4, 4]
    assert st._pad([], 2) == [0, 0]
    assert st._pad(np.array([3, 4]), 4).tolist() == [3, 4, 4, 4]
    assert st._pad(np.zeros(0, dtype=np.int64), 2).tolist() == [0, 0]


def test_align_rows_repeats_the_last_row_to_a_tile():
    assert st.align_rows(np.arange(64)).tolist() == list(range(64))
    out = st.align_rows(np.array([5, 6, 7]), 4)
    assert out.tolist() == [5, 6, 7, 7]


def _bulk_tiles(dst, kernel_rows=512, tile=64):
    """Fraction of tiles the tiled scatter writes with one DMA."""
    dst = np.asarray(dst)
    dst = np.concatenate((dst, np.full(-len(dst) % tile, dst[-1])))
    page, off = dst // kernel_rows, dst % kernel_rows
    t = len(dst) // tile
    page, off = page.reshape(t, tile), off.reshape(t, tile)
    bulk = (page == page[:, :1]).all(1) & (off == off[:, :1] + np.arange(tile)).all(1)
    return bulk.mean()


def test_aligned_segments_fill_whole_tiles():
    rank = st.RankSlice(tp_size=8, tp_rank=0, num_kv_heads=2, main_dcp=4)
    rng = np.random.default_rng(0)
    segs = []
    for _ in range(12):  # main KV layers of one 100k-token request
        dst_pages = rng.permutation(50_000)[:300].tolist()
        _, d = st.main_rows_np(100_000, list(range(1, 40)), 3200, dst_pages, 512, rank)
        segs.append(st.align_rows(d))
    for _ in range(12):  # compressed keys
        dst_pages = rng.permutation(50_000)[:300].tolist()
        _, d = st.compressed_rows_np(100_000, 4, list(range(1, 40)), 3200, dst_pages, 512)
        segs.append(st.align_rows(d))
    assert _bulk_tiles(np.concatenate(segs)) > 0.99


# --- q38 F2: QWEN4_EXP_PD_DRAFT_KV (the producer ships the MTP draft's KV) ---
# The decode-side byte identity, the refusals through StagedConsumer and the
# multi-turn path are in tests/distributed/test_qwen4_exp_decode_prefix_cache.py;
# these pin the two pure helpers that StagedConsumer calls.

_DRAFT = "mtp.layers.48.self_attn"
_TARGET = "model.layers.3.self_attn"
_DST = [_TARGET, _TARGET + ".raw_ring", _DRAFT, _DRAFT + ".raw_ring"]


def _decode_cfg(k=3, method="mtp"):
    import dataclasses as dc
    from types import SimpleNamespace

    @dc.dataclass
    class Cfg:
        speculative_config: object
        other: int = 3

    spec = None if k is None else SimpleNamespace(
        method=method, num_speculative_tokens=k, draft_tensor_parallel_size=None
    )
    return Cfg(speculative_config=spec)


def test_producer_geometry_without_the_flag_is_the_tp1_plan_without_the_draft(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_KV", raising=False)
    cfg = _decode_cfg(3)
    layout_cfg, tp, dcp, pack, draft_cacheable = st.producer_geometry(cfg)
    assert (tp, dcp, pack, draft_cacheable) == (1, 1, False, True)
    assert layout_cfg.speculative_config is None and layout_cfg.other == 3
    assert cfg.speculative_config.num_speculative_tokens == 3


def test_producer_geometry_with_the_flag_is_the_k1_mtp_producer(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    cfg = _decode_cfg(3)
    layout_cfg, tp, dcp, pack, draft_cacheable = st.producer_geometry(cfg)
    # Option (a): the producer's draft owners are not prefix-cacheable.
    assert (tp, dcp, pack, draft_cacheable) == (1, 1, False, False)
    spec = layout_cfg.speculative_config
    assert (spec.method, spec.num_speculative_tokens) == ("mtp", 1)
    assert spec.draft_tensor_parallel_size is None and layout_cfg.other == 3
    # The decode's own config is untouched: its draft still verifies K = 3.
    assert layout_cfg is not cfg and spec is not cfg.speculative_config
    assert cfg.speculative_config.num_speculative_tokens == 3


@pytest.mark.parametrize(
    "k,method", [pytest.param(None, "mtp", id="no_spec"), pytest.param(3, "eagle3", id="eagle3")]
)
def test_producer_geometry_with_the_flag_needs_an_mtp_decode(monkeypatch, k, method):
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    with pytest.raises(ValueError, match="MTP speculative config"):
        st.producer_geometry(_decode_cfg(k, method))


def test_check_draft_kv_admits_a_complete_split_landing(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    st.check_draft_kv(set(), _DST, 4, (_TARGET, _DRAFT))
    # Without main-KV DCP every rank keeps all of the draft's rows.
    st.check_draft_kv(set(), _DST, 1, ())


def test_check_draft_kv_refuses_missing_owners(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    missing = {_DRAFT, _DRAFT + ".compressed_k"}
    with pytest.raises(ValueError, match="no pages for") as err:
        st.check_draft_kv(missing, _DST, 4, (_TARGET, _DRAFT))
    assert all(name in str(err.value) for name in missing)


def test_check_draft_kv_refuses_zero_draft(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", "1")
    with pytest.raises(ValueError, match="QWEN4_EXP_PD_ZERO_DRAFT_KV=1 would zero"):
        st.check_draft_kv(set(), _DST, 4, (_TARGET, _DRAFT))


def test_check_draft_kv_refuses_an_unsplit_draft_on_a_dcp_decode(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    with pytest.raises(ValueError, match="needs VLLM_TPU_QSA_MTP_KV_DCP=1") as err:
        st.check_draft_kv(set(), _DST, 4, (_TARGET,))
    assert _DRAFT in str(err.value)


# --- q38 F2 option (a): the producer never caches draft KV --------------------
# The producer computes draft rows only from its prefix hit Hp; a decode with a
# shorter local hit Hd zero-fills the draft rows in [Hd, Hp) (and a forced gap,
# QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP) instead of landing unwritten ones.


def test_check_draft_kv_refuses_compressed_dcp(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    owners = (_TARGET, _DRAFT, _TARGET + ".compressed_k", _DRAFT + ".compressed_k")
    with pytest.raises(ValueError, match="VLLM_TPU_QSA_COMPRESSED_DCP=1"):
        st.check_draft_kv(set(), _DST, 4, owners)


@pytest.mark.parametrize(
    "raw,want",
    [(None, 0.0), ("", 0.0), ("0", 0.0), ("0.25", 0.25), ("1", None), ("-0.1", None)],
    ids=["unset", "empty", "zero", "quarter", "one", "negative"],
)
def test_force_gap_fraction_reads_a_fraction_below_one(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", raising=False)
    else:
        monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", raw)
    if want is None:
        with pytest.raises(ValueError, match="QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP"):
            st.force_gap_fraction()
    else:
        assert st.force_gap_fraction() == want


@pytest.mark.parametrize(
    "params,want",
    [
        ({st.PRODUCER_HIT_PARAM: 6400}, 6400),
        ({}, 0),
        (None, 0),
        ({st.PRODUCER_HIT_PARAM: -5}, 0),
        ({st.PRODUCER_HIT_PARAM: "6400"}, 0),
        ({st.PRODUCER_HIT_PARAM: True}, 0),
    ],
    ids=["int", "absent", "none", "negative", "string", "bool"],
)
def test_published_hit_tokens_is_zero_unless_a_positive_int(params, want):
    assert st.published_hit_tokens(params) == want


def _zero_end_consumer(hit=0, draft_kv=True):
    c = object.__new__(st.StagedConsumer)
    c.draft_kv, c.ratio = draft_kv, 4
    if hit:
        c.draft_zero = {7: hit}
    return c


@pytest.mark.parametrize(
    "hit,fraction,want",
    [
        pytest.param(0, None, 0, id="no_hit"),
        pytest.param(9600, None, 9600, id="hit"),
        pytest.param(0, "0.3", 5992, id="forced"),  # int(0.3 * 19999) = 5999 -> 5992
        pytest.param(9600, "0.3", 9600, id="hit_above_forced"),
        pytest.param(3200, "0.3", 5992, id="forced_above_hit"),
        pytest.param(25600, None, 19999, id="capped"),
    ],
)
def test_draft_zero_end_is_the_hit_raised_to_the_forced_gap(monkeypatch, hit, fraction, want):
    if fraction is None:
        monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", raising=False)
    else:
        monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", fraction)
    assert _zero_end_consumer(hit).draft_zero_end(7, 19999) == want


def test_draft_zero_end_is_zero_with_the_flag_off(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", "0.5")
    assert _zero_end_consumer(9600, draft_kv=False).draft_zero_end(7, 19999) == 0


@pytest.mark.parametrize("dcp,tp_rank", [(1, 0), (4, 0), (4, 3), (4, 6)])
@pytest.mark.parametrize("num_tokens,start", [(1, 0), (7521, 0), (20000, 4096), (20000, 18432)])
def test_token_enumerations_match_the_row_planners(dcp, tp_rank, num_tokens, start):
    """The draft gap splits landing rows by token: main_tokens_np and
    pair_tokens_np must list exactly the tokens of main_rows_np's and
    compressed_rows_np's rows, in order."""
    rank = st.RankSlice(tp_size=8, tp_rank=tp_rank, num_kv_heads=2, main_dcp=dcp)
    dst, src, first = list(range(100, 160)), np.arange(1, 9), start // 3200
    s, d = st.main_rows_np(num_tokens, src, 3200, dst, 512, rank, start=start, src_first=first)
    t = st.main_tokens_np(num_tokens, 512, rank, start)
    assert len(t) == len(d)
    # The source rows name the token itself (destinations repeat across a
    # DCP entry's ranks).
    assert np.array_equal(src[t // 3200 - first] * 3200 + t % 3200, s)
    assert np.array_equal(np.asarray(dst)[t // (dcp * 512)] * 512 + t % 512, d)
    s, d = st.compressed_rows_np(num_tokens, 4, src, 3200, dst, 512, start=start, src_first=first)
    p = st.pair_tokens_np(num_tokens, 4, start)
    assert len(p) == len(d)
    assert np.array_equal(src[p // 3200 - first] * 3200 + (p % 3200) // 8, s)
    assert np.array_equal(np.asarray(dst)[p // 512] * 512 + (p % 512) // 8, d)


def test_draft_gap_zero_fills_only_the_draft_tokens_below_its_end():
    t = np.arange(4096, 8192, dtype=np.int64)
    s, d = t + 1_000_000, t + 2_000_000
    gap = st._DraftGap(6000)
    s2, d2 = gap.split(_DRAFT, s, d, lambda: t)
    assert np.array_equal(d2, d[t >= 6000]) and np.array_equal(s2, s[t >= 6000])
    # A target owner is untouched, whatever the gap.
    s3, d3 = gap.split(_TARGET, s, d, lambda: (_ for _ in ()).throw(AssertionError))
    assert s3 is s and d3 is d
    copy_s, copy_d = [], []
    gap.emit(copy_s, copy_d)
    assert np.array_equal(np.concatenate(copy_d), d[t < 6000])
    assert not np.concatenate(copy_s).any()  # from the null staging row
    assert (gap.landed, gap.zeroed) == (int((t >= 6000).sum()), int((t < 6000).sum()))
    # No gap: everything lands.
    none = st._DraftGap(0)
    s4, d4 = none.split(_DRAFT, s, d, lambda: t)
    assert s4 is s and d4 is d and none.zeroed == 0
    copy_s, copy_d = [], []
    none.emit(copy_s, copy_d)
    assert copy_d == []


def test_draft_group_problems_names_cacheable_or_mixed_draft_groups():
    from types import SimpleNamespace as NS

    def group(names, cacheable):
        return NS(layer_names=names, kv_cache_spec=NS(prefix_cacheable=cacheable))

    good = [group([_TARGET], True), group([_DRAFT], False), group([_DRAFT + ".raw_ring"], False)]
    assert st.draft_group_problems(good) == []
    bad = [group([_TARGET], True), group([_DRAFT], True), group([_TARGET + "x", _DRAFT + ".c"], False)]
    problems = st.draft_group_problems(bad)
    assert len(problems) == 2 and _DRAFT in problems[0]


def _gap_consumer(mode):
    """A staged TP8 / DCP4 rank (tp_rank 1) with the draft's and one target's
    main KV, packed compressed slots and ring: broadcast staging, routed
    compact staging (QWEN4_EXP_PD_ROUTED_DST), or routed with S4's landing
    gather (QWEN4_EXP_PD_SHARED_COMPRESSED)."""
    c = object.__new__(st.StagedConsumer)
    c.rank = st.RankSlice(tp_size=8, tp_rank=1, num_kv_heads=2, main_dcp=4)
    c.ratio, c.dst_block, c.src_block, c._state_entry = 4, 512, 3200, 0
    c.route = None if mode == "broadcast" else st.pd_route.RouteGeometry(
        tp=8, dcp=4, dst_block=512, src_block=3200, ratio=4,
        ssm_rows_src=2048, conv_rows_src=256, ssm_rows_dst=256,
    )
    c.shared = st.pd_shared_compressed.geometry_for(c.route) if mode == "s4_shared" else None
    c._dst_owner = {_DRAFT: 0, _TARGET: 1, _DRAFT + ".raw_ring": 2}
    c._dst_compressed = {_DRAFT + ".compressed_k": (3, 64), _TARGET + ".compressed_k": (3, 0)}
    names = list(c._dst_owner) + list(c._dst_compressed)
    c._src_owner = {name: i for i, name in enumerate(names)}
    c.src_groups = [None] * len(names)
    c._gdn_names, c._zero_draft = [], {}
    return c


@pytest.mark.parametrize("mode", ["broadcast", "routed", "s4_shared"])
def test_every_landing_path_zero_fills_the_draft_gap(mode):
    """``indices`` with a draft gap [skip, hi): the draft's main-KV rows of
    this rank's DCP slice and its compressed pairs in the gap are zero-filled
    from the null row and not landed; everything else plans as with no gap.
    Under S4 the gather lands compressed pairs, and the repack's zero rows
    (enqueued after it) overwrite the gap's."""
    c = _gap_consumer(mode)
    n, skip, hi = 20000, 4096, 9600
    assert c._shared_load(n) == (mode == "s4_shared")
    pages = {name: list(range(1, 400)) for name in c._src_owner}
    tables = [list(range(1000, 1100)), list(range(2000, 2100)), [3000],
              list(range(4000, 4100))]
    first = [skip // 3200] * len(c.src_groups)  # staging begins at that source page
    ms0, md0, cs0, cd0, _, _ = c.indices(pages, tables, n, skip, first)
    ms1, md1, cs1, cd1, _, _ = c.indices(pages, tables, n, skip, first, draft_zero_until=hi)
    gap_main = set(st.main_dst_np(hi, tables[0], 512, c.rank, skip).tolist())
    gap_pairs = set(st.compressed_dst_np(hi, 4, tables[3], 512, skip, 64).tolist())
    assert gap_main and gap_pairs
    # Main KV: the draft's gap rows leave the landing; nothing else changes.
    assert set(md1.tolist()) == set(md0.tolist()) - gap_main
    assert not set(md1.tolist()) & gap_main
    # Copies: the zero rows (source row 0) are exactly the draft's gap.
    zero = {int(d) for s, d in zip(cs1, cd1) if s == 0}
    assert zero == gap_main | gap_pairs
    landed1 = {int(d) for s, d in zip(cs1, cd1) if s != 0}
    landed0 = {int(d) for s, d in zip(cs0, cd0) if s != 0}
    assert landed1 == landed0 - gap_pairs
    assert not landed1 & zero
    assert c.last_draft_rows[1] == len(gap_main) + len(gap_pairs)


@pytest.mark.parametrize(
    "env", [{"QWEN4_EXP_PD_PRODUCER_PCP": "2"}, {"QWEN4_EXP_PD_PRODUCER_MERGED": "1"}]
)
def test_draft_kv_refuses_a_pcp_or_merged_key_producer(monkeypatch, env):
    """QWEN4_EXP_PD_DRAFT_KV models a plain TP1 producer's draft owners. The
    staged decode also takes PCP and merged-key producers, so the combination
    must fail when the decode is built."""
    pkg = pytest.importorskip("vllm_torchtpu.models.vllm.qwen4_exp.pd_staging")
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="plain TP1 pipeline producer"):
        pkg.StagedConsumer(_decode_cfg(3), None, None, None, None, 1 << 20)
    # Control: without the PCP / merged producer the decode gets past this
    # check and fails later, on the stand-in config, for a different reason.
    for key in env:
        monkeypatch.delenv(key)
    with pytest.raises(Exception) as other:
        pkg.StagedConsumer(_decode_cfg(3), None, None, None, None, 1 << 20)
    assert "plain TP1 pipeline producer" not in str(other.value)


# --- VLLM_TPU_QSA_COMPRESSED_DCP on a staged decode ----------------------------
# Every rank still receives every compressed pair; each lands only the pairs of
# its own B-token blocks, at row (t % B) // (2 ratio) of its page for the
# block's D x B table entry (pooled_qsa.compressed_pairs with dcp). The
# reference lands the load unsplit (today's planner, into per-block virtual
# pages) and then copies each owned block's rows to its split page.

_SPLIT_TP, _SPLIT_D, _SPLIT_B, _SPLIT_SRC, _SPLIT_R = 8, 4, 128, 384, 4
_MEMBERS = ("model.layers.3.self_attn.compressed_k", "model.layers.7.self_attn.compressed_k")
# Slot 2 of the packed group is the MTP draft's (no producer pages: never landed).
_MEMBER_ROWS = _SPLIT_B // (2 * _SPLIT_R)


def _split_consumer(tp_rank, split=True):
    c = object.__new__(st.StagedConsumer)
    c.rank = st.RankSlice(tp_size=_SPLIT_TP, tp_rank=tp_rank, num_kv_heads=2, main_dcp=_SPLIT_D)
    c.ratio, c.dst_block, c.src_block, c._state_entry = _SPLIT_R, _SPLIT_B, _SPLIT_SRC, 0
    c._dst_owner = {}
    c._dst_compressed = {name: (0, slot * _MEMBER_ROWS) for slot, name in enumerate(_MEMBERS)}
    c._src_owner = {name: i for i, name in enumerate(_MEMBERS)}
    c.src_groups = [None] * len(_MEMBERS)
    c._gdn_names, c._zero_draft = [], {}
    c._dcp_owners = frozenset(_MEMBERS) if split else frozenset()
    return c


def _copy_fn():
    # Compressed keys only: a repack whose main and GDN parts are empty.
    return st.make_repack(src_block=_SPLIT_SRC, dst_block=_SPLIT_B, src_gdn=None, dst_gdn=None,
                          kv_head=0, v_head_start=0, channels=[0], write=scatter)


def _run_copies(pool, staging, copy_s, copy_d):
    size = st.bucket(max(len(copy_s), 1), 16)
    out = jnp.asarray(pool)
    for plan in st.repack_chunks([], [], copy_s, copy_d, [], [], size, 1):
        out = _copy_fn()(out, jnp.asarray(staging), *(jnp.asarray(v, jnp.int32) for v in plan))
    return np.asarray(out)


def _split_load(num_tokens, skip, seed):
    """Staging with every producer compressed page (random keys), the staged
    suffix's page lists, split tables (one page per D x B entry, the same for
    every rank's page) and per-block virtual tables for the unsplit landing."""
    rng = np.random.default_rng(seed)
    n_src = -(-num_tokens // _SPLIT_SRC)
    first = skip // _SPLIT_SRC
    staging = carrier(1 + len(_MEMBERS) * n_src, _SPLIT_SRC)
    staging[1:] = rng.integers(0, 256, staging[1:].shape, dtype=np.uint8)
    full = {
        name: list(range(1 + i * n_src, 1 + (i + 1) * n_src)) for i, name in enumerate(_MEMBERS)
    }
    staged = {name: pages[first:] for name, pages in full.items()}
    entries = -(-num_tokens // (_SPLIT_D * _SPLIT_B))
    blocks = -(-num_tokens // _SPLIT_B)
    ids = rng.permutation(np.arange(1, 200))
    split_table = [int(x) for x in ids[:entries]]
    virtual = {name: [int(x) for x in ids[100 + 40 * i : 100 + 40 * i + blocks]]
               for i, name in enumerate(_MEMBERS)}
    stale = rng.integers(0, 256, (200, _SPLIT_B, 1, ROWS, LANES), dtype=np.uint8)
    return staging, staged, [first] * len(_MEMBERS), split_table, virtual, stale


def _reference_split_pool(
    num_tokens, skip, tp_rank, staging, staged, first, split_table, virtual, stale
):
    span = 2 * _SPLIT_R
    pairs = math.ceil(math.ceil(num_tokens / _SPLIT_R) / 2)
    copy_s, copy_d = [], []
    for name, (gi, offset) in _split_consumer(tp_rank)._dst_compressed.items():
        s, d = st.compressed_rows_np(num_tokens, _SPLIT_R, staged[name], _SPLIT_SRC, virtual[name],
                                     _SPLIT_B, start=skip, src_first=first[0], dst_offset=offset)
        copy_s.append(s)
        copy_d.append(d)
    unsplit = flat_rows(_run_copies(np.zeros_like(stale), staging,
                                    np.concatenate(copy_s), np.concatenate(copy_d)))
    ref = flat_rows(stale.copy())
    for name, (gi, offset) in _split_consumer(tp_rank)._dst_compressed.items():
        for k in range(skip // _SPLIT_B, -(-num_tokens // _SPLIT_B)):
            if k % _SPLIT_D != tp_rank % _SPLIT_D:
                continue
            rows = min(_MEMBER_ROWS, pairs - k * _SPLIT_B // span)  # pairs of the block
            src = virtual[name][k] * _SPLIT_B + offset
            dst = split_table[k // _SPLIT_D] * _SPLIT_B + offset
            ref[dst : dst + rows] = unsplit[src : src + rows]
    return ref


@pytest.mark.parametrize(
    "num_tokens,skip",
    [
        pytest.param(1500, 0, id="no_prefix"),
        pytest.param(2600, 1024, id="local_prefix"),  # skip on a D x B entry, mid source page
        pytest.param(300, 0, id="shorter_than_an_entry"),
        pytest.param(100, 0, id="shorter_than_a_block"),
    ],
)
def test_split_compressed_landing_is_the_owned_part_of_the_unsplit_landing(num_tokens, skip):
    staging, staged, first, split_table, virtual, stale = _split_load(num_tokens, skip, num_tokens)
    tables = [split_table]
    owned_any = False
    for r in range(_SPLIT_TP):
        c = _split_consumer(r)
        ms, md, cs, cd, gs, gd = c.indices(staged, tables, num_tokens, skip, first)
        assert len(ms) == len(gs) == 0
        got = flat_rows(_run_copies(stale, staging, cs, cd))
        want = _reference_split_pool(num_tokens, skip, r, staging, staged, first, split_table,
                                     virtual, stale)
        # Page 0 is the null page: an empty plan rewrites its row 0.
        np.testing.assert_array_equal(got[_SPLIT_B:], want[_SPLIT_B:], err_msg=f"rank {r}")
        changed = got[_SPLIT_B:] != flat_rows(stale)[_SPLIT_B:]
        written = int(changed.any(axis=(1, 2)).sum())
        owned_any |= written > 0
        if num_tokens <= _SPLIT_B and r % _SPLIT_D:
            assert written == 0  # a block of rank 0's only
        # The draft's slot (rows from 2 x member rows) is never written.
        pages = np.asarray(split_table)
        draft_rows = pages[:, None] * _SPLIT_B + 2 * _MEMBER_ROWS + np.arange(_MEMBER_ROWS)
        draft_rows = draft_rows.reshape(-1)
        np.testing.assert_array_equal(got[draft_rows], flat_rows(stale)[draft_rows])
    assert owned_any


@pytest.mark.parametrize("dcp,tp_rank", [(4, 0), (4, 3), (4, 6), (2, 5)])
@pytest.mark.parametrize("num_tokens,start", [(1, 0), (300, 0), (7521, 0), (20000, 4096)])
def test_split_compressed_planners_agree(dcp, tp_rank, num_tokens, start):
    """compressed_rows_np, compressed_rows_split_np (one producer rank) and
    compressed_dst_np list the same pairs, those pair_tokens_np names, with the
    kernel's split rows."""
    rank = st.RankSlice(tp_size=8, tp_rank=tp_rank, num_kv_heads=2, main_dcp=dcp)
    src, dst = np.arange(1, 9), list(range(100, 140))
    first = start // 3200
    t = st.pair_tokens_np(num_tokens, 4, start, 512, rank)
    every = st.pair_tokens_np(num_tokens, 4, start)
    assert np.array_equal(t, every[(every // 512) % dcp == tp_rank % dcp])
    want_d = np.asarray(dst)[t // (dcp * 512)] * 512 + 64 + (t % 512) // 8
    s, d = st.compressed_rows_np(num_tokens, 4, src, 3200, dst, 512, start=start,
                                 src_first=first, dst_offset=64, rank=rank)
    assert np.array_equal(d, want_d)
    assert np.array_equal(s, src[t // 3200 - first] * 3200 + (t % 3200) // 8)
    s1, d1 = st.compressed_rows_split_np(num_tokens, 4, [src], 3200, dst, 512, start=start,
                                         src_first=first, dst_offset=64, rank=rank)
    assert np.array_equal(s1, s) and np.array_equal(d1, d)
    assert np.array_equal(
        st.compressed_dst_np(num_tokens, 4, dst, 512, start, 64, rank=rank), want_d)
    # Without a rank (or at DCP 1) nothing changes.
    plain = st.RankSlice(tp_size=8, tp_rank=tp_rank, num_kv_heads=2, main_dcp=1)
    assert np.array_equal(st.compressed_dst_np(num_tokens, 4, dst, 512, start, 64, rank=plain),
                          st.compressed_dst_np(num_tokens, 4, dst, 512, start, 64))


def test_zero_draft_rows_follow_the_split():
    """QWEN4_EXP_PD_ZERO_DRAFT_KV: the draft's rows of the landed tokens, split
    like its pages (main KV under VLLM_TPU_QSA_MTP_KV_DCP, compressed keys
    under VLLM_TPU_QSA_COMPRESSED_DCP), else every row."""
    draft = "mtp.layers.48.self_attn"
    n, skip = 5000, 2048
    for split in (True, False):
        c = _split_consumer(5, split=split)
        c.dst_block, c.ratio = 512, 4
        c._dst_compressed = {}
        c._zero_draft = {draft: (1, 0), draft + ".compressed_k": (2, 128)}
        c._dcp_owners = frozenset(c._zero_draft) if split else frozenset()
        main_table = list(range(10, 13)) if split else list(range(10, 20))
        comp_table = list(range(30, 33)) if split else list(range(30, 40))
        _, _, cs, cd, _, _ = c.indices({}, [[], main_table, comp_table], n, skip, [0, 0, 0])
        assert not cs.any()  # from the null staging row
        d = 4 if split else 1
        rank = st.RankSlice(tp_size=8, tp_rank=5, num_kv_heads=2, main_dcp=d)
        t = np.arange(skip, n)
        t = t[(t // 512) % d == rank.dcp_rank] if split else t
        main = np.asarray(main_table)[t // (d * 512)] * 512 + t % 512
        p = np.arange(skip // 8, math.ceil(math.ceil(n / 4) / 2)) * 8
        p = p[(p // 512) % d == rank.dcp_rank] if split else p
        comp = np.asarray(comp_table)[p // (d * 512)] * 512 + 128 + (p % 512) // 8
        assert set(cd.tolist()) == set(main.tolist()) | set(comp.tolist())


def test_verify_split_keys_reads_zero_mismatches_and_catches_a_wrong_row():
    """QWEN4_EXP_PD_VERIFY_SPLIT_KEYS on the landing above, with a host reader
    in place of the device read-back: no mismatch after the split landing, one
    after a corrupted pool row, many after an unsplit (wrong) landing."""
    n, skip, r = 2600, 1024, 6
    staging, staged, first, split_table, virtual, stale = _split_load(n, skip, 7)
    c = _split_consumer(r)
    c.route, c.src_merged = None, False
    c.reservations, c.skips = {9: staged}, {9: (skip, first)}
    _, _, cs, cd, _, _ = c.indices(staged, [split_table], n, skip, first)
    pool = flat_rows(_run_copies(stale, staging, cs, cd))
    stage = flat_rows(staging)

    def reader(p):
        return lambda s, d: (stage[s], p[d])

    out = c.verify_split_keys(9, [split_table], n, read_rows=reader(pool))
    owned = sum(len(x[1]) for x in c.split_key_check_rows(9, [split_table], n))
    assert out["mismatched"] == 0 and out["pairs"] == owned > 0
    assert out["pool_digest"] == out["staging_digest"]
    assert out["nonzero"] == owned and out["control_equal"] == 0
    bad = pool.copy()
    row = c.split_key_check_rows(9, [split_table], n)[1][2][3]
    bad[row] ^= 1
    out = c.verify_split_keys(9, [split_table], n, read_rows=reader(bad))
    assert out["mismatched"] == 1 and out["by_owner"] == {_MEMBERS[0]: 0, _MEMBERS[1]: 1}
    # Rank 6's rows landed by rank 5's plan: nearly every pair differs.
    other = _split_consumer(5)
    _, _, cs, cd, _, _ = other.indices(staged, [split_table], n, skip, first)
    wrong = flat_rows(_run_copies(stale, staging, cs, cd))
    out = c.verify_split_keys(9, [split_table], n, read_rows=reader(wrong))
    assert out["mismatched"] == owned


# --- QWEN4_EXP_PACK_MAIN_KV on a staged decode ----------------------------------
# Consecutive target layers share one main-KV owner: member s keeps K and V in
# head slots 2s and 2s + 1 of the owner's rows (pool_layout.pooled_cache_specs);
# an odd last layer keeps whole rows. The repack writes both members' pairs in
# one row. The reference lands the load the slow way, unpacked (every layer on
# pages of its own, today's planner), then merges each member's landed pair into
# the owner's row by the rule of the decode's own paired writer
# (pooled_qsa._write_main_cache_paired: a lane's four FP8 head slots are one
# 32-bit word, low byte first; main_slot_rows and the keep mask).

_PACK_LAYERS = tuple(f"model.layers.{i}.self_attn" for i in (3, 7, 11, 15, 19))


def _main_owners(layers):
    """``layout.main_owners`` as pooled_cache_specs pairs ``layers``."""
    owners = []
    for start in range(0, len(layers) - 1, 2):
        chunk = layers[start : start + 2]
        owners += [(name, chunk[0], slot) for slot, name in enumerate(chunk)]
    return tuple(owners)


def _pack_pairs(layers):
    from types import SimpleNamespace

    return st.main_pairs(SimpleNamespace(main_owners=_main_owners(layers)))


def _pack_consumer(tp_rank, layers, pairs, compressed=(), d=_SPLIT_D):
    """A staged TP8 rank landing ``layers``' main KV (paired by ``pairs``) and,
    for F1's split, the split compressed keys of ``compressed`` (group "comp")."""
    c = object.__new__(st.StagedConsumer)
    c.rank = st.RankSlice(tp_size=_SPLIT_TP, tp_rank=tp_rank, num_kv_heads=2, main_dcp=d)
    c.ratio, c.dst_block, c.src_block, c._state_entry = _SPLIT_R, _SPLIT_B, _SPLIT_SRC, 0
    away = {m for members in pairs.values() for m in members[1:]}
    c._dst_owner = {n: n for n in layers if n not in away}
    c._main_pairs = pairs
    c._dst_compressed = {n: ("comp", slot * _MEMBER_ROWS) for slot, n in enumerate(compressed)}
    names = list(layers) + list(compressed)
    c._src_owner = {n: i for i, n in enumerate(names)}
    c.src_groups = [None] * len(names)
    c._gdn_names, c._zero_draft = [], {}
    c._dcp_owners = frozenset(compressed) | frozenset(c._dst_owner)
    return c


def _main_fn(rank):
    return st.make_repack(src_block=_SPLIT_SRC, dst_block=_SPLIT_B, src_gdn=None, dst_gdn=None,
                          kv_head=rank.kv_head, v_head_start=0, channels=[0], write=scatter)


def _land(c, pool, staging, idx, size=None, null_rows=None):
    size = size or st.bucket(max(len(idx[0]), len(idx[2]), 1), 16)
    out = jnp.asarray(pool)
    fn = _main_fn(c.rank)
    for plan in st.repack_chunks(*idx, size, 1, null_rows=null_rows):
        out = fn(out, jnp.asarray(staging), *(jnp.asarray(v, jnp.int32) for v in plan))
    return np.asarray(out)


def _main_load(num_tokens, skip, layers, rng, page_base=200, pages=200):
    """Staging with every layer's producer main-KV pages (random rows), the
    staged suffix's page lists and one destination table per layer (a paired
    member's own table is the unpacked reference's virtual pages)."""
    n_src = -(-num_tokens // _SPLIT_SRC)
    first = skip // _SPLIT_SRC
    staging = carrier(1 + len(layers) * n_src, _SPLIT_SRC)
    staging[1:] = rng.integers(0, 256, staging[1:].shape, dtype=np.uint8)
    staged = {name: list(range(1 + i * n_src, 1 + (i + 1) * n_src))[first:]
              for i, name in enumerate(layers)}
    entries = -(-num_tokens // (_SPLIT_D * _SPLIT_B))
    ids = [int(x) for x in rng.permutation(np.arange(page_base, page_base + pages))]
    tables = {name: ids[i * entries : (i + 1) * entries] for i, name in enumerate(layers)}
    return staging, staged, first, tables


def _merge_slot(old, new, slot):
    """``pooled_qsa._write_main_cache_paired`` on rows ``[n, 4, 256]``: ``new``
    holds the layer's pair at slots 0-1 (``pack_rows``), moved to ``slot``."""
    pq = pytest.importorskip("vllm_torchtpu.kernels.qwen4_exp.pooled_qsa")
    moved = np.asarray(pq.main_slot_rows(jnp.asarray(new), slot))

    def words(rows):  # [n, 256] uint32: sublane 0 is the low byte
        return np.ascontiguousarray(rows.transpose(0, 2, 1)).view(np.uint32)[..., 0]

    merged = (words(old) & np.uint32(pq._main_slot_keep_mask(slot))) | words(moved)
    return merged[..., None].view(np.uint8).transpose(0, 2, 1)


def _reference_packed_main(num_tokens, skip, tp_rank, layers, pairs, staging, staged, first,
                           tables, stale):
    plain = _pack_consumer(tp_rank, layers, {})
    idx = plain.indices(staged, tables, num_tokens, skip, [first] * len(layers))
    assert idx[0].ndim == 1
    unpacked = flat_rows(_land(plain, np.zeros_like(stale), staging, idx))
    ref = flat_rows(stale.copy())
    b, d = _SPLIT_B, _SPLIT_D
    t = np.arange(skip, num_tokens)
    t = t[(t // b) % d == tp_rank % d]

    def rows(name):
        return np.asarray(tables[name], dtype=np.int64)[t // (d * b)] * b + t % b

    paired = {m for members in pairs.values() for m in members}
    for owner, members in pairs.items():
        for slot, member in enumerate(members):
            ref[rows(owner)] = _merge_slot(ref[rows(owner)], unpacked[rows(member)], slot)
    for name in layers:
        if name not in paired:  # whole rows (K, V, 0, 0), as today
            ref[rows(name)] = unpacked[rows(name)]
    return ref


_PACK_CASES = [
    pytest.param(1500, 0, 5, None, id="no_prefix_odd"),
    pytest.param(2600, 1024, 5, None, id="local_prefix_odd"),
    pytest.param(2600, 1024, 4, 96, id="local_prefix_even_chunked"),
    pytest.param(300, 0, 3, None, id="shorter_than_an_entry_odd"),
    pytest.param(100, 0, 2, 64, id="shorter_than_a_block_chunked"),
]


@pytest.mark.parametrize("spread", [False, True], ids=["last_row_pad", "spread_pad"])
@pytest.mark.parametrize("num_tokens,skip,n_layers,chunk", _PACK_CASES)
def test_packed_main_landing_is_the_slot_merge_of_the_unpacked_landing(
    num_tokens, skip, n_layers, chunk, spread
):
    layers = _PACK_LAYERS[:n_layers]
    pairs = _pack_pairs(layers)
    assert len(pairs) == n_layers // 2
    rng = np.random.default_rng(num_tokens + n_layers)
    staging, staged, first, tables = _main_load(num_tokens, skip, layers, rng)
    stale = rng.integers(0, 256, (400, _SPLIT_B, 1, ROWS, LANES), dtype=np.uint8)
    written_any = False
    for r in range(_SPLIT_TP):
        c = _pack_consumer(r, layers, pairs)
        idx = c.indices(staged, tables, num_tokens, skip, [first] * len(layers))
        assert idx[0].ndim == 2 and idx[0].shape == (len(idx[1]), 2)
        # Spread padding (repack_chunks null_rows) pads both source columns
        # with distinct zero staging rows onto the pool's page 0.
        null_rows = (_SPLIT_SRC, _SPLIT_B) if spread else None
        got = flat_rows(_land(c, stale, staging, idx, chunk, null_rows))
        want = _reference_packed_main(num_tokens, skip, r, layers, pairs, staging, staged, first,
                                      tables, stale)
        # Page 0 is the null page: an empty plan rewrites its row 0.
        np.testing.assert_array_equal(got[_SPLIT_B:], want[_SPLIT_B:], err_msg=f"rank {r}")
        written = (got[_SPLIT_B:] != flat_rows(stale)[_SPLIT_B:]).any(axis=(1, 2))
        written_any |= bool(written.any())
        if num_tokens <= _SPLIT_B and r % _SPLIT_D:
            assert not written.any()  # a block of rank 0's only
        # The paired-away members' own pages are never written.
        for owner, (_, member) in pairs.items():
            pages = np.asarray(tables[member])
            np.testing.assert_array_equal(np.asarray(got).reshape(400, -1)[pages],
                                          stale.reshape(400, -1)[pages])
    assert written_any


def test_packed_main_pairs_differ_and_an_odd_layer_keeps_zero_slots():
    """Controls on one rank: the two head-slot pairs of an owner's row carry
    different layers (a repack writing member 0 twice would pass a check of
    member 0 alone), and the odd last layer's slots 2-3 are zero."""
    layers = _PACK_LAYERS[:3]
    pairs = _pack_pairs(layers)
    n, skip, r = 1500, 0, 5
    staging, staged, first, tables = _main_load(n, skip, layers, np.random.default_rng(3))
    c = _pack_consumer(r, layers, pairs)
    got = flat_rows(_land(c, carrier(400, _SPLIT_B), staging,
                          c.indices(staged, tables, n, skip, [first] * 3)))
    t = np.arange(n)
    t = t[(t // _SPLIT_B) % _SPLIT_D == r % _SPLIT_D]

    def rows(name):
        return np.asarray(tables[name])[t // (_SPLIT_D * _SPLIT_B)] * _SPLIT_B + t % _SPLIT_B

    owner = got[rows(layers[0])]
    assert (owner[:, :2] != owner[:, 2:]).any(axis=(1, 2)).all()
    h = c.rank.kv_head

    def src(name):
        pages = np.asarray(staged[name])
        return flat_rows(staging)[pages[t // _SPLIT_SRC] * _SPLIT_SRC + t % _SPLIT_SRC]

    np.testing.assert_array_equal(owner[:, :2], src(layers[0])[:, 2 * h : 2 * h + 2])
    np.testing.assert_array_equal(owner[:, 2:], src(layers[1])[:, 2 * h : 2 * h + 2])
    odd = got[rows(layers[2])]
    np.testing.assert_array_equal(odd[:, :2], src(layers[2])[:, 2 * h : 2 * h + 2])
    assert not odd[:, 2:].any()


@pytest.mark.parametrize(
    "num_tokens,skip",
    [pytest.param(1500, 0, id="no_prefix"), pytest.param(2600, 1024, id="local_prefix"),
     pytest.param(300, 0, id="shorter_than_an_entry")],
)
def test_packed_main_with_split_compressed_keys(num_tokens, skip):
    """F1 (VLLM_TPU_QSA_MTP_KV_DCP + VLLM_TPU_QSA_COMPRESSED_DCP) with packing:
    one repack lands paired main KV (an odd layer included) and split
    compressed keys, in chunks of mixed kinds; each part equals its slow
    reference and nothing else changes."""
    layers = _PACK_LAYERS[:3]
    pairs = _pack_pairs(layers)
    rng = np.random.default_rng(num_tokens * 3 + skip)
    m_stage, m_staged, first, tables = _main_load(num_tokens, skip, layers, rng)
    c_stage, c_staged, _, split_table, virtual, _ = _split_load(num_tokens, skip, num_tokens)
    # One carrier: main pages first, then the compressed pages shifted after them.
    shift = len(m_stage) - 1
    staging = np.concatenate([m_stage, c_stage[1:]])
    c_staged_s = {n: [p + shift for p in pages] for n, pages in c_staged.items()}
    stale = rng.integers(0, 256, (400, _SPLIT_B, 1, ROWS, LANES), dtype=np.uint8)
    for r in range(_SPLIT_TP):
        c = _pack_consumer(r, layers, pairs, compressed=_MEMBERS)
        src_pages = {**m_staged, **c_staged_s}
        dst = {**tables, "comp": split_table}
        idx = c.indices(src_pages, dst, num_tokens, skip, [first] * (len(layers) + len(_MEMBERS)))
        assert idx[0].ndim == 2
        got = flat_rows(_land(c, stale, staging, idx, 96, (_SPLIT_SRC, _SPLIT_B)))
        want = _reference_packed_main(num_tokens, skip, r, layers, pairs, m_stage, m_staged, first,
                                      tables, stale)
        comp = _reference_split_pool(num_tokens, skip, r, c_stage, c_staged, [first] * 2,
                                     split_table, virtual, stale)
        # Compressed pages (ids < 200) from the split reference, main pages
        # (ids >= 200) from the packed reference.
        want[: 200 * _SPLIT_B] = comp[: 200 * _SPLIT_B]
        np.testing.assert_array_equal(got[_SPLIT_B:], want[_SPLIT_B:], err_msg=f"rank {r}")


@pytest.mark.parametrize("num_tokens,skip", [(2600, 0), (2600, 1024), (100, 0)])
def test_paired_main_sources_and_tile_alignment(num_tokens, skip):
    """``main_sources`` lists member 0's and member 1's staging rows for the
    owner's destination rows, in the same order; an unpaired owner pads
    column 1 with the null row; ``align_rows`` pads both columns alike."""
    layers = _PACK_LAYERS[:3]
    pairs = _pack_pairs(layers)
    staging, staged, first, tables = _main_load(num_tokens, skip, layers, np.random.default_rng(9))
    for r in (0, 3, 6):
        c = _pack_consumer(r, layers, pairs)
        ms, md, *_ = c.indices(staged, tables, num_tokens, skip, [first] * 3)
        rank = c.rank
        segs = []
        for name in (layers[0], layers[2]):
            s0, d = st.main_rows_np(num_tokens, staged[name], _SPLIT_SRC, tables[name], _SPLIT_B,
                                    rank, start=skip, src_first=first)
            if name == layers[0]:
                s1, _ = st.main_rows_np(num_tokens, staged[layers[1]], _SPLIT_SRC, tables[name],
                                        _SPLIT_B, rank, start=skip, src_first=first)
            else:
                s1 = np.zeros_like(s0)
            if len(d):
                segs.append((st.align_rows(np.stack([s0, s1], axis=1)), st.align_rows(d)))
        want_s = np.concatenate([s for s, _ in segs]) if segs else np.zeros((0, 2), np.int64)
        want_d = np.concatenate([d for _, d in segs]) if segs else np.zeros(0, np.int64)
        np.testing.assert_array_equal(ms, want_s)
        np.testing.assert_array_equal(md, want_d)
        assert ms.shape == (len(md), 2) and len(md) % st.REPACK_TILE == 0
        # Every program keeps the [rows, 2] shape, an empty rank's included.
        for plan in st.repack_chunks(ms, md, [], [], [], [], 64, 1):
            assert plan[0].shape == (64, 2)


def test_verify_main_pairs_reads_zero_mismatches_and_catches_a_wrong_slot():
    """QWEN4_EXP_PD_VERIFY_SPLIT_KEYS's packed main-KV check on the landing
    above, with a host reader in place of the device read-back: no mismatch
    after the packed landing, one after a corrupted byte, every row after a
    landing that swaps the two members' slots."""
    layers = _PACK_LAYERS[:5]
    pairs = _pack_pairs(layers)
    n, skip, r = 2600, 1024, 6
    staging, staged, first, tables = _main_load(n, skip, layers, np.random.default_rng(11))
    c = _pack_consumer(r, layers, pairs)
    c.route, c._src_split = None, frozenset()
    uuid = 9
    c.reservations, c.skips = {uuid: staged}, {uuid: (skip, [first] * 5)}
    pool = flat_rows(_land(c, carrier(400, _SPLIT_B), staging,
                           c.indices(staged, tables, n, skip, [first] * 5)))
    stage = flat_rows(staging)

    def reader(p):
        return lambda s, d: (stage[s], p[d])

    checked = c.main_pair_check_rows(uuid, tables, n)
    owner = list(pairs)[0]
    assert [m for m, *_ in checked] == list(pairs[owner])
    out = c.verify_main_pairs(uuid, tables, n, read_rows=reader(pool))
    assert out["owner"] == owner
    # Each completed check moves to the next owner.
    assert [m for m, *_ in c.main_pair_check_rows(uuid, tables, n)] == list(pairs[list(pairs)[1]])
    assert out["mismatched"] == 0 and out["rows"] == sum(len(x[2]) for x in checked) > 0
    assert out["pool_digest"] == out["staging_digest"] and out["partner_equal"] == 0
    assert out["nonzero"] > 0.9 * out["rows"]
    c._verify_main_total["checks"] = 0  # back to the first owner
    bad = pool.copy()
    bad[checked[1][3][5], 3, 17] ^= 1  # member 1's V byte of one row
    out = c.verify_main_pairs(uuid, tables, n, read_rows=reader(bad))
    assert out["mismatched"] == 1 and out["by_member"] == {pairs[owner][0]: 0, pairs[owner][1]: 1}
    c._verify_main_total["checks"] = 0
    swapped = pool.copy()
    rows = checked[0][3]
    swapped[rows] = np.concatenate([pool[rows][:, 2:], pool[rows][:, :2]], axis=1)
    out = c.verify_main_pairs(uuid, tables, n, read_rows=reader(swapped))
    assert out["mismatched"] == out["rows"] and out["partner_equal"] == out["rows"]


def test_verify_main_pairs_reports_a_rank_without_rows():
    """A DCP rank that stores no token of a load's suffix (here, a load
    shorter than one block, on rank 1) logs an empty check instead of
    raising."""
    layers = _PACK_LAYERS[:2]
    pairs = _pack_pairs(layers)
    n, skip, r = 100, 0, 1
    staging, staged, first, tables = _main_load(n, skip, layers, np.random.default_rng(12))
    c = _pack_consumer(r, layers, pairs)
    c.route, c._src_split = None, frozenset()
    c.reservations, c.skips = {4: staged}, {4: (skip, [first] * 2)}
    assert all(len(x[2]) == 0 for x in c.main_pair_check_rows(4, tables, n))

    def never(s, d):
        raise AssertionError("no rows to read")

    out = c.verify_main_pairs(4, tables, n, read_rows=never)
    assert out["rows"] == 0 and out["mismatched"] == 0 and out["nonzero"] == 0


def test_verify_split_keys_of_a_rank_without_pairs_reports_zero():
    """A load whose suffix holds no block of this rank: nothing to read back."""
    n, skip, r = 100, 0, 3
    staging, staged, first, split_table, virtual, stale = _split_load(n, skip, 11)
    c = _split_consumer(r)
    c.route, c.src_merged = None, False
    c.reservations, c.skips = {9: staged}, {9: (skip, first)}

    def reader(s, d):
        raise AssertionError("no rows to read")

    out = c.verify_split_keys(9, [split_table], n, read_rows=reader)
    assert out["pairs"] == out["mismatched"] == out["nonzero"] == 0

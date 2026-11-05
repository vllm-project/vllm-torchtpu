# SPDX-License-Identifier: Apache-2.0
"""Decode-side repack of TP1 pipeline-producer pages (qwen4_exp pd_staging).

Producer pages are built from logical values with an independent writer that
follows the pool layout (``pack_rows`` main-KV rows, compressed-key pairs, and
GDN state in Mosaic typed-view order: an F32 is the 4 sublanes at one lane, low
byte first); each decode rank's repacked pages must then hold exactly its
slice in its own geometry.
"""

import importlib.util
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

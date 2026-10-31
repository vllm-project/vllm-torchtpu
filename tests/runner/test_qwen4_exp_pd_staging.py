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
    return np.zeros((pages * block // 128, 128, 1, ROWS, LANES), np.uint8)


def flat_rows(c: np.ndarray) -> np.ndarray:
    return c.reshape(-1, ROWS, LANES)


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


@pytest.mark.parametrize("tp,dcp", [(8, 2), (8, 1), (2, 1)])
def test_repack_rank_slices(tp, dcp):
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
        size = st.bucket(len(ms), 16)
        fn = st.make_repack(
            src_block=src_block, dst_block=dst_block, src_gdn=s, dst_gdn=d,
            kv_head=rank.kv_head, v_head_start=r * d.v_heads,
            channels=st.conv_channels(s, d, rank),
        )
        out = fn(
            jnp.asarray(pool), jnp.asarray(staging),
            jnp.asarray(st._pad(ms, size), jnp.int32), jnp.asarray(st._pad(md, size), jnp.int32),
            jnp.asarray(st._pad(copy_s, st.bucket(len(copy_s), 16)), jnp.int32),
            jnp.asarray(st._pad(copy_d, st.bucket(len(copy_d), 16)), jnp.int32),
            jnp.asarray([src_gdn_page], jnp.int32), jnp.asarray([dst_gdn], jnp.int32),
        )
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

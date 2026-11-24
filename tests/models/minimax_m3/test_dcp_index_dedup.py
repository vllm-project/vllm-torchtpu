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
"""MSA index cache stored once per TP group under DCP
(TPU_MINIMAX_M3_DCP_INDEX_DEDUP), interpret mode.

Without it, each of the ``lanes = tp / dcp`` DCP groups keeps the shared
index-key head of its blocks: ``lanes`` copies. With it, lane ``l`` of DCP
rank ``r`` keeps sub-block ``l`` (``block_size / lanes`` tokens) of each of
rank ``r``'s blocks, scores every lane's index heads on it, and the lanes'
per-head candidates merge (`msa_dcp.merge_lane_candidates`) into exactly
the candidates of the unsplit cache. Checked bit for bit against the
current path: K/V and scale caches, the index pages (as halves / quarters
of the unsplit ones) and the candidates, for DCP 4 (2 lanes, 2 KV heads per
group) and DCP 2 (4 lanes, one KV head), decode- and prefill-sized steps.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

PAGE, HEAD_DIM, MAX_SEQS, PAGES = 128, 128, 4, 24
TOPK, CONFIG = 3, dict(init_blocks=0, local_blocks=1)
STEPS = {
    # decode rows (1 + spec) over long contexts: the decode scoring path
    "decode": ([4, 4, 4, 1], [1100, 700, 260, 40]),
    # a chunk, decode and spec rows, a short request: the class path
    # (> 4 x 64 rows for one index head per lane too, DCP 2)
    "prefill": ([1, 300, 3, 1], [1100, 200, 0, 40]),
}


def _setup(seed, dcp, q_lens, ctx_lens):
    rng = np.random.default_rng(seed)
    logical = -(-max(np.add(q_lens, ctx_lens)) // (PAGE * dcp))
    num_seqs, num_real = len(q_lens), sum(q_lens)
    num_tokens = num_real + 3
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.add(ctx_lens, q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = np.zeros((MAX_SEQS, logical), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, PAGES))[
        : num_seqs * logical
    ].reshape(num_seqs, logical)
    meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
    return rng, num_tokens, meta, len(q_lens)


def _split(index_page, lanes, lane, rng):
    """Lane `lane`'s share of folded 4-bit index pages: slots
    ``[lane * S / lanes, (lane + 1) * S / lanes)`` with their scales in the
    first lanes of the scale row, the rest random (unused)."""
    nib = msa_fp4.unpack_rows(index_page)
    scales = np.asarray(msa_fp4.folded_scales(index_page))
    sub = PAGE // lanes
    part = slice(lane * sub, (lane + 1) * sub)
    pad = rng.uniform(7, 9, (scales.shape[0], PAGE - sub)).astype(np.float32)
    return msa_fp4.pack_rows(
        nib[:, part], jnp.asarray(np.concatenate([scales[:, part], pad], axis=1))
    )


def _lanes_setup(seed, dcp, lanes, step):
    """Per-lane layers of one TP group: lane ``l`` (one DCP group) has its
    own K/V heads and caches; every lane holds the same index keys."""
    q_lens, ctx_lens = STEPS[step]
    g = max(1, dcp // 2)  # TP 8, 4 KV heads: 2 ranks per KV head
    rng, num_tokens, meta, num_seqs = _setup(seed, dcp, q_lens, ctx_lens)

    def bf16(*shape):
        return jnp.asarray(rng.standard_normal(shape)).astype(jnp.bfloat16)

    def nib(width):
        return jnp.asarray(rng.integers(0, 16, (PAGES, PAGE, width)))

    index = [
        msa_fp4.pack_rows(
            nib(HEAD_DIM),
            jnp.asarray(rng.uniform(0.05, 0.2, (PAGES, PAGE)), jnp.float32),
        )
        for _ in range(dcp)
    ]
    ik = bf16(num_tokens, HEAD_DIM)
    iq = bf16(num_tokens, lanes * g, HEAD_DIM)
    layers = []
    for lane in range(lanes):
        layers.append(
            {
                "kv": [msa_fp4.pack_rows(nib(2 * g * HEAD_DIM)) for _ in range(dcp)],
                "scale": [
                    jnp.asarray(
                        rng.uniform(0.05, 0.2, (PAGES, msa_fp4.scale_rows(g), PAGE)),
                        jnp.float32,
                    )
                    for _ in range(dcp)
                ],
                "index": index,
                "lane_index": [_split(p, lanes, lane, rng) for p in index],
                "k": bf16(num_tokens, g, HEAD_DIM),
                "v": bf16(num_tokens, g, HEAD_DIM),
                "iq": iq[:, lane * g : (lane + 1) * g],
            }
        )
    return layers, iq, ik, meta, num_seqs, g


def _run(layer, iq, ik, meta, num_seqs, dcp, rank, index, **kw):
    distribution = jnp.asarray([0, 0, num_seqs], jnp.int32)
    return msa_dcp.msa_write_and_candidates_dcp(
        layer["k"],
        layer["v"],
        iq,
        ik,
        layer["kv"][rank],
        index,
        *meta,
        distribution,
        topk=TOPK,
        rank=rank,
        dcp=dcp,
        scale_cache=layer["scale"][rank],
        interpret=True,
        **CONFIG,
        **kw,
    )


@pytest.mark.parametrize("step", sorted(STEPS))
@pytest.mark.parametrize("dcp,lanes", [(4, 2), (2, 4)], ids=["dcp4", "dcp2"])
def test_lane_split_matches_full_index(dcp, lanes, step):
    layers, iq, ik, meta, num_seqs, g = _lanes_setup(50 + dcp, dcp, lanes, step)
    width = g * 2 * TOPK
    base, split = {}, {}
    for lane, layer in enumerate(layers):
        for rank in range(dcp):
            base[lane, rank] = _run(
                layer, layer["iq"], ik, meta, num_seqs, dcp, rank, layer["index"][rank]
            )
            split[lane, rank] = _run(
                layer,
                iq,
                ik,
                meta,
                num_seqs,
                dcp,
                rank,
                layer["lane_index"][rank],
                lanes=lanes,
                lane=lane,
            )
    rng = np.random.default_rng(0)
    for lane in range(lanes):
        for rank in range(dcp):
            b_kv, b_index, b_scale, b_cands = base[lane, rank]
            s_kv, s_index, s_scale, s_cands = split[lane, rank]
            assert s_cands.shape == (b_cands.shape[0], lanes * width)
            np.testing.assert_array_equal(np.asarray(s_kv), np.asarray(b_kv))
            np.testing.assert_array_equal(np.asarray(s_scale), np.asarray(b_scale))
            # The lane's index pages are its share of the unsplit pages.
            want = _split(b_index, lanes, lane, rng)
            np.testing.assert_array_equal(
                msa_fp4.unpack_rows(s_index), msa_fp4.unpack_rows(want)
            )
            sub = PAGE // lanes
            np.testing.assert_array_equal(
                np.asarray(msa_fp4.folded_scales(s_index))[:, :sub],
                np.asarray(msa_fp4.folded_scales(want))[:, :sub],
            )
            # Every lane's candidates of this lane's heads -> its candidates.
            parts = jnp.stack(
                [
                    split[src, rank][3][:, lane * width : (lane + 1) * width]
                    for src in range(lanes)
                ]
            )
            got = msa_dcp.merge_lane_candidates(parts, TOPK)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(b_cands))
    # Not vacuous: the lanes' partial candidates differ from the merged ones.
    lane, rank = 0, 0
    first = split[lane, rank][3][:, :width]
    assert not np.array_equal(np.asarray(first), np.asarray(base[lane, rank][3]))


def test_merge_lane_candidates_dedups_and_orders():
    """Hand case, topk 3, one head, two lanes: block 4 listed by both lanes
    keeps its larger score; ties go to the lower block; a short list pads
    with (-inf, -1)."""
    inf = float("-inf")

    def packed(scores, blocks):
        return np.asarray([scores + blocks], np.float32)

    lane0 = packed([9.0, 5.0, 2.0], [4.0, 8.0, 0.0])
    lane1 = packed([7.0, 5.0, inf], [12.0, 4.0, -1.0])
    got = msa_dcp.merge_lane_candidates(jnp.asarray(np.stack([lane0, lane1])), 3)
    np.testing.assert_array_equal(np.asarray(got), [[9.0, 7.0, 5.0, 4.0, 12.0, 8.0]])
    lane1 = packed([inf, inf, inf], [-1.0, -1.0, -1.0])
    lane0 = packed([3.0, inf, inf], [4.0, -1.0, -1.0])
    got = msa_dcp.merge_lane_candidates(jnp.asarray(np.stack([lane0, lane1])), 3)
    np.testing.assert_array_equal(np.asarray(got), [[3.0, inf, inf, 4.0, -1.0, -1.0]])


def test_index_lane_layout():
    """Sub-block s = b * lanes + l of block b sits on sub-rank s % (dcp *
    lanes) = r * lanes + l, in the page of b's logical block."""
    dcp, lanes, block = 4, 2, 128
    for rank in range(dcp):
        for lane in range(lanes):
            sub, sub_rank, sub_dcp = msa_dcp.index_lane_layout(
                block, rank, dcp, lanes, lane
            )
            assert (sub, sub_dcp) == (64, 8) and int(sub_rank) == rank * lanes + lane
            for b in range(rank, 64, dcp):
                s = b * lanes + lane
                assert s % sub_dcp == int(sub_rank)
                assert s // sub_dcp == b // dcp
    with pytest.raises(ValueError, match="multiple"):
        msa_dcp.index_lane_layout(100, 0, 4, 3, 0)

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
"""Layout contract of the token-parallel DCP prefill split
(`kernels/mla/dcp_token_parallel.py`), on a host-device mesh.

The caches are written DCP-sharded exactly as `_get_dcp_page_and_slot` lays
them out, the kernels are jnp references, and every result is compared with
the same reference run on the unsharded, natural-order KV of each request:
the gather, the flat-row remap, the per-rank causal bounds, the index-K
transpose and both all-to-alls must make the two identical.

Needs `dcp` devices; CPU virtual devices are enough
(XLA_FLAGS=--xla_force_host_platform_device_count=16).
"""

import dataclasses

import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp

AXIS = "dcp"
DN, DR = 8, 4  # nope / rope width of the reference latent
ROPE_PER_ROW = 4  # tokens per rope cache row, as in the sparse MLA cache
IDX_ROWS, IDX_SUB = 3, 4  # index-K cache [pages, rows, sub, page_size]
TOPK = 16


@dataclasses.dataclass
class Req:
    prefix: int
    chunk: int


CASES = {
    # 3 decode rows, a chunk straddling the split and a rank boundary, then a
    # new request whose rows share the last busy rank with the first's tail.
    4: dict(
        layout=tp.TokenParallelLayout(
            4, 4, 32, split_rows=16, max_pages=8, gather_caps=(2, 4, 6, 8)
        ),
        num_tokens=256,
        reqs=[Req(700, 1), Req(90, 1), Req(260, 1), Req(300, 150), Req(0, 40)],
    ),
    16: dict(
        layout=tp.TokenParallelLayout(16, 4, 64, split_rows=32, max_pages=8),
        num_tokens=512,
        reqs=[Req(3000, 1), Req(5000, 1), Req(2100, 300), Req(0, 150)],
    ),
}


def _mesh(dcp: int) -> Mesh:
    if jax.device_count() < dcp:
        pytest.skip(f"requires {dcp} devices; CPU virtual devices are sufficient")
    return Mesh(np.array(jax.devices()[:dcp]), (AXIS,))


class Step:
    """One step's metadata, natural per-request KV, and DCP-sharded caches
    holding every position before the step (the chunk is written by the op)."""

    def __init__(self, layout, num_tokens, reqs, max_reqs=8, seed=0):
        rng = np.random.default_rng(seed)
        self.layout, self.num_tokens, self.reqs = layout, num_tokens, reqs
        vps, ps, dcp = layout.virtual_page_size, layout.page_size, layout.dcp_size
        self.num_seqs = len(reqs)
        self.seq_lens = np.zeros(max_reqs, np.int32)
        qsl = [0]
        for i, r in enumerate(reqs):
            self.seq_lens[i] = r.prefix + r.chunk
            qsl.append(qsl[-1] + r.chunk)
        assert qsl[-1] <= num_tokens
        self.total = qsl[-1]
        self.qsl = np.full(max_reqs + 1, qsl[-1], np.int32)
        self.qsl[: len(qsl)] = qsl
        self.dist = np.array([0, 0, self.num_seqs], np.int32)
        vpages_per_seq = -(-int(self.seq_lens.max()) // vps) + 1
        num_pages = 4 * max_reqs * vpages_per_seq
        self.num_pages = num_pages
        free = rng.permutation(num_pages)
        self.block_tables = np.zeros((max_reqs, vpages_per_seq), np.int32)
        n = 0
        for i, s in enumerate(self.seq_lens[: self.num_seqs]):
            k = -(-int(s) // vps)
            # Virtual ordinals past num_pages must resolve with `% num_pages`.
            self.block_tables[i, :k] = free[n : n + k] + num_pages * (i % 2)
            n += k
        # Natural KV of each request, chunk included.
        self.nope = [rng.standard_normal((s, DN), np.float32) for s in self.seq_lens]
        self.rope = [rng.standard_normal((s, DR), np.float32) for s in self.seq_lens]
        self.idx = [
            rng.integers(0, 256, (s, IDX_ROWS * IDX_SUB), dtype=np.uint8)
            for s in self.seq_lens
        ]
        nope_c = np.zeros((dcp, num_pages, ps, DN), np.float32)
        rope_c = np.zeros(
            (dcp, num_pages, ps // ROPE_PER_ROW, ROPE_PER_ROW, DR), np.float32
        )
        idx_c = np.zeros((dcp, num_pages, IDX_ROWS, IDX_SUB, ps), np.uint8)
        for i, r in enumerate(reqs):
            for p in range(r.prefix + r.chunk):
                owner, slot = (int(v) for v in tp.owner_and_slot(layout, np.int32(p)))
                page = self.block_tables[i, p // vps] % num_pages
                # The index-K cache is written in full here (the indexer op's K
                # write is the existing DCP writer, not under test); the latent
                # chunk is left for `write_kv`.
                idx_c[owner, page, :, :, slot] = self.idx[i][p].reshape(
                    IDX_ROWS, IDX_SUB
                )
                if p < r.prefix:
                    nope_c[owner, page, slot] = self.nope[i][p]
                    rope_c[owner, page, slot // ROPE_PER_ROW, slot % ROPE_PER_ROW] = (
                        self.rope[i][p]
                    )
        self.nope_cache = nope_c.reshape(dcp * num_pages, ps, DN)
        self.rope_cache = rope_c.reshape(dcp * num_pages, ps // ROPE_PER_ROW, -1)
        self.idx_cache = idx_c.reshape(dcp * num_pages, IDX_ROWS, IDX_SUB, ps)
        rows = np.arange(num_tokens)
        seq_id = np.minimum(
            np.searchsorted(self.qsl[1:], rows, side="right"), max_reqs - 1
        )
        self.row_req = seq_id
        self.row_pos = (
            self.seq_lens[seq_id]
            - (self.qsl[seq_id + 1] - self.qsl[seq_id])
            + rows
            - self.qsl[seq_id]
        )
        self.kv_c = np.zeros((num_tokens, DN), np.float32)
        self.k_pe = np.zeros((num_tokens, DR), np.float32)
        for t in range(self.total):
            self.kv_c[t] = self.nope[seq_id[t]][self.row_pos[t]]
            self.k_pe[t] = self.rope[seq_id[t]][self.row_pos[t]]

    def args(self):
        return [
            jnp.asarray(a)
            for a in (self.seq_lens, self.block_tables.reshape(-1), self.qsl, self.dist)
        ]

    def tp_rows(self):
        """(rank, local row) -> step row, for the real token-parallel rows."""
        lay = self.layout
        per = lay.rows_per_rank(self.num_tokens)
        for t in range(lay.split_rows, self.total):
            yield divmod(t - lay.split_rows, per), t


def _write_kv_ref(layout, num_pages):
    """The owner-masked DCP write of every row of the step (reference)."""

    def write(nope, rope, kv_c, k_pe, seq_lens, bt, qsl, rank):
        t = jnp.arange(kv_c.shape[0])
        seq = jnp.minimum(
            jnp.searchsorted(qsl[1:], t, side="right"), seq_lens.shape[0] - 1
        )
        pos = seq_lens[seq] - (qsl[seq + 1] - qsl[seq]) + t - qsl[seq]
        owner, slot = tp.owner_and_slot(layout, pos)
        table = bt.reshape(seq_lens.shape[0], -1)
        page = table[
            seq, jnp.clip(pos // layout.virtual_page_size, 0, table.shape[1] - 1)
        ]
        ok = (t < qsl[-1]) & (owner == rank)
        page = jnp.where(ok, page % num_pages, 2**30)
        nope = nope.at[page, slot].set(kv_c, mode="drop")
        r4 = rope.reshape(*rope.shape[:2], ROPE_PER_ROW, DR)
        r4 = r4.at[page, slot // ROPE_PER_ROW, slot % ROPE_PER_ROW].set(
            k_pe, mode="drop"
        )
        return nope, r4.reshape(rope.shape)

    return write


def _attend_rows(q, keys_nope, keys_rope, valid):
    """Exact softmax over each row's keys; value = the nope part (as MLA)."""
    k = jnp.concatenate([keys_nope, keys_rope], axis=-1)  # [R, K, D]
    s = jnp.einsum("rhd,rkd->rhk", q, k)
    s = jnp.where(valid[:, None, :], s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    p = jnp.where(valid[:, None, :], p, 0.0)
    return jnp.einsum("rhk,rkd->rhd", p, keys_nope)


def _attend_ref(q, g_nope, g_rope, topk, num_rows):
    valid = topk >= 0
    idx = jnp.where(valid, topk, 0)
    kn = g_nope.reshape(-1, DN)[idx]
    kr = g_rope.reshape(-1, ROPE_PER_ROW, DR)[idx // ROPE_PER_ROW, idx % ROPE_PER_ROW]
    out = _attend_rows(q[..., : DN + DR], kn, kr, valid)
    real = jnp.arange(q.shape[0]) < num_rows
    return jnp.where(real[:, None, None], out, 0.0)


def _random_topk(rng, pos):
    """TOPK distinct visible positions (<= pos), `-1` padded."""
    n = min(TOPK, pos + 1)
    sel = rng.choice(pos + 1, n, replace=False)
    return np.concatenate([sel, -np.ones(TOPK - n, np.int64)]).astype(np.int32)


@pytest.mark.parametrize("q_own_rows", [False, True])
@pytest.mark.parametrize("dcp", sorted(CASES))
def test_sparse_mla_attention_matches_natural_reference(dcp, q_own_rows):
    mesh = _mesh(dcp)
    case = CASES[dcp]
    lay, num_tokens = case["layout"], case["num_tokens"]
    st = Step(lay, num_tokens, case["reqs"])
    rng = np.random.default_rng(1)
    heads = 2 * dcp  # 2 per rank
    per = lay.rows_per_rank(num_tokens)
    q_full = rng.standard_normal((num_tokens, heads, DN + DR)).astype(np.float32)

    pages, offsets = (
        np.asarray(a)
        for a in tp.step_pages(
            lay, *[jnp.asarray(x) for x in (st.seq_lens, st.qsl)], st.num_seqs
        )
    )
    assert pages.sum() <= lay.max_pages
    topk_pos = np.full((dcp, per, TOPK), -1, np.int32)
    topk_flat = np.full((dcp, per, TOPK), -1, np.int32)
    for (rank, i), t in st.tp_rows():
        topk_pos[rank, i] = _random_topk(rng, int(st.row_pos[t]))
        topk_flat[rank, i] = np.asarray(
            tp.flat_rows(
                lay,
                jnp.asarray(topk_pos[rank, i][None]),
                jnp.asarray(offsets[st.row_req[t]][None]),
            )
        )[0]

    # Each rank's head slice of every row, stacked rank-major.
    q_sharded = q_full.reshape(num_tokens, dcp, 2, -1).transpose(1, 0, 2, 3)
    # Every head of each rank's rows S + j * R + [0, R) (zero past T).
    q_rows = np.concatenate([q_full, np.zeros_like(q_full[: lay.split_rows])])
    q_own = q_rows[lay.split_rows :].reshape(dcp * per, heads, -1)
    q_in = q_own if q_own_rows else q_sharded.reshape(dcp * num_tokens, 2, -1)
    nope, rope, out = jax.jit(
        lambda *a: tp.sparse_mla_attention(
            lay,
            mesh,
            AXIS,
            *a,
            write_kv=_write_kv_ref(lay, st.num_pages),
            attend=_attend_ref,
            q_own_rows=q_own_rows,
        )
    )(
        jnp.asarray(st.nope_cache),
        jnp.asarray(st.rope_cache),
        jnp.asarray(q_in),
        jnp.asarray(st.kv_c),
        jnp.asarray(st.k_pe),
        jnp.asarray(topk_flat.reshape(dcp * per, TOPK)),
        *st.args(),
    )
    out = np.asarray(out).reshape(dcp, num_tokens, 2, DN)
    # Back to [rows, heads]: step row S + i is out[:, i].
    out = out.transpose(1, 0, 2, 3).reshape(num_tokens, heads, DN)

    checked = 0
    for (rank, i), t in st.tp_rows():
        r, sel = st.row_req[t], topk_pos[rank, i]
        valid = sel >= 0
        idx = np.where(valid, sel, 0)
        want = _attend_rows(
            jnp.asarray(q_full[t][None]),
            jnp.asarray(st.nope[r][idx][None]),
            jnp.asarray(st.rope[r][idx][None]),
            jnp.asarray(valid[None]),
        )[0]
        np.testing.assert_allclose(out[t - lay.split_rows], want, rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked == st.total - lay.split_rows

    # The op wrote the chunk exactly where the DCP readers expect it.
    for i, req in enumerate(st.reqs):
        for p in range(req.prefix, req.prefix + req.chunk):
            owner, slot = (int(v) for v in tp.owner_and_slot(lay, np.int32(p)))
            page = st.block_tables[i, p // lay.virtual_page_size] % st.num_pages
            np.testing.assert_array_equal(
                np.asarray(nope)[owner * st.num_pages + page, slot], st.nope[i][p]
            )
            np.testing.assert_array_equal(
                np.asarray(rope).reshape(-1, ROPE_PER_ROW, DR)[
                    (owner * st.num_pages + page) * (lay.page_size // ROPE_PER_ROW)
                    + slot // ROPE_PER_ROW,
                    slot % ROPE_PER_ROW,
                ],
                st.rope[i][p],
            )


def _select_topk_ref(layout):
    """Exact top-k over each local row's visible positions, from the natural
    buffer through the page table, as `streamindex_topk(cp_size=1)` reads it."""
    ps = layout.page_size

    def select(q, weights, natural, rr, table):
        del weights
        rows = q.shape[0]
        num_reqs = rr.seq_lens.shape[0]
        tbl = table.reshape(num_reqs, -1)
        i = jnp.arange(rows)
        k = jnp.minimum(
            jnp.searchsorted(rr.cu_q_lens[1:], i, side="right"), num_reqs - 1
        )
        q_len = rr.cu_q_lens[k + 1] - rr.cu_q_lens[k]
        visible = rr.seq_lens[k] - q_len + (i - rr.cu_q_lens[k]) + 1
        visible = jnp.where(i < rr.num_rows, visible, 0)
        p = jnp.arange(tbl.shape[1] * ps)
        page = tbl[k][:, p // ps]  # [rows, P]
        keys = natural[page, :, :, (p % ps)[None, :]]  # [rows, P, A, B]
        keys = keys.reshape(rows, p.shape[0], -1).astype(jnp.float32)
        s = jnp.einsum("rd,rpd->rp", q.reshape(rows, -1), keys)
        s = jnp.where(p[None, :] < visible[:, None], s, -jnp.inf)
        _, top = jax.lax.top_k(s, TOPK)
        n = jnp.minimum(visible, TOPK)
        return jnp.where(jnp.arange(TOPK)[None, :] < n[:, None], top, -1)

    return select


@pytest.mark.parametrize("q_own_rows", [False, True])
@pytest.mark.parametrize("dcp", sorted(CASES))
def test_indexer_topk_matches_natural_reference(dcp, q_own_rows):
    mesh = _mesh(dcp)
    case = CASES[dcp]
    lay, num_tokens = case["layout"], case["num_tokens"]
    st = Step(lay, num_tokens, case["reqs"], seed=2)
    rng = np.random.default_rng(3)
    per = lay.rows_per_rank(num_tokens)
    q = rng.standard_normal((num_tokens, 1, IDX_ROWS * IDX_SUB)).astype(np.float32)
    weights = np.ones((num_tokens, 1), np.float32)
    if q_own_rows:
        q_in = np.concatenate(
            [
                q[np.minimum(lay.split_rows + r * per + np.arange(per), num_tokens - 1)]
                for r in range(dcp)
            ]
        )
        w_in = np.concatenate(
            [
                weights[
                    np.minimum(
                        lay.split_rows + r * per + np.arange(per), num_tokens - 1
                    )
                ]
                for r in range(dcp)
            ]
        )
    else:
        q_in, w_in = q, weights

    flat = jax.jit(
        lambda *a: tp.indexer_topk(
            lay,
            mesh,
            AXIS,
            *a,
            select_topk=_select_topk_ref(lay),
            q_own_rows=q_own_rows,
        )
    )(jnp.asarray(st.idx_cache), jnp.asarray(q_in), jnp.asarray(w_in), *st.args())
    flat = np.asarray(flat).reshape(dcp, per, TOPK)

    _, offsets = (
        np.asarray(a)
        for a in tp.step_pages(
            lay, *[jnp.asarray(x) for x in (st.seq_lens, st.qsl)], st.num_seqs
        )
    )
    seen = np.zeros((dcp, per), bool)
    for (rank, i), t in st.tp_rows():
        r, pos = st.row_req[t], int(st.row_pos[t])
        s = st.idx[r][: pos + 1].astype(np.float32) @ q[t, 0]
        want = np.argsort(-s, kind="stable")[:TOPK].astype(np.int32)
        want = np.concatenate([want, -np.ones(TOPK - want.size, np.int32)])
        want_flat = np.asarray(
            tp.flat_rows(lay, jnp.asarray(want[None]), jnp.asarray(offsets[r][None]))
        )[0]
        np.testing.assert_array_equal(flat[rank, i], want_flat, err_msg=f"row {t}")
        seen[rank, i] = True
    # Padding rows (past the step, or past T) are all -1.
    assert (flat[~seen] == -1).all()


@pytest.mark.parametrize("dcp", sorted(CASES))
def test_gather_page_ids_and_capacity(dcp):
    case = CASES[dcp]
    lay = case["layout"]
    st = Step(lay, case["num_tokens"], case["reqs"])
    seq_lens, qsl = jnp.asarray(st.seq_lens), jnp.asarray(st.qsl)
    pages, offsets = tp.step_pages(lay, seq_lens, qsl, st.num_seqs)
    pages, offsets = np.asarray(pages), np.asarray(offsets)
    vps = lay.virtual_page_size
    for i, s in enumerate(st.seq_lens):
        listed = i < st.num_seqs and st.qsl[i + 1] > lay.split_rows
        assert pages[i] == (-(-int(s) // vps) if listed else 0), i
    ids = np.asarray(
        tp.gather_page_ids(
            lay,
            jnp.asarray(st.block_tables.reshape(-1)),
            jnp.asarray(pages),
            jnp.asarray(offsets),
            st.num_pages,
        )
    )
    for i in range(st.num_seqs):
        for v in range(pages[i]):
            assert ids[offsets[i] + v] == st.block_tables[i, v] % st.num_pages
    assert (ids[pages.sum() :] == 0).all()
    assert int(tp.total_step_pages(lay, seq_lens, qsl, st.num_seqs)) == pages.sum()


def test_layout_rejects_unservable_buckets():
    lay = CASES[4]["layout"]
    with pytest.raises(ValueError):
        lay.rows_per_rank(lay.split_rows)
    with pytest.raises(ValueError):
        lay.rows_per_rank(258)
    with pytest.raises(ValueError):
        tp.TokenParallelLayout(16, 4, 32, split_rows=16, max_pages=4)
    with pytest.raises(ValueError):
        tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8, gather_caps=(4,))


def test_should_bound_natural_page_table_to_gather_cap_when_cap_below_max_pages():
    dcp = 4
    mesh = _mesh(dcp)
    lay = tp.TokenParallelLayout(
        dcp, 4, 32, split_rows=16, max_pages=8, gather_caps=(2, 4, 8)
    )
    st = Step(lay, 256, [Req(50, 150)], max_reqs=4, seed=11)
    seq_lens, qsl = jnp.asarray(st.seq_lens), jnp.asarray(st.qsl)
    _, offsets = tp.step_pages(lay, seq_lens, qsl, st.num_seqs)
    rr = tp.rank_rows(lay, jnp.int32(0), 256, seq_lens, qsl, st.num_seqs)

    capped_tbl = np.asarray(
        tp.natural_page_table(lay, rr, offsets, vpages_per_seq=8, cap=2)
    )
    uncapped_tbl = np.asarray(tp.natural_page_table(lay, rr, offsets, vpages_per_seq=8))
    assert capped_tbl.shape == (4 * 2 * dcp,)
    assert uncapped_tbl.shape == (4 * 8 * dcp,)
    assert int(capped_tbl.max()) <= 2 * dcp - 1

    vpages_per_seq = st.block_tables.shape[1]
    seen_table_widths: list[int] = []
    ref_select = _select_topk_ref(lay)

    def strict_select(q, weights, natural, rr_arg, table):
        num_reqs = rr_arg.seq_lens.shape[0]
        cap = natural.shape[0] // dcp
        seen_table_widths.append(table.shape[0] // num_reqs)
        assert table.shape == (num_reqs * min(vpages_per_seq, cap) * dcp,)
        in_bounds = (table >= 0) & (table < natural.shape[0])
        out = ref_select(q, weights, natural, rr_arg, table)
        return jnp.where(jnp.all(in_bounds), out, jnp.full_like(out, -999))

    rng = np.random.default_rng(13)
    q = rng.standard_normal((256, 1, IDX_ROWS * IDX_SUB)).astype(np.float32)
    weights = np.ones((256, 1), np.float32)
    flat = np.asarray(
        jax.jit(
            lambda *a: tp.indexer_topk(
                lay, mesh, AXIS, *a, select_topk=strict_select, q_own_rows=False
            )
        )(jnp.asarray(st.idx_cache), jnp.asarray(q), jnp.asarray(weights), *st.args())
    )
    assert seen_table_widths == [min(vpages_per_seq, c) * dcp for c in lay.gather_caps]
    assert (flat != -999).all()


@pytest.mark.parametrize("q_own_rows", [False, True])
def test_should_match_restrided_attention_when_indexer_emits_cap_strided_flat_topk(
    q_own_rows,
):
    dcp = 4
    mesh = _mesh(dcp)
    # cap=2 < max_pages=8 forces anti-cancellation divergence if max_pages
    # stride leaks into cap-strided flat_topk on any owner > 0.
    lay = tp.TokenParallelLayout(
        dcp, 4, 32, split_rows=16, max_pages=8, gather_caps=(2, 4, 8)
    )
    num_tokens = 256
    st = Step(lay, num_tokens, [Req(12, 130), Req(18, 95)], max_reqs=4, seed=23)
    rng = np.random.default_rng(29)
    per = lay.rows_per_rank(num_tokens)

    q_idx = rng.standard_normal((num_tokens, 1, IDX_ROWS * IDX_SUB)).astype(np.float32)
    w_idx = np.ones((num_tokens, 1), np.float32)
    if q_own_rows:
        q_idx_in = np.concatenate(
            [
                q_idx[
                    np.minimum(
                        lay.split_rows + r * per + np.arange(per), num_tokens - 1
                    )
                ]
                for r in range(dcp)
            ]
        )
        w_idx_in = np.concatenate(
            [
                w_idx[
                    np.minimum(
                        lay.split_rows + r * per + np.arange(per), num_tokens - 1
                    )
                ]
                for r in range(dcp)
            ]
        )
    else:
        q_idx_in, w_idx_in = q_idx, w_idx

    idx_args = (
        jnp.asarray(st.idx_cache),
        jnp.asarray(q_idx_in),
        jnp.asarray(w_idx_in),
        *st.args(),
    )
    flat_max = np.asarray(
        jax.jit(
            lambda *a: tp.indexer_topk(
                lay,
                mesh,
                AXIS,
                *a,
                select_topk=_select_topk_ref(lay),
                q_own_rows=q_own_rows,
                flat_topk_cap_strided=False,
            )
        )(*idx_args)
    )
    flat_cap = np.asarray(
        jax.jit(
            lambda *a: tp.indexer_topk(
                lay,
                mesh,
                AXIS,
                *a,
                select_topk=_select_topk_ref(lay),
                q_own_rows=q_own_rows,
                flat_topk_cap_strided=True,
            )
        )(*idx_args)
    )

    pages, _ = tp.step_pages(
        lay, jnp.asarray(st.seq_lens), jnp.asarray(st.qsl), st.num_seqs
    )
    active_cap = next(c for c in lay.gather_caps if int(pages.sum()) <= c)
    assert active_cap < lay.max_pages
    expected_cap = np.asarray(
        tp.restride_flat_rows(lay, jnp.asarray(flat_max), active_cap)
    )
    np.testing.assert_array_equal(flat_cap, expected_cap)
    assert np.any(flat_cap != flat_max)

    heads = 2 * dcp
    q_full = rng.standard_normal((num_tokens, heads, DN + DR)).astype(np.float32)
    q_sharded = q_full.reshape(num_tokens, dcp, 2, -1).transpose(1, 0, 2, 3)
    q_rows = np.concatenate([q_full, np.zeros_like(q_full[: lay.split_rows])])
    q_own = q_rows[lay.split_rows :].reshape(dcp * per, heads, -1)
    q_in = q_own if q_own_rows else q_sharded.reshape(dcp * num_tokens, 2, -1)

    _, _, out_max = jax.jit(
        lambda *a: tp.sparse_mla_attention(
            lay,
            mesh,
            AXIS,
            *a,
            write_kv=_write_kv_ref(lay, st.num_pages),
            attend=_attend_ref,
            q_own_rows=q_own_rows,
            flat_topk_cap_strided=False,
        )
    )(
        jnp.asarray(st.nope_cache),
        jnp.asarray(st.rope_cache),
        jnp.asarray(q_in),
        jnp.asarray(st.kv_c),
        jnp.asarray(st.k_pe),
        jnp.asarray(flat_max),
        *st.args(),
    )
    _, _, out_cap = jax.jit(
        lambda *a: tp.sparse_mla_attention(
            lay,
            mesh,
            AXIS,
            *a,
            write_kv=_write_kv_ref(lay, st.num_pages),
            attend=_attend_ref,
            q_own_rows=q_own_rows,
            flat_topk_cap_strided=True,
        )
    )(
        jnp.asarray(st.nope_cache),
        jnp.asarray(st.rope_cache),
        jnp.asarray(q_in),
        jnp.asarray(st.kv_c),
        jnp.asarray(st.k_pe),
        jnp.asarray(flat_cap),
        *st.args(),
    )
    np.testing.assert_array_equal(np.asarray(out_cap), np.asarray(out_max))


@pytest.mark.parametrize("il", [1, 4])
def test_should_preserve_all_index_k_bytes_across_interleave_gather_paths(il):
    dcp = 4
    mesh = _mesh(dcp)
    lay = tp.TokenParallelLayout(
        dcp, il, 32, split_rows=16, max_pages=4, gather_caps=(2, 4)
    )
    st = Step(lay, 256, [Req(20, 140), Req(10, 80)], max_reqs=4, seed=31)
    pages, offsets = tp.step_pages(
        lay, jnp.asarray(st.seq_lens), jnp.asarray(st.qsl), st.num_seqs
    )
    page_ids = tp.gather_page_ids(
        lay,
        jnp.asarray(st.block_tables.reshape(-1)),
        pages,
        offsets,
        st.num_pages,
    )
    gathered = np.asarray(
        jax.jit(
            jax.shard_map(
                lambda c, pids, tot: tp.gather_index_k(lay, c, pids, tot, AXIS),
                mesh=mesh,
                in_specs=(P(AXIS), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )(jnp.asarray(st.idx_cache), page_ids, jnp.sum(pages))
    )
    # Reference natural byte layout for each active virtual page v and global slot t.
    n_active = int(pages.sum())
    cache_by_rank = st.idx_cache.reshape(dcp, st.num_pages, IDX_ROWS, IDX_SUB, 32)
    pids_np = np.asarray(page_ids)
    for v in range(n_active):
        p = int(pids_np[v])
        for q in range(dcp):
            nat_page = gathered[v * dcp + q]
            for s in range(32):
                pos = q * 32 + s
                owner = (pos // il) % dcp
                local_slot = (pos // (dcp * il)) * il + (pos % il)
                np.testing.assert_array_equal(
                    nat_page[:, :, s],
                    cache_by_rank[owner, p, :, :, local_slot],
                )


def test_should_accumulate_lead_partial_psum_in_float32_without_bf16_cancellation():
    dcp = 4
    mesh = _mesh(dcp)
    split_rows = 4
    num_heads = 2
    lay = tp.TokenParallelLayout(
        dcp, 4, 32, split_rows=split_rows, max_pages=4, gather_caps=(2, 4)
    )
    st = Step(lay, 64, [Req(32, 32)], max_reqs=4, seed=71)
    rows_per_rank = lay.rows_per_rank(64)

    # Anti-cancellation inputs across ranks: asymmetric LSE weights multiply
    # opposing large partials so bf16 pre-cast before psum incurs >0.1 error,
    # whereas float32 psum matches the float64 reference to bf16 rounding.
    lp = np.zeros((dcp, split_rows, num_heads, DN), dtype=np.float32)
    ll = np.full((dcp, split_rows, num_heads), -np.inf, dtype=np.float32)
    # Row 0: opposing large partials with slightly different LSEs (0.0 vs 0.01).
    ll[:, 0, :] = np.array([0.0, -4.0, 0.01, -4.0], dtype=np.float32)[:, None]
    for r, val in enumerate((100.0, 0.5, -99.0, 0.5)):
        lp[r, 0, :, :] = val
    # Row 1: asymmetric LSE + opposing large/small terms.
    ll[:, 1, :] = np.array([0.0, 0.25, 0.02, 0.25], dtype=np.float32)[:, None]
    for r, val in enumerate((256.0, 0.5, -252.0, 0.5)):
        lp[r, 1, :, :] = val
    # Row 2: all -inf LSE -> must produce exact finite 0.0 (no NaN).
    lp[:, 2, :, :] = 7.0
    # Row 3: single finite rank -> exact passthrough of rank 1's partial.
    ll[1, 3, :] = 2.5
    lp[:, 3, :, :] = np.arange(1, dcp + 1, dtype=np.float32)[:, None, None]

    lp_bf16 = jnp.asarray(lp.reshape(dcp * split_rows, num_heads, DN), jnp.bfloat16)
    ll_f32 = jnp.asarray(ll.reshape(dcp * split_rows, num_heads), jnp.float32)
    w_uv = jnp.broadcast_to(jnp.eye(DN, dtype=jnp.bfloat16)[None], (num_heads, DN, DN))
    q_tail = jnp.zeros((dcp * rows_per_rank, num_heads, DN), jnp.bfloat16)
    flat_topk = jnp.full((dcp * rows_per_rank, TOPK), -1, jnp.int32)

    def _zero_attend(q_tok, g_n, g_r, flat, num_rows):
        del g_n, g_r, flat, num_rows
        return jnp.zeros_like(q_tok)

    def _run_lead_merge(lp_in, ll_in):
        return tp.sparse_mla_attention_with_lead(
            lay,
            mesh,
            AXIS,
            jnp.asarray(st.nope_cache),
            jnp.asarray(st.rope_cache),
            q_tail,
            None,
            None,
            flat_topk,
            jnp.asarray(st.seq_lens),
            jnp.asarray(st.block_tables.reshape(-1)),
            jnp.asarray(st.qsl),
            jnp.asarray([0, 0, st.num_seqs], jnp.int32),
            lead_partial=lp_in,
            lead_lse=ll_in,
            write_kv=None,
            attend=_zero_attend,
            w_uv=w_uv,
            q_own_rows=True,
            w_uv_replicated=True,
            num_tokens=64,
        )[3]

    lead_v = _run_lead_merge(lp_bf16, ll_f32)
    got = np.asarray(lead_v, dtype=np.float32)

    # Structural check: cross-rank lead psum must execute in float32 (not bf16)
    # so TPU all-reduce lowering does not accumulate bf16 rounding drift.
    psum_dtypes: list[jnp.dtype] = []

    def _collect_psum(jpr):
        for eqn in jpr.eqns:
            if eqn.primitive.name == "psum":
                psum_dtypes.extend(v.aval.dtype for v in eqn.invars)
            for sub in jax.extend.core.jaxprs_in_params(eqn.params):
                _collect_psum(sub)

    _collect_psum(jax.make_jaxpr(_run_lead_merge)(lp_bf16, ll_f32).jaxpr)
    assert psum_dtypes and all(dt == jnp.float32 for dt in psum_dtypes)

    # Independent float64 reference using quantized bf16 inputs.
    lp_q = np.asarray(lp_bf16, dtype=np.float64).reshape(dcp, split_rows, num_heads, DN)
    ll_64 = ll.astype(np.float64)
    finite = np.isfinite(ll_64)
    any_finite = np.any(finite, axis=0)
    safe_ll = np.where(finite, ll_64, 0.0)
    max_ll = np.max(np.where(finite, ll_64, -np.inf), axis=0)
    safe_max = np.where(any_finite, max_ll, 0.0)
    exp_shifted = np.where(finite, np.exp(safe_ll - safe_max[None]), 0.0)
    sum_exp = np.sum(exp_shifted, axis=0)
    m = np.where(any_finite, np.log(np.where(any_finite, sum_exp, 1.0)) + safe_max, 0.0)
    w_64 = np.where(finite, np.exp(safe_ll - m[None]), 0.0)
    expected = np.sum(lp_q * w_64[..., None], axis=0)
    expected_bf16 = np.asarray(jnp.asarray(expected, jnp.bfloat16), dtype=np.float32)

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected_bf16, rtol=0.0, atol=1e-2)
    np.testing.assert_array_equal(got[2], 0.0)
    np.testing.assert_array_equal(got[3], 2.0)


@pytest.mark.parametrize("stride_pages", [4, 12, 16, 24, 44])
@pytest.mark.parametrize(
    ("dcp", "il", "ps"),
    [(16, 4, 256), (4, 4, 16), (3, 4, 24)],
)
def test_should_match_divmod_reference_for_flat_rows_power_of_two_and_fallback(
    stride_pages: int, dcp: int, il: int, ps: int
):
    lay = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=il,
        page_size=ps,
        split_rows=4,
        max_pages=44,
    )
    rng = np.random.default_rng(1111 + stride_pages + dcp * 17)
    raw = rng.integers(-4, dcp * ps * 10, size=(32, 64), dtype=np.int32)
    pos = np.where(raw < 0, -1, raw).astype(np.int32)
    # Include exact boundary positions around virtual pages and negative sentinels.
    pos[0, :6] = np.array([-1, 0, il - 1, il, dcp * il, dcp * ps - 1], dtype=np.int32)
    row_offsets = rng.integers(0, 4, size=(32,), dtype=np.int32)

    got = np.asarray(
        tp.flat_rows(
            lay, jnp.asarray(pos), jnp.asarray(row_offsets), stride_pages=stride_pages
        )
    )
    # Independent scalar reference oracle.
    want = np.full_like(pos, -1)
    vps = dcp * ps
    for r in range(pos.shape[0]):
        ro = int(row_offsets[r])
        for c in range(pos.shape[1]):
            p = int(pos[r, c])
            if p >= 0:
                owner = (p // il) % dcp
                in_vp = p % vps
                slot = (in_vp // (dcp * il)) * il + (in_vp % il)
                page = owner * stride_pages + ro + (p // vps)
                want[r, c] = page * ps + slot
    np.testing.assert_array_equal(got, want)

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
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

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


@pytest.mark.parametrize("dcp", sorted(CASES))
def test_indexer_topk_matches_natural_reference(dcp):
    mesh = _mesh(dcp)
    case = CASES[dcp]
    lay, num_tokens = case["layout"], case["num_tokens"]
    st = Step(lay, num_tokens, case["reqs"], seed=2)
    rng = np.random.default_rng(3)
    per = lay.rows_per_rank(num_tokens)
    q = rng.standard_normal((num_tokens, 1, IDX_ROWS * IDX_SUB)).astype(np.float32)
    weights = np.ones((num_tokens, 1), np.float32)

    flat = jax.jit(
        lambda *a: tp.indexer_topk(
            lay, mesh, AXIS, *a, select_topk=_select_topk_ref(lay)
        )
    )(jnp.asarray(st.idx_cache), jnp.asarray(q), jnp.asarray(weights), *st.args())
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

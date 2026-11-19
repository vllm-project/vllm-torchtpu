# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS (q38 BLSTACK-R): q38 F2's draft KV on a
staged decode whose compressed keys are split over its main-KV DCP ranks
(VLLM_TPU_QSA_COMPRESSED_DCP=1), on CPU.

The key only lifts check_draft_kv's refusal: the landing planners already split
every compressed owner by ``_split_rank``, the MTP draft's included, and route
the draft gap (F2 option (a)) through the same split pair tokens. These tests
land synthetic producer pages on a TP8 / DCP4 decode (2 KV heads, 128-token
blocks, 320-token producer pages, ratio 4, packed compressed slots: targets in
slots 0-1, the draft in slot 2) through all three planners - broadcast
``indices``, routed ``_routed_indices`` (QWEN4_EXP_PD_ROUTED_DST, real producer
spans and the span applier of test_qwen4_exp_pd_route.py) and its
QWEN4_EXP_PD_FAST_ADMIT form ``_routed_indices_shared_rows`` - with the real
repack, and check, on every rank:

* every compressed owner's rows, the draft's included, are the owned part of
  the unsplit landing (today's planner, each owner on per-block pages): block
  ``k`` on rank ``k % D``, rows ``slot * 16 + i`` of the page for entry
  ``k // D``;
* the landing differs from one without the draft's compressed owner (what
  d523c2ce0 plans for a decode-only draft) only at the draft's owned rows:
  target slots, main KV and rings are byte-identical, and rows no rank owns
  keep their stale bytes;
* with a draft gap, exactly the draft's owned main-KV rows and owned
  compressed pairs of tokens in [skip, end) are zero, by the (t // B) % D rule
  computed here, and nothing else changes.
"""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from tests.runner import test_qwen4_exp_pd_route as rtt  # noqa: E402

st = rtt.st
rt = st.pd_route

ROWS, LANES = rtt.ROWS, rtt.LANES
TP, D, B, SRC, R = 8, 4, 128, 320, 4
SPAN = 2 * R
MEMBER_ROWS = B // SPAN  # pair rows of one packed member per block
GEO = rt.RouteGeometry(tp=TP, dcp=D, dst_block=B, src_block=SRC, ratio=R,
                       ssm_rows_src=16, conv_rows_src=8, ssm_rows_dst=2, stage_rows=2)
TARGETS = ("model.layers.3.self_attn", "model.layers.7.self_attn")
DRAFT = "mtp.layers.48.self_attn"
LAYERS = TARGETS + (DRAFT,)
MAIN = list(LAYERS)
RING = [n + ".raw_ring" for n in LAYERS]
COMP = [n + ".compressed_k" for n in LAYERS]  # packed slots 0, 1, 2 (the draft's)
DRAFT_CK = DRAFT + ".compressed_k"
NAMES = MAIN + COMP + RING
POOL_PAGES = 240
CASES = [
    pytest.param(1500, 0, id="no_prefix"),
    pytest.param(2600, 1024, id="local_prefix"),  # skip on a D x B entry
    pytest.param(300, 0, id="shorter_than_an_entry"),
    pytest.param(100, 0, id="shorter_than_a_block"),
]
PLANNERS = ["broadcast", "routed", "routed_fast"]


@pytest.fixture(autouse=True)
def geometry(monkeypatch):
    # The route test's staging helpers read its module geometry.
    monkeypatch.setattr(rtt, "GEO", GEO)
    assert (rtt.SRC_BLOCK, rtt.DST_BLOCK, rtt.RATIO) == (SRC, B, R)


def rank_slice(r):
    return st.RankSlice(tp_size=TP, tp_rank=r, num_kv_heads=2, main_dcp=D)


_FNS = {}


def repack_fn(rank):
    if rank.kv_head not in _FNS:
        _FNS[rank.kv_head] = st.make_repack(
            src_block=SRC, dst_block=B, src_gdn=None, dst_gdn=None, kv_head=rank.kv_head,
            v_head_start=0, channels=[0], write=rtt.scatter,
        )
    return _FNS[rank.kv_head]


def land(rank, pool, staging, idx):
    size = st.bucket(max(len(idx[0]), len(idx[2]), 1), 16)
    out = jnp.asarray(pool)
    fn = repack_fn(rank)
    # No GDN owners here: empty GDN lists (the repack is built without GDN state).
    for plan in st.repack_chunks(idx[0], idx[1], idx[2], idx[3],
                                 np.zeros(0, np.int64), np.zeros(0, np.int64), size, 1):
        out = fn(out, jnp.asarray(staging), *(jnp.asarray(v, jnp.int32) for v in plan))
    return np.asarray(out).reshape(-1, ROWS, LANES)


def consumer(rank, planner, *, split=True, draft_ck=True):
    """A staged decode rank: main KV and rings on tables of their own,
    compressed keys packed in group "comp" (slot s at row s * 16) when split,
    else each on its own per-block group (the unsplit reference)."""
    c = object.__new__(st.StagedConsumer)
    c.rank, c.ratio, c.dst_block, c.src_block, c._state_entry = rank, R, B, SRC, 0
    c.route = None if planner == "broadcast" else GEO
    c.__dict__["_fast_admit"] = planner == "routed_fast"
    c._dst_owner = {n: n for n in MAIN + RING}
    if split:
        c._dst_compressed = {n: ("comp", s * MEMBER_ROWS) for s, n in enumerate(COMP)}
    else:
        c._dst_compressed = {n: (n, 0) for n in COMP}
    if not draft_ck:
        del c._dst_compressed[DRAFT_CK]
    c._src_owner = {n: i for i, n in enumerate(NAMES)}
    c.src_groups = [None] * len(NAMES)
    c._gdn_names, c._zero_draft = [], {}
    # VLLM_TPU_QSA_MTP_KV_DCP + VLLM_TPU_QSA_COMPRESSED_DCP (pool_layout).
    c._dcp_owners = frozenset(MAIN) | (frozenset(c._dst_compressed) if split else frozenset())
    return c


class Load:
    """One load: producer pages (random bytes), destination tables, staging for
    both layouts, and a stale pool (random bytes)."""

    def __init__(self, n, skip, seed):
        rng = np.random.default_rng(seed)
        self.n, self.skip = n, skip
        self.prod = rtt.producer_pages(n, rng, names=NAMES)
        ids = [int(x) for x in rng.permutation(np.arange(1, POOL_PAGES))]
        take = lambda k: [ids.pop() for _ in range(k)]  # noqa: E731
        entries, blocks = -(-n // (D * B)), -(-n // B)
        self.tables = {n_: take(entries) for n_ in MAIN}
        self.tables.update({n_: take(1) for n_ in RING})
        self.tables["comp"] = take(entries)  # split packed pages, one per entry
        self.virtual = {n_: take(blocks) for n_ in COMP}  # unsplit per-block pages
        self.stale = rng.integers(0, 256, (POOL_PAGES, B, 1, ROWS, LANES), dtype=np.uint8)
        # Broadcast staging: every producer page of every owner, whole.
        total = sum(len(self.prod[x]) for x in NAMES) + 1
        self.bstage = np.zeros((total, SRC, 1, ROWS, LANES), np.uint8)
        self.bpages, cur = {}, 1
        for name in NAMES:
            k = len(self.prod[name])
            self.bpages[name] = list(range(cur, cur + k))
            self.bstage[cur:cur + k, :, 0] = self.prod[name]
            cur += k
        self.routed = {}

    def staging(self, planner, rank):
        if planner == "broadcast":
            return self.bstage, self.bpages
        if rank.tp_rank not in self.routed:
            staging, pages, _ = rtt.routed_staging(self.n, self.skip, rank, self.prod, NAMES)
            self.routed[rank.tp_rank] = (staging, pages)
        return self.routed[rank.tp_rank]

    def run(self, c, planner, *, pool=None, tables=None, gap=0):
        staging, pages = self.staging(planner, c.rank)
        idx = c.indices(pages, tables or self.tables, self.n, self.skip, None,
                        draft_zero_until=gap)
        return land(c.rank, self.stale if pool is None else pool, staging, idx)

    def owned_blocks(self, rank):
        return [k for k in range(self.skip // B, -(-self.n // B)) if k % D == rank.tp_rank % D]

    def comp_rows(self, rank, slot, k, n_rows):
        return self.tables["comp"][k // D] * B + slot * MEMBER_ROWS + np.arange(n_rows)


def _pairs(n):
    return math.ceil(math.ceil(n / R) / 2)


def unsplit_reference(load, rank, planner):
    """Today's unsplit landing of every compressed owner on per-block pages."""
    c = consumer(rank, planner, split=False)
    return load.run(c, planner, pool=np.zeros_like(load.stale),
                    tables={**load.tables, **load.virtual})


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("num_tokens,skip", CASES)
def test_split_draft_landing_is_the_owned_part_and_touches_nothing_else(planner, num_tokens, skip):
    load = Load(num_tokens, skip, seed=num_tokens * 3 + skip + PLANNERS.index(planner))
    stale = load.stale.reshape(-1, ROWS, LANES)
    draft_any = 0
    for r in range(TP):
        rank = rank_slice(r)
        got = load.run(consumer(rank, planner), planner)
        ref = unsplit_reference(load, rank, planner)
        owned_all = []
        draft_rows = []
        for slot, name in enumerate(COMP):
            for k in load.owned_blocks(rank):
                n_rows = min(MEMBER_ROWS, _pairs(num_tokens) - k * MEMBER_ROWS)
                rows = load.comp_rows(rank, slot, k, n_rows)
                src = load.virtual[name][k] * B + np.arange(n_rows)
                np.testing.assert_array_equal(
                    got[rows], ref[src], err_msg=f"{planner} rank {r} {name} block {k}")
                owned_all.append(rows)
                if name == DRAFT_CK:
                    draft_rows.append(rows)
        owned = np.concatenate(owned_all) if owned_all else np.zeros(0, np.int64)
        drafted = np.concatenate(draft_rows) if draft_rows else np.zeros(0, np.int64)
        draft_any += len(drafted)
        # Every other row of the packed pages keeps its stale bytes.
        comp_rows = (np.asarray(load.tables["comp"])[:, None] * B + np.arange(B)).reshape(-1)
        rest = np.setdiff1d(comp_rows, owned)
        np.testing.assert_array_equal(got[rest], stale[rest], err_msg=f"{planner} rank {r}")
        # Against the landing without the draft's compressed owner: only the
        # draft's owned rows differ (page 0 is the null page the padding rewrites).
        base = load.run(consumer(rank, planner, draft_ck=False), planner)
        diff = np.nonzero((got != base).any(axis=(1, 2)))[0]
        diff = diff[diff >= B]
        assert set(diff.tolist()) <= set(drafted.tolist()), f"{planner} rank {r}: clobbered rows"
        assert set(diff.tolist()) == set(drafted.tolist()), f"{planner} rank {r}: draft not landed"
        if num_tokens <= B and r % D:
            assert len(drafted) == 0  # a block of rank 0's DCP slice only
    assert draft_any > 0


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("num_tokens,skip", CASES)
def test_split_draft_gap_zero_fills_exactly_the_owned_draft_rows_below_its_end(
    planner, num_tokens, skip
):
    load = Load(num_tokens, skip, seed=num_tokens * 5 + skip + 7 * PLANNERS.index(planner))
    end = skip + (num_tokens - skip) // 2
    zeroed_any = 0
    for r in range(TP):
        rank = rank_slice(r)
        c = consumer(rank, planner)
        full = load.run(c, planner)
        gapped = load.run(c, planner, gap=end)
        # The (t // B) % D rule, written out here.
        t = np.arange(skip, end)
        t = t[(t // B) % D == r % D]
        main = np.asarray(load.tables[DRAFT])[t // (D * B)] * B + t % B
        p = np.arange(skip // SPAN, _pairs(num_tokens)) * SPAN
        p = p[(p < end) & ((p // B) % D == r % D)]
        comp = np.asarray(load.tables["comp"])[p // (D * B)] * B + 2 * MEMBER_ROWS + (p % B) // SPAN
        zero = np.union1d(main, comp)
        assert not gapped[zero].any(), f"{planner} rank {r}"
        others = np.setdiff1d(np.arange(B, gapped.shape[0]), zero)
        np.testing.assert_array_equal(gapped[others], full[others], err_msg=f"{planner} rank {r}")
        assert c.last_draft_rows[1] == len(main) + len(comp)
        zeroed_any += len(zero)
    assert zeroed_any > 0


# --- the key: admission only ---------------------------------------------------

_ST_DRAFT, _ST_TARGET = "mtp.layers.48.self_attn", "model.layers.3.self_attn"
_ST_DST = [_ST_TARGET, _ST_TARGET + ".raw_ring", _ST_DRAFT, _ST_DRAFT + ".raw_ring"]
_SPLIT_OWNERS = (_ST_TARGET, _ST_DRAFT, _ST_TARGET + ".compressed_k", _ST_DRAFT + ".compressed_k")


def test_key_off_keeps_the_refusal(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS", raising=False)
    assert st.draft_kv_split_keys_enabled() is False
    with pytest.raises(ValueError, match="refuses VLLM_TPU_QSA_COMPRESSED_DCP=1"):
        st.check_draft_kv(set(), _ST_DST, 4, _SPLIT_OWNERS)


def test_key_on_admits_split_compressed_keys(monkeypatch):
    # test_qwen4_exp_pd_staging.py's test_check_draft_kv_refuses_compressed_dcp, inverted.
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS", "1")
    assert st.draft_kv_split_keys_enabled() is True
    st.check_draft_kv(set(), _ST_DST, 4, _SPLIT_OWNERS)


def test_key_on_keeps_every_other_refusal(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS", "1")
    monkeypatch.delenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", raising=False)
    with pytest.raises(ValueError, match="no pages for"):
        st.check_draft_kv({_ST_DRAFT + ".compressed_k"}, _ST_DST, 4, _SPLIT_OWNERS)
    with pytest.raises(ValueError, match="needs VLLM_TPU_QSA_MTP_KV_DCP=1"):
        st.check_draft_kv(set(), _ST_DST, 4, (_ST_TARGET, _ST_TARGET + ".compressed_k"))
    monkeypatch.setenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", "1")
    with pytest.raises(ValueError, match="QWEN4_EXP_PD_ZERO_DRAFT_KV=1 would zero"):
        st.check_draft_kv(set(), _ST_DST, 4, _SPLIT_OWNERS)


def test_key_registered_off_by_default_and_compile_neutral(monkeypatch):
    envs = pytest.importorskip("vllm_torchtpu.envs")
    from vllm_torchtpu.compilation import tpu_compiler

    key = "QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS"
    assert key in envs.environment_variables
    assert key in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv(key, raising=False)
    assert envs.QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS is False
    off = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(key, "1")
    assert envs.QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS is True
    assert key not in off and tpu_compiler._tpu_compile_env_factors() == off

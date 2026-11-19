# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_REPACK_V2 (q38 XFER-4 F1-F3): a staged decode lands each load
in fewer repack programs and writes the same bytes.

Served decodes have 36 GDN owners; with 4 GDN pages a program, every load
(even a warm hit with one chunk of main / copy rows) ran 9 programs, matching
the multiples of 9 landing kernels in arm S's PJRT VLOG. v2 puts GDN owners in
programs of their own and buckets main / copy rows. These tests check, on CPU:
the pool ends byte-identical to today's chunks; the program counts; the packed
index buffer round-trips; the off path is today's; and the plan made at landing
(F3) is the one the named step runs.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

_PATH = (
    pathlib.Path(__file__).parents[2]
    / "src/vllm_torchtpu/models/vllm/qwen4_exp/pd_staging.py"
)
_spec = importlib.util.spec_from_file_location("pd_staging_v2_under_test", _PATH)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

ROWS, LANES = 4, 256


def carrier(pages: int, block: int) -> np.ndarray:
    return np.zeros((pages, block, 1, ROWS, LANES), np.uint8)


def scatter(pool, rows, page, logical, valid):
    page = jnp.where(valid, page, pool.shape[0])
    return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")


def geometry():
    s = st.GdnGeometry(ssm_tokens=8, conv_tokens=8, v_heads=16, k_heads=8,
                       head_k_dim=8, head_v_dim=16, conv_width=3,
                       ssm_itemsize=4, conv_itemsize=4)
    tp = 2
    v_local, k_local = 16 // tp, 8 // tp
    ssm_rows = -(-v_local * 8 * 16 // LANES)
    conv_rows = -(-3 * (2 * k_local * 8 + v_local * 16) // LANES)
    d = st.GdnGeometry(ssm_tokens=ssm_rows, conv_tokens=1 << (conv_rows - 1).bit_length(),
                       v_heads=v_local, k_heads=k_local, head_k_dim=8, head_v_dim=16,
                       conv_width=3, ssm_itemsize=4, conv_itemsize=4)
    return s, d, tp


def build(n_gdn: int, n_tokens: int, seed: int):
    """A random staging carrier, a rank's repack fn, and its index lists."""
    rng = np.random.default_rng(seed)
    src_block, dst_block, ratio = 384, 128, 4
    s, d, tp = geometry()
    rank = st.RankSlice(tp_size=tp, tp_rank=1, num_kv_heads=2, main_dcp=1)
    n_src_pages = 3 + -(-n_tokens // src_block) * 2 + n_gdn
    staging = rng.integers(0, 255, (n_src_pages, src_block, 1, ROWS, LANES)).astype(np.uint8)
    staging[0] = 0  # the null page, as in serving
    src_main = list(range(1, 1 + -(-n_tokens // src_block)))
    src_comp = list(range(src_main[-1] + 1, src_main[-1] + 1 + len(src_main)))
    src_ring = src_comp[-1] + 1
    src_gdn = list(range(src_ring + 1, src_ring + 1 + n_gdn))
    n_dst = -(-n_tokens // dst_block)
    dst_main = list(range(1, 1 + n_dst))
    dst_comp = list(range(1 + n_dst, 1 + 2 * n_dst))
    dst_ring = 1 + 2 * n_dst
    dst_gdn = list(range(dst_ring + 1, dst_ring + 1 + n_gdn))
    pool_pages = dst_gdn[-1] + 2 if dst_gdn else dst_ring + 2
    ms, md = st.main_rows(n_tokens, src_main, src_block, dst_main, dst_block, rank)
    cs, cd = st.compressed_rows(n_tokens, ratio, src_comp, src_block, dst_comp, dst_block)
    rs, rd = st.page_prefix_rows(src_ring, src_block, dst_ring, dst_block, dst_block)
    fn = st.make_repack(
        src_block=src_block, dst_block=dst_block, src_gdn=s, dst_gdn=d,
        kv_head=rank.kv_head, v_head_start=rank.tp_rank * d.v_heads,
        channels=st.conv_channels(s, d, rank), write=scatter,
    )
    idx = (np.asarray(ms, np.int64), np.asarray(md, np.int64),
           np.asarray(cs + rs, np.int64), np.asarray(cd + rd, np.int64),
           np.asarray(src_gdn, np.int64), np.asarray(dst_gdn, np.int64))
    return staging, carrier(pool_pages, dst_block), fn, idx


def run(fn, pool, staging, programs):
    out = jnp.asarray(pool)
    for program in programs:
        out = fn(out, jnp.asarray(staging), *(jnp.asarray(np.asarray(v), jnp.int32) for v in program))
    return np.asarray(out)


@pytest.mark.parametrize("n_gdn,n_tokens,gdn_pages,size", [
    (9, 700, 9, 64),     # all GDN in one program; main / copy in several chunks
    (9, 700, 4, 64),     # GDN split 4 + 4 + 1 (last group padded)
    (5, 200, 36, 512),   # fewer owners than a program holds
    (0, 900, 4, 128),    # no GDN at all
])
def test_v2_pool_is_byte_identical_to_todays_chunks(n_gdn, n_tokens, gdn_pages, size):
    staging, pool, fn, idx = build(n_gdn, n_tokens, seed=n_gdn * 1000 + n_tokens)
    v1 = list(st.repack_chunks(*idx, size=size, gdn_pages=4))
    v2 = st.repack_programs_v2(*idx, gdn_pages, size=size, buckets=(16, 32, size), prefer_fewer=False)
    want = run(fn, pool, staging, v1)
    got = run(fn, pool, staging, v2)
    assert want.any()
    np.testing.assert_array_equal(got, want)
    chosen = st.repack_programs_v2(*idx, gdn_pages, size=size, buckets=(16, 32, size))
    assert len(chosen) == min(len(v1), len(v2))
    np.testing.assert_array_equal(run(fn, pool, staging, chosen), want)


def test_v2_writes_what_a_single_unchunked_program_writes():
    staging, pool, fn, idx = build(9, 700, seed=7)
    one = [(idx[0], idx[1], idx[2], idx[3], idx[4], idx[5])]
    v2 = st.repack_programs_v2(*idx, 36, size=64, buckets=(16, 32, 64), prefer_fewer=False)
    np.testing.assert_array_equal(run(fn, pool, staging, v2), run(fn, pool, staging, one))


def test_served_shape_program_counts():
    # 36 GDN owners; a warm hit's rows fit one chunk; a big miss needs three.
    gdn = np.arange(36, dtype=np.int64)
    hit = np.arange(15_000, dtype=np.int64)
    assert len(list(st.repack_chunks(hit, hit, hit[:3000], hit[:3000], gdn, gdn))) == 9
    v2 = st.repack_programs_v2(hit, hit, hit[:3000], hit[:3000], gdn, gdn, 36)
    assert len(v2) == 2
    assert [len(p[0]) for p in v2] == [st.REPACK_TILE, 16384]
    assert [len(p[4]) for p in v2] == [36, 1]
    big = np.arange(40_000, dtype=np.int64)
    v2 = st.repack_programs_v2(big, big, hit, hit, gdn, gdn, 36)
    assert [len(p[0]) for p in v2] == [st.REPACK_TILE, 16384, 16384, 8192]
    # A cold miss whose rows need >= 9 chunks keeps today's programs (never more programs).
    cold = np.arange(250_000, dtype=np.int64)
    today = list(st.repack_chunks(cold, cold, hit, hit, gdn, gdn))
    v2 = st.repack_programs_v2(cold, cold, hit, hit, gdn, gdn, 36)
    assert len(today) == 16 and len(v2) == 16
    assert [st.StagedConsumer._v2_layout(p) for p in v2] == [st.StagedConsumer._v2_layout(p) for p in today]
    assert len(st.repack_programs_v2(cold, cold, hit, hit, gdn, gdn, 36, prefer_fewer=False)) == 17


def test_routed_gdn_entries_keep_their_columns():
    gdn_s = np.arange(10 * 3, dtype=np.int64).reshape(10, 3)
    gdn_d = np.arange(10, dtype=np.int64)
    rows = np.arange(100, dtype=np.int64)
    v2 = st.repack_programs_v2(rows, rows, rows, rows, gdn_s, gdn_d, 4, buckets=(128,), prefer_fewer=False)
    assert [p[4].shape for p in v2] == [(4, 3), (4, 3), (4, 3), (1, 3)]
    np.testing.assert_array_equal(v2[2][4], np.stack([gdn_s[8], gdn_s[9], gdn_s[9], gdn_s[9]]))
    assert not v2[3][4].any()  # the null entry


def test_packed_buffer_round_trips():
    gdn_s = np.arange(12, dtype=np.int64).reshape(4, 3)
    program = (np.arange(8), np.arange(8) + 100, np.arange(8) + 200, np.arange(8) + 300,
               gdn_s, np.array([7, 8, 9, 10]))
    layout = st.StagedConsumer._v2_layout(program)
    buf = st.StagedConsumer._v2_pack(program)
    assert buf.dtype == np.int32 and buf.size == 4 * 8 + 12 + 4
    for got, want in zip(st.v2_unpack(buf, layout), program):
        np.testing.assert_array_equal(got, np.asarray(want))


def test_off_plans_todays_chunks(monkeypatch):
    c = object.__new__(st.StagedConsumer)
    assert c._v2_gdn_pages == 0  # class default: today's path
    seen = {}
    monkeypatch.setattr(st, "repack_chunks", lambda *a, **k: seen.setdefault("v1", [("c",)]))
    monkeypatch.setattr(st, "repack_programs_v2", lambda *a, **k: seen.setdefault("v2", [("p",)]))
    c.reservations = {5: {}}
    c.skips = {}
    c.draft_zero_end = lambda uuid, n: 0
    c.indices = lambda *a, **k: (np.zeros(1),) * 6
    c._count_draft = lambda *a: None
    assert c.plan_repack(5, [], 10).chunks == [("c",)]
    c._v2_gdn_pages = 36
    assert c.plan_repack(5, [], 10).chunks == [("p",)]


def test_null_programs_cover_every_layout_and_write_only_null_pages():
    c = object.__new__(st.StagedConsumer)
    c._v2_gdn_pages = 36
    c._gdn_names = ["g%d" % i for i in range(36)]
    c.route = SimpleNamespace(gdn_rows=6, stage_rows=2, gdn_pages=3)  # RouteGeometry.gdn_pages = ceil(gdn_rows / stage_rows)
    programs = c.v2_null_programs()
    layouts = [st.StagedConsumer._v2_layout(p) for p in programs]
    assert len(set(layouts)) == len(layouts) == 1 + len(st.REPACK_V2_BUCKETS) + 1
    assert ((st.REPACK_TILE,), (st.REPACK_TILE,), (st.REPACK_TILE,), (st.REPACK_TILE,), (36, 3), (36,)) in layouts
    big = (st.REPACK_CHUNK_ROWS,)
    assert (big, big, big, big, (st.REPACK_GDN_PAGES, 3), (st.REPACK_GDN_PAGES,)) in layouts  # a cold miss's
    assert all(not np.asarray(v).any() for p in programs for v in p)


def test_null_programs_run_where_the_routed_gdn_entry_is_not_whole_stage_pages():
    """T10 x REPACK_V2: the F2-TP2 smoke j-10ef57cf died in warm_repack_v2 on every decode attempt with
    'cannot reshape array of shape (9216, 4, 256) into shape (-1, 448, 4, 256)'. A TP2 producer's routed GDN entry
    is 448 rows over 256-row staging pages, so the repack reads route.gdn_pages = 2 columns per entry while the null
    programs carried gdn_rows // stage_rows = 1. Same shape here: a 12-row entry over 8-row pages. Every null program
    carries route.gdn_pages columns and runs through the routed repack, writing only null pages."""
    s, d, tp = geometry()
    stage_rows = 8
    gdn_rows = d.ssm_tokens + s.conv_tokens
    assert gdn_rows % stage_rows, "the case needs an entry that is not whole stage pages"
    route = SimpleNamespace(gdn_rows=gdn_rows, stage_rows=stage_rows, gdn_pages=-(-gdn_rows // stage_rows))
    c = object.__new__(st.StagedConsumer)
    c._v2_gdn_pages = 36
    c._gdn_names = ["g%d" % i for i in range(36)]
    c.route = route
    programs = c.v2_null_programs()
    assert {np.asarray(p[4]).shape[1:] for p in programs} == {(route.gdn_pages,)}
    rank = st.RankSlice(tp_size=tp, tp_rank=1, num_kv_heads=2, main_dcp=1)
    fn = st.make_repack(
        src_block=384, dst_block=128, src_gdn=s, dst_gdn=d, kv_head=rank.kv_head,
        v_head_start=rank.tp_rank * d.v_heads, channels=st.conv_channels(s, d, rank), write=scatter,
        compact_gdn=(stage_rows, route.gdn_pages),
    )
    out = run(fn, carrier(4, 128), np.zeros((4, 384, 1, ROWS, LANES), np.uint8), programs)
    assert not out.any()


def _pp8_route(stage_rows):
    """A PP8 (TP1-producer) RouteGeometry: no gdn_pages, whole-page GDN extents (12 rows here)."""
    s, d, tp = geometry()
    if hasattr(st.pd_route.RouteGeometry, "gdn_pages"):
        pytest.skip("this tree's RouteGeometry is T10's (it has gdn_pages): covered by the T10 test above")
    route = st.pd_route.RouteGeometry(
        tp=tp, dcp=1, dst_block=128, src_block=384, ratio=4, ssm_rows_src=tp * d.ssm_tokens,
        conv_rows_src=s.conv_tokens, ssm_rows_dst=d.ssm_tokens, stage_rows=stage_rows,
    )
    return s, d, tp, route


@pytest.mark.parametrize("stage_rows", [4, 2])
def test_null_programs_on_a_pp8_route_carry_its_whole_page_gdn_columns(stage_rows):
    """PP8 x REPACK_V2: 01e56ce26 read route.gdn_pages, which only T10's RouteGeometry has, so every PP8 decode with
    QWEN4_EXP_PD_REPACK_V2 died in warm_repack_v2 with "AttributeError: 'RouteGeometry' object has no attribute
    'gdn_pages'" (P-MOE-6, T64). On a PP8 route the null programs carry gdn_rows // stage_rows columns, the columns
    the landing reads (plan: n_gdn = g.gdn_rows // g.stage_rows; make_repack: compact_gdn=(stage_rows, that)), and
    they run through that routed repack writing only null pages."""
    s, d, tp, route = _pp8_route(stage_rows)
    assert not hasattr(route, "gdn_pages")
    cols = route.gdn_rows // route.stage_rows
    c = object.__new__(st.StagedConsumer)
    c._v2_gdn_pages = 36
    c._gdn_names = ["g%d" % i for i in range(36)]
    c.route = route
    programs = c.v2_null_programs()
    assert {np.asarray(p[4]).shape[1:] for p in programs} == {(cols,)}
    assert all(not np.asarray(v).any() for p in programs for v in p)
    rank = st.RankSlice(tp_size=tp, tp_rank=1, num_kv_heads=2, main_dcp=1)
    fn = st.make_repack(
        src_block=384, dst_block=128, src_gdn=s, dst_gdn=d, kv_head=rank.kv_head,
        v_head_start=rank.tp_rank * d.v_heads, channels=st.conv_channels(s, d, rank), write=scatter,
        compact_gdn=(route.stage_rows, cols),
    )
    out = run(fn, carrier(4, 128), np.zeros((4, 384, 1, ROWS, LANES), np.uint8), programs)
    assert not out.any()


def test_null_programs_refuse_a_route_without_gdn_pages_whose_gdn_entry_is_not_whole_pages():
    """No silent truncation: a route without gdn_pages must have whole-page GDN extents (PP8's RouteGeometry
    enforces it); one that does not is refused instead of landing gdn_rows // stage_rows columns."""
    c = object.__new__(st.StagedConsumer)
    c._v2_gdn_pages = 36
    c._gdn_names = ["g%d" % i for i in range(36)]
    c.route = SimpleNamespace(gdn_rows=12, stage_rows=8)
    with pytest.raises(ValueError, match="not whole 8-row staging pages"):
        c.v2_null_programs()


def test_repack_v2_env_default_and_compile_key():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import (
        _TPU_COMPILE_ENV_IGNORED,
        _tpu_compile_env_factors,
    )

    assert envs.environment_variables["QWEN4_EXP_PD_REPACK_V2"]() == 0
    for name in ("QWEN4_EXP_PD_REPACK_V2", "QWEN4_EXP_PD_REPACK_PLAN_AT_LAND"):
        assert name in _TPU_COMPILE_ENV_IGNORED
        assert name not in set(_tpu_compile_env_factors())
    assert envs.environment_variables["QWEN4_EXP_PD_REPACK_PLAN_AT_LAND"]() is False


# --- F3: the plan is made when the load lands on the rank ---------------------


class FakeStaging:
    shared = None

    def __init__(self, v2):
        self._v2_gdn_pages = v2
        self.reservations, self.calls, self.released = {}, [], []
        self.skips = {}

    def plan_repack(self, uuid, tables, num_tokens):
        assert uuid in self.reservations
        self.calls.append(("plan", uuid, [list(t) for t in tables], num_tokens))
        return SimpleNamespace(chunks=["P%d" % uuid], passes=[])

    def run_repack(self, plan):
        self.calls.append(("run", plan.chunks))

    def repack(self, uuid, tables, num_tokens):
        self.calls.append(("repack", uuid))

    def release(self, uuid):
        self.reservations.pop(uuid, None)
        self.released.append(uuid)


def worker(v2):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker

    w = object.__new__(Qwen4ExpRaidenWorker)
    w._staging = FakeStaging(v2)
    w._stage3_submitted_load_metas = {}
    w._pending_repacks = {}
    w._sync_repack = True
    w._sync_on = lambda: True
    w.tp_rank = 0
    w._kvconn = None  # landing log off
    return w


def land(w, req_id, uuid):
    w._staging.reservations[uuid] = {}
    w._stage3_submitted_load_metas[req_id] = SimpleNamespace(
        uuid=uuid, group_block_ids=[[3, 4], [9]], num_tokens=7000
    )
    w._stage3_finish_destination(req_id, True)


@pytest.fixture
def plan_at_land(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_REPACK_PLAN_AT_LAND", "1")


def test_f3_needs_its_own_switch(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_REPACK_PLAN_AT_LAND", raising=False)
    w = worker(v2=36)
    land(w, "a", 11)
    assert w._staging.calls == []  # v2 alone plans in the named step
    w.repack_landed(["a"])
    assert w._staging.calls == [("repack", 11)]


def test_f3_on_plans_at_landing_and_the_named_step_only_runs(plan_at_land):
    w = worker(v2=36)
    land(w, "a", 11)
    assert w._staging.calls == [("plan", 11, [[3, 4], [9]], 7000)]
    w._stage3_submitted_load_metas.clear()
    w.repack_landed(["a"])
    assert w._staging.calls[1:] == [("run", ["P11"])]
    assert w._staging.released == [11]


def test_f3_off_repacks_in_the_named_step_as_today(plan_at_land):
    w = worker(v2=0)
    land(w, "a", 11)
    assert w._staging.calls == []
    w.repack_landed(["a"])
    assert w._staging.calls == [("repack", 11)]
    assert w._staging.released == [11]


def test_f3_cleanup_before_naming_releases_without_running(plan_at_land):
    w = worker(v2=36)
    land(w, "a", 11)
    w._stage3_finish_destination("a", False)
    assert w._staging.released == [11]
    w.repack_landed(["a"])
    assert [c[0] for c in w._staging.calls] == ["plan"]


def test_f3_with_the_landing_log_logs_the_named_loads_own_rows(monkeypatch, plan_at_land):
    import json as _json

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pc

    lines = []
    monkeypatch.setattr(pc, "logger", SimpleNamespace(info=lambda fmt, *a: lines.append(fmt % a if a else fmt),
                                                      warning=lambda *a, **k: None))
    monkeypatch.setenv("QWEN4_EXP_PD_LANDING_LOG", "5")
    w = worker(v2=36)
    del w._kvconn  # landing log on (rank 0)

    def plan(uuid, tables, num_tokens):
        w._staging.last_plan_rows = (uuid, 0, 0)  # the staging's last plan: whichever landed last
        return SimpleNamespace(chunks=["P%d" % uuid], passes=[], rows=(uuid, 0, 0))

    w._staging.plan_repack = plan
    land(w, "a", 11)
    land(w, "b", 12)  # lands after a, before a is named
    w.repack_landed(["a"])
    (rec,) = [_json.loads(x) for x in lines if x.startswith("{")]
    assert (rec["req_id"], rec["rows_main"], rec["plan_ms"]) == ("a", 11, -1.0)


def test_v2_jax_op_passes_torch_tpus_signature_check():
    """j-4c6c1808 died here on the TPU image: torch_tpu's jax_op refuses an
    unannotated JAX function at construction (pallas._verify_signature), the
    same check the CPU torch_tpu runs. Build the v2 op as warm_repack_v2 does."""
    pytest.importorskip("torch_tpu._internal.pallas")
    c = object.__new__(st.StagedConsumer)
    c._fn = lambda pool, staging, *idx: pool
    op = c._v2_jax_op()
    assert op is not None and c._v2_jax_op() is op  # built once, then reused

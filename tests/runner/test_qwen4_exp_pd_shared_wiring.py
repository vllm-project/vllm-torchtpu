# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_SHARED_COMPRESSED wired into the routed staged path, on CPU.

Built on S12's routed compact staging (``test_qwen4_exp_pd_route``, whose
geometry, span applier and broadcast reference this reuses): with the flag,
a shared load's compressed pairs land on one rank each (page b on rank
b % TP), the consumer's staging counts and clips follow
``pd_shared_compressed``, the repack leaves them out, and the landing gather
(pack, all-gather across the ranks' staging, scatter) writes them. Every
rank's pool must equal today's broadcast repack byte for byte.

Also: the producer's spans with and without compressed-key sub-pages (stage
C), the fingerprint, the scheduler's naming guard, and the workers'
agreement step in a real 4-process gloo group (a rank missing a load must
make every rank drop it, not hang).
"""

import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

_HERE = pathlib.Path(__file__).parent


def _load_test(name):
    spec = importlib.util.spec_from_file_location(f"shared_wiring_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rtest = _load_test("test_qwen4_exp_pd_route")
st, rt = rtest.st, rtest.rt
psc = st.pd_shared_compressed
GEO, TP, RB = rtest.GEO, rtest.TP, rtest.RB
SH = psc.geometry_for(GEO)  # 4 ranks, 2-row pages, 320-token producer pages
UUID = 7


# --- the consumer: counts, clips, plan, landing --------------------------------


def consumer(r, shared=True):
    c = rtest.consumer(rtest.rank_slice(r))
    names = rtest.owners()
    c.shared = SH if shared else None
    c.landing = None
    c.reservations, c.skips, c.tokens = {}, {}, {}
    c.manifest = SimpleNamespace(pools=[SimpleNamespace(layer_name=n, tag=f"t.{n}") for n in names])
    c.src_groups = [
        SimpleNamespace(
            page_count=lambda n, k=st.owner_kind(name): (
                -(-n // rtest.SRC_BLOCK) if k in ("main", "compressed") else 1
            )
        )
        for name in names
    ]
    c._src_owner = {n: i for i, n in enumerate(names)}
    return c


_SCATTER = jax.jit(st.make_shared_scatter(write=rtest.scatter))
_REPACK = {}


def repack_fn(r):
    """The rank's routed repack program, jitted once (fixed chunk shapes)."""
    if r not in _REPACK:
        _REPACK[r] = jax.jit(rtest.make_fn(rtest.rank_slice(r), True))
    return _REPACK[r]


class CpuLanding:
    """``SharedLanding.run`` on CPU: the all-gather is the concatenation of
    every rank's packed staging pages; the scatter is the real program with
    the CPU writer."""

    def __init__(self, stagings, holder):
        self.stagings, self.holder = stagings, holder
        self.fn = _SCATTER

    def run(self, passes):
        for p in passes:
            gathered = np.concatenate([s[p.pack] for s in self.stagings])
            self.holder["pool"] = self.fn(
                self.holder["pool"], jnp.asarray(gathered),
                jnp.asarray(p.scatter_src, jnp.int32), jnp.asarray(p.scatter_dst, jnp.int32),
            )


def shared_landing(num_tokens, skip, prod, tables, shared=True):
    """Every rank: staging counts and clips from the consumer, the producer's
    spans through Raiden's rules, then ``plan_repack`` / ``run_repack``."""
    cons = [consumer(r, shared) for r in range(TP)]
    _, names, counts = cons[0]._owner_counts(num_tokens, skip)
    for c in cons:
        assert c._owner_counts(num_tokens, skip)[2] == counts  # rank-independent
    pages, cur = {}, 1
    for name, k in zip(names, counts):
        pages[name] = list(range(cur, cur + k))
        cur += k
    for c in cons:
        c.reservations[UUID], c.skips[UUID], c.tokens[UUID] = pages, (skip, None), num_tokens
    clips = cons[0].skip_bytes(UUID)
    assert all(c.skip_bytes(UUID) == clips for c in cons)
    stagings = []
    for r in range(TP):
        staging = np.zeros((cur, GEO.stage_rows, 1, rtest.ROWS, rtest.LANES), np.uint8)
        flat = staging.reshape(-1, rtest.ROWS, rtest.LANES)
        for name, clip in zip(names, clips):
            kind, n_src = st.owner_kind(name), len(prod[name])
            if shared and kind == "compressed" and SH.shared(num_tokens):
                spans = psc.shared_spans(num_tokens, n_src, SH, RB)
            else:
                spans = rt.producer_spans(kind, num_tokens, n_src, GEO)
            rtest.apply_spans(spans, sum(s[3] for s in spans), prod[name], r, pages[name], clip, flat)
        stagings.append(staging)
    pools = []
    for r, c in enumerate(cons):
        holder = {"pool": jnp.zeros((400, rtest.DST_BLOCK, 1, rtest.ROWS, rtest.LANES), jnp.uint8)}
        c.landing = CpuLanding(stagings, holder)
        fn, staged = repack_fn(r), jnp.asarray(stagings[r])

        def run_chunks(chunks, fn=fn, staged=staged, holder=holder):
            for chunk in chunks:
                holder["pool"] = fn(holder["pool"], staged, *(jnp.asarray(v, jnp.int32) for v in chunk))

        c._run_chunks = run_chunks
        plan = c.plan_repack(UUID, tables, num_tokens)
        assert bool(plan.passes) == (shared and SH.shared(num_tokens))
        c.run_repack(plan)
        pools.append(np.asarray(holder["pool"]))
    return pools, counts


CASES = [(1, 0), (48, 0), (49, 0), (127, 0), (320, 0), (321, 0), (640, 0), (1000, 0), (2049, 0),
         (600, 256), (1000, 768), (2049, 512), (2049, 1792)]


@pytest.mark.parametrize("num_tokens,skip", CASES)
def test_shared_landing_is_byte_identical_to_the_broadcast_repack(num_tokens, skip):
    rng = np.random.default_rng(num_tokens + skip)
    prod, tables = rtest.producer_pages(num_tokens, rng), rtest.dst_tables(num_tokens, rng)
    pools, counts = shared_landing(num_tokens, skip, prod, tables)
    for r in range(TP):
        want = rtest.broadcast_landing(num_tokens, skip, rtest.rank_slice(r), prod, tables)
        np.testing.assert_array_equal(pools[r], want, err_msg=f"rank {r}, {num_tokens}, skip {skip}")
        assert want.any()
    if SH.shared(num_tokens):
        # A rank stages ~1/TP of the compressed pages it stages broadcast.
        routed = rt.staging_pages("compressed", num_tokens, -(-num_tokens // rtest.SRC_BLOCK), skip, GEO)
        comp = counts[rtest.owners().index("l0.self_attn.compressed_k")]
        assert comp == SH.staged_pages(num_tokens, skip) <= routed


def test_flag_off_keeps_the_routed_path():
    rng = np.random.default_rng(3)
    prod, tables = rtest.producer_pages(1000, rng), rtest.dst_tables(1000, rng)
    pools, counts = shared_landing(1000, 0, prod, tables, shared=False)
    _, routed_counts = rtest.routed_landing(1000, 0, rtest.rank_slice(0), prod, tables)
    assert counts == [routed_counts[n] for n in rtest.owners()]
    np.testing.assert_array_equal(
        pools[0], rtest.broadcast_landing(1000, 0, rtest.rank_slice(0), prod, tables)
    )


# --- the producer: spans, sub-pages, fingerprint -------------------------------

pc = pytest.importorskip("vllm_torchtpu.models.vllm.qwen4_exp.pd_connector")


def producer(route, shared=True):
    w = object.__new__(pc.Qwen4ExpRaidenWorker)
    w._qwen_route = route
    w._qwen_shared = psc.geometry_for(route) if shared else None
    return w


def test_producer_routes_shared_compressed_and_leaves_the_rest():
    w = producer(GEO)
    names = rtest.owners()
    n = 1000
    n_src = -(-n // rtest.SRC_BLOCK)
    plan = SimpleNamespace(pool_groups=lambda: [
        (SimpleNamespace(tag=f"t.{nm}", layer_name=nm), i) for i, nm in enumerate(names)
    ])
    pages = tuple(tuple(range(10 * i, 10 * i + (n_src if st.owner_kind(nm) in ("main", "compressed") else 1)))
                  for i, nm in enumerate(names))
    regs = w._routed_spans(plan, pages, n, None, GEO.pairs_per_src_page * RB)
    for reg, name, ids in zip(regs, names, pages):
        kind = st.owner_kind(name)
        want = (psc.shared_spans(n, n_src, SH, RB) if kind == "compressed"
                else rt.producer_spans(kind, n, len(ids), GEO))
        got = [(s.src_block_ordinal, s.src_offset_bytes, s.dst_offset_bytes, s.size_bytes,
                s.dst_unit_ordinal) for s in reg.spans]
        assert got == want, name
        assert reg.declared_bytes == sum(s.size_bytes for s in reg.spans)
        assert reg.dst_space_version == 1


def test_shared_spans_with_subpages_land_the_same_bytes_as_without():
    """Stage C: a sub-page producer lists each shared page once and reads each
    block at its slot; the shared runs never leave a producer block, so the
    same bytes land on every rank. Producer blocks of 3 pair rows against
    2-row staging pages: global pages straddle blocks, as 400 vs 256 do."""
    if not hasattr(pc, "_subpage_address"):
        pytest.skip("stage C (compressed-key sub-pages) is not in this tree")
    slots = 8
    geo = rt.RouteGeometry(tp=4, dcp=2, dst_block=128, src_block=24, ratio=4,
                           ssm_rows_src=8, conv_rows_src=8, ssm_rows_dst=2, stage_rows=2)
    sh = psc.geometry_for(geo)
    slot = geo.pairs_per_src_page * RB  # 3 rows: one 24-token block's pairs
    page = slots * slot
    num_tokens = 4 * geo.src_block
    assert sh.shared(num_tokens)
    ids_v = [9 * slots + 2, 9 * slots + 3, 4 * slots + 7, 9 * slots + 0]
    carrier = np.random.default_rng(5).integers(1, 255, (16, page), dtype=np.uint8)
    blocks = [carrier[v // slots, (v % slots) * slot:(v % slots + 1) * slot] for v in ids_v]
    pool = SimpleNamespace(tag="t.c", layer_name="qsa.compressed_k", block_stride_bytes=page)
    plan = SimpleNamespace(pool_groups=lambda: [(pool, 0)])
    lay = SimpleNamespace(compressed_slots=1, block_size=geo.src_block, kernel_block_size=geo.src_block,
                          compressed_kernel_bytes=slot, page_bytes=page, compressed_subpages=slots)
    lookup = lambda g, ids: tuple(ids_v[: len(ids)])  # noqa: E731
    addr, _ = pc._subpage_address(lay, lookup)
    w = producer(geo)
    (sub,) = w._routed_spans(plan, ((11, 12, 13, 14),), num_tokens, lookup, slot, addr)
    (whole,) = w._routed_spans(plan, ((11, 12, 13, 14),), num_tokens, None, slot, None)
    assert list(sub.block_ids) == [9, 4]

    def land(reg, read):
        out = {}
        for s in reg.spans:
            buf = out.setdefault(s.dst_unit_ordinal, bytearray(sh.unit_pages(num_tokens) * sh.page_rows * RB))
            buf[s.dst_offset_bytes:s.dst_offset_bytes + s.size_bytes] = read(s)
        return out

    got = land(sub, lambda s: carrier[sub.block_ids[s.src_block_ordinal],
                                      s.src_offset_bytes:s.src_offset_bytes + s.size_bytes].tobytes())
    want = land(whole, lambda s: blocks[s.src_block_ordinal][s.src_offset_bytes:
                                                             s.src_offset_bytes + s.size_bytes].tobytes())
    assert sorted(got) == list(range(4)) and got == want
    assert sub.declared_bytes == whole.declared_bytes


def test_fingerprint_pins_the_flag():
    layout = dict(torch_tpu="x", libtpu="y", schema="s",
                  carriers=[{"shape_per_kernel_page": [256, 1, 4, 256], "tiles": [[4, 128], [4, 1]]}],
                  pages=[{"block_stride_bytes": 262144, "dtype_tag": "qwen4-whole-fp8-page-v1"}])
    on = pc._routed_layout(layout, GEO, shared=True)
    assert on == pc._routed_layout(layout, GEO, shared=True)
    assert on != pc._routed_layout(layout, GEO)
    assert pc._routed_layout(layout, GEO) == pc._routed_layout(layout, GEO, shared=False)


def test_producer_needs_routing(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_SHARED_COMPRESSED", "1")
    with pytest.raises(ValueError, match="QWEN4_EXP_PD_ROUTED_DST"):
        pc.Qwen4ExpRaidenWorker._producer_shared(None)
    assert vars(pc.Qwen4ExpRaidenWorker._producer_shared(GEO)) == vars(SH)
    monkeypatch.setenv("QWEN4_EXP_PD_SHARED_COMPRESSED", "0")
    assert pc.Qwen4ExpRaidenWorker._producer_shared(GEO) is None


def test_flag_stays_out_of_the_compile_key():
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert "QWEN4_EXP_PD_SHARED_COMPRESSED" in _TPU_COMPILE_ENV_IGNORED


# --- the scheduler's naming guard ------------------------------------------------


class Req:
    def __init__(self, req_id, status):
        from vllm.v1.request import RequestStatus

        self.request_id, self.status, self._s = req_id, status, RequestStatus

    def is_finished(self):
        return self._s.is_finished(self.status)


class FakeScheduler:
    def update_connector_output(self, out):
        pass

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

        return TPUConnectorMetadata()


def scheduler(shared=True):
    c = object.__new__(pc.Qwen4ExpRaidenConnector)
    c.connector_scheduler = FakeScheduler()
    c.connector_worker = None
    c._sync_repack = True
    c._landed_on_all_ranks, c._failed_on_some_rank = [], []
    c._shared_compressed = shared
    c._load_requests = {}
    return c


def out(finished=(), failed=()):
    return SimpleNamespace(finished_sending=set(), finished_recving=set(finished),
                           failed_recving=set(failed))


@pytest.mark.parametrize("shared", [True, False])
def test_scheduler_names_only_loads_still_waiting(shared):
    from vllm.v1.request import RequestStatus

    c = scheduler(shared)
    a = Req("a", RequestStatus.WAITING_FOR_REMOTE_KVS)
    b = Req("b", RequestStatus.WAITING_FOR_REMOTE_KVS)
    for r in (a, b):
        c.update_state_after_alloc(r, None, 4096)
    b.status = RequestStatus.FINISHED_ABORTED  # aborted while landing
    c.update_connector_output(out(finished={"a", "b", "c"}))  # c: no record
    # The guard acts when the landing is reported, not only at build, and
    # under L1 alone too (shared=False): repacking b would write into blocks
    # vLLM already freed (lead, 21:15Z).
    assert c._landed_on_all_ranks == ["a", "c"] and c._failed_on_some_rank == ["b"]
    meta = c.build_connector_meta(None)
    assert meta.qwen_repack_req_ids == ["a", "c"] and meta.qwen_drop_req_ids == ["b"]
    assert c._load_requests == {}


def test_without_l1_nothing_is_named_or_recorded():
    from vllm.v1.request import RequestStatus

    c = scheduler(shared=False)
    c._sync_repack = False
    c.update_state_after_alloc(Req("a", RequestStatus.WAITING_FOR_REMOTE_KVS), None, 4096)
    assert c._load_requests == {}
    c.update_connector_output(out(finished={"a"}))
    assert c._landed_on_all_ranks == [] and c._failed_on_some_rank == []


def test_guard_works_on_a_connector_built_without_load_records():
    """L1's own tests assemble the connector with object.__new__ and no
    _load_requests: an unrecorded load is named as before."""
    c = scheduler(shared=False)
    del c._load_requests
    c.update_connector_output(out(finished={"a"}))
    assert c.build_connector_meta(None).qwen_repack_req_ids == ["a"]


def test_scheduler_drops_a_name_whose_request_finished_before_the_metadata():
    from vllm.v1.request import RequestStatus

    c = scheduler()
    a = Req("a", RequestStatus.WAITING_FOR_REMOTE_KVS)
    c.update_state_after_alloc(a, None, 4096)
    c.update_connector_output(out(finished={"a"}))
    a.status = RequestStatus.FINISHED_ABORTED  # between naming and the next schedule
    meta = c.build_connector_meta(None)
    assert meta.qwen_repack_req_ids == [] and meta.qwen_drop_req_ids == ["a"]


def test_scheduler_failed_load_is_dropped_and_forgotten():
    from vllm.v1.request import RequestStatus

    c = scheduler()
    c.update_state_after_alloc(Req("a", RequestStatus.WAITING_FOR_REMOTE_KVS), None, 4096)
    c.update_connector_output(out(finished={"a"}, failed={"a"}))
    meta = c.build_connector_meta(None)
    assert meta.qwen_repack_req_ids == [] and meta.qwen_drop_req_ids == ["a"]
    assert c._load_requests == {}


# --- the workers: agreement, plan before collectives ------------------------------


class RecordingStaging:
    shared = object()

    def __init__(self):
        self.events = []

    def plan_repack(self, uuid, tables, num_tokens):
        self.events.append(("plan", uuid))
        return uuid

    def run_repack(self, plan):
        self.events.append(("run", plan))

    def release(self, uuid):
        self.events.append(("release", uuid))


def test_every_load_is_planned_before_the_first_landing():
    w = object.__new__(pc.Qwen4ExpRaidenWorker)
    w._staging = RecordingStaging()
    w._sync_on = lambda: True
    w.tp_rank = 0
    w._tp_cpu_group = lambda: None
    w._pending_repacks = {"a": (1, [[1]], 4096), "b": (2, [[2]], 4096)}
    w.repack_landed(["a", "b"], drop=())
    assert w._staging.events == [("plan", 1), ("plan", 2), ("run", 1), ("release", 1),
                                 ("run", 2), ("release", 2)]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_rank_missing_a_load_makes_every_rank_drop_it_without_hanging():
    """4 processes, one gloo group; rank 1 no longer holds load b. Each
    landing runs an all-reduce standing in for the device all-gather, so a
    disagreement would hang the ranks holding b (killed after 120 s)."""
    world, port, missing = 4, _free_port(), 1
    helper = _HERE / "qwen4_exp_pd_shared_agree_helper.py"
    procs = [
        subprocess.Popen([sys.executable, str(helper), str(r), str(world), str(port), str(missing)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ))
        for r in range(world)
    ]
    results = {}
    try:
        for p in procs:
            stdout, stderr = p.communicate(timeout=120)
            assert p.returncode == 0, stderr[-2000:]
            line = [x for x in stdout.splitlines() if x.startswith("{")][-1]
            res = json.loads(line)
            results[res["rank"]] = res
    except subprocess.TimeoutExpired:
        pytest.fail("a rank hung in the landing collective")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    for r in range(world):
        runs = [u for e, u in results[r]["events"] if e == "run"]
        assert runs == [1, 3], (r, results[r])
        assert results[r]["left"] == []
        released = [u for e, u in results[r]["events"] if e == "release"]
        assert sorted(released) == ([1, 3] if r == missing else [1, 2, 3])


def test_admission_snapshot_restores_load_tokens():
    import torch

    w = object.__new__(pc.Qwen4ExpRaidenWorker)
    w._staging = SimpleNamespace(free=torch.ones(4, dtype=torch.int32), reservations={1: {}},
                                 skips={1: (0, None)}, tokens={1: 4096})
    saved = w._stage3_admission_snapshot()
    w._staging.reservations[2] = {}
    w._staging.tokens[2] = 9000
    w._stage3_admission_restore(saved)
    assert w._staging.tokens == {1: 4096} and set(w._staging.reservations) == {1}

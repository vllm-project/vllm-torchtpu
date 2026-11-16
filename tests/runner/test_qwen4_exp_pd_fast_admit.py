# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_FAST_ADMIT on CPU: the staged decode's admission and landing
host work gives the same pages, rows and chunks as the reference path.

Two consumers are assembled alike, one with the flag and one without, at the
served Qwen3.8-Flash geometry (routed TP8 / DCP4 / 512-token blocks, 3,200-
token producer pages, 12 full-attention layers with packed compressed keys,
36 GDN layers, one PLE owner) and at the small geometry of
test_qwen4_exp_pd_route. Each check runs the same calls on both: owner
counts and pages_needed (first sight and re-examined), reserve and release
sequences on fragmented free maps with the pass's reduced map, parked loads
re-examined across passes, the repack's index arrays and chunks (prefix
skips, short loads, a draft gap, shared compressed keys, ragged tables), and
the packed index upload.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_DIR = pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu/models/vllm/qwen4_exp"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"fast_admit_test_{name}", _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load("pd_staging")
rt = st.pd_route
psc = st.pd_shared_compressed

SERVED = dict(
    geo=rt.RouteGeometry(tp=8, dcp=4, dst_block=512, src_block=3200, ratio=4,
                         ssm_rows_src=3072, conv_rows_src=128, ssm_rows_dst=384),
    fa=[3 + 4 * i for i in range(12)], layers=48, kv_heads=2, slots=4, pages=65536,
    shape=(16384, 16384, 36), state_entries=4,
)
SMALL = dict(
    geo=rt.RouteGeometry(tp=4, dcp=2, dst_block=128, src_block=320, ratio=4,
                         ssm_rows_src=8, conv_rows_src=8, ssm_rows_dst=2, stage_rows=2),
    fa=[1, 3], layers=5, kv_heads=2, slots=2, pages=1 << 15,
    shape=(256, 256, 4), state_entries=2,
)


def owners(spec, mtp=False):
    names = []
    for i in range(spec["layers"]):
        if i in spec["fa"]:
            p = f"model.layers.{i}.self_attn"
            names += [p, p + ".compressed_k", p + ".raw_ring"]
        else:
            names.append(f"model.layers.{i}.linear_attn")
    names.append("model.ple")
    if mtp:
        p = "mtp.layers.0.self_attn"
        names += [p, p + ".compressed_k", p + ".raw_ring"]
    return names


def consumer(spec, fast, rank=0, mtp=False, shared=False, free=None, pairs=None,
             split_keys=False, null_rows=False, shape=None):
    """``pairs``: QWEN4_EXP_PACK_MAIN_KV main-KV owners -> members (W4);
    ``split_keys``: compressed keys DCP-split (VLLM_TPU_QSA_COMPRESSED_DCP, W3);
    ``null_rows``: spread chunk padding; ``shape``: QWEN4_EXP_PD_REPACK_SHAPE."""
    g = spec["geo"]
    c = object.__new__(st.StagedConsumer)
    c._fast_admit = fast
    c.route, c.ratio, c.dst_block, c.src_block = g, g.ratio, g.dst_block, g.src_block
    c.rank = st.RankSlice(tp_size=g.tp, tp_rank=rank, num_kv_heads=spec["kv_heads"], main_dcp=g.dcp)
    names = owners(spec, mtp)
    c.src_groups = [
        SimpleNamespace(page_count=lambda n, k=st.owner_kind(name): (
            0 if n == 0 else -(-n // g.src_block) if k in ("main", "compressed") else 1
        ))
        for name in names
    ]
    c._src_owner = {n: i for i, n in enumerate(names)}
    c.manifest = SimpleNamespace(pools=[SimpleNamespace(layer_name=n, tag=f"t{i}") for i, n in enumerate(names)])
    c.shared = psc.geometry_for(g) if shared else None
    c.landing = None
    if free is None:
        free = torch.ones(spec["pages"], dtype=torch.int32)
        free[0] = 0
    c.free = free.clone()
    c.reservations, c.skips, c.tokens = {}, {}, {}
    dst_owner, dst_compressed, gi = {}, {}, 0
    for n in names:
        if st.owner_kind(n) != "compressed":
            dst_owner[n] = gi
            gi += 1
    for j, n in enumerate(x for x in names if st.owner_kind(x) == "compressed"):
        if j % spec["slots"] == 0:
            owner_gi = gi
            gi += 1
        dst_compressed[n] = (owner_gi, (j % spec["slots"]) * (g.dst_block // (2 * g.ratio)))
    if pairs:
        # A paired member has no destination group of its own.
        away = {m for members in pairs.values() for m in members[1:]}
        dst_owner = {n: gi for n, gi in dst_owner.items() if n not in away}
        c._main_pairs = dict(pairs)
    if split_keys:
        c._dcp_owners = frozenset(dst_compressed)
    if null_rows:
        c._null_rows = (g.stage_rows, g.dst_block)
    c._dst_owner, c._dst_compressed = dst_owner, dst_compressed
    c._gdn_names = [n for n in dst_owner if st.owner_kind(n) == "gdn"]
    c._state_entry = 0
    c._repack_shape = shape or spec["shape"]
    c._split_gdn = 0 < len(c._gdn_names) <= c._repack_shape[2]
    c._zero_draft = {}
    c.n_dst_groups = gi
    return c


def dst_tables(c, spec, num_tokens, seed=0, ragged=False):
    g = spec["geo"]
    kinds = {gi: st.owner_kind(n) for n, gi in c._dst_owner.items()}
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, 1_000_000))
    out, cur = [], 0
    for gi in range(c.n_dst_groups):
        kind = kinds.get(gi, "compressed")
        if kind == "compressed":
            k = -(-num_tokens // g.dst_block)
        elif kind == "main":
            k = -(-num_tokens // (g.dcp * g.dst_block)) + (gi % 3 if ragged else 0)
        else:
            k = spec["state_entries"]
        out.append([int(x) for x in ids[cur:cur + k]])
        cur += k
    return out


def fragmented(spec, seed):
    rng = np.random.default_rng(seed)
    free = torch.from_numpy((rng.random(spec["pages"]) < 0.7).astype(np.int32))
    free[0] = 0
    return free


def _same_counts(a, b):
    assert list(a[0]) == list(b[0])
    assert list(a[1]) == list(b[1])
    assert list(a[2]) == list(b[2])


def _entry(g):
    return g.dcp * g.dst_block


def quantum(g):
    """The prefix step a routed load accepts: whole table entries whose main
    and compressed rows fill whole staging pages."""
    return int(np.lcm.reduce([_entry(g), 2 * g.ratio * g.stage_rows, g.dcp * g.stage_rows]))


def loads(spec):
    """(tokens, prefix) cases: full, prefix-aware, short."""
    g = spec["geo"]
    q = quantum(g)
    big = 95_000 if spec is SERVED else 2049
    out = [(big, 0), (big, (big // q - 1) * q), (big // 2 + 1, q), (1, 0), (g.dst_block - 1, 0),
           (g.dst_block + 1, 0), (g.src_block + 1, 0), (3 * q + 7, 2 * q)]
    if spec is SERVED:
        out += [(83_488, 79_872), (193_377, 0), (193_377, 180_224), (16_385, 0)]
    return out


GEOS = [pytest.param(SERVED, id="served"), pytest.param(SMALL, id="small")]


@pytest.fixture(autouse=True)
def _no_quarantine(monkeypatch):
    """Released pages return at once unless a test sets its own steps."""
    monkeypatch.setattr(st, "quarantine_steps", lambda: 0)


@pytest.mark.parametrize("spec", GEOS)
def test_owner_counts_and_pages_needed_match(spec):
    ref, fast = consumer(spec, False), consumer(spec, True)
    for n, skip in loads(spec):
        want = ref._owner_counts(n, skip)
        for _ in range(3):  # first sight, then the memo
            _same_counts(fast._owner_counts(n, skip), want)
            assert fast.pages_needed(99, n, skip_tokens=skip) == ref.pages_needed(99, n, skip_tokens=skip)
    n, skip = loads(spec)[1]
    for c in (ref, fast):
        assert c.reserve(77, n, skip_tokens=skip)
        assert c.pages_needed(77, n, skip_tokens=skip) == 0
    # An invalid prefix raises the same error, every time (never memoized).
    g = spec["geo"]
    bad = _entry(g) // 2
    with pytest.raises(ValueError) as e_ref:
        ref.pages_needed(1, 10 * _entry(g), skip_tokens=bad)
    for _ in range(2):
        with pytest.raises(ValueError) as e_fast:
            fast.pages_needed(1, 10 * _entry(g), skip_tokens=bad)
        assert str(e_fast.value) == str(e_ref.value)


def _state(c):
    return (
        c.free.clone(), {u: {k: list(v) for k, v in p.items()} for u, p in c.reservations.items()},
        {u: (s, list(f) if f is not None else None) for u, (s, f) in c.skips.items()},
        dict(c.tokens), [[e[0], list(e[1])] for e in c.__dict__.get("quarantine", [])],
    )


def _assert_same_state(a, b):
    sa, sb = _state(a), _state(b)
    assert torch.equal(sa[0], sb[0])
    assert sa[1:] == sb[1:]


@pytest.mark.parametrize("spec", GEOS)
@pytest.mark.parametrize("steps", [0, 3])
def test_reserve_and_release_sequences_match(spec, steps, monkeypatch):
    monkeypatch.setattr(st, "quarantine_steps", lambda: steps)
    free = fragmented(spec, 1)
    ref, fast = consumer(spec, False, free=free), consumer(spec, True, free=free)
    rng = np.random.default_rng(2)
    cases = loads(spec)
    uuid, held = 100, []
    for step in range(40):
        n, skip = cases[step % len(cases)]
        uuid += 1
        # The pass's reduced map: this rank's map with another rank's holes.
        holes = torch.from_numpy((rng.random(spec["pages"]) < 0.05).astype(np.int32))
        red_ref = ref.free.clone() * (1 - holes)
        red_fast = red_ref.clone()
        if step % 5 == 4:  # the per-load path (no reduced map, no TP group)
            ok_ref = ref.reserve(uuid, n, skip_tokens=skip)
            ok_fast = fast.reserve(uuid, n, skip_tokens=skip)
        else:
            ok_ref = ref.reserve(uuid, n, skip_tokens=skip, reduced_free=red_ref)
            ok_fast = fast.reserve(uuid, n, skip_tokens=skip, reduced_free=red_fast)
            assert torch.equal(red_ref, red_fast)
        assert ok_ref == ok_fast
        if ok_ref:
            held.append(uuid)
            # A second reserve of a held load is a no-op on both.
            assert fast.reserve(uuid, n, skip_tokens=skip) and ref.reserve(uuid, n, skip_tokens=skip)
        _assert_same_state(ref, fast)
        if held and step % 3 == 2:
            victim = held.pop(int(rng.integers(len(held))))
            assert ref.release(victim) == fast.release(victim)
        assert ref.advance_quarantine() == fast.advance_quarantine()
        _assert_same_state(ref, fast)
    # Exhaust the map: the no-room answer matches too.
    n, skip = cases[0]
    for u in range(10_000, 10_200):
        a = ref.reserve(u, n, skip_tokens=skip, reduced_free=ref.free.clone())
        b = fast.reserve(u, n, skip_tokens=skip, reduced_free=fast.free.clone())
        assert a == b
        if not a:
            break
    else:
        pytest.fail("the staging map never filled")
    _assert_same_state(ref, fast)


@pytest.mark.parametrize("spec", GEOS)
def test_parked_loads_reexamined_across_passes(spec, monkeypatch):
    """A pass examines parked loads first (oldest first), then new ones,
    with the S2 check (pages_needed against the reduced map's free count)
    before reserve; releases between passes let some fit later."""
    monkeypatch.setattr(st, "quarantine_steps", lambda: 1)
    free = torch.zeros(spec["pages"], dtype=torch.int32)
    free[1 : spec["pages"] // 3] = 1
    ref, fast = consumer(spec, False, free=free), consumer(spec, True, free=free)
    cases = loads(spec)
    parked, uuid, admitted = [], 500, []
    decisions = {"ref": [], "fast": []}
    for pass_no in range(30):
        new = []
        for _ in range(2):
            uuid += 1
            new.append((uuid,) + cases[uuid % len(cases)])
        candidates = parked + new
        red = {"ref": ref.free.clone(), "fast": fast.free.clone()}
        still = []
        for u, n, skip in candidates:
            out = []
            for tag, c in (("ref", ref), ("fast", fast)):
                available = int(torch.count_nonzero(red[tag]))
                if c.pages_needed(u, n, skip_tokens=skip) > available:
                    out.append("no_room")
                elif c.reserve(u, n, skip_tokens=skip, reduced_free=red[tag]):
                    out.append("reserved")
                else:
                    out.append("no_room")
            decisions["ref"].append(out[0])
            decisions["fast"].append(out[1])
            assert out[0] == out[1], (pass_no, u, n, skip)
            if out[0] == "reserved":
                admitted.append(u)
            else:
                still.append((u, n, skip))
        parked = still
        _assert_same_state(ref, fast)
        if admitted and pass_no % 2:
            u = admitted.pop(0)
            ref.release(u)
            fast.release(u)
        ref.advance_quarantine()
        fast.advance_quarantine()
    assert "no_room" in decisions["ref"] and "reserved" in decisions["ref"]


def _plans_equal(ref, fast, uuid, tables, n):
    want = ref.plan_repack(uuid, tables, n)
    got = fast.plan_repack(uuid, tables, n)
    assert len(want.chunks) == len(got.chunks)
    for w, gch in zip(want.chunks, got.chunks):
        gch = st.unpacked_chunk(gch)
        assert len(w) == len(gch) == 6
        for a, b in zip(w, gch):
            if a is None:
                assert b is None
                continue
            np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
            assert np.shape(a) == np.shape(b)
    assert ref.last_draft_rows == fast.last_draft_rows


def _indices_equal(ref, fast, pages, tables, n, skip, zero_until=0):
    want = ref.indices(pages, tables, n, skip, draft_zero_until=zero_until)
    got = fast.indices(pages, tables, n, skip, draft_zero_until=zero_until)
    for a, b in zip(want, got):
        assert a.dtype == b.dtype == np.int64
        np.testing.assert_array_equal(b, a)
    assert ref.last_draft_rows == fast.last_draft_rows


@pytest.mark.parametrize("spec", GEOS)
def test_repack_indices_and_chunks_match_on_every_rank(spec):
    g = spec["geo"]
    for rank in range(g.tp):
        ref, fast = consumer(spec, False, rank), consumer(spec, True, rank)
        for k, (n, skip) in enumerate(loads(spec)):
            tables = dst_tables(ref, spec, n, seed=k)
            uuid = 1000 + k
            assert ref.reserve(uuid, n, skip_tokens=skip)
            assert fast.reserve(uuid, n, skip_tokens=skip)
            assert ref.reservations[uuid] == fast.reservations[uuid]
            _indices_equal(ref, fast, ref.reservations[uuid], tables, n, skip)
            _plans_equal(ref, fast, uuid, tables, n)
            ref.release(uuid)
            fast.release(uuid)
        if spec is SERVED and rank == 1:
            break  # the served geometry: ranks 0 and 1 (two DCP slices, both KV heads)


@pytest.mark.parametrize("spec", GEOS)
def test_ragged_tables_take_the_per_layer_path(spec):
    ref, fast = consumer(spec, False, 1), consumer(spec, True, 1)
    n, skip = loads(spec)[2]
    tables = dst_tables(ref, spec, n, seed=3, ragged=True)
    assert ref.reserve(5, n, skip_tokens=skip) and fast.reserve(5, n, skip_tokens=skip)
    _indices_equal(ref, fast, ref.reservations[5], tables, n, skip)


@pytest.mark.parametrize("spec", GEOS)
def test_draft_gap_rows_match(spec):
    """MTP draft owners with a zero-filled gap (q38 F2 option (a))."""
    g = spec["geo"]
    for rank in (0, g.tp - 1):
        ref = consumer(spec, False, rank, mtp=True)
        fast = consumer(spec, True, rank, mtp=True)
        for c in (ref, fast):
            c.draft_kv = True
        for k, (n, skip) in enumerate(loads(spec)):
            tables = dst_tables(ref, spec, n, seed=k)
            assert ref.reserve(7, n, skip_tokens=skip) and fast.reserve(7, n, skip_tokens=skip)
            pages = ref.reservations[7]
            for zero_until in (0, skip, (skip + n) // 2, n):
                _indices_equal(ref, fast, pages, tables, n, skip, zero_until)
            ref.release(7)
            fast.release(7)


@pytest.mark.parametrize("spec", GEOS)
def test_shared_compressed_rows_match(spec):
    g = spec["geo"]
    q = quantum(g)
    n = 95_000 if spec is SERVED else 2049
    for rank in (0, 2):
        ref = consumer(spec, False, rank, shared=True)
        fast = consumer(spec, True, rank, shared=True)
        assert ref._shared_load(n)
        for skip in (0, 2 * q, (n // q - 1) * q):
            tables = dst_tables(ref, spec, n, seed=skip)
            _same_counts(fast._owner_counts(n, skip), ref._owner_counts(n, skip))
            assert ref.reserve(9, n, skip_tokens=skip) and fast.reserve(9, n, skip_tokens=skip)
            assert ref.reservations[9] == fast.reservations[9]
            _indices_equal(ref, fast, ref.reservations[9], tables, n, skip)
            ref.release(9)
            fast.release(9)


def test_pack_and_unpack_index_round_trip():
    rng = np.random.default_rng(0)
    full = (rng.integers(0, 1 << 30, 64), rng.integers(0, 1 << 30, 64), rng.integers(0, 9, 32),
            rng.integers(0, 9, 32), rng.integers(0, 99, (4, 2)), rng.integers(0, 99, 4))
    for lists in (full, full[:4]):
        host, shapes = st.pack_index(lists)
        assert host.dtype == np.int32 and host.ndim == 1
        args = st.unpack_index(torch.from_numpy(host), shapes)
        assert len(args) == 6
        for a, b in zip(args, lists):
            np.testing.assert_array_equal(a.numpy(), b)
        if len(lists) == 4:
            assert [tuple(a.shape) for a in args[4:]] == [(1,), (1,)]


@pytest.mark.parametrize("split", [False, True])
def test_packed_chunks_are_repack_chunks(split):
    rng = np.random.default_rng(1)
    for main, copy, gdn in ((0, 0, 0), (5, 70, 3), (64, 64, 4), (130, 1, 2), (300, 129, 4)):
        idx = (rng.integers(0, 1 << 20, main), rng.integers(0, 1 << 20, main),
               rng.integers(0, 1 << 20, copy), rng.integers(0, 1 << 20, copy),
               rng.integers(1, 99, (gdn, 2)), rng.integers(1, 99, gdn))
        want = list(st.repack_chunks(*idx, 64, 4, copy_size=128, split_gdn=split))
        got = list(st.repack_chunks_packed(*idx, 64, 4, copy_size=128, split_gdn=split))
        assert len(want) == len(got)
        for w, gch in zip(want, got):
            u = st.unpacked_chunk(gch)
            for a, b in zip(w, u):
                if a is None:
                    assert b is None
                else:
                    np.testing.assert_array_equal(b, a)
                    assert np.shape(a) == np.shape(b)


def _recording_consumer(fast):
    c = consumer(SMALL, fast)
    c.pool = SimpleNamespace(device="cpu")
    c.staging = None
    c._fn = c._fn_rest = "fn"
    c._op = c._op_rest = None
    calls = []

    def build(fn, name, packed=None):
        def op(pool, staging, *args):
            if packed is not None:
                (index,) = args
                calls.append(("packed", name, [a.clone() for a in st.unpack_index(index, packed)]))
            else:
                calls.append(("lists", name, [a.clone() for a in args]))
        return op

    c._build_op = build
    return c, calls


def test_packed_upload_sends_the_same_arguments_once_per_chunk():
    ref, ref_calls = _recording_consumer(False)
    fast, fast_calls = _recording_consumer(True)
    n, skip = loads(SMALL)[0]
    tables = dst_tables(ref, SMALL, n)
    for c in (ref, fast):
        assert c.reserve(3, n, skip_tokens=skip)
        c.run_repack(c.plan_repack(3, tables, n))
    assert len(ref_calls) == len(fast_calls) > 1
    assert all(kind == "packed" for kind, _, _ in fast_calls)
    for (_, name, want), (_, _, got) in zip(ref_calls, fast_calls):
        # The GDN-free program never reads its two GDN arguments.
        n_read = 4 if name.endswith("_no_gdn") else 6
        for a, b in zip(want[:n_read], got[:n_read]):
            assert torch.equal(a, b)


def test_packed_upload_falls_back_to_per_list_uploads():
    ref, ref_calls = _recording_consumer(False)
    fast, fast_calls = _recording_consumer(True)
    build = fast._build_op

    def failing(fn, name, packed=None):
        if packed is not None and len(packed) == 4:
            raise RuntimeError("no packed program")
        return build(fn, name, packed)

    fast._build_op = failing
    n, skip = loads(SMALL)[0]
    tables = dst_tables(ref, SMALL, n)
    for c in (ref, fast):
        assert c.reserve(3, n, skip_tokens=skip)
        c.run_repack(c.plan_repack(3, tables, n))
    assert [k for k, _, _ in fast_calls] == ["packed"] + ["lists"] * (len(ref_calls) - 1)
    assert fast._packed_ok is False
    for (_, _, want), (_, _, got) in zip(ref_calls[1:], fast_calls[1:]):
        for a, b in zip(want, got):
            assert torch.equal(a, b)


# --- W3 (DCP-split compressed keys) + W4 (paired main KV) at 8192:8192:36 ----------


def main_pairs_of(spec, paired):
    """``paired`` consecutive full-attention layers paired in twos; the rest
    keep one layer a row (the paired layout's odd owners)."""
    fa = [f"model.layers.{i}.self_attn" for i in spec["fa"]]
    return {fa[k]: (fa[k], fa[k + 1]) for k in range(0, paired, 2)}


PACKED = [
    pytest.param(SERVED, 10, (8192, 8192, 36), id="served-8192"),
    pytest.param(SMALL, 2, (256, 256, 4), id="small"),
]


def _packed_pair(spec, paired, shape, fast, rank):
    return consumer(spec, fast, rank, pairs=main_pairs_of(spec, paired), split_keys=True,
                    null_rows=True, shape=shape)


def _packed_loads(spec):
    if spec is SERVED:
        # Full (many programs), prefix-aware with one and with two programs,
        # short, and a 193k-token full load.
        return [(95_000, 0), (83_488, 79_872), (95_000, 90_112), (16_385, 0), (513, 0),
                (193_377, 180_224)]
    q = quantum(spec["geo"])
    return [(2049, 0), (2049, 1792), (1025, q), (321, 0), (1, 0)]


@pytest.mark.parametrize("spec,paired,shape", PACKED)
def test_split_keys_and_paired_main_match_the_reference(spec, paired, shape):
    """FAST_ADMIT=1 against FAST_ADMIT=0 with paired main KV, split compressed
    keys and spread padding: the same reservations, index arrays ([rows, 2]
    main sources), chunks and program count, with and without a prefix."""
    g = spec["geo"]
    programs = set()
    for rank in ((0, 1, 6) if spec is SERVED else range(g.tp)):
        ref = _packed_pair(spec, paired, shape, False, rank)
        fast = _packed_pair(spec, paired, shape, True, rank)
        for k, (n, skip) in enumerate(_packed_loads(spec)):
            uuid = 3000 + k
            _same_counts(fast._owner_counts(n, skip), ref._owner_counts(n, skip))
            assert ref.reserve(uuid, n, skip_tokens=skip) and fast.reserve(uuid, n, skip_tokens=skip)
            assert ref.reservations[uuid] == fast.reservations[uuid]
            tables = dst_tables(ref, spec, n, seed=k)
            pages = ref.reservations[uuid]
            want = ref.indices(pages, tables, n, skip)
            got = fast.indices(pages, tables, n, skip)
            assert want[0].ndim == got[0].ndim == 2
            for a, b in zip(want, got):
                assert a.dtype == b.dtype == np.int64 and a.shape == b.shape
                np.testing.assert_array_equal(b, a)
            _plans_equal(ref, fast, uuid, tables, n)
            n_prog = len(fast.plan_repack(uuid, tables, n).chunks)
            assert n_prog == len(ref.plan_repack(uuid, tables, n).chunks)
            programs.add(n_prog)
            ref.release(uuid)
            fast.release(uuid)
    if spec is SERVED:
        assert 1 in programs and max(programs) > 2, programs  # single and chunked programs


def _recording_packed(spec, paired, shape, fast):
    c = _packed_pair(spec, paired, shape, fast, 1)
    c.pool = SimpleNamespace(device="cpu")
    c.staging = None
    c._fn = c._fn_rest = "fn"
    c._op = c._op_rest = None
    calls = []

    def build(fn, name, packed=None):
        def op(pool, staging, *args):
            if packed is not None:
                (index,) = args
                calls.append(("packed", name, [a.clone() for a in st.unpack_index(index, packed)]))
            else:
                calls.append(("lists", name, [a.clone() for a in args]))
        return op

    c._build_op = build
    return c, calls


@pytest.mark.parametrize("spec,paired,shape", PACKED)
def test_packed_upload_carries_the_paired_main_source(spec, paired, shape):
    """One host-to-device copy per program, unpacked to the same arguments the
    per-list uploads pass, main source [rows, 2]; warm and served chunks share
    one shape, so warm compiles the programs loads run."""
    n, skip = _packed_loads(spec)[0]
    for case in (0, 1):
        if case:
            n, skip = _packed_loads(spec)[1]
        ref, ref_calls = _recording_packed(spec, paired, shape, False)
        fast, fast_calls = _recording_packed(spec, paired, shape, True)
        tables = dst_tables(ref, spec, n)
        for c in (ref, fast):
            assert c.reserve(3, n, skip_tokens=skip)
            c.run_repack(c.plan_repack(3, tables, n))
        assert len(ref_calls) == len(fast_calls) >= 1
        assert all(kind == "packed" for kind, _, _ in fast_calls)
        for (_, name, want), (_, _, got) in zip(ref_calls, fast_calls):
            assert tuple(got[0].shape) == (shape[0], 2) == tuple(want[0].shape)
            n_read = 4 if name.endswith("_no_gdn") else 6
            for a, b in zip(want[:n_read], got[:n_read]):
                assert a.shape == b.shape and torch.equal(a, b)
    # warm_repack's null chunks take the same packed shapes as a load's chunks.
    fast, calls = _recording_packed(spec, paired, shape, True)
    fast.warm_repack()
    warm_shapes = set(fast._packed_ops)
    assert any(s[0] == (shape[0], 2) for s in warm_shapes)
    assert fast.reserve(4, n, skip_tokens=skip)
    fast.run_repack(fast.plan_repack(4, dst_tables(fast, spec, n), n))
    assert set(fast._packed_ops) == warm_shapes

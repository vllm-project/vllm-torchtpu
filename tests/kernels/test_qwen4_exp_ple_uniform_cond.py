# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_UNIFORM_COND (q38 PLE-UNIFORM-COND, layer U on the R form; port of L1's D07-b2
``uniform_window_conv``, credit L1): a token-path bucket with T % window == 0 chooses ON THE
DEVICE (``lax.cond``) between the uniform-window path and the stock equations; the pool bytes and
the output bits are the stock op's either way, with the checkpoint writer stock (base) or the
VLLM_TPU_PLE_WRDEFER ring (W, WU).

Legs:
- bitwise (TPU interpreter, on_wait DMAs, value-level typed load / store in every arm; each case in
  a subprocess): arms base (flags unset), W (WRDEFER 4) and WU (WRDEFER 4 + UNIFORM_COND) at
  T 128 / 160 / 192 / 256 x count 40 / 48 / 64 (C 256) and T 160 / count 48 at C 10240, on uniform
  metadata (pred True: the uniform branch must run, counted by a debug callback inside it) and
  non-uniform metadata (pred False: the stock branch runs); request-path and T % 4 != 0 buckets
  fall back statically (no cond in the jaxpr). Some cases also run the literal rw3 tree
  (Q38_RPLEW_BASE_SRC) and must give the same bytes.
- pred: ``uniform_window_pred`` on every metadata case against the expected set, and against an
  independent numpy oracle on random metadata.
- engagement: trace lines, one cond in the jaxpr with the flag and none without, no
  ple_history_rows call in the uniform branch, the uniform branch reads n history rows.
"""

import functools
import importlib.util
import json
import logging
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

HERE = os.path.abspath(__file__)
_spec = importlib.util.spec_from_file_location(
    "q38_rplew_w_helpers", os.path.join(os.path.dirname(HERE), "test_qwen4_exp_ple_wrdefer.py")
)
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

FLAG_U = "VLLM_TPU_PLE_UNIFORM_COND"
FLAG_W = "VLLM_TPU_PLE_WRDEFER"
HRK = "VLLM_TPU_PLE_HISTORY_ROW_KERNEL"
SKIP = "VLLM_TPU_PLE_SKIP_IDLE_ROWS"
WINDOW, STATE_LEN, DILATION = 4, 9, 3
SUB_TIMEOUT_S = 1200
ARMS = {"base": {}, "W": {FLAG_W: "4"}, "WU": {FLAG_W: "4", FLAG_U: "1"}}


def _clear(monkeypatch):
    for name in (FLAG_U, FLAG_W, HRK, SKIP, "VLLM_KV_CACHE_LAYOUT"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------- metadata cases
def metadata(kind, T, count):
    """(lengths, windowed, initial, ids_on, expected pred) for one metadata kind at T / count.
    expected None = the bucket falls back statically (no cond)."""
    n = T // WINDOW
    fallback = T % WINDOW != 0 or n > count
    full = min(n, count)
    if kind == "U-full":  # L = n, a few idle ids (pred ignores ids)
        lengths = [WINDOW] * full
        ids_on = [i % 7 != 5 for i in range(full)]
        windowed, initial = None, None
    elif kind == "U-tail":  # L < n, empty tail
        lengths = [WINDOW] * (full - 3)
        ids_on, windowed, initial = None, None, None
    elif kind == "U-initmix":  # initial alternating
        lengths = [WINDOW] * full
        ids_on, windowed = None, None
        initial = [i % 2 == 0 for i in range(full)]
    elif kind == "U-nonwin":  # windowed and non-windowed requests of length 4
        lengths = [WINDOW] * full
        ids_on, initial = None, None
        windowed = [i % 3 != 1 for i in range(full)]
    elif kind == "N-one1":  # one request of 1 token, live rows after it
        lengths = [WINDOW, 1] + [WINDOW] * (full - 3)
        ids_on, windowed, initial = None, None, None
    elif kind == "N-one2":  # one request of 2 tokens
        lengths = [WINDOW, WINDOW, 2] + [WINDOW] * (full - 4)
        ids_on, windowed, initial = None, None, None
    elif kind == "N-gap":  # a live row after an empty row
        lengths = [WINDOW, 0] + [WINDOW] * (full - 2)
        ids_on = [True, False] + [True] * (full - 2)
        windowed, initial = None, None
    elif kind == "N-tailrem":  # full rows, then two short rows past L = starts[-1] // window
        lengths = [WINDOW] * (full - 2) + [1, 1]
        ids_on, windowed, initial = None, None, None
    else:
        raise ValueError(kind)
    if fallback:
        expected = None
    else:
        expected = kind.startswith("U-")
    return lengths, windowed, initial, ids_on, expected


KINDS = ["U-full", "U-tail", "U-initmix", "U-nonwin", "N-one1", "N-one2", "N-gap", "N-tailrem"]
SHAPES = [(t, c, 256) for t in (128, 160, 192, 256) for c in (40, 48, 64)] + [(160, 48, 10240)]
FALLBACK_SHAPES = [(162, 48, 256), (130, 40, 256)]  # T % 4 != 0


def _case_kw(kind, T, count, channels, seed):
    lengths, windowed, initial, ids_on, expected = metadata(kind, T, count)
    kw = dict(seed=seed, count=count, tokens=T, lengths=lengths, channels=channels)
    if windowed is not None:
        kw["windowed"] = windowed
    if initial is not None:
        kw["initial"] = initial
    if ids_on is not None:
        kw["ids_on"] = ids_on
    return kw, expected


# ---------------------------------------------------------------------------- subprocess child
_CHILD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("q38_rplew_u_tests", sys.argv[1])
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
print("Q38_RPLEW_U_CHILD " + json.dumps(T._child_main(json.loads(sys.argv[2]))), flush=True)
"""


def _sub(base_tree=False, **kw):
    env = dict(os.environ)
    for name in (FLAG_U, FLAG_W, HRK, SKIP, "VLLM_KV_CACHE_LAYOUT"):
        env.pop(name, None)
    want_origin = os.path.realpath(pool_mod.__file__)
    if base_tree:
        base = os.environ.get("Q38_RPLEW_BASE_SRC")
        assert base, "Q38_RPLEW_BASE_SRC (the rw3 tree's src/) is not set"
        env["PYTHONPATH"] = base + os.pathsep + env.get("PYTHONPATH", "")
        want_origin = os.path.realpath(os.path.join(base, "vllm_torchtpu/kernels/qwen4_exp/pool.py"))
    try:
        r = subprocess.run([sys.executable, "-c", _CHILD, HERE, json.dumps(kw)], capture_output=True,
                           text=True, timeout=SUB_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"TPU-interpreter run did not finish in {SUB_TIMEOUT_S} s") from None
    lines = [x for x in r.stdout.splitlines() if x.startswith("Q38_RPLEW_U_CHILD ")]
    assert lines, f"child failed (rc {r.returncode}): {(r.stdout + r.stderr)[-3000:]}"
    out = json.loads(lines[-1].split(" ", 1)[1])
    print("Q38 RPLEW U %s%s: %s" % ("BASE-TREE " if base_tree else "", {k: v for k, v in kw.items() if k != "case"}
                                    | {"case": kw["case"]["tokens"]}, out))
    assert out["origin"] == want_origin, (out["origin"], want_origin)
    return out


def _child_main(a):
    import hashlib

    import vllm_torchtpu.kernels.qwen4_exp.pool as P

    W._interpret_on_wait()
    out = {"origin": os.path.realpath(P.__file__), "arms": {}}
    ran = []
    if hasattr(P, "uniform_window_conv"):
        real = P.uniform_window_conv

        def counted(*args, **kw):
            res = real(*args, **kw)
            jax.debug.callback(lambda: ran.append(1))
            return res

        P.uniform_window_conv = counted
    if a.get("hrk"):
        os.environ[HRK] = "1"
    case = W._conv_case(**a["case"])
    args = W._conv_args(case, "spec")
    fn = functools.partial(P.pooled_short_conv_speculative, dilation=DILATION, state_len=STATE_LEN)
    before = np.asarray(case["pool"]).view(np.uint8)
    for arm in a["arms"]:
        for name in (FLAG_U, FLAG_W):
            os.environ.pop(name, None)
        os.environ.update(ARMS[arm])
        n0 = len(ran)
        jaxpr = jax.make_jaxpr(lambda *x: fn(*x))(*args)
        conds = sum(1 for e in _walk(jaxpr.jaxpr) if e.primitive.name == "cond")
        pool1, out1 = W._fresh_jit(fn)(*args)
        pb, ob = W._bytes(pool1), W._bits16(out1)
        out["arms"][arm] = {
            "pool_sha": hashlib.sha256(pb.tobytes()).hexdigest()[:16],
            "out_sha": hashlib.sha256(ob.tobytes()).hexdigest()[:16],
            "wrote": int((pb != before).sum()),
            "uniform_ran": len(ran) - n0,
            "conds": conds,
        }
    if hasattr(P, "uniform_window_pred"):
        T = a["case"]["tokens"]
        starts = case["starts"][: min(T, a["case"]["count"]) + 1] if T < a["case"]["count"] else case["starts"]
        out["pred"] = bool(P.uniform_window_pred(starts, T // WINDOW, WINDOW))
    return out


def _walk(jaxpr):
    """Every equation, into nested jaxprs (cond branches, loops, pjit) but not into Pallas kernel
    bodies, whose pl.when blocks are conds of their own."""
    for e in jaxpr.eqns:
        yield e
        if e.primitive.name == "pallas_call":
            continue
        for p in e.params.values():
            for sub in (p if isinstance(p, (list, tuple)) else [p]):
                if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    yield from _walk(sub.jaxpr)
                elif hasattr(sub, "eqns"):
                    yield from _walk(sub)


def _check_bitwise(out, expected, *, all_idle=False):
    arms = out["arms"]
    base = arms["base"]
    if not all_idle:
        assert base["wrote"] > 0, "no checkpoint written (vacuous case)"
    for arm in ("W", "WU"):
        assert arms[arm]["pool_sha"] == base["pool_sha"], f"pool bytes: {arm} vs base"
        assert arms[arm]["out_sha"] == base["out_sha"], f"output bits: {arm} vs base"
    assert base["conds"] == 0 and arms["W"]["conds"] == 0, "a cond without the flag"
    assert base["uniform_ran"] == 0 and arms["W"]["uniform_ran"] == 0
    if expected is None:  # static fallback
        assert arms["WU"]["conds"] == 0, "static fallback bucket traced a cond"
        assert arms["WU"]["uniform_ran"] == 0
    else:
        assert arms["WU"]["conds"] == 1, arms["WU"]["conds"]
        assert out["pred"] is expected, (out["pred"], expected)
        assert arms["WU"]["uniform_ran"] == (1 if expected else 0), "the branch that ran is not pred's"


def _sid(s):
    return f"T{s[0]}-c{s[1]}-C{s[2]}"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES, ids=_sid)
def test_bitwise_base_w_wu(shape, kind):
    T, count, channels = shape
    kw, expected = _case_kw(kind, T, count, channels, seed=T * 7 + count * 3 + KINDS.index(kind))
    _check_bitwise(_sub(case=kw, arms=["base", "W", "WU"], hrk=1), expected)


@pytest.mark.parametrize("kind", ["U-full", "N-one1"])
@pytest.mark.parametrize("shape", FALLBACK_SHAPES, ids=_sid)
def test_bitwise_static_fallback_t_mod_window(shape, kind):
    T, count, channels = shape
    kw, expected = _case_kw(kind, T, count, channels, seed=T + count + KINDS.index(kind))
    assert expected is None
    _check_bitwise(_sub(case=kw, arms=["base", "W", "WU"], hrk=1), None)


@pytest.mark.parametrize("kind", ["U-full", "N-one1", "U-initmix"])
def test_bitwise_without_history_row_kernel(kind):
    kw, expected = _case_kw(kind, 160, 48, 256, seed=900 + KINDS.index(kind))
    _check_bitwise(_sub(case=kw, arms=["base", "W", "WU"], hrk=0), expected)


@pytest.mark.parametrize("kind", ["U-full", "N-one1"])
@pytest.mark.parametrize("shape", SHAPES, ids=_sid)
def test_literal_rw3_tree_matches(shape, kind):
    """The rw3 tree itself (Q38_RPLEW_BASE_SRC) gives the head base arm's pool bytes and output bits,
    which the W and WU arms equal (test_bitwise_base_w_wu)."""
    T, count, channels = shape
    kw, _ = _case_kw(kind, T, count, channels, seed=T * 7 + count * 3 + KINDS.index(kind))
    head = _sub(case=kw, arms=["base", "WU"], hrk=1)
    rw3 = _sub(base_tree=True, case=kw, arms=["base"], hrk=1)
    assert rw3["arms"]["base"]["pool_sha"] == head["arms"]["base"]["pool_sha"] == head["arms"]["WU"]["pool_sha"]
    assert rw3["arms"]["base"]["out_sha"] == head["arms"]["base"]["out_sha"] == head["arms"]["WU"]["out_sha"]


def test_shapes_cover_cond_and_fallback():
    cond = [s for s in SHAPES if s[0] // WINDOW <= s[1]]
    req = [s for s in SHAPES if s[0] // WINDOW > s[1]]
    print("Q38 RPLEW U shapes: cond %s request-path fallback %s" % (cond, req))
    assert len(cond) == 10 and sorted(req) == [(192, 40, 256), (256, 40, 256), (256, 48, 256)]


# ---------------------------------------------------------------------------- pred units
def _starts(lengths):
    return jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))


def _oracle(lengths, n, window):
    """Independent statement: every row with tokens is one of the leading rows, each of them holds
    exactly ``window`` tokens, and there are at most n of them."""
    lengths = list(lengths)
    nz = [i for i, x in enumerate(lengths) if x > 0]
    lead = len(nz)
    return nz == list(range(lead)) and all(lengths[i] == window for i in nz) and lead <= n


PRED_EDGE = [
    ("all-empty", [0] * 8, 16, True),
    ("exact-n", [4] * 4, 16, True),
    ("short-last", [4, 4, 1, 0], 16, False),
    ("tail-1-1", [4, 4, 1, 1], 16, False),
    ("lead-zero", [0, 4, 4, 0], 16, False),
    ("lead-short", [3, 4, 4, 0], 16, False),
    ("long-row", [8, 0, 4, 0], 16, False),
    ("five-three", [5, 3, 4, 0], 16, False),
    ("over-n", [4] * 5, 16, False),  # starts[-1] > T (a contract violation): L = 5 > n = 4
    ("one-row", [4, 0, 0, 0], 16, True),
]


@pytest.mark.parametrize("name,lengths,T,want", PRED_EDGE, ids=[e[0] for e in PRED_EDGE])
def test_pred_edge_cases(name, lengths, T, want):
    got = bool(pool_mod.uniform_window_pred(_starts(lengths), T // WINDOW, WINDOW))
    assert got is want, (name, lengths, got)
    assert _oracle(lengths, T // WINDOW, WINDOW) is want


@pytest.mark.parametrize("shape", SHAPES, ids=_sid)
def test_pred_on_every_metadata_case(shape):
    T, count, _ = shape
    n = T // WINDOW
    got_set, want_set = {}, {}
    for kind in KINDS:
        lengths, _, _, _, expected = metadata(kind, T, count)
        if expected is None:  # static fallback: pred is not evaluated by the op
            continue
        cut = lengths + [0] * (min(T, count) - len(lengths)) if T < count else lengths + [0] * (count - len(lengths))
        got_set[kind] = bool(pool_mod.uniform_window_pred(_starts(cut), n, WINDOW))
        want_set[kind] = expected
        assert _oracle(cut, n, WINDOW) is expected, (kind, "oracle disagrees with the expected set")
    print("Q38 RPLEW U pred %s: %s" % (_sid(shape), got_set))
    assert got_set == want_set


def test_pred_matches_oracle_on_random_metadata():
    rng = np.random.default_rng(7)
    seen = {True: 0, False: 0}
    for _ in range(3000):
        rows = int(rng.integers(1, 20))
        n = int(rng.integers(1, 24))
        mode = rng.integers(0, 3)
        if mode == 0:  # uniform-looking with random perturbations
            live = int(rng.integers(0, rows + 1))
            lengths = [WINDOW] * live + [0] * (rows - live)
            for _ in range(int(rng.integers(0, 3))):
                lengths[int(rng.integers(0, rows))] = int(rng.integers(0, 6))
        elif mode == 1:
            lengths = list(rng.integers(0, 6, size=rows))
        else:
            live = int(rng.integers(0, rows + 1))
            lengths = [WINDOW] * live + [0] * (rows - live)
        want = _oracle(lengths, n, WINDOW)
        got = bool(pool_mod.uniform_window_pred(_starts(lengths), n, WINDOW))
        assert got is want, (lengths, n, got, want)
        seen[want] += 1
    print("Q38 RPLEW U pred random:", seen)
    assert seen[True] > 300 and seen[False] > 300


# ---------------------------------------------------------------------------- engagement
def _capture():
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    names = ("vllm.vllm_torchtpu.kernels.qwen4_exp.pool", "vllm_torchtpu.kernels.qwen4_exp.pool")
    loggers = [logging.getLogger(n) for n in names]
    h = _Keep(level=logging.INFO)
    olds = []
    for log in loggers:
        log.addHandler(h)
        olds.append(log.level)
        log.setLevel(logging.INFO)

    def done():
        for log, old in zip(loggers, olds):
            log.removeHandler(h)
            log.setLevel(old)
        return records

    return done


def _jaxpr(monkeypatch, T, count, flags, kind="U-full", channels=256):
    _clear(monkeypatch)
    for k, v in flags.items():
        monkeypatch.setenv(k, v)
    kw, _ = _case_kw(kind, T, count, channels, seed=1)
    case = W._conv_case(**kw)
    fn = functools.partial(pool_mod.pooled_short_conv_speculative, dilation=DILATION, state_len=STATE_LEN)
    return jax.make_jaxpr(lambda *a: fn(*a))(*W._conv_args(case, "spec"))


def _conds(jaxpr):
    return [e for e in _walk(jaxpr.jaxpr) if e.primitive.name == "cond"]


def _pallas_names(jaxpr):
    return [str(e.params.get("name_and_src_info", e.params.get("name", "")))
            for e in _walk(jaxpr) if e.primitive.name == "pallas_call"]


@pytest.mark.parametrize("flags,want", [({}, 0), ({FLAG_W: "4"}, 0), ({FLAG_U: "1"}, 1), ({FLAG_U: "1", FLAG_W: "4"}, 1),
                                        ({FLAG_U: "0"}, 0)], ids=["unset", "W", "U", "WU", "U0"])
def test_cond_only_with_the_flag(monkeypatch, flags, want):
    jaxpr = _jaxpr(monkeypatch, 160, 48, {HRK: "1", **flags})
    print("Q38 RPLEW U cond count flags=%s: %d" % (flags, len(_conds(jaxpr))))
    assert len(_conds(jaxpr)) == want


@pytest.mark.parametrize("T,count", [(130, 48), (192, 40), (256, 40)])
def test_no_cond_on_static_fallback(monkeypatch, T, count):
    jaxpr = _jaxpr(monkeypatch, T, count, {HRK: "1", FLAG_U: "1"}, kind="N-one1")
    assert _conds(jaxpr) == []


def test_no_cond_with_skip_idle_rows(monkeypatch):
    jaxpr = _jaxpr(monkeypatch, 160, 48, {HRK: "1", FLAG_U: "1", SKIP: "1"})
    assert _conds(jaxpr) == []


@pytest.mark.parametrize("T,count", [(128, 40), (160, 40), (192, 48), (256, 64)])
def test_uniform_branch_has_no_history_row_kernel(monkeypatch, T, count):
    """branches[1] (pred True) = the uniform path: no ple_history_rows call, and its history read
    covers n = T // 4 rows; branches[0] (pred False) = the stock path, with the history-row kernel."""
    jaxpr = _jaxpr(monkeypatch, T, count, {HRK: "1", FLAG_U: "1", FLAG_W: "4"})
    (cond,) = _conds(jaxpr)
    stock, uniform = cond.params["branches"]
    un, sn = _pallas_names(uniform.jaxpr), _pallas_names(stock.jaxpr)
    print("Q38 RPLEW U branches T%d c%d: uniform %s stock %s" % (T, count, un, sn))
    assert not any("ple_history_rows" in x for x in un)
    assert sum("ple_history_rows" in x for x in sn) >= 1
    reads = [e for e in _walk(uniform.jaxpr) if e.primitive.name == "pallas_call"]
    grids = [tuple(e.params["grid_mapping"].grid) for e in reads]
    print("Q38 RPLEW U uniform-branch pallas grids T%d: %s" % (T, grids))
    assert (T // WINDOW,) in grids
    top = [n for n in _pallas_names(jaxpr.jaxpr) if "q38_ple_wrdefer" in n]
    assert len(top) == 1, "one writer, outside the cond"


def test_engagement_lines(monkeypatch):
    monkeypatch.setattr(pool_mod, "_UNIFORM_SEEN", set())
    done = _capture()
    try:
        _jaxpr(monkeypatch, 160, 48, {HRK: "1", FLAG_U: "1"})
        _jaxpr(monkeypatch, 160, 48, {HRK: "1", FLAG_U: "1"})
        _jaxpr(monkeypatch, 130, 48, {HRK: "1", FLAG_U: "1"}, kind="N-one1")
        _jaxpr(monkeypatch, 160, 48, {HRK: "1"})
    finally:
        records = done()
    lines = [m for m in records if "PLE-UNIFORM-COND" in m]
    print("Q38 RPLEW U engagement:", lines)
    assert lines == [
        "q38 PLE-UNIFORM-COND traced: 160 tokens, device-side cond between the uniform path "
        "(40 of 48 request rows, window 4, static slices) and the stock path",
        "q38 PLE-UNIFORM-COND static fallback: 130 tokens, 48 request rows, window 4 (T % window != 0); "
        "stock path, no cond",
    ], records


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_flag_reads(monkeypatch, value, expected):
    from vllm_torchtpu import envs

    envs.__dict__.pop(FLAG_U, None)
    monkeypatch.delenv(FLAG_U, raising=False)
    if value is not None:
        monkeypatch.setenv(FLAG_U, value)
    assert pool_mod.uniform_cond_enabled() is expected


def test_flag_is_a_compile_key(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    envs.__dict__.pop(FLAG_U, None)
    assert FLAG_U in envs.environment_variables
    assert FLAG_U not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv(FLAG_U, raising=False)
    unset = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG_U, "0")
    zero = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG_U, "1")
    one = tpu_compiler._tpu_compile_env_factors()
    assert unset == zero
    assert one != unset and {k for k in one if one[k] != unset.get(k)} == {FLAG_U}


def test_bounded_request_rows():
    assert pool_mod.bounded_request_rows(160, 48, 4) == 40
    assert pool_mod.bounded_request_rows(256, 48, 4) == 48
    assert pool_mod.bounded_request_rows(130, 48, 4) == 33

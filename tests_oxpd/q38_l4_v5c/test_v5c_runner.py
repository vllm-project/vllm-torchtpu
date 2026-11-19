# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c), runner side: VFY-ONE value 2 with VPSHOST's sampsnap and statsasync on,
against the stock D83 chain with the same levers. CDK CPU pod only. V5COMP-BUILD (helper of L4-E).

Driven through VFY-ONE's own harness (tests/l4vfy/test_vfy_one.py: the runner's methods extracted with ast into a
torch-CPU harness, the compiled bodies run eagerly, a fake TP1 vocab shard). The levers are switched through a
fake l6_ab in the harness namespace (the runner reads `_l6_ab.on(lever)`); the snapshot and the async counts are
the tree's own runner/l4_vpshost.py (a private copy whose device event is a fake with a programmable lag).

R1  8 steps at 8 / 32 / 48 / 64 live rows, fallback rows 0 and 8: the VFY step's tokens, random stream,
    fallback stats, read-offset buffer, fold mark and chunk counts equal the stock chain's with the same levers,
    and the levers-off stock chain's (the levers change no value); the VPS_FALLBACK lines are logged on the same
    steps; sampsnap serves the VFY step's parameter and num_active uploads exactly as often as the stock step's
    (786f7bac2); statsasync issues and drains the same copies on the same steps.
R2  statsasync across path switches (stock -> VFY -> stock ..., and VFY -> stock -> ...), copy lag 0 and 1: a
    copy issued on one path is drained on the next step whichever path it takes; per-step lines and drained
    copies equal an all-stock runner's; no copy is left un-drained beyond the lag.
R3  window switching after a real VFY warm-up (`_vfy_precompile`): alternating stock / VFY steps over every steady
    serving shape, levers on: no VFY step runs cold (no compile after warm-up), lines and stats equal all-stock.
R4  the VFY copies carry the stock copies' lever branches statement for statement (786f7bac2 sites).
Run: python -m pytest -q -p no:cacheprovider --noconftest -s tests/q38_l4_v5c/test_v5c_runner.py
"""

import ast
import collections
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38mm_dstock"))
sys.path.insert(0, os.path.join(HERE, "..", "l4vfy"))
import test_vfy_one as tv  # noqa: E402  (VFY-ONE's harness: NS, M, Runner, problem, run_step)
import vfy_harness as H  # noqa: E402

NS, VF, K, V = tv.NS, tv.VF, tv.K, tv.V
L4PATH = os.path.join(tv.w.PKG, "runner", "l4_vpshost.py")
COUNTS = collections.Counter()
LEVERS_ON = frozenset({"sampsnap", "statsasync"})


class FakeL6:
    """The runner's `_l6_ab` (harness namespace) and l4_vpshost's: `on` from a lever set, notes counted."""

    def __init__(self):
        self.levers = set()
        self.counts = collections.Counter()

    def on(self, lever):
        return lever in self.levers

    def note(self, lever, key, n=1):
        self.counts[f"{lever}.{key}"] += n

    @staticmethod
    def check_allowed():
        return True


L6 = FakeL6()
_core = types.ModuleType("vllm_torchtpu.core")
_core.__path__ = []
_core.l6_ab = L6
L4 = H._load("q38_v5c_l4_vpshost", L4PATH, {"vllm_torchtpu.core": _core})


class Events:
    """l4_vpshost._event: each copy's event completes after `lag` unready queries."""

    def __init__(self, lag=0):
        self.lag, self.made = lag, []

    def __call__(self):
        ev = types.SimpleNamespace(left=self.lag, queries=0)

        def query():
            ev.queries += 1
            if ev.left <= 0:
                return True
            ev.left -= 1
            return False

        ev.query = query
        ev.synchronize = lambda: setattr(ev, "left", 0)
        self.made.append(ev)
        return ev


class CountingSnap(L4.SampSnap):
    def __init__(self):
        super().__init__()
        self.calls = collections.Counter()

    def params(self, *a, **k):
        self.calls["params"] += 1
        return super().params(*a, **k)

    def num_active(self, *a, **k):
        self.calls["num_active"] += 1
        return super().num_active(*a, **k)


class CountingAsync(L4.AsyncCounts):
    def __init__(self):
        super().__init__()
        self.issued, self.drained = [], []

    def issue(self, tag, dev):
        self.issued.append(tag)
        super().issue(tag, dev)

    def ready(self, wait=False):
        out = super().ready(wait)
        self.drained += [t for t, _ in out]
        return out


class Batch:
    """The InputBatch fields the snapshot keys and checks (the parameters equal the harness Runner's)."""

    def __init__(self, B):
        self.req_ids = [f"req{i}" for i in range(B)]
        self.temperature_cpu_tensor = torch.full((B,), 0.6)
        self.top_k_cpu_tensor = torch.full((B,), 20, dtype=torch.int32)
        self.top_p_cpu_tensor = torch.full((B,), 0.95)

    def add_request(self, *a, **k):
        return None


class Lines:
    """The harness logger: VPS_FALLBACK lines recorded per runner tag."""

    def __init__(self):
        self.tag, self.rec = None, collections.defaultdict(list)

    def info(self, fmt, *a):
        msg = fmt % a if a else fmt
        if msg.startswith("VPS_FALLBACK"):
            self.rec[self.tag].append(msg)

    def warning(self, *a, **k):
        return None

    warning_once = error = debug = warning


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    """VFY-ONE's harness modules installed, the fake l6_ab / logger in the namespace, stats every 2 chunks."""
    with tv.w.HARNESS.active():
        pkg = sys.modules["vllm_torchtpu.runner"]
        had = hasattr(pkg, "vfy_one")
        pkg.vfy_one = VF
        with H.modules({"vllm_torchtpu.runner.vfy_one": VF}):
            saved = {k: NS[k] for k in ("_l6_ab", "logger")}
            lines = Lines()
            NS["_l6_ab"], NS["logger"] = L6, lines
            L6.levers, L6.counts = set(), collections.Counter()
            monkeypatch.setenv("VLLM_TPU_VPS_STATS_EVERY", "2")
            monkeypatch.setattr(L4, "_event", Events(0))
            try:
                yield lines
            finally:
                NS.update(saved)
        if not had:
            del pkg.vfy_one


def make(B, *, mode, cand, seed, flagged=True):
    r = tv.Runner(B, cand=cand, mode=mode, seed=seed)
    if flagged:
        r.top_k[1] = r.top_k[5] = V  # VPS cannot cover these two rows: non-zero fallback stats (tv._pair)
    if mode:
        r._vfy.logits_dtype = torch.bfloat16
        r._vfy.warm = tv.AlwaysWarm()
    r._samp_snap = CountingSnap()
    r._vps_fb_async = CountingAsync()
    r.input_batch = Batch(B)
    return r


def step(lines, tag, r, inputs, R, F, seed, vfy, levers):
    hidden, idx, md, chunk, groups = inputs
    L6.levers = set(levers)
    lines.tag = tag
    n0 = len(lines.rec[tag])
    out, gen = tv.run_step(r, hidden, idx, md, groups, R, F, seed, chunk, vfy=vfy)
    r.__dict__.pop("_vfy_seed", None)
    return out, gen, lines.rec[tag][n0:]


def inputs(seed, B, R):
    hidden, idx, md, chunk = tv.problem(seed, B, R)
    return hidden, idx, md, chunk, tv.slot_groups(seed, 3, 64, R)


ROWS = [(8, 8), (32, 32), (48, 64), (64, 64)]  # (live rows R, padded rows B)


@pytest.mark.parametrize("F,cand", [(0, False), (8, True)])
@pytest.mark.parametrize("R,B", ROWS)
def test_r1_vfy_with_levers_equals_the_stock_chain_with_levers_over_8_steps(harness, R, B, F, cand):
    lines = harness
    base = make(B, mode=0, cand=cand, seed=R + F)
    vfy = make(B, mode=2, cand=cand, seed=R + F)
    off = make(B, mode=0, cand=cand, seed=R + F)
    per_step = []
    for i in range(8):
        seed = 1000 * R + 10 * F + i
        x = inputs(seed, B, R)
        o0, g0, l0 = step(lines, "base", base, x, R, F, seed, False, LEVERS_ON)
        o1, g1, l1 = step(lines, "vfy", vfy, inputs(seed, B, R), R, F, seed, True, LEVERS_ON)
        o2, g2, _ = step(lines, "off", off, inputs(seed, B, R), R, F, seed, False, ())
        assert torch.equal(o0, o1) and torch.equal(o0, o2), f"tokens differ at step {i}"
        assert torch.equal(g0, g1) and torch.equal(g0, g2), f"random stream differs at step {i}"
        tv._same_state(base, vfy)
        tv._same_state(base, off)
        assert l0 == l1, (i, l0, l1)  # the VPS_FALLBACK lines on the same steps
        assert len(base._vps_fb_async.pending) == len(vfy._vps_fb_async.pending) <= 1, i
        per_step.append((len(l0), len(l1)))
    assert vfy._vfy.stats["vfy"] == 8 and vfy._vfy.stats["cold"] == 0, dict(vfy._vfy.stats)
    # sampsnap: the VFY step takes the snapshot exactly as the stock step does (786f7bac2)
    assert vfy._samp_snap.calls == base._samp_snap.calls == {"params": 8, "num_active": 8}, (
        dict(vfy._samp_snap.calls), dict(base._samp_snap.calls))
    assert off._samp_snap.calls == {} and not off._vps_fb_async.issued
    # statsasync: the same copies issued and drained on the same steps; lever off logs the same lines, blocking
    assert vfy._vps_fb_async.issued == base._vps_fb_async.issued == [2, 4, 6, 8]
    assert vfy._vps_fb_async.drained == base._vps_fb_async.drained == [2, 4, 6]
    tail_b, tail_v = base._vps_fb_async.ready(wait=True), vfy._vps_fb_async.ready(wait=True)
    assert [c for c, _ in tail_b] == [c for c, _ in tail_v] == [8]
    for t, r in (("base", base), ("vfy", vfy)):
        lines.tag = t
        for chunks, counts in (tail_b if t == "base" else tail_v):
            r._log_vps_fallback(chunks, counts)
    assert lines.rec["base"] == lines.rec["vfy"] == lines.rec["off"], (lines.rec["base"], lines.rec["off"])
    assert len(lines.rec["off"]) == 4 and int(base._vps_fb_stats[0]) > 0  # not vacuous: rows were flagged
    COUNTS["r1_cases"] += 1
    COUNTS["r1_steps"] += 8
    print(f"\nV5C_RUNNER R1 R={R} B={B} F={F} cand={int(cand)} steps=8 tokens/stream/stats/offsets bitwise "
          f"(VFY with sampsnap+statsasync == stock with them == stock levers off); snap {dict(vfy._samp_snap.calls)}; "
          f"async issued {vfy._vps_fb_async.issued} drained {vfy._vps_fb_async.drained}+tail; lines {len(lines.rec['vfy'])}")


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("first", ["stock", "vfy"])
def test_r2_statsasync_copies_drain_across_path_switches(harness, monkeypatch, first, lag):
    lines = harness
    monkeypatch.setenv("VLLM_TPU_VPS_STATS_EVERY", "1")  # a copy on every step
    monkeypatch.setattr(L4, "_event", Events(lag))
    B, R, F = 16, 12, 8
    alt = make(B, mode=2, cand=True, seed=7)
    ref = make(B, mode=0, cand=True, seed=7)
    path = []
    for i in range(8):
        seed = 5100 + i
        # path blocks of lag + 1 steps: the copy issued at step i drains at step i + 1 + lag, on the other path
        use_vfy = ((i // (lag + 1)) % 2 == 0) == (first == "vfy")
        path.append("V" if use_vfy else "S")
        oa, ga, la = step(lines, "alt", alt, inputs(seed, B, R), R, F, seed, use_vfy, LEVERS_ON)
        orf, gr, lr = step(lines, "ref", ref, inputs(seed, B, R), R, F, seed, False, LEVERS_ON)
        assert torch.equal(oa, orf) and torch.equal(ga, gr)
        tv._same_state(alt, ref)
        assert la == lr, (i, path[-1], la, lr)
        assert alt._vps_fb_async.drained == ref._vps_fb_async.drained, (i, alt._vps_fb_async.drained)
        # nothing older than the lag stays in flight: after step i (chunk i + 1) only chunks > i - lag remain
        assert [t for t, *_ in alt._vps_fb_async.pending] == list(range(max(1, i + 1 - lag), i + 2)), (
            i, [t for t, *_ in alt._vps_fb_async.pending])
    assert alt._vps_fb_async.issued == list(range(1, 9)) and alt._vfy.stats["vfy"] == 4
    # chunk c is issued at step c - 1 and drained at step c + lag; with the block pattern every drain is on the
    # other path than its issue
    drained_on = {c: path[c + lag] for c in alt._vps_fb_async.drained if c + lag < 8}
    crossed = sum(1 for c, p in drained_on.items() if p != path[c - 1])
    assert crossed == len(drained_on) >= 6, (path, drained_on)
    assert lines.rec["alt"] == lines.rec["ref"]
    COUNTS["r2_cases"] += 1
    print(f"\nV5C_RUNNER R2 first={first} lag={lag} path={''.join(path)} issued={alt._vps_fb_async.issued} "
          f"drained={alt._vps_fb_async.drained} crossed_paths={crossed} pending_at_end={len(alt._vps_fb_async.pending)} "
          f"lines equal to all-stock ({len(lines.rec['alt'])})")


def test_r3_window_switching_after_warmup_never_runs_cold(harness):
    lines = harness
    r = make(8, mode=2, cand=True, seed=3, flagged=False)
    ref = make(8, mode=0, cand=True, seed=3, flagged=False)
    for x in (r, ref):
        x.max_num_reqs, x.num_tokens_paddings, x.num_reqs_paddings = 20, [16, 32, 64, 128], [8, 16, 32]
    r._vfy.warm = set()
    buf0 = r.mamba_slot_read_offsets.clone()
    r._vfy_precompile()
    assert torch.equal(r.mamba_slot_read_offsets, buf0) and r._vps_fb_stats is None and r._vps_fb_chunks == 0
    v = r._vfy
    assert v.programs == len(v.warm) > 0
    G, widths = r._read_offset_fold_geometry()
    variants = tv.VW.fallback_variants(r._vps_fb_max, r._vps_tiegap)
    n_vfy = n_stock = 0
    for n in range(1, r.max_num_reqs + 1):
        for F in variants:
            for wd in widths:
                if n > wd:
                    continue
                use_vfy = (n + F + wd) % 2 == 0
                L6.levers = set(LEVERS_ON)
                for x in (r, ref):
                    x.input_batch = Batch(n)  # the batch of this step (the snapshot re-keys on it)
                hidden, idx, md, chunk, groups = tv._steady(r, n, F, wd, 100 * n + F + wd)
                lines.tag = "r"
                o1, g1 = tv.run_step(r, hidden, idx, md, groups, n, F, n, chunk, vfy=use_vfy)
                r.__dict__.pop("_vfy_seed", None)
                hidden, idx, md, chunk, groups = tv._steady(ref, n, F, wd, 100 * n + F + wd)
                lines.tag = "ref"
                o0, g0 = tv.run_step(ref, hidden, idx, md, groups, n, F, n, chunk)
                assert torch.equal(o0, o1) and torch.equal(g0, g1), (n, F, wd, use_vfy)
                tv._same_state(ref, r)
                n_vfy += use_vfy
                n_stock += not use_vfy
    assert v.stats["cold"] == 0 and v.stats["vfy"] == n_vfy > 0 and n_stock > 0, dict(v.stats)
    assert lines.rec["r"] == lines.rec["ref"] and r._vps_fb_async.issued == ref._vps_fb_async.issued
    assert r._samp_snap.calls == ref._samp_snap.calls
    COUNTS["r3_vfy_steps"] = n_vfy
    COUNTS["r3_stock_steps"] = n_stock
    print(f"\nV5C_RUNNER R3 warm_programs={v.programs} vfy_steps={n_vfy} stock_steps={n_stock} cold=0 "
          f"(levers on, alternating paths after the warm-up); lines/stats equal all-stock")


def _fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _if_on(fn, lever):
    """The `if` statements of `fn` whose test calls _l6_ab.on(lever), unparsed, with their line numbers."""
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and f"_l6_ab.on('{lever}')" in ast.unparse(n.test):
            out.append((n.lineno, ast.unparse(n)))
    return sorted(out)


def test_r4_vfy_copies_carry_the_stock_branches_statement_for_statement():
    src = open(tv.RUNNER).read()
    tree = ast.parse(src)
    fb, tick = _fn(tree, "_vps_verify_fb"), _fn(tree, "_vfy_fb_tick")
    i0 = next(i for i, s in enumerate(fb.body) if ast.unparse(s) == "self._vps_fb_chunks += 1")
    j0 = next(i for i, s in enumerate(fb.body) if isinstance(s, ast.Return))
    stock_tail = [ast.unparse(s) for s in fb.body[i0:j0]]
    tick_body = [ast.unparse(s) for s in tick.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert tick_body == stock_tail, (tick_body, stock_tail)
    sites = {}
    stock_na = _if_on(fb, "sampsnap")
    assert len(stock_na) == 1
    for name in ("_vfy_chain", "_vfy_verify"):
        got = [s for s in _if_on(_fn(tree, name), "sampsnap") if "num_active" in s[1]]
        assert len(got) == 1 and got[0][1] == stock_na[0][1], (name, got, stock_na)
        sites[f"{name}:num_active"] = got[0][0]
    stock_p = _if_on(_fn(tree, "_sample_spec_verify_vps"), "sampsnap")
    vfy_p = [s for s in _if_on(_fn(tree, "_vfy_verify"), "sampsnap") if "params" in s[1]]
    assert len(stock_p) == len(vfy_p) == 1
    norm = stock_p[0][1].replace("local_logits.dtype", "dtype").replace("local_logits.device", "device")
    assert vfy_p[0][1] == norm, (vfy_p, norm)
    sites["_vfy_verify:params"] = vfy_p[0][0]
    sites["_vfy_fb_tick:statsasync"] = _if_on(tick, "statsasync")[0][0]
    print("\nV5C_RUNNER R4 sites " + " ".join(f"{k}=tpu_runner.py:{v}" for k, v in sorted(sites.items())))
    COUNTS["r4_sites"] = len(sites)


def test_zz_counts():
    print("\nV5C_RUNNER_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))

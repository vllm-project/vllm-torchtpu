# SPDX-License-Identifier: Apache-2.0
"""q38 STALL-HUNT: VLLM_TPU_VFY_WARM_ALIAS (reports/v8_stall_hunt_1116.md). CDK CPU pod only.

v8 T94u (j-64613705, decode pod dvbv5) recompiled its VFY-ONE x VRNG program at the first serving step of every
token bucket (11 compiles, 9.0-9.6 s host steps in vfy_rejection / ~vr_post), although VFY-ONE's own warm-key test
counted 0 cold steps. Serving's `_vfy_verify` passes ONE tensor as the verify's draft_lengths and as the seed's
num_draft (execute_model: DraftChunkInputs.draft_lengths = spec_decode_metadata.draft_lengths); `_vfy_precompile`
passed two distinct zero tensors. runner/vfy_one.key sees shapes only, so the key matched; Dynamo's warm entry is
guarded on no tensor aliasing, so serving missed it and compiled the aliased graph.

Harness: tests/l4vfy/test_vfy_one.py (the runner's real `_vfy_precompile` / `_vfy_verify` / `_sample_spec_verify_vps`
and eagle3's seed methods extracted with ast, stub packages, a fake TP1 vocab shard); `_vfy_one_compiled` is the
real `_vfy_one_impl` under torch.compile(dynamic=False) with a counting CPU backend (graphs run eagerly).

S1 the new-shape path: warm-up, then a steady serving step at a warmed key (vfy_one cold == 0). Flag off: the
   serving step compiles again (the v8 bug). Flag on: 0 compiles, and the step's tokens equal the flag-off run's.
S2 control: flag off with only the serving seed's num_draft replaced by a distinct copy: 0 compiles (in this harness
   the alias is the only warm/serving difference Dynamo sees).
S3 flag-on warm-up calls: every seed call's num_draft IS its own draft_lengths (positions 12 and 27); keys, tensor
   bytes and program count equal the flag-off warm-up; the engagement line counts N of N.
S4 flag-off identity: head's warm-up calls with the flag off (unset and "0") equal base e2511c6df's call for call
   (keys, alias pairs, tensor bytes); the serving-path methods are source-identical to base.
S5 `_vfy_warm_alias` unit: replaces args[0] (and base_args[0] when it is the same object); refuses a shape / dtype
   mismatch or a non-tensor first argument; the VRNG composed call (`_vfy_rng_call`) keeps the alias.
S6 / S7 (real vllm_torchtpu imports, so in tests/q38_stallhunt/test_vfy_warm_alias_real.py): DRAFT-ASM's superset
   seed and the key's registration.
Base tree: env Q38_STALLHUNT_BASE (a checkout of e2511c6df); S4 fails (not skips) without it.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/q38_stallhunt/test_vfy_warm_alias.py
"""

import ast
import dataclasses
import hashlib
import logging
import os
import re
import sys
import textwrap
from typing import Any

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "l4vfy"))
import test_vfy_one as T  # noqa: E402

w, H, VF, NS = T.w, T.H, T.VF, T.NS
NS.setdefault("dataclasses", dataclasses)
NS.setdefault("Any", Any)
w._fns(T.RUNNER, ["_vfy_warm_alias"], NS)
ALIAS = NS["_vfy_warm_alias"]
EXTRA = dict(zip(["_vfy_rng_call"], w._fns(T.RUNNER, ["_vfy_rng_call"], NS, cls="TPUModelRunner")))
for _n, _f in EXTRA.items():
    setattr(T.Runner, _n, _f)
SERVING_METHODS = ["_vfy_verify", "_sample_spec_verify_vps", "_vfy_seed_spec", "_vfy_chain", "_vfy_rng_call",
                   "_vfy_meta_draws", "_apply_folded_read_offsets"]
POS_DL, POS_ND = 12, 27  # call positions of the verify's draft_lengths and the seed's num_draft (see _vfy_verify)


@pytest.fixture(autouse=True, scope="module")
def _harness():
    with w.HARNESS.active():
        pkg = sys.modules["vllm_torchtpu.runner"]
        had = hasattr(pkg, "vfy_one")
        pkg.vfy_one = VF
        with H.modules({"vllm_torchtpu.runner.vfy_one": VF}):
            yield
        if not had:
            del pkg.vfy_one


@pytest.fixture
def flag(monkeypatch):
    def set_(value):
        if value is None:
            monkeypatch.delattr(NS["envs"], "VLLM_TPU_VFY_WARM_ALIAS", raising=False)
        else:
            monkeypatch.setattr(NS["envs"], "VLLM_TPU_VFY_WARM_ALIAS", value, raising=False)
    return set_


def _flat(x, path=()):
    if isinstance(x, (list, tuple)):
        for i, e in enumerate(x):
            yield from _flat(e, path + (i,))
    else:
        yield path, x


def alias_pairs(call) -> tuple:
    """Pairs of positions (nested in lists) holding the same tensor object."""
    seen, pairs = {}, []
    for p, x in _flat(tuple(call)):
        if isinstance(x, torch.Tensor):
            if id(x) in seen:
                pairs.append((seen[id(x)], p))
            else:
                seen[id(x)] = p
    return tuple(pairs)


def digest(call) -> tuple:
    out = []
    for p, x in _flat(tuple(call)[1:]):
        if isinstance(x, torch.Tensor):
            raw = x.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
            out.append((p, str(x.dtype), tuple(x.shape), hashlib.sha1(raw).hexdigest()[:16]))
        elif callable(x):
            out.append((p, "F", getattr(x, "__qualname__", repr(x))))
        else:
            out.append((p, "V", repr(x)))
    return tuple(out)


class Counting:
    """A CPU Dynamo backend that counts compiled graphs and runs them eagerly."""

    def __init__(self):
        self.graphs = 0

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        return gm.forward


def _dynamo_fresh():
    torch._dynamo.reset()
    for name in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 512)
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 4096
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = 4096


def _runner():
    r = T.Runner(8, cand=False, mode=2)
    r.max_num_reqs = 8
    r.num_tokens_paddings = [16, 32]
    r.num_reqs_paddings = [8]
    return r


def _serving_step(r, n=4, F=0, seed=41, break_alias=False):
    hidden, idx, md, chunk, groups = T._steady(r, n, F, r.num_reqs_max_model_len, seed)
    assert chunk.draft_lengths is md.draft_lengths  # execute_model's DraftChunkInputs (tpu_runner.py)
    if break_alias:
        chunk.draft_lengths = md.draft_lengths.clone()
    out, _ = T.run_step(r, hidden, idx, md, groups, n, F, n, chunk, vfy=True)
    r.__dict__.pop("_vfy_seed", None)
    return out


class _RecompileLog(logging.Handler):
    """Dynamo's recompile reasons (torch._logging recompiles artifact), for the record only."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if "ecompil" in msg or "guard" in msg.lower():
            self.lines.append(msg)


def _compiled(cnt, fails):
    """`_vfy_one_impl` under Dynamo with the counting backend, dynamic=False (as `_tpu_compile`), recording guard
    failures through guard_fail_fn where this torch has it."""
    impl = NS["_vfy_one_impl"]
    try:
        return torch._dynamo.optimize(cnt, guard_fail_fn=lambda f: fails.append(str(getattr(f, "reason", f))),
                                      dynamic=False)(impl)
    except TypeError:
        return torch.compile(impl, backend=cnt, dynamic=False)


def _compiled_run(monkeypatch, flag, value, break_alias=False):
    flag(value)
    _dynamo_fresh()
    cnt, fails = Counting(), []
    monkeypatch.setitem(NS, "_vfy_one_compiled", _compiled(cnt, fails))
    r = _runner()
    r._vfy_precompile()
    warm_graphs, warm_fails = cnt.graphs, len(fails)
    rl = _RecompileLog()
    tl = logging.getLogger("torch")
    tl.addHandler(rl)
    try:
        torch._logging.set_logs(recompiles=True)
        out = _serving_step(r, break_alias=break_alias)
    finally:
        torch._logging.set_logs()
        tl.removeHandler(rl)
    v = r._vfy
    assert v.stats["cold"] == 0 and v.stats["vfy"] == 1, dict(v.stats)  # vfy_one's key was warm either way
    for line in fails[warm_fails:]:
        print("S-GUARDFAIL serving:", line.replace("\n", " | ")[:2000])
    for line in rl.lines:
        print("S-RECOMPILE serving:", line.replace("\n", " | ")[:2000])
    return warm_graphs, cnt.graphs - warm_graphs, out, r


# ---------------------------------------------------------------- S1 / S2: the new-shape path under real Dynamo

def test_s1_flag_off_serving_recompiles_a_warm_key(monkeypatch, flag):
    warm, serving, _, r = _compiled_run(monkeypatch, flag, None)
    assert warm >= r._vfy.programs > 0, (warm, r._vfy.programs)
    assert serving >= 1, "flag off: the serving step should miss the warm entry (the v8 bug)"
    print(f"S1 flag-off: warm-up graphs {warm} (programs {r._vfy.programs}), serving graphs {serving}")


def test_s1_flag_on_serving_hits_the_warm_entry(monkeypatch, flag):
    warm_on, serving_on, out_on, r = _compiled_run(monkeypatch, flag, True)
    assert serving_on == 0, f"flag on: the serving step compiled {serving_on} graph(s)"
    _, _, out_off, _ = _compiled_run(monkeypatch, flag, None)
    assert torch.equal(out_on, out_off), "tokens differ between the flag-on and flag-off serving step"
    print(f"S1 flag-on: warm-up graphs {warm_on} (programs {r._vfy.programs}), serving graphs {serving_on}")


def test_s2_control_alias_is_the_only_difference(monkeypatch, flag):
    _, serving, _, _ = _compiled_run(monkeypatch, flag, None, break_alias=True)
    assert serving == 0, f"flag off, seed num_draft a distinct copy: the serving step compiled {serving} graph(s)"


# ---------------------------------------------------------------- S3 / S4: the warm-up calls

def _record(monkeypatch, runner_cls=None):
    calls = []

    def rec(*c):
        calls.append(c)
        return None

    monkeypatch.setitem(NS, "_vfy_one_compiled", rec)
    r = _runner() if runner_cls is None else runner_cls()
    r._vfy_precompile()
    return calls, r


class RecLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *a, **k):
        self.messages.append(msg % a if a else msg)

    warning = error = debug = info

    def warning_once(self, msg, *a, **k):
        self.info(msg, *a)


def test_s3_flag_on_warm_seed_calls_alias_like_serving(monkeypatch, flag):
    flag(None)
    off, r_off = _record(monkeypatch)
    flag(True)
    rl = RecLogger()
    monkeypatch.setitem(NS, "logger", rl)
    on, r_on = _record(monkeypatch)
    assert len(on) == len(off) and r_on._vfy.programs == r_off._vfy.programs > 0
    assert r_on._vfy.warm == r_off._vfy.warm
    seeded = 0
    for a, b in zip(off, on):
        assert VF.key(*a[1:]) == VF.key(*b[1:])
        assert digest(a) == digest(b)
        if len(b) > POS_ND:
            seeded += 1
            assert b[POS_ND] is b[POS_DL], "flag on: the seed's num_draft is not the call's draft_lengths"
            assert alias_pairs(b) == (((POS_DL,), (POS_ND,)),), alias_pairs(b)
            assert alias_pairs(a) == (), alias_pairs(a)
        else:
            assert alias_pairs(a) == alias_pairs(b) == ()
    assert seeded > 0
    lines = [m for m in rl.messages if "VFY_WARM_ALIAS" in m]
    m = re.search(r"VFY_WARM_ALIAS: (\d+) of (\d+) seed warm-up calls", lines[-1] if lines else "")
    assert m and m.group(1) == m.group(2) and int(m.group(1)) >= seeded, lines  # every seed spec built, N of N
    print(f"S3 engagement: {lines[-1]} (compiled seed calls {seeded} of {len(on)} calls)")


def _base_root() -> str:
    root = os.environ.get("Q38_STALLHUNT_BASE", "")
    assert root and os.path.isfile(os.path.join(root, "src", "vllm_torchtpu", "runner", "tpu_runner.py")), \
        "Q38_STALLHUNT_BASE must point at a checkout of the base tree e2511c6df"
    return root


def _method_src(path, cls, name):
    src = open(path).read()
    tree = ast.parse(src)
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(src, node))


@pytest.mark.parametrize("value", [None, False])  # the parsed value (envs.py env_bool; "0" -> False: S7)
def test_s4_flag_off_warmup_equals_base(monkeypatch, flag, value):
    base_runner = os.path.join(_base_root(), "src", "vllm_torchtpu", "runner", "tpu_runner.py")
    bns = dict(NS)  # the base method resolves the same harness names
    (base_pre,) = w._fns(base_runner, ["_vfy_precompile"], bns, cls="TPUModelRunner")

    class BaseRunner(T.Runner):
        _vfy_precompile = base_pre

    def rec_into(store):
        def rec(*c):
            store.append(c)
        return rec

    flag(None)
    base_calls = []
    bns["_vfy_one_compiled"] = rec_into(base_calls)
    rb = BaseRunner(8, cand=False, mode=2)
    rb.max_num_reqs, rb.num_tokens_paddings, rb.num_reqs_paddings = 8, [16, 32], [8]
    rb._vfy_precompile()
    flag(value)
    head_calls, rh = _record(monkeypatch)
    assert len(head_calls) == len(base_calls) > 0
    for a, b in zip(base_calls, head_calls):
        assert VF.key(*a[1:]) == VF.key(*b[1:])
        assert alias_pairs(a) == alias_pairs(b) == ()
        assert digest(a) == digest(b)
    assert rh._vfy.warm == rb._vfy.warm and rh._vfy.programs == rb._vfy.programs


def test_s4_serving_path_is_source_identical_to_base():
    base_runner = os.path.join(_base_root(), "src", "vllm_torchtpu", "runner", "tpu_runner.py")
    for name in SERVING_METHODS:
        assert _method_src(T.RUNNER, "TPUModelRunner", name) == _method_src(base_runner, "TPUModelRunner", name), name


# ---------------------------------------------------------------- S5 / S6: the helper and the seed hooks

def test_s5_alias_helper_unit():
    dl = torch.zeros(8, dtype=torch.int32)
    own = torch.zeros(8, dtype=torch.int32)
    spec = VF.VfySeedSpec(lambda *a: a, (own, torch.zeros(65, dtype=torch.int32), 4), (own, 1))
    got, ok = ALIAS(spec, dl)
    assert ok and got.args[0] is dl and got.base_args[0] is dl and got.args[1] is spec.args[1]
    assert VF.key(got.fn, *got.args) == VF.key(spec.fn, *spec.args)
    sep = VF.VfySeedSpec(spec.fn, (own, 4), (torch.zeros(8, dtype=torch.int32), 4))
    got, ok = ALIAS(sep, dl)
    assert ok and got.args[0] is dl and got.base_args[0] is sep.base_args[0]  # base_args alias only when it was args[0]
    for bad in (torch.zeros(16, dtype=torch.int32), torch.zeros(8, dtype=torch.int64)):
        s2 = VF.VfySeedSpec(spec.fn, (bad, 4), (bad, 4))
        got, ok = ALIAS(s2, dl)
        assert not ok and got is s2
    s3 = VF.VfySeedSpec(spec.fn, (4, own), (4, own))
    got, ok = ALIAS(s3, dl)
    assert not ok and got is s3


def test_s5_vrng_composed_call_keeps_the_alias(monkeypatch, flag):
    flag(True)
    calls, r = _record(monkeypatch)
    seeded = [c for c in calls if len(c) > POS_ND]
    assert seeded
    counter = torch.zeros((), dtype=torch.int32)
    for c in seeded:
        rc = r._vfy_rng_call(c, counter)
        dl = c[POS_DL]
        hits = [i for i, x in enumerate(rc) if x is dl]
        assert len(hits) == 2, hits  # the verify's draft_lengths and the seed's num_draft, as in serving

# SPDX-License-Identifier: Apache-2.0
"""No HC routing test pins the traced token count (q38 L3-HCGATE; T95u fix 521ca02f7 + its generalization).

T95u (L2's localization, M) and L2-LADDER (j-9357fc67 / j-4b5f2e6d, M): an HC routing test that puts a
bound on the traced token SymInt can meet another module's bound at the same value -- O1s' cap
`tokens <= slots x (k + 1)` is 256 at 64 slots and 512 at 128 slots -- and pin the symbol; the pinned
trace loses SymInt placeholders and the shape-variant session refuses it. Measured: the old band
(`tokens >= 256`) at 64 slots; n128 / Form A (`_prefill_full_rows`' `tokens >= 512`) at 128 slots.

One decoder layer's HC pattern (attn mix -> combine_mix into the mlp HC -> mlp combine) is captured with
Dynamo at every ladder bucket 16 ... 2048 with the token dimension dynamic. ``ub`` (0 / 128 / 256 / 512)
adds a stand-in for another module's `tokens <= ub` test (the O1s cap at 32 / 64 / 128 slots). From each
capture: the SymInt inputs, the token symbol's value range, and the ShapeEnv guards. Required:

1. every arm (FINAL, FINAL - HC fused, FINAL - HC fused + HCX 256, and the gate arms 128, 128:256,
   128 + HCX 256 = Form A) keeps the token symbol symbolic at every bucket and every ub;
2. bound inventory: across every arm, the only value bounds a guard puts on the token symbol are
   FINAL's own (decode_token_tile: upper 512 / lower 513, the single-tile split upper 128 / lower 129,
   lower 1) and the stand-in's (upper ub / lower ub + 1); no HC test bounds it at 256, at a lower 128 /
   512, or anywhere else;
3. the gate arms add no guard beyond their reference arm's except tests against 128 (the single-tile
   split): FINAL on the fused side, FINAL - HC fused (+ HCX for Form A) on the other;
4. controls (the harness can see the bug): the old band order (`tokens >= 256`) pins the symbol at 256
   with ub 256, and the old `_prefill_full_rows` (`tokens >= 512`) pins it at 512 with ub 512 (gate 128).
   "Symbolic" needs a SymInt input, a non-singleton range for it, AND no guard set that fixes a symbol to
   one value (an Eq or a lower bound meeting an upper bound on the same symbol).

Ops: the real torch_tpu jax_ops in the CDK CPU image, torch.library stand-ins elsewhere (shared with
test_hcgate_engage.py). Mutation-checked by q38_l3_hcgate_mutrun.py.
"""

import os
import re
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "q38_l3_hcx"))
import hcx_load  # noqa: E402
import test_hcgate_engage as eng  # noqa: E402

LADDER = (16, 96, 128, 160, 224, 256, 512, 1024, 2048)
UBS = (0, 128, 256, 512)
ARMS = {
    "final": dict(fused=True),
    "nohcf": dict(fused=False),
    "nohcf_hcx256": dict(fused=False, hcx="256"),
    "gate128": dict(fused=True, gate="128"),
    "band": dict(fused=True, gate="128:256"),
    "formA": dict(fused=True, gate="128", hcx="256"),
}
GATES = {"gate128": ("128", "nohcf"), "band": ("128:256", "nohcf"), "formA": ("128", "nohcf_hcx256")}
FINAL_LOWER, FINAL_UPPER = {1, 2, 129, 513}, {128, 512}


def _old_allows(tokens, limit):
    """The pre-fix gate (1ef7c4bd6), for the control only."""
    lo, hi = (limit, 0) if isinstance(limit, int) else limit
    return lo <= 0 or tokens <= lo or (hi > 0 and tokens >= hi)


def _old_prefill_full_rows(self, tokens):
    """The pre-fix _prefill_full_rows (served step), for the control only."""
    return tokens >= 2 * self.prefill_full_tile and tokens % self._prefill_tile_step == 0


class _Captured(Exception):
    pass


def _capture(attn, mlp, rows, ub):
    seen = {}

    def backend(gm, example_inputs):
        fake = torch._guards.detect_fake_mode(example_inputs)
        env = fake.shape_env if fake is not None else None
        syms = [e for e in example_inputs if isinstance(e, torch.SymInt)]
        rng = None
        if env is not None and syms:
            r = env.var_to_range.get(syms[0].node.expr)
            rng = (r.lower, r.upper) if r is not None else None
        seen["out"] = dict(symints=len(syms), rng=rng,
                           guards=sorted({str(g.expr) for g in (env.guards if env is not None else [])}))
        raise _Captured()

    def layer(x, block):
        mixed, residual = attn.mix(x)
        if ub and x.shape[0] <= ub:   # stand-in for another module's `tokens <= ub` (the O1s cap)
            block = block * 1.0
        mixed2, residual2 = attn.combine_mix(block, residual, mlp)
        return mixed, mixed2, mlp.combine(block, residual2)

    g = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, eng.HC * eng.H, generator=g).bfloat16()
    block = torch.randn(rows, eng.H, generator=g).bfloat16()
    torch._dynamo.maybe_mark_dynamic(x, 0)
    torch._dynamo.maybe_mark_dynamic(block, 0)
    torch._dynamo.reset()
    try:
        torch.compile(layer, backend=backend, fullgraph=True)(x, block)
    except Exception as e:  # noqa: BLE001 - BackendCompilerFailed wraps _Captured
        if "out" in seen and (isinstance(e, _Captured) or isinstance(getattr(e, "inner_exception", None), _Captured)):
            return seen["out"]
        return dict(error="%s: %s" % (type(e).__name__, (str(e).splitlines() or [""])[0][:200]))
    return dict(error="the capture backend never ran")


def _bounds_meet(guards):
    """True if the guards alone fix some symbol to one value: an Eq(s, c) guard, or a lower bound that
    reaches an upper bound on the same symbol (symbols joined by Eq(sA, sB) count as one), whether or not
    the ShapeEnv already turned it into a replacement. smoke-7 j-387fed08 (M): the old prefill-rows control
    at ub 512 carried `s77 >= 512` and `s77 <= 512` while one SymInt input survived, so the input count
    alone missed the pin."""
    parent = {}

    def root(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    rel = []
    for g in guards:
        if _EQ.match(g):
            return True
        m = _EQSYM.match(g)
        if m:
            parent[root(m.group(1))] = root(m.group(2))
            continue
        m = _REL.match(g)
        if m:
            rel.append(m.groups())
    lo, hi = {}, {}
    for a, op, b in rel:
        if a.startswith("s") == b.startswith("s"):
            continue
        if b.startswith("s"):
            a, b, op = b, a, {"<=": ">=", "<": ">", ">=": "<=", ">": "<"}[op]
        a, c = root(a), int(b)
        if op in (">=", ">"):
            lo[a] = max(lo.get(a, c + (op == ">")), c + (op == ">"))
        else:
            hi[a] = min(hi.get(a, c - (op == "<")), c - (op == "<"))
    return any(a in hi and lo[a] >= hi[a] for a in lo)


def _symbolic(c):
    return ("error" not in c and c["symints"] >= 1 and c["rng"] is not None and c["rng"][0] != c["rng"][1]
            and not _bounds_meet(c["guards"]))


_REL = re.compile(r"^\s*(s\d+|-?\d+)\s*(<=|<|>=|>)\s*(s\d+|-?\d+)\s*$")
_EQ = re.compile(r"^\s*Eq\((s\d+), (-?\d+)\)\s*$|^\s*Eq\((-?\d+), (s\d+)\)\s*$")
_EQSYM = re.compile(r"^\s*Eq\((s\d+), (s\d+)\)\s*$")


def value_bounds(guards):
    """[(kind, value)] for every guard that bounds a token symbol by a constant: kind lower / upper / pin,
    value the tightest inclusive bound (128 < s0 -> lower 129)."""
    out = []
    for g in guards:
        m = _EQ.match(g)
        if m:
            out.append(("pin", int(m.group(2) or m.group(3))))
            continue
        m = _REL.match(g)
        if not m:
            continue
        a, op, b = m.groups()
        if a.startswith("s") == b.startswith("s"):
            continue
        if b.startswith("s"):   # c OP s  ->  s OP' c
            a, b, op = b, a, {"<=": ">=", "<": ">", ">=": "<=", ">": "<"}[op]
        c = int(b)
        out.append({"<=": ("upper", c), "<": ("upper", c - 1), ">=": ("lower", c), ">": ("lower", c + 1)}[op])
    return out


def _extra_ok(e):
    """An extra guard a gate arm may add over its reference: a test against 128 only."""
    nums = set(re.findall(r"\d+", re.sub(r"s\d+", "", e))) - {"0"}
    return nums == {"128"}


@pytest.fixture(scope="module")
def table():
    m = hcx_load.load()
    ops = eng._ops_mode()
    mp = pytest.MonkeyPatch()
    out = {}
    try:
        for arm, kw in ARMS.items():
            attn, mlp = eng._build(m, mp, 0, ops, **kw), eng._build(m, mp, 5, ops, **kw)
            for ub in UBS:
                for rows in LADDER:
                    out[(arm, ub, rows)] = _capture(attn, mlp, rows, ub)
        # control 1: the old band order, patched into band-arm modules
        attn, mlp = eng._build(m, mp, 0, ops, **ARMS["band"]), eng._build(m, mp, 5, ops, **ARMS["band"])
        mp.setattr(m.hc, "decode_fused_allows", _old_allows)
        for ub in UBS:
            for rows in LADDER:
                out[("band_old", ub, rows)] = _capture(attn, mlp, rows, ub)
        mp.undo()
        mp = pytest.MonkeyPatch()
        # control 2: the old _prefill_full_rows, patched into gate-128 modules
        attn, mlp = eng._build(m, mp, 0, ops, **ARMS["gate128"]), eng._build(m, mp, 5, ops, **ARMS["gate128"])
        for mod in (attn, mlp):
            mod._prefill_full_rows = types.MethodType(_old_prefill_full_rows, mod)
        for ub in UBS:
            for rows in LADDER:
                out[("n128_old", ub, rows)] = _capture(attn, mlp, rows, ub)
    finally:
        mp.undo()
    for (arm, ub, rows), c in sorted(out.items()):
        print("HCGATE-SYMINT ops=%s arm=%s ub=%d rows=%d symints=%s range=%s symbolic=%s bounds=%s %s" % (
            ops, arm, ub, rows, c.get("symints"), c.get("rng"), _symbolic(c),
            sorted(set(value_bounds(c.get("guards", [])))), c.get("error", "")), flush=True)
    return out


def _ref(rows, gate, unfused_ref):
    lo, _, hi = gate.partition(":")
    fused_side = rows <= int(lo) or (bool(hi) and rows % 128 == 0)
    return "final" if fused_side else unfused_ref


def test_captures_ran(table):
    bad = {k: v["error"] for k, v in table.items() if "error" in v}
    assert not bad, bad


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_keeps_the_token_symbol_symbolic(table, arm):
    for ub in UBS:
        for rows in LADDER:
            assert _symbolic(table[(arm, ub, rows)]), (arm, ub, rows, table[(arm, ub, rows)])


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_bound_inventory(table, arm):
    """Only FINAL's own bounds and the stand-in's: no HC routing bound at any other value."""
    for ub in UBS:
        lower_ok = FINAL_LOWER | ({ub + 1} if ub else set())
        upper_ok = FINAL_UPPER | ({ub} if ub else set())
        for rows in LADDER:
            for kind, v in value_bounds(table[(arm, ub, rows)]["guards"]):
                ok = (kind == "lower" and v in lower_ok) or (kind == "upper" and v in upper_ok)
                assert ok, (arm, ub, rows, kind, v, table[(arm, ub, rows)]["guards"])


@pytest.mark.parametrize("arm", sorted(GATES))
def test_gate_adds_no_guard_beyond_the_reference_except_against_128(table, arm):
    gate, unfused_ref = GATES[arm]
    for ub in UBS:
        for rows in LADDER:
            ref = _ref(rows, gate, unfused_ref)
            extra = sorted(set(table[(arm, ub, rows)]["guards"]) - set(table[(ref, ub, rows)]["guards"]))
            assert all(_extra_ok(e) for e in extra), (arm, ub, rows, ref, extra)
            assert table[(arm, ub, rows)]["symints"] == table[(ref, ub, rows)]["symints"], (arm, ub, rows)


def test_bounds_meet_detector():
    """The pin detector on the guard sets smoke-7 j-387fed08 printed (M) and on near misses."""
    assert _bounds_meet(["Eq(Mod(s77, 512), 0)", "s77 <= 512", "s77 > 128", "s77 >= 512"])
    assert _bounds_meet(["s72 > 128", "s72 >= 256", "s16 <= 256", "Eq(s72, s16)"])  # T95u: joined by Eq(s72, s16)
    assert _bounds_meet(["Eq(s3, 256)"])
    assert not _bounds_meet(["s77 > 128", "s77 <= 256", "Ne(Mod(s77, 512), 0)", "Eq(Mod(s77, 128), 0)"])
    assert not _bounds_meet(["s1 >= 512", "s2 <= 512"])  # different symbols
    assert not _bounds_meet(["s77 > 256", "s77 <= 512"])


def test_control_old_band_order_pins_at_256(table):
    assert not _symbolic(table[("band_old", 256, 256)]), table[("band_old", 256, 256)]
    assert any(k == "lower" and v == 256 for k, v in value_bounds(table[("band_old", 0, 256)]["guards"]))


def test_control_old_prefill_rows_test_pins_at_512(table):
    assert not _symbolic(table[("n128_old", 512, 512)]), table[("n128_old", 512, 512)]
    assert _bounds_meet(table[("n128_old", 512, 512)]["guards"])
    assert any(k == "lower" and v == 512 for k, v in value_bounds(table[("n128_old", 0, 512)]["guards"]))

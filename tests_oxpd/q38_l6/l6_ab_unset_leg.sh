#!/bin/bash
# extra_leg.sh (v4) -- q38 L6-HOST P7b: syncgc / MAINPIN-2 unset under an ACTIVE l6_ab (STACK-D v5's own L6 lines).
# Asked by L1 successor 3 for v5's H-L6 record (gap found by the V5_COMPOSE verifier wf_caa80ae4-19a: no head-vs-base run
# reached the `if _l6_ab.ACTIVE` block of execute_model or the propose wrapper with syncgc and MAINPIN-2 unset).
# Runs from a bundle root holding head/ (src + tests) and base3/ (src; b488e1cb0 in L6's v5c7 stage), like run_cpu.sh.
#
# Legs (embedded leg.py, sha f4121072e6f8797f; one python per tree x config, PYTHONPATH=src of that tree, the head's harnesses):
#   configs  lines = v5's L6 lines byte for byte (l6_ab ACTIVE, class 0 for the whole run);
#            s2    = the same with VLLM_TPU_L6_AB_STEPS=2 / CONFIRM=1, so all 16 classes and every lever's on / off
#                    windows occur inside the run.
#   trees    head; KO = head with on_step's _syncgc_step() / _mainpin2_step() calls removed (embedded mk.py, sha bd843e988045529a):
#            the base on which syncgc / MAINPIN-2 cannot act; base3; mutants M1 M1b M2 M2b M3 M4 of head's l6_jitter.py;
#            M5 = L4's sampfb added to l6_ab.LEVERS (as on the composed v5 heads) with the leg's prewarm fix OFF (must fail
#            the class checks); control M5c = the same LEVERS edit with the fix ON (must pass them and equal KO).
# v2 (2026-11-14): v1 FAILED s2_all_16_classes on F-v5c / F-vfy (j-73838313 / j-c4746ae3): the ADC harness's prewarm pops
#   only VLLM_TPU_L6_* keys, and their l6_ab.LEVERS also carries L4 levers, so VLLM_TPU_L4_SAMP_FB_ORDER=1 kept l6_ab
#   ACTIVE in class 'uniform' through 16 x 9 prewarm steps. The leg now pops every l6_ab lever's env around the prewarm,
#   and checks the exact step count and the lines config's class set too.
# v3 (2026-11-14, L6-V6): on a tree with P-METER (q38-l6-pmeter 80ea2224a) execute_model's l6 block also holds
#   `if _l6_ab.ACTIVE: _l6_pmeter.on_step()` after the STEADY drop; the AST check accepts it exactly when the tree
#   has runner/l6_pmeter.py, exactly as [begin_step, envcache, jitter, STEADY drop, P-METER hook, GATE hook]; the leg's
#   block replay calls the hook in that position. New mutants M6 (hook dropped) and M6r (hook moved before the drop)
#   on runner/tpu_runner.py must fail the AST pin. Trees without P-METER keep v2's rule. Otherwise as v2.
# v4 (2026-11-14, L6-V6): with L6-H2 (q38-l6-h2cheap-v2 a106b9a51) the block also holds `if _l6_aot._handles:
#   _l6_aot.on_step()` right after the P-METER hook; on a tree with both the AST pin is exactly [begin_step, envcache,
#   jitter, STEADY drop, P-METER hook, AOTSKIP hook, GATE hook] and the replay calls the AOTSKIP hook there. New mutants
#   M7 (AOTSKIP hook dropped) and M7r (moved before the P-METER hook) must fail the pin. Otherwise as v3.
# Verdicts (embedded chk.py, sha 692ca0f244ab304f):
#   head invariants: execute_model's l6 block pinned by AST; l6_ab ACTIVE with syncgc / mainpin2 parsed (False, None);
#     both step hooks reached on every step; no JITTER-lever note; gc enabled and no l6_jitter gc callback;
#     Eagle3Proposer.propose unwrapped (import must succeed); host state constant over every step; s2 reaches all 16
#     classes with carry / steady / defer / envcache / fctx on somewhere; lines reaches class 0 only; begin_step ran exactly
#     the leg's own steps (prewarm excluded).
#   head == KO in both configs: digests of every _prepare_inputs / adapter output, the input batch after every
#     _modify_prev_results, upload counts, per-step host state (gc / switch interval / thread affinities / propose
#     wrapper), and l6_jitter's syncgc / MAINPIN-2 state dicts.
#   head == base3 in both configs: verdict when head is q38-l6-v5 e80411f84's src (e7b37d81fa); printed as a diagnostic
#     on a composed head (KO is then the verdict base).
#   every mutant killed (its digests differ from KO's, a JITTER note appears, or a class / step check fails);
#   control M5c passes the class / step checks, popped VLLM_TPU_L4_SAMP_FB_ORDER around the prewarm, and equals KO.
#   head src unchanged by the run.
# Last line: Q38_L6_ABUNSET_SUMMARY PASS|FAIL [failed checks]; exit 0 only on PASS.
set -u
W=$(pwd); H=$W/head; B3=$W/base3
echo "== P7b L6 ab-active syncgc / MAINPIN-2 unset leg $(date -u +%FT%TZ) host $(hostname) nproc $(nproc)"
FAIL=""
[ -d "$H/src/vllm_torchtpu" ] && [ -d "$H/tests/q38_l6" ] || { echo "Q38_L6_ABUNSET_SUMMARY FAIL no-head"; exit 1; }
[ -d "$B3/src/vllm_torchtpu" ] || { echo "Q38_L6_ABUNSET_SUMMARY FAIL no-base3"; exit 1; }
L=$(mktemp -d /tmp/l6abu.XXXXXX) || exit 1
cat > $L/leg.py <<'PYEOF'
# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST P7b: syncgc / MAINPIN-2 unset under an ACTIVE l6_ab (v5's own L6 lines).

Run with the working directory at one tree and PYTHONPATH=src:

    python3 leg.py <harness dir: a head's tests/q38_l6> <config: lines | s2>

Sets the STACK-D v5 decode-leg L6 lines (scratch/decL1/v5/l6_lines_v5.txt, embedded below byte for byte; every
MUST-BE-ABSENT key is removed), so l6_ab is ACTIVE and VLLM_TPU_L6_SYNC_GC / VLLM_TPU_L6_MAIN_PIN2 are unset.
Config ``s2`` is the same lines with VLLM_TPU_L6_AB_STEPS=2 and VLLM_TPU_L6_AB_CONFIRM=1, so the 16 classes turn over
inside the run and every lever's on and off windows are reached (``lines`` stays in class 0 for the whole run).

Drives the L6 carry harness (real ``_prepare_inputs``) and the ADC harness (real ``_prepare_inputs`` plus the real
Qwen3.8 adapter), as P7 / P7a do, but with l6_ab ACTIVE. Each harness step calls ``l6_ab.begin_step()``; this leg
wraps it so that the rest of execute_model's ``if _l6_ab.ACTIVE or _l6_envcache.CENSUS:`` block follows, in its
order: ``_l6_envcache.on_step()``, ``_l6_jitter.on_step()`` (syncgc's and MAINPIN-2's step hooks), and the STEADY
drop, then (v3, q38 L6-V6) P-METER's ``_l6_pmeter.on_step()`` and (v4) L6-H2's ``_l6_aot.on_step()`` on trees
that carry them. The block's statements are
pinned by an AST check of this tree's tpu_runner.py (L6ABUNSET_AST).
The worker's ``l6_jitter.mark_warm()`` is called first where the tree has it (the post-warm-up state).
v2: the ADC harness's prewarm runs with the env of every l6_ab lever popped (not only VLLM_TPU_L6_*), so no
begin_step runs during it on trees whose LEVERS carries L4 levers (L6ABU_PREWARM_FIX=0 turns that off: mutant M5).

Prints, per harness run, the digest of every output that leaves ``_prepare_inputs`` and the adapter, the input-batch
state right after every ``_modify_prev_results``, and the per-step upload counts; per step, the host state syncgc and
MAINPIN-2 would change (gc enabled / thresholds / callbacks / freeze count, switch interval, the main thread's and
every thread's CPU affinity, whether Eagle3Proposer.propose is wrapped), and (where the tree has them) l6_jitter's
syncgc / MAINPIN-2 state dicts; every l6_ab.note of a JITTER lever; and how often the syncgc / MAINPIN-2 step hooks
were reached. L6ABUNSET_ALL (outputs + host state) is the digest compared across trees.
"""

import ast
import gc
import hashlib
import json
import os
import sys

HDIR, CFG = sys.argv[1], sys.argv[2]
assert CFG in ("lines", "s2"), CFG

LINES = """VLLM_TPU_HOST_PHASES=250
VLLM_TPU_L6_AB_STEPS=250
VLLM_TPU_L6_AB_CLASSES=16
VLLM_TPU_L6_AB_CONFIRM=16
VLLM_TPU_L6_ENVS_CACHE=ab:1+3+5+7+9+11+13+15
VLLM_TPU_L6_FAST_FWD_CTX=ab:1+3+5+7+9+11+13+15
VLLM_TPU_L6_DEFER_PREV=ab:2+3+6+7+10+11+14+15
VLLM_TPU_L6_DECODE_CARRY=ab:4+5+6+7+12+13+14+15
VLLM_TPU_L6_DECODE_CARRY_CHECK=200
VLLM_TPU_L6_STEADY_STEP=ab:8+9+10+11+12+13+14+15
VLLM_TPU_L6_STEADY_STEP_CHECK=200
VLLM_TPU_L6_CHECK_STEPS=30000
VLLM_TPU_L6_EOS_CANARY=248044+248046
VLLM_TPU_L6_ENGINE_ALLOC_FAST=1
VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK=200
VLLM_TPU_ENGINE_PHASES=200
VLLM_TPU_L4_SAMP_FB_ORDER=1
VLLM_TPU_HOST_GC=threshold
"""
MUST_BE_ABSENT = ("VLLM_TPU_L6_GC_FREEZE VLLM_TPU_L6_SYNC_GC VLLM_TPU_L6_SYNC_GC_EVERY VLLM_TPU_L6_SYNC_GC_VALVE "
                  "VLLM_TPU_L6_MAIN_PIN VLLM_TPU_L6_MAIN_PIN_CPU VLLM_TPU_L6_SWITCH VLLM_TPU_L6_SWITCH_US "
                  "VLLM_TPU_HOST_AB_PERIOD_MS VLLM_TPU_L6_AB_PERIOD_MS").split() + ["VLLM_TPU_L6_MAIN_PIN2"]
LINES_SHA = hashlib.sha256(LINES.encode()).hexdigest()[:16]

for _k in list(os.environ):
    if _k.startswith("VLLM_TPU_L6_") or _k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
        del os.environ[_k]
for _ln in LINES.splitlines():
    _k, _v = _ln.split("=", 1)
    os.environ[_k] = _v
if CFG == "s2":
    os.environ["VLLM_TPU_L6_AB_STEPS"] = "2"
    os.environ["VLLM_TPU_L6_AB_CONFIRM"] = "1"
for _k in MUST_BE_ABSENT:
    assert _k not in os.environ, _k
os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"

sys.path.insert(0, HDIR)
import l6_adc_harness as A  # noqa: E402
import l6_carry_harness as H  # noqa: E402
import vllm_torchtpu  # noqa: E402
from vllm_torchtpu.core import l6_ab, l6_envcache, l6_jitter  # noqa: E402
from vllm_torchtpu.runner import l6_steady  # noqa: E402

try:  # q38 L6-V6: P-METER's step hook sits in the block on trees that carry it (not on base3 b488e1cb0)
    from vllm_torchtpu.runner import l6_pmeter as _PM  # noqa: E402
except ImportError:
    _PM = None
try:  # q38 L6-V6 (v4): L6-H2's AOTSKIP step hook, likewise
    from vllm_torchtpu.compilation import l6_aotskip as _AOT  # noqa: E402
except ImportError:
    _AOT = None

JIT = {"gcfreeze", "switch", "mainpin", "mainpin2", "syncgc", "gc"}


def ast_block() -> list[str]:
    """The statements of execute_model's `if _l6_ab.ACTIVE or _l6_envcache.CENSUS:` block in this tree."""
    p = os.path.join(os.path.dirname(vllm_torchtpu.__file__), "runner", "tpu_runner.py")
    tree = ast.parse(open(p).read())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model":
            for sub in ast.walk(node):
                if isinstance(sub, ast.If) and ast.unparse(sub.test) == "_l6_ab.ACTIVE or _l6_envcache.CENSUS":
                    hits.append([ast.unparse(s).replace("\n", " ") for s in sub.body])
    assert len(hits) == 1, f"execute_model's l6 block found {len(hits)} times"
    return hits[0]


WANT_BLOCK = [
    "_l6_ab.begin_step(fp_fn=lambda: _l6_ab.sched_fingerprint(scheduler_output))",
    "_l6_envcache.on_step()",
    "_l6_jitter.on_step()",
    "if _l6_ab.ACTIVE and (not _l6_steady.on(self)):     _l6_steady.drop(self)",
]
WANT_GATE = "if _l6_ab.GATE:     self._l6_gate_step(scheduler_output)"
# q38 L6-V6: P-METER's step hook (q38-l6-pmeter 80ea2224a), after the STEADY drop and before the GATE hook.
WANT_PM = "if _l6_ab.ACTIVE:     _l6_pmeter.on_step()"
# q38 L6-V6 (v4): L6-H2's AOTSKIP step hook (q38-l6-h2cheap-v2 a106b9a51), after P-METER's and before the GATE hook.
WANT_AOT = "if _l6_aot._handles:     _l6_aot.on_step()"


def norm(x):
    if isinstance(x, dict):
        return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (set, frozenset)):
        return sorted(norm(v) for v in x)
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return type(x).__name__


def _propose_cls():
    try:
        from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
        return Eagle3Proposer, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"[:200]


PCLS, PERR = _propose_cls()
PORIG = PCLS.__dict__.get("propose") if PCLS is not None else None


def propose_state():
    if PCLS is None:
        return f"import-failed {PERR}"
    f = PCLS.__dict__.get("propose")
    return [f is PORIG, bool(getattr(f, "_q38_syncgc", False)), hasattr(f, "__wrapped__")]


def affinities():
    out = {}
    for t in os.listdir("/proc/self/task"):
        try:
            out[t] = tuple(sorted(os.sched_getaffinity(int(t))))
        except OSError:
            pass
    return sorted(set(out.values()))


def host_state():
    cbs = [f"{getattr(c, '__module__', '?')}.{getattr(c, '__qualname__', type(c).__name__)}" for c in gc.callbacks]
    return {
        "gc_enabled": gc.isenabled(),
        "gc_threshold": list(gc.get_threshold()),
        "gc_callbacks": cbs,
        "freeze_count": gc.get_freeze_count(),
        "switch": sys.getswitchinterval(),
        "main_aff": sorted(os.sched_getaffinity(0)),
        "aff_set": [list(a) for a in affinities()],
        "propose": propose_state(),
    }


def internal_state():
    sg = getattr(l6_jitter, "_sg", None)
    p2 = getattr(l6_jitter, "_p2", None)
    warm = getattr(l6_jitter, "_warm", None)
    return {"sg": norm(sg) if sg is not None else "absent", "p2": norm(p2) if p2 is not None else "absent",
            "warm": norm(warm) if warm is not None else "absent"}


HOST, INTERNAL, NOTES, CLASSES = [], [], [], []
REACH = {"syncgc": 0, "mainpin2": 0, "steps": 0}
CUR = {"r": None, "prev": []}

_orig_note = l6_ab.note


def note(lever, key, n=1):
    if lever in JIT:
        NOTES.append((lever, key, n))
    return _orig_note(lever, key, n)


l6_ab.note = note

for _name, _key in (("_syncgc_step", "syncgc"), ("_mainpin2_step", "mainpin2")):
    _f = getattr(l6_jitter, _name, None)
    if _f is not None:
        def _counted(_f=_f, _key=_key):
            REACH[_key] += 1
            return _f()
        setattr(l6_jitter, _name, _counted)

_orig_begin = l6_ab.begin_step


def begin_step(*a, **k):
    """execute_model's block, in its order, after the harness's begin_step call."""
    _orig_begin(*a, **k)
    l6_envcache.on_step()
    l6_jitter.on_step()
    r = CUR["r"]
    if l6_ab.ACTIVE and not l6_steady.on(r):
        l6_steady.drop(r)
    if l6_ab.ACTIVE and _PM is not None:
        _PM.on_step()
    if _AOT is not None and _AOT._handles:
        _AOT.on_step()
    REACH["steps"] += 1
    CLASSES.append((l6_ab.current_class(), tuple(sorted(lv for lv, on in l6_ab._state["on"].items() if on))))
    HOST.append(json.dumps(host_state(), sort_keys=True))
    INTERNAL.append(json.dumps(internal_state(), sort_keys=True))


l6_ab.begin_step = begin_step


def ib_state(r) -> bytes:
    ib = r.input_batch
    h = hashlib.sha256()
    for name in sorted(vars(ib)):
        v = getattr(ib, name)
        if hasattr(v, "tobytes"):
            h.update(name.encode())
            h.update(v.tobytes())
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, type(None))) for x in v):
            h.update(name.encode())
            h.update(repr(v).encode())
    h.update(type(r.__dict__.get("_pre_async_results")).__name__.encode())
    return h.digest()


def on_step(sim):
    r = sim.r
    CUR["r"] = r
    if "_modify_prev_results" not in r.__dict__:
        orig = r._modify_prev_results

        def mpr(*a, **k):
            out = orig(*a, **k)
            CUR["prev"].append(ib_state(r))
            return out

        r._modify_prev_results = mpr


PREWARM_FIX = os.environ.get("L6ABU_PREWARM_FIX", "1") != "0"
PREWARM = {"calls": 0, "popped": set()}
_orig_prewarm = A.prewarm_keys


def prewarm_keys(shape, seed):
    """The ADC harness's prewarm (served warm-up shapes on scratch runners) pops only VLLM_TPU_L6_* keys. On trees
    whose l6_ab.LEVERS also carries other levers (L4's sampfb / sampsnap / statsasync / vfyone / draftasm*), one of
    v5's lines (VLLM_TPU_L4_SAMP_FB_ORDER=1) kept l6_ab ACTIVE through the prewarm, so its 16 x 9 steps ran
    begin_step in class 'uniform'. Pop the env of EVERY l6_ab lever (and gate) around the prewarm, restore after."""
    PREWARM["calls"] += 1
    names = set(l6_ab.LEVERS.values()) | set(getattr(l6_ab, "GATES", {}).values())
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k in names and not k.startswith("VLLM_TPU_L6_")}
    PREWARM["popped"] |= set(saved)
    try:
        return _orig_prewarm(shape, seed)
    finally:
        os.environ.update(saved)
        l6_ab._resolve()


if PREWARM_FIX:
    A.prewarm_keys = prewarm_keys

EXPECT_STEPS = 0  # set in main(): the leg's own run steps (prewarm steps excluded)

CARRY_SHAPES = {
    "b1": dict(nreq=1, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b3": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b5p": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    "b8n": dict(nreq=8, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
}
CARRY_EVENTS = {
    5: ("join",), 8: ("swap",), 10: ("partial",), 12: ("nonspec",), 14: ("pad",),
    16: ("skew",), 18: ("remap",), 20: ("discard",), 23: ("leave",), 26: ("flush",),
}
ADC_SHAPES = {
    "b3": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b5p": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    "b2s": dict(nreq=2, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True, prompt=1, ngram=8),
}
ADC_EVENTS = {
    5: ("join_short",), 8: ("swap",), 10: ("partial",), 12: ("nonspec",), 14: ("pad",), 16: ("skew",),
    18: ("remap",), 20: ("discard",), 22: ("preempt",), 24: ("idle_begin",), 25: ("resume",), 27: ("leave",),
    29: ("flush",),
}


def main() -> int:
    print("L6ABUNSET_FILE", os.path.realpath(vllm_torchtpu.__file__))
    print("L6ABUNSET_CFG", CFG, "lines_sha", LINES_SHA, "steps", os.environ["VLLM_TPU_L6_AB_STEPS"],
          "confirm", os.environ["VLLM_TPU_L6_AB_CONFIRM"])
    blk = ast_block()
    if _PM is not None and _AOT is not None:  # v4: P-METER and L6-H2 (the composed v6 head): the block exactly
        ok = blk == WANT_BLOCK + [WANT_PM, WANT_AOT, WANT_GATE]
    elif _PM is not None:  # v3: a P-METER tree without L6-H2
        ok = blk == WANT_BLOCK + [WANT_PM, WANT_GATE]
    else:  # v2's rule, unchanged (base3 b488e1cb0 has no GATE hook)
        ok = blk[:4] == WANT_BLOCK and (blk[4:] == [] or blk[4:] == [WANT_GATE])
    print("L6ABUNSET_AST", "ok" if ok else "MISMATCH", " | ".join(blk))
    l6_ab._resolve()
    print("L6ABUNSET_ACTIVE", l6_ab.ACTIVE, "levers_set",
          ",".join(sorted(lv for lv, c in l6_ab._cfg.items() if c != (False, None))),
          "syncgc_cfg", l6_ab._cfg.get("syncgc", "unregistered"), "mainpin2_cfg", l6_ab._cfg.get("mainpin2", "unregistered"))
    assert l6_ab.ACTIVE
    if hasattr(l6_jitter, "mark_warm"):
        l6_jitter.mark_warm()
    print("L6ABUNSET_PROPOSE_IMPORT", "ok" if PCLS is not None else f"FAILED {PERR}")
    HOST.append(json.dumps(host_state(), sort_keys=True))
    INTERNAL.append(json.dumps(internal_state(), sort_keys=True))
    lines = []
    undo = H.install_eager_programs()
    try:
        for i, (name, shape) in enumerate(CARRY_SHAPES.items()):
            CUR["prev"] = []
            events = {k: v for k, v in CARRY_EVENTS.items() if not (name == "b8n" and "join" in v)}
            digests, sim = H.run(shape, 30, seed=100 + i, events=events, on_step=on_step)
            up = hashlib.sha256(repr(sim.uploads).encode()).hexdigest()[:16]
            pv = hashlib.sha256(b"".join(CUR["prev"])).hexdigest()[:16]
            lines.append(f"carry {name} steps={len(digests)} out={H.digest_all(digests)} "
                         f"prev={len(CUR['prev'])}:{pv} uploads={sum(sim.uploads)}:{up}")
    finally:
        undo()
    undo = A.install_eager_programs()
    try:
        for cname, cenv in {"served": A.adapter_env(), "store": A.adapter_env(window=False),
                            "inplace": A.adapter_env(functional=False)}.items():
            os.environ.update(cenv)
            l6_ab._resolve()
            assert l6_ab.ACTIVE
            for i, (name, shape) in enumerate(ADC_SHAPES.items()):
                CUR["prev"] = []
                digests, sim = A.run(shape, 32, seed=200 + i, events=ADC_EVENTS, on_step=on_step)
                up = hashlib.sha256(repr(sim.uploads).encode()).hexdigest()[:16]
                pv = hashlib.sha256(b"".join(CUR["prev"])).hexdigest()[:16]
                lines.append(f"adc {cname} {name} steps={len(digests)} out={A.digest_all(digests)} "
                             f"prev={len(CUR['prev'])}:{pv} uploads={sum(sim.uploads)}:{up}")
    finally:
        undo()
    HOST.append(json.dumps(host_state(), sort_keys=True))
    INTERNAL.append(json.dumps(internal_state(), sort_keys=True))
    for line in lines:
        print("L6ABUNSET", line)
    hd = hashlib.sha256("\n".join(HOST).encode()).hexdigest()
    idg = hashlib.sha256("\n".join(INTERNAL).encode()).hexdigest()
    seen = sorted(set(CLASSES))
    print("L6ABUNSET_CLASSES n_steps", len(CLASSES), "distinct", len(seen), "classes",
          ",".join(sorted({c for c, _ in seen}, key=lambda s: int(s) if s.isdigit() else -1)),
          "levers_on_somewhere", ",".join(sorted({lv for _, ons in seen for lv in ons})))
    print("L6ABUNSET_HOST_FIRST", HOST[0])
    print("L6ABUNSET_HOST_LAST", HOST[-1])
    print("L6ABUNSET_HOST_DISTINCT", len(set(HOST)))
    print("L6ABUNSET_HOST", hd)
    print("L6ABUNSET_INTERNAL", idg, "first", INTERNAL[0][:400])
    print("L6ABUNSET_INTERNAL_LAST", INTERNAL[-1][:600])
    print("L6ABUNSET_JITNOTES", len(NOTES), repr(NOTES[:10]))
    print("L6ABUNSET_PREWARM", "fix" if PREWARM_FIX else "NOFIX", "calls", PREWARM["calls"], "popped",
          ",".join(sorted(PREWARM["popped"])) or "-", "expect_steps", 30 * len(CARRY_SHAPES) + 3 * 32 * len(ADC_SHAPES))
    print("L6ABUNSET_REACH", " ".join(f"{k}={v}" for k, v in REACH.items()),
          "hooks_present", int(hasattr(l6_jitter, "_syncgc_step")), int(hasattr(l6_jitter, "_mainpin2_step")))
    print("L6ABUNSET_ALL", hashlib.sha256(("\n".join(lines) + "\n" + hd).encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF
cat > $L/mk.py <<'PYEOF'
# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST P7b variants of a head tree's core/l6_jitter.py (applied to a copy; each anchor must match exactly once).

    python3 mk.py <tree copy> <KO | M1 | M1b | M2 | M2b | M3 | M4 | M5 | M5c | M6 | M6r | M7 | M7r>

KO   knockout base: on_step no longer calls _syncgc_step() / _mainpin2_step(), so syncgc and MAINPIN-2 cannot act;
     the head with both unset must be indistinguishable from it.
M1   syncgc's step hook ignores that the lever is unset (gate dropped): installs its gc timer and the propose wrapper.
M1b  M1, and unset syncgc engages after warm-up (automatic gc off).
M2   MAINPIN-2's step hook ignores that the lever is unset (gate dropped): runs its static thread exclusion.
M2b  M2, and unset MAINPIN-2 pins the main thread.
M3   _configured() is true for every registered lever, set or not.
M4   the syncgc propose wrapper is installed at every ACTIVE step, whatever the flags.
M5   (core/l6_ab.py) L4's sampfb joins l6_ab.LEVERS, as on the composed STACK-D v5 heads, and the leg runs WITHOUT its
     prewarm fix (L6ABU_PREWARM_FIX=0): l6_ab stays ACTIVE through the ADC prewarm and the class checks must fail.
M5c  control: the same LEVERS edit WITH the prewarm fix; must pass the class checks and equal KO.
M6   (runner/tpu_runner.py, v3) P-METER's step hook dropped from execute_model's l6 block: the AST pin must fail.
M6r  (runner/tpu_runner.py, v3) P-METER's step hook moved before the STEADY drop: the AST pin must fail.
M7   (runner/tpu_runner.py, v4) L6-H2's AOTSKIP step hook dropped from execute_model's l6 block: the AST pin must fail.
M7r  (runner/tpu_runner.py, v4) L6-H2's AOTSKIP step hook moved before P-METER's: the AST pin must fail.
"""

import os
import sys

root, var = sys.argv[1], sys.argv[2]
if var in ("M6", "M6r", "M7", "M7r"):
    p = os.path.join(root, "src", "vllm_torchtpu", "runner", "tpu_runner.py")
else:
    p = os.path.join(root, "src", "vllm_torchtpu", "core", "l6_ab.py" if var in ("M5", "M5c") else "l6_jitter.py")
s = open(p).read()

DROP = "            if _l6_ab.ACTIVE and not _l6_steady.on(self):\n                _l6_steady.drop(self)\n"
PMH = ("            if _l6_ab.ACTIVE:\n"
       "                # q38 L6 P-METER: close the last step's stamps, fix this step's inject spin.\n"
       "                _l6_pmeter.on_step()\n")
AOTH = ("            if _l6_aot._handles:\n"
        "                # q38 L6-H2 CF-2 AOTSKIP-AB: this step's guard-check state for the kept artifacts.\n"
        "                _l6_aot.on_step()\n")
GATE_SG = '    if not _configured("syncgc") or _sg["broken"]:\n        return\n'
GATE_P2 = '    if not _configured("mainpin2") or _p2["broken"]:\n        return\n'
EDITS = {
    "KO": [("    _syncgc_step()\n    _mainpin2_step()\n", "")],
    "M1": [(GATE_SG, '    if _sg["broken"]:\n        return\n')],
    "M1b": [(GATE_SG, '    if _sg["broken"]:\n        return\n'),
            ('    want = l6_ab.on("syncgc") and _warm["on"]\n', '    want = _warm["on"]\n')],
    "M2": [(GATE_P2, '    if _p2["broken"]:\n        return\n')],
    "M2b": [(GATE_P2, '    if _p2["broken"]:\n        return\n'),
            ('    want = l6_ab.on("mainpin2")\n', '    want = True\n')],
    "M3": [('    return l6_ab._cfg.get(lever, (False, None)) != (False, None)\n', '    return lever in l6_ab._cfg\n')],
    "M5": [('    "syncgc": "VLLM_TPU_L6_SYNC_GC",\n',
            '    "syncgc": "VLLM_TPU_L6_SYNC_GC",\n    "sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER",\n')],
    "M5c": [('    "syncgc": "VLLM_TPU_L6_SYNC_GC",\n',
             '    "syncgc": "VLLM_TPU_L6_SYNC_GC",\n    "sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER",\n')],
    "M6": [(DROP + PMH, DROP)],
    "M6r": [(DROP + PMH, PMH + DROP)],
    "M7": [(PMH + AOTH, PMH)],
    "M7r": [(PMH + AOTH, AOTH + PMH)],
    "M4": [('def on_step() -> None:\n    if not l6_ab.ACTIVE:\n        return\n',
            'def on_step() -> None:\n    if not l6_ab.ACTIVE:\n        return\n    _install_sync_point()\n')],
}
for old, new in EDITS[var]:
    n = s.count(old)
    if n != 1:
        print(f"MK {var} ANCHOR count={n}: {old[:60]!r}")
        sys.exit(3)
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"MK {var} ok ({len(EDITS[var])} edit(s))")
PYEOF
cat > $L/chk.py <<'PYEOF'
# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST P7b verdicts over the leg logs.   python3 chk.py <log dir> <head src hash> <runs file>

<runs file>: one line per run, `<tree tag> <cfg> <tree path> <rc>`. Tags: head, ko, base3, M1 M1b M2 M2b M3 M4 M5 M6 M6r M7 M7r, M5c.
Prints one P7b line per check and `P7B_FAILS <space-separated failed checks>` (empty = PASS)."""

import json
import os
import sys

LOGS, HEAD_SRC, RUNS = sys.argv[1], sys.argv[2], sys.argv[3]
V5_L6_SRC = "e7b37d81fa"  # q38-l6-v5 e80411f84's src: the base3 comparison is a verdict only on that tree
MUTANTS = ("M1", "M1b", "M2", "M2b", "M3", "M4", "M5", "M6", "M6r", "M7", "M7r")  # v3: M6 / M6r (P-METER hook);
# v4: M7 / M7r (AOTSKIP hook dropped / moved)
CONTROLS = ("M5c",)
CFGS = ("lines", "s2")
WANT_LEVERS_S2 = {"carry", "steady", "defer", "envcache", "fctx"}
fails = []


def fail(tag):
    fails.append(tag)


def parse(path):
    d = {"lines": []}
    if not os.path.exists(path):
        return d
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        if not ln.startswith("L6ABUNSET"):
            continue
        key, _, rest = ln.partition(" ")
        if key == "L6ABUNSET":
            d["lines"].append(rest)
        else:
            d[key] = rest
    return d


runs = {}
for ln in open(RUNS):
    tag, cfg, tree, rc = ln.split()
    runs[(tag, cfg)] = (tree, int(rc), parse(os.path.join(LOGS, f"{tag}_{cfg}.log")))


def get(tag, cfg, key, default=""):
    return runs[(tag, cfg)][2].get(key, default)


def first(s):
    return s.split()[0] if s else ""


def class_checks(tag, cfg):
    """{name: ok}: the run's begin_step steps are exactly the leg's own (prewarm excluded) and its classes are exact."""
    d = runs[(tag, cfg)][2]
    reach = dict(kv.split("=") for kv in d.get("L6ABUNSET_REACH", "").split()[:3] if "=" in kv)
    pw = d.get("L6ABUNSET_PREWARM", "").split()
    try:
        exp = int(pw[pw.index("expect_steps") + 1])
    except (ValueError, IndexError):
        exp = -1
    parts = d.get("L6ABUNSET_CLASSES", "").split()
    try:
        classes = set(parts[parts.index("classes") + 1].split(","))
    except (ValueError, IndexError):
        classes = set()
    want = {str(i) for i in range(16)} if cfg == "s2" else {"0"}
    return {"steps_exact": exp > 0 and reach.get("steps") == str(exp),
            ("s2_all_16_classes" if cfg == "s2" else "lines_class_0_only"): classes == want}


# 1. every run completed, printed its digest and imported its own tree
for (tag, cfg), (tree, rc, d) in sorted(runs.items()):
    f = d.get("L6ABUNSET_FILE", "")
    own = f.startswith(os.path.realpath(tree) + "/src/")
    ok = rc == 0 and "L6ABUNSET_ALL" in d and own and len(d["lines"]) == 13
    print(f"P7B run {tag} {cfg} rc={rc} prewarm={d.get('L6ABUNSET_PREWARM', '')[:120]!r} all={first(d.get('L6ABUNSET_ALL', ''))[:16]} lines={len(d['lines'])} "
          f"own_src={int(own)} {'ok' if ok else 'BAD'}")
    if not ok and tag not in MUTANTS + CONTROLS:
        fail(f"P7b-run-{tag}-{cfg}")

# 2. head invariants (both configs)
for cfg in CFGS:
    _, rc, d = runs[("head", cfg)]
    act = d.get("L6ABUNSET_ACTIVE", "")
    reach = dict(kv.split("=") for kv in d.get("L6ABUNSET_REACH", "").split()[:3] if "=" in kv)
    steps = int(reach.get("steps", "0") or 0)
    last = json.loads(d.get("L6ABUNSET_HOST_LAST", "{}") or "{}")
    checks = {
        "ast": d.get("L6ABUNSET_AST", "").startswith("ok "),
        "active": act.startswith("True ") and "syncgc_cfg (False, None)" in act and "mainpin2_cfg (False, None)" in act,
        "propose_import": d.get("L6ABUNSET_PROPOSE_IMPORT", "") == "ok",
        "no_jitter_notes": d.get("L6ABUNSET_JITNOTES", "x").startswith("0 "),
        "hooks_reached_every_step": steps > 0 and reach.get("syncgc") == str(steps) and reach.get("mainpin2") == str(steps),
        "gc_untouched": last.get("gc_enabled") is True and not any("l6_jitter" in c for c in last.get("gc_callbacks", ["x"])),
        "propose_unwrapped": last.get("propose") == [True, False, False],
        "host_constant": d.get("L6ABUNSET_HOST_DISTINCT", "") == "1",
    }
    cl = d.get("L6ABUNSET_CLASSES", "")
    if cfg == "s2":
        parts = cl.split()
        try:
            classes = set(parts[parts.index("classes") + 1].split(","))
            on = set(parts[parts.index("levers_on_somewhere") + 1].split(","))
        except (ValueError, IndexError):
            classes, on = set(), set()
        checks["s2_levers_on"] = WANT_LEVERS_S2 <= on
    checks.update(class_checks("head", cfg))
    for k, v in checks.items():
        print(f"P7B head {cfg} {k} {'ok' if v else 'FAIL'}")
        if not v:
            fail(f"P7b-head-{cfg}-{k}")
    print(f"P7B head {cfg} reach {d.get('L6ABUNSET_REACH', '')} classes {cl}")

# 3. head == knockout (the verdict base on every tree): outputs + host state, and l6_jitter's internal state
for cfg in CFGS:
    ah, ak = first(get("head", cfg, "L6ABUNSET_ALL")), first(get("ko", cfg, "L6ABUNSET_ALL"))
    ih, ik = first(get("head", cfg, "L6ABUNSET_INTERNAL")), first(get("ko", cfg, "L6ABUNSET_INTERNAL"))
    ok = bool(ah) and ah == ak and bool(ih) and ih == ik
    print(f"P7B head-vs-KO {cfg} all {ah[:16]} / {ak[:16]} internal {ih[:16]} / {ik[:16]} {'EQUAL' if ok else 'DIFFER'}")
    if not ok:
        fail(f"P7b-KO-{cfg}")
    rk = get("ko", cfg, "L6ABUNSET_REACH")
    if "syncgc=0 mainpin2=0" not in rk:
        print(f"P7B KO {cfg} hooks still reached: {rk}")
        fail(f"P7b-KO-{cfg}-reach")

# 4. head == base3 (v5 L6 tree only; diagnostic on a composed head)
verdict = HEAD_SRC == V5_L6_SRC
for cfg in CFGS:
    ah, ab = first(get("head", cfg, "L6ABUNSET_ALL")), first(get("base3", cfg, "L6ABUNSET_ALL"))
    eq = bool(ah) and ah == ab
    lh, lb = runs[("head", cfg)][2]["lines"], runs[("base3", cfg)][2]["lines"]
    diff = [f"{x.split(' out=')[0]}" for x, y in zip(lh, lb) if x != y]
    hh, hb = first(get("head", cfg, "L6ABUNSET_HOST")), first(get("base3", cfg, "L6ABUNSET_HOST"))
    print(f"P7B head-vs-base3 {cfg} all {ah[:16]} / {ab[:16]} {'EQUAL' if eq else 'DIFFER'} host "
          f"{'equal' if hh and hh == hb else 'differ'} differing_runs={diff[:6]} "
          f"({'verdict' if verdict else 'diagnostic: composed head, KO is the verdict base'})")
    if verdict and not eq:
        fail(f"P7b-base3-{cfg}")

# 5. every mutant killed: outputs / host / internal state differ from KO in some config, or a head invariant fails
killed = 0
for m in MUTANTS:
    why = []
    for cfg in CFGS:
        tree, rc, d = runs[(m, cfg)]
        if first(d.get("L6ABUNSET_ALL", "")) != first(get("ko", cfg, "L6ABUNSET_ALL")):
            why.append(f"{cfg}:all")
        if first(d.get("L6ABUNSET_INTERNAL", "")) != first(get("ko", cfg, "L6ABUNSET_INTERNAL")):
            why.append(f"{cfg}:internal")
        if not d.get("L6ABUNSET_JITNOTES", "x").startswith("0 "):
            why.append(f"{cfg}:notes")
        if rc != 0:
            why.append(f"{cfg}:rc={rc}")
        if not d.get("L6ABUNSET_AST", "").startswith("ok "):
            why.append(f"{cfg}:ast")
        why += [f"{cfg}:{k}" for k, v in class_checks(m, cfg).items() if not v]
    built = all(runs[(m, c)][1] != 97 for c in CFGS)
    k = built and bool(why)
    killed += k
    print(f"P7B mutant {m} {'KILLED' if k else ('NOT-BUILT' if not built else 'SURVIVED')} {' '.join(why)}")
    if not k:
        fail(f"P7b-mut-{m}")
print(f"P7B mutants killed {killed}/{len(MUTANTS)}")

# 6. control M5c (L4 lever in LEVERS, prewarm fix on): class checks pass and it equals KO
for c in CONTROLS:
    bad = []
    for cfg in CFGS:
        tree, rc, d = runs[(c, cfg)]
        if rc != 0:
            bad.append(f"{cfg}:rc={rc}")
        bad += [f"{cfg}:{k}" for k, v in class_checks(c, cfg).items() if not v]
        if first(d.get("L6ABUNSET_ALL", "")) != first(get("ko", cfg, "L6ABUNSET_ALL")):
            bad.append(f"{cfg}:all")
        if first(d.get("L6ABUNSET_INTERNAL", "")) != first(get("ko", cfg, "L6ABUNSET_INTERNAL")):
            bad.append(f"{cfg}:internal")
        if "VLLM_TPU_L4_SAMP_FB_ORDER" not in d.get("L6ABUNSET_PREWARM", ""):
            bad.append(f"{cfg}:not-popped")
    print(f"P7B control {c} {'OK' if not bad else 'BAD ' + ' '.join(bad)}")
    if bad:
        fail(f"P7b-ctl-{c}")
print("P7B_FAILS", " ".join(fails))
PYEOF
for f in leg mk chk; do echo "P7b embedded $f.py sha256 $(sha256sum < $L/$f.py | cut -c1-16)"; done
[ "$(sha256sum < $L/leg.py | cut -c1-16)" = f4121072e6f8797f ] && [ "$(sha256sum < $L/mk.py | cut -c1-16)" = bd843e988045529a ] \
  && [ "$(sha256sum < $L/chk.py | cut -c1-16)" = 692ca0f244ab304f ] || { echo "Q38_L6_ABUNSET_SUMMARY FAIL embedded-sha"; exit 1; }
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
HS=$(srch $H); B3S=$(srch $B3)
echo "P7b head src $HS base3 src $B3S"
mkdir -p $L/logs
: > $L/runs.txt
for v in ko M1 M1b M2 M2b M3 M4 M5 M5c M6 M6r M7 M7r; do
  mkdir -p $L/t_$v && cp -r $H/src $L/t_$v/src
  V=$v; [ $v = ko ] && V=KO
  python3 $L/mk.py $L/t_$v $V || { echo "P7b variant $v not built"; [ $v = ko ] && FAIL="$FAIL P7b-KO-build"; touch $L/t_$v/NOTBUILT; }
done
CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
PAR=$(( $(nproc) > 2 ? $(nproc) - 1 : 1 )); [ $PAR -gt 9 ] && PAR=9
echo "P7b runs: 15 trees x 2 configs, $PAR in parallel, timeout 2400 s each"
run1() {  # tag cfg tree
  local tag=$1 cfg=$2 tree=$3 rc
  if [ -f $tree/NOTBUILT ]; then echo "$tag $cfg $tree 97" >> $L/runs.txt; return; fi
  local fix=1; [ $tag = M5 ] && fix=0
  (cd $tree && env $CPUENV L6ABU_PREWARM_FIX=$fix timeout -s KILL 2400 python3 $L/leg.py $H/tests/q38_l6 $cfg) > $L/logs/${tag}_$cfg.log 2>&1; rc=$?
  echo "$tag $cfg $tree $rc" >> $L/runs.txt
  echo "P7b done $tag $cfg rc=$rc $(date -u +%T)"
}
T0=$(date +%s)
for cfg in lines s2; do
  for tag in head ko base3 M1 M1b M2 M2b M3 M4 M5 M5c M6 M6r M7 M7r; do
    case $tag in head) tree=$H ;; base3) tree=$B3 ;; *) tree=$L/t_$tag ;; esac
    while [ "$(jobs -rp | wc -l)" -ge $PAR ]; do sleep 2; done
    run1 $tag $cfg $tree &
  done
done
wait
echo "P7b runs finished in $(( $(date +%s) - T0 )) s"
for tag in head ko base3; do for cfg in lines s2; do
  echo "--- $tag $cfg (L6ABUNSET lines)"; grep -a '^L6ABUNSET' $L/logs/${tag}_$cfg.log | cut -c1-400
  grep -aq '^L6ABUNSET_ALL ' $L/logs/${tag}_$cfg.log || tail -40 $L/logs/${tag}_$cfg.log
done; done
for m in M1 M1b M2 M2b M3 M4 M5 M5c M6 M6r M7 M7r; do
  echo "--- $m s2 (selected)"; grep -aE '^L6ABUNSET_(ALL|INTERNAL|JITNOTES|REACH|PREWARM|CLASSES|HOST_LAST)' $L/logs/${m}_s2.log | cut -c1-300
  grep -aq '^L6ABUNSET_ALL ' $L/logs/${m}_s2.log || tail -15 $L/logs/${m}_s2.log
done
python3 $L/chk.py $L/logs $HS $L/runs.txt > $L/chk.out 2>&1; crc=$?
cat $L/chk.out
F=$(grep '^P7B_FAILS' $L/chk.out | cut -d' ' -f2-)
[ $crc = 0 ] && grep -q '^P7B_FAILS' $L/chk.out || FAIL="$FAIL P7b-chk-rc$crc"
FAIL="$FAIL $F"
HS2=$(srch $H)
echo "P7b head src after $HS2 (before $HS)"
[ "$HS2" = "$HS" ] || FAIL="$FAIL P7b-src-changed"
FAIL=$(echo $FAIL)
if [ -z "$FAIL" ]; then echo "Q38_L6_ABUNSET_SUMMARY PASS head_src=$HS"; exit 0; fi
echo "Q38_L6_ABUNSET_SUMMARY FAIL head_src=$HS $FAIL"; exit 1

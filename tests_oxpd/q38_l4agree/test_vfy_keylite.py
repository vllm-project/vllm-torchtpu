# SPDX-License-Identifier: Apache-2.0
"""q38 L4 KEYLITE (VLLM_TPU_VPS_VFY_ONE_KEYLITE; H2 card T1-VFY-KEYLITE), torch CPU only (CDK CPU pod). AGREE-BUILD.

VFY-ONE's warm-key test with KEYLITE: runner/vfy_one.lite_key (the scheduler integers that fix every shape and value
of the full key) and VfyOne.warm_check (the lite key's first verdict reused; the full key on CHECK steps and every
KEYLITE_AUDIT_EVERY-th call; an unequal full key turns KEYLITE off). Calls are built as tpu_runner._vfy_verify builds
them (real tensors; 37 read-offset groups of one width, as served Flash logs 'groups=37 width=16').

1. Over a grid of served-like calls (P, T, drafts, target rows, F, groups, width, aliasing, pre-selected rows, the
   seed off / base / DraftAsm superset with its loop bucket and capacity), lite key and full key determine each other.
2. Over a random serving sequence with half the keys warm, every KEYLITE verdict equals the full-key verdict; the
   counters are exact.
3. Each lite component alone: a call that differs from a warm one only there has another lite key and falls to the
   cold path (a lite key that missed the change would reuse the warm verdict). With the superset seed on, T and P move
   exactly their own component and one seed argument each (L4-E: a missed seed component is caught too).
4. A change the lite key cannot see (groups of unequal width, which the served metadata walk never makes): the audit
   on the next CHECK step, or at the latest the KEYLITE_AUDIT_EVERY-th call, turns KEYLITE off and the full verdict
   decides from then on.
5. Flag: off = the full key on every call; the env default; from_env; the config hash.
"""

import json
import os
import random
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import vfy_one as V

K, VOCAB, H, MAXR = 3, 97, 64, 64
COUNTS: dict = {}


def seed_base(*a):
    pass


def seed_super(*a):
    pass


def _t(*s, d=torch.int32):
    return torch.zeros(s, dtype=d)


def make(P=16, T=64, D=None, Nt=None, F=0, G=37, W=16, alias=False, pre=False, spec=None, p=16, cap=64, wlast=None):
    """(call, lite inputs) as _vfy_verify builds them. spec: None | 'base' | 'super'."""
    D = P * K if D is None else D
    Nt = D if Nt is None else Nt
    hidden, indices = _t(T, H, d=torch.bfloat16), (None if pre else _t(P * (K + 1)))
    md = SimpleNamespace(bonus_logits_indices=_t(P), target_logits_indices=_t(Nt), draft_token_ids=_t(D),
                         draft_lengths=_t(P), segment_ids=_t(D), group_indices=_t(D))
    args = (md.bonus_logits_indices, md.target_logits_indices, _t(P, 1, d=torch.float32), _t(P, 1),
            _t(P, 1, d=torch.float32), _t(P, VOCAB, d=torch.float32), _t(D, d=torch.float32),
            _t(Nt, VOCAB, d=torch.float32), md.draft_token_ids, md.draft_lengths, md.segment_ids, md.group_indices,
            None, K, "conditional")
    if alias:
        one = _t(W)
        groups = [one] * G
    else:
        groups = [_t(W) for _ in range(G)]
    if wlast is not None:
        groups[-1] = _t(wlast)
    fb_u = _t(F, VOCAB, d=torch.float32) if F else None
    call = (object(), hidden, indices, *args, _t(1), F, True, fb_u, list(groups), _t(), torch.int32, _t(3, d=torch.int64))
    sp = None
    if spec == "base":
        sp = V.VfySeedSpec(seed_base, (_t(P), _t(cap + 1), _t(), _t(T), p, cap), ())
    elif spec == "super":
        sp = V.VfySeedSpec(seed_super, (_t(P), _t(cap + 1), _t(), _t(T), _t(cap), p, cap, MAXR), ())
    if sp is not None:
        call = (*call, sp.fn, *sp.args)
    lite = V.lite_key(hidden, indices, P, md, F, call[22], sp)
    return call, lite


def _grid():
    out = []
    for P, T in ((8, 32), (16, 64), (16, 80)):
        for D in (P * K, P * K - 2):
            for F in (0, 8):
                for G, W, alias in ((1, 16, False), (37, 16, False), (37, 16, True), (37, 64, False)):
                    for pre in (False, True):
                        for spec, p, cap in ((None, 0, 0), ("base", P, 64), ("base", P, 128), ("super", P, 64)):
                            out.append(dict(P=P, T=T, D=D, F=F, G=G, W=W, alias=alias, pre=pre, spec=spec, p=p, cap=cap))
    out.append(dict(P=16, T=64, D=48, Nt=40))  # target rows apart from the drafts
    return out


GRID = _grid()


def test_lite_key_and_full_key_determine_each_other():
    by_lite, by_full = {}, {}
    for kw in GRID:
        call, lite = make(**kw)
        full = V.key(*call[1:])
        by_lite.setdefault(lite, set()).add(full)
        by_full.setdefault(full, set()).add(lite)
    assert all(len(v) == 1 for v in by_lite.values()), "a lite key with two full keys"
    assert all(len(v) == 1 for v in by_full.values()), "a full key with two lite keys"
    assert len(by_lite) == len(by_full) == len(GRID)
    COUNTS["grid"] = len(GRID)


def test_keylite_verdicts_equal_the_full_key_verdicts():
    rng = random.Random(4242)
    calls = [make(**kw) for kw in GRID]
    warm = {V.key(*c[1:]) for i, (c, _) in enumerate(calls) if i % 2 == 0}
    v = V.VfyOne(1, 0, "", keylite=True)
    v.warm = warm
    seq = [rng.randrange(len(calls)) for _ in range(3000)]
    seen, audits = set(), 0
    for i, j in enumerate(seq):
        call, lite = calls[j]
        audit = i % 97 == 0
        hit = lite in v.lite
        nxt = v.lite_calls + 1  # warm_check counts the call first
        want = V.key(*call[1:]) in warm
        assert v.warm_check(call, lite, audit) is want, (i, j)
        audits += hit and (audit or nxt % V.KEYLITE_AUDIT_EVERY == 0)
        seen.add(lite)
    s = v.stats
    assert not v.keylite_off and s["keylite_miss"] == len(seen) and s["keylite_audit"] == audits
    assert s["keylite_hit"] + s["keylite_miss"] + s["keylite_audit"] == len(seq) == v.lite_calls
    COUNTS["verdicts"] = len(seq)
    COUNTS["hits"] = s["keylite_hit"]


BASE = dict(P=16, T=64, F=8, G=37, W=16, spec="super", p=16, cap=64)
# name -> (base overrides, variant overrides): each variant moves exactly one lite component
VARIANTS = {
    "T": (dict(spec=None), dict(spec=None, T=80)),  # the seed's input_ids are T long: seed off isolates T
    "pre_selected": ({}, dict(pre=True)),
    "P": (dict(pre=True, spec=None), dict(pre=True, spec=None, P=8, D=48)),  # the seed's num_draft is P long
    "drafts": ({}, dict(D=46, Nt=48)),
    "target_rows": ({}, dict(Nt=40)),
    "F": ({}, dict(F=0)),
    "groups": ({}, dict(G=36)),
    "width": ({}, dict(W=64)),
    "aliasing": ({}, dict(alias=True)),
    "seed_off": ({}, dict(spec=None)),
    "seed_body": ({}, dict(spec="base")),
    "seed_loop_bucket": ({}, dict(p=8)),
    "seed_capacity": ({}, dict(cap=128)),
}


@pytest.mark.parametrize("what", sorted(VARIANTS))
def test_each_lite_component_alone_falls_to_the_cold_path(what):
    bo, vo = VARIANTS[what]
    base, base_lite = make(**{**BASE, **bo})
    call, lite = make(**{**BASE, **bo, **vo})
    assert sum(a != b for a, b in zip(lite, base_lite)) == 1, (what, lite, base_lite)
    v = V.VfyOne(1, 0, "", keylite=True)
    v.warm = {V.key(*base[1:])}
    assert v.warm_check(base, base_lite, False) is True
    assert V.key(*call[1:]) not in v.warm
    assert v.warm_check(call, lite, False) is False and v.warm_check(call, lite, False) is False
    assert v.warm_check(base, base_lite, False) is True and not v.keylite_off
    COUNTS[f"cold_{what}"] = 1


# With DraftAsm's superset seed on, T and P also move one seed argument each (its input_ids are T long, its num_draft
# P long): name -> (base overrides, variant overrides, changed lite components, changed seed arguments). Lite layout:
# 0 T, 1 N, 2 P, 3 drafts, 4 target rows, 5 F, 6 groups, 7 width, 8 aliasing, 9 seed (fn, per-argument shapes / values).
SEED_ON = {
    "T_seed_on": ({}, dict(T=80), {0, 9}, {3}),
    "P_seed_on": (dict(pre=True), dict(pre=True, P=8, D=48), {2, 9}, {0}),
}


@pytest.mark.parametrize("what", sorted(SEED_ON))
def test_seed_on_t_and_p_move_exactly_their_component_and_their_seed_argument(what):
    bo, vo, want_top, want_seed = SEED_ON[what]
    base, base_lite = make(**{**BASE, **bo})
    call, lite = make(**{**BASE, **bo, **vo})
    assert {i for i, (a, b) in enumerate(zip(lite, base_lite)) if a != b} == want_top, (what, lite, base_lite)
    assert lite[9][0] is base_lite[9][0]
    assert {i for i, (a, b) in enumerate(zip(lite[9][1], base_lite[9][1])) if a != b} == want_seed, what
    v = V.VfyOne(2, 0, "", keylite=True)
    v.warm = {V.key(*base[1:])}
    assert v.warm_check(base, base_lite, False) is True
    assert V.key(*call[1:]) not in v.warm and v.warm_check(call, lite, False) is False
    COUNTS[f"cold_{what}"] = 1


def test_the_audit_turns_keylite_off_on_a_change_the_lite_key_cannot_see():
    base, base_lite = make(**BASE)
    odd, odd_lite = make(**BASE, wlast=8)  # one group narrower: never made by the served metadata walk
    assert odd_lite == base_lite and V.key(*odd[1:]) != V.key(*base[1:])
    # (a) a CHECK step audits at once
    v = V.VfyOne(1, 0, "", keylite=True)
    v.warm = {V.key(*base[1:])}
    assert v.warm_check(base, base_lite, False) is True
    assert v.warm_check(odd, odd_lite, True) is False and v.keylite_off
    assert v.warm_check(base, base_lite, False) is True and v.stats["keylite_audit"] == 1  # full keys from now on
    # (b) without CHECK steps: the stale verdict lasts until the KEYLITE_AUDIT_EVERY-th call at most
    v = V.VfyOne(1, 0, "", keylite=True)
    v.warm = {V.key(*base[1:])}
    got = [v.warm_check(base if i == 0 else odd, base_lite, False) for i in range(V.KEYLITE_AUDIT_EVERY + 3)]
    first_off = got.index(False)
    assert first_off == V.KEYLITE_AUDIT_EVERY - 1 and not any(got[first_off:]) and v.keylite_off, first_off
    COUNTS["audit_stale_calls"] = first_off - 1


def test_the_memo_follows_its_warm_set():
    base, base_lite = make(**BASE)
    v = V.VfyOne(1, 0, "", keylite=True)
    v.warm = {V.key(*base[1:])}
    assert v.warm_check(base, base_lite, False) is True
    v.warm = set()  # replaced: the verdicts made against the old set are dropped
    assert v.warm_check(base, base_lite, False) is False
    v.warm.add(V.key(*base[1:]))  # grown (as warm-up grows it)
    assert v.warm_check(base, base_lite, False) is True and v.stats["keylite_miss"] == 3


def _runner_src():
    import ast

    root = os.environ.get("Q38MM_SRC_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
    src = open(os.path.join(root, "vllm_torchtpu", "runner", "tpu_runner.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_vfy_verify")
    lite_init = [n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "lite"]
    lite_if = [n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "vfy.keylite"]
    cold_if = [n for n in fn.body if isinstance(n, ast.If) and "warm_check" in ast.unparse(n.test)]
    assert len(lite_init) == len(lite_if) == len(cold_if) == 1, (len(lite_init), len(lite_if), len(cold_if))
    stmts = compile(ast.Module(body=lite_init + lite_if, type_ignores=[]), "<_vfy_verify>", "exec")
    test = compile(ast.Expression(body=cold_if[0].test), "<_vfy_verify test>", "eval")
    return stmts, test


@pytest.mark.parametrize("keylite", [False, True])
def test_the_runner_statements_route_the_warm_check_through_keylite(keylite):
    stmts, test = _runner_src()
    kw = {**BASE, "spec": "super"}
    call, lite = make(**kw)
    v = V.VfyOne(2, 0, "", keylite=keylite)
    v.warm = {V.key(*call[1:])}
    chunk = SimpleNamespace(hidden=call[1], indices=call[2])
    md = SimpleNamespace(draft_token_ids=call[11], target_logits_indices=call[4])
    spec = V.VfySeedSpec(seed_super, tuple(call[27:]), ())
    loc = {"vfy": v, "call": call, "chunk": chunk, "padded_num_reqs": BASE["P"], "md": md,
           "fallback_rows": BASE["F"], "ro": (call[22], call[23], call[24]), "spec": spec, "mode": None,
           "_vfy_one": V,
           # v5 composition (V5COMP-BUILD; tests only): VRNG-VPS's composition fix 3 makes the warm check read `kcall`
           # (the meta-draw key call; the raw call on a base-draw step, as here) beside `rng` (None: no VRNG step).
           "kcall": call, "rng": None}
    exec(stmts, {}, loc)
    assert (loc["lite"] == lite) if keylite else (loc["lite"] is None)
    cold = eval(test, {}, loc)
    assert cold is False
    assert v.stats["keylite_miss"] == (1 if keylite else 0) and v.lite_calls == (1 if keylite else 0)
    COUNTS[f"wiring_{keylite}"] = 1


def test_flag_off_env_default_from_env_and_config_hash(monkeypatch):
    base, base_lite = make(**BASE)
    v = V.VfyOne(1, 0, "", keylite=False)
    v.warm = {V.key(*base[1:])}
    for _ in range(3):
        assert v.warm_check(base, None, False) is True
    assert not v.lite and v.lite_calls == 0 and not any(k.startswith("keylite") for k in v.stats)
    monkeypatch.delenv("VLLM_TPU_VPS_VFY_ONE_KEYLITE", raising=False)
    off = V.VfyOne(1, 0, "")
    assert off.keylite is False
    monkeypatch.setenv("VLLM_TPU_VPS_VFY_ONE_KEYLITE", "1")
    on = V.VfyOne(1, 0, "")
    assert on.keylite is True and on.config_hash != off.config_hash
    import hashlib

    assert off.config_hash == int(hashlib.sha1(repr((1, "", 0)).encode()).hexdigest()[:15], 16)  # unchanged off
    envs = SimpleNamespace(vfy_one_value=lambda: 2, VLLM_TPU_VPS_VFY_ONE_CHECK=0, VLLM_TPU_VPS_VFY_ONE_AB="",
                           VLLM_TPU_VPS_VFY_ONE_KEYLITE=True)
    monkeypatch.delenv("VLLM_TPU_VPS_VFY_ONE_KEYLITE", raising=False)
    assert V.from_env(envs).keylite is True
    envs.VLLM_TPU_VPS_VFY_ONE_KEYLITE = False
    monkeypatch.setenv("VLLM_TPU_VPS_VFY_ONE_KEYLITE", "1")
    assert V.from_env(envs).keylite is False  # the runner's value wins over the environment


def test_zz_counts():
    out = os.environ.get("Q38_KEYLITE_COUNTS")
    if out:
        with open(out, "w") as f:
            json.dump(COUNTS, f, sort_keys=True)
    print("Q38_KEYLITE_COUNTS " + json.dumps(COUNTS, sort_keys=True))

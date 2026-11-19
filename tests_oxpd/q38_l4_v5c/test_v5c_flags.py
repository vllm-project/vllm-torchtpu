# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c): flag table, compile-key rows and the intended v5 decode env. CDK CPU pod only.
V5COMP-BUILD (helper of L4-E).

F1  every L4 v5 flag is registered in envs.py, resolves to its off value when unset, and is compile-hashed or
    compile-ignored exactly as the flag table says (hashed = adds or changes compiled programs; ignored = host-only
    or picks between precompiled programs). The env vars new on the branch since ecf4cc34a are exactly these.
F2  l6_ab.LEVERS is exactly the union (14 levers with v6's k1fuse); every lever's env var is registered and compile-ignored.
F3  render-grep exemption on the intended v5 decode env: q38-L1-v4's il registry (q38_l1sv4_flags_il.e1a3c835.tsv,
    sha256 pinned; its decode / both rows with an on-value) plus the v5 levers' on-values, uniform (no *_AB key):
    no VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS value with a colon, no key containing REQUEST_SHARD_MIN_ROWS, no comma in a
    v5 on-value; the check bites on each injected violation.
F4  the v5 on-values resolve through envs.py to the intended settings.
F5  v6 (q38-l4-v6c2, V6COMP-BUILD): the two v6 configurations the lead picks between, each the v5 env plus H3
    (VLLM_TPU_MK_DRAFT_ATTN=1) plus K1 (VLLM_TPU_K1FUSE=1): config K drops VRNG (VLLM_TPU_VPS_INGRAPH_RNG unset; K1
    engages), config V keeps it (K1 refuses for the run); both render-grep clean and resolving; H3's env prerequisite
    VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE=1 is in the il decode rows (shared selection and the MTP tail are reported).
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v5c_flags.py
"""

import hashlib
import os
import pathlib
import re

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import l6_ab

HERE = pathlib.Path(__file__).resolve().parent
REG = HERE / "q38_l1sv4_flags_il.e1a3c835.tsv"
REG_SHA256 = "6ee776f4c33a219c52c141ba93b091be43bee151463df7832357c3b01fa762e2"

# flag -> (compile class, off value as envs resolves it, piece)
FLAGS = {
    "VLLM_TPU_L4_SAMP_SNAP": ("ignored", "", "sampsnap"),
    "VLLM_TPU_L4_SAMP_SNAP_CHECK": ("ignored", 0, "sampsnap"),
    "VLLM_TPU_L4_VPS_STATS_ASYNC": ("ignored", "", "statsasync"),
    "VLLM_TPU_DRAFT_ASM_FOLD": ("hashed", False, "DRAFT-ASM"),
    "VLLM_TPU_DRAFT_ASM_CHECK_STEPS": ("ignored", 0, "DRAFT-ASM"),
    "VLLM_TPU_DRAFT_ASM_AB": ("ignored", "", "DRAFT-ASM"),
    "VLLM_TPU_VPS_VFY_ONE": ("hashed", False, "VFY-ONE"),
    "VLLM_TPU_VPS_VFY_ONE_CHECK": ("ignored", 0, "VFY-ONE"),
    "VLLM_TPU_VPS_VFY_ONE_AB": ("ignored", "", "VFY-ONE"),
    "VLLM_TPU_VPS_VFY_ONE_KEYLITE": ("ignored", False, "KEYLITE"),
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP": ("hashed", False, "DH-B"),
    "VLLM_TPU_DRAFT_ASM_ONE": ("hashed", False, "DH-D"),
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB": ("ignored", "", "DH-B"),
    "VLLM_TPU_DRAFT_ASM_ONE_AB": ("ignored", "", "DH-D"),
    "VLLM_TPU_DRAFT_EMBED_REPL": ("hashed", False, "DRAFT-2 (c)"),
    "VLLM_TPU_TARGET_EMBED_REPL": ("hashed", False, "EMB-REP-TGT"),
    "VLLM_TPU_DRAFT_ASM_CAP": ("hashed", False, "DRAFT-CAP"),
    "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT": ("hashed", False, "LMH-STREAM s0"),
    "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY": ("hashed", False, "LMH-STREAM s0 verify (class B: not a v5 member, never set)"),
    "VLLM_TPU_SEEDED_SAMPLING": ("ignored", True, "SEEDFIX (default ON)"),
    "VLLM_TPU_VPS_INGRAPH_RNG": ("hashed", False, "VRNG-VPS"),
    "VLLM_TPU_VPS_INGRAPH_RNG_AB": ("ignored", "", "VRNG-VPS"),
    "VLLM_TPU_VPS_TOPK_CUTOFF": ("hashed", False, "S2"),
    # v6 (V6COMP-BUILD)
    "VLLM_TPU_K1FUSE": ("hashed", False, "T1-K1FUSE"),
    "VLLM_TPU_K1FUSE_AB": ("ignored", "", "T1-K1FUSE"),
    "VLLM_TPU_K1FUSE_CHECK": ("ignored", 256, "T1-K1FUSE"),
    "VLLM_TPU_MK_DRAFT_ATTN": ("hashed", False, "L1-H3 / ATTNWIRE"),
}
UNION = {
    "envcache": "VLLM_TPU_L6_ENVS_CACHE", "fctx": "VLLM_TPU_L6_FAST_FWD_CTX", "gcfreeze": "VLLM_TPU_L6_GC_FREEZE",
    "switch": "VLLM_TPU_L6_SWITCH", "mainpin": "VLLM_TPU_L6_MAIN_PIN", "defer": "VLLM_TPU_L6_DEFER_PREV",
    "sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER", "sampsnap": "VLLM_TPU_L4_SAMP_SNAP",
    "statsasync": "VLLM_TPU_L4_VPS_STATS_ASYNC", "vfyone": "VLLM_TPU_VPS_VFY_ONE_AB",
    "draftasm": "VLLM_TPU_DRAFT_ASM_AB", "draftasmcl": "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
    "draftasmone": "VLLM_TPU_DRAFT_ASM_ONE_AB", "k1fuse": "VLLM_TPU_K1FUSE_AB",
}
# the v5 levers' on-values (L4-E's v5 set: DRAFT-ASM + VFY-ONE=2 + DH-D + sampsnap + statsasync + DRAFT-2 (c) +
# EMB-REP-TGT; uniform, so no window variable)
V5_ON = {
    "VLLM_TPU_DRAFT_ASM_FOLD": "1",
    "VLLM_TPU_DRAFT_ASM_ONE": "1",
    "VLLM_TPU_VPS_VFY_ONE": "2",
    "VLLM_TPU_L4_SAMP_SNAP": "1",
    "VLLM_TPU_L4_VPS_STATS_ASYNC": "1",
    "VLLM_TPU_DRAFT_EMBED_REPL": "1",
    "VLLM_TPU_TARGET_EMBED_REPL": "1",
    "VLLM_TPU_DRAFT_ASM_CAP": "1",
    "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT": "1",
    "VLLM_TPU_VPS_INGRAPH_RNG": "1",
    "VLLM_TPU_VPS_TOPK_CUTOFF": "1",
}
KEY_HCF = "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"
KEY_MR = "REQUEST_SHARD_MIN_ROWS"
BASE_ENV_FILE = os.environ.get("Q38_V5C_BASE_ENVS")  # base (ecf4cc34a) envs.py, set by run_cpu.sh


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(FLAGS) + list(UNION.values()):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_f1_flag_table(clean_env):
    rows = []
    for name, (cls, off, piece) in FLAGS.items():
        assert name in envs.environment_variables, name
        got = getattr(envs, name)
        assert got == off and type(got) is type(off), (name, got, off)
        ignored = name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert ignored == (cls == "ignored"), (name, cls, ignored)
        rows.append(f"{name}={cls}/{piece}")
    if BASE_ENV_FILE:
        base = set(re.findall(r'^    "(VLLM_[A-Z0-9_]+)":', open(BASE_ENV_FILE).read(), re.M))
        head = set(re.findall(r'^    "(VLLM_[A-Z0-9_]+)":', open(envs.__file__).read(), re.M))
        assert head - base == set(FLAGS) and not base - head, (sorted(head - base), sorted(base - head))
    print("\nV5C_FLAGS F1 " + " ".join(rows) + f" new_vs_base={'checked' if BASE_ENV_FILE else 'SKIPPED'}")
    assert BASE_ENV_FILE, "Q38_V5C_BASE_ENVS unset: the new-flag set was not compared against ecf4cc34a"


def test_f2_levers_union_registered_and_ignored():
    assert dict(l6_ab.LEVERS) == UNION, set(l6_ab.LEVERS.items()) ^ set(UNION.items())
    assert len(set(UNION.values())) == len(UNION)
    for lever, var in UNION.items():
        assert var in envs.environment_variables, (lever, var)
        assert var in tpu_compiler._TPU_COMPILE_ENV_IGNORED, (lever, var)
    print("\nV5C_FLAGS F2 LEVERS " + ",".join(f"{k}={v}" for k, v in l6_ab.LEVERS.items()))


def _decode_env():
    raw = REG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REG_SHA256
    env = {}
    for ln in raw.decode().splitlines():
        if not ln or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if f[0] == "id" or len(f) < 4:
            continue
        if f[2] in ("decode", "both") and f[3] != "-":
            env[f[1]] = f[3]
    env.update(V5_ON)
    return env


def rendergrep(env):
    """(dirty count, reasons) of the render-grep exemption check on one decode env dict."""
    why = [f"{k}={v} (colon)" for k, v in env.items() if k == KEY_HCF and ":" in v]
    why += [f"{k} present" for k in env if KEY_MR in k]
    why += [f"{k}={v} (comma)" for k, v in env.items() if k in V5_ON and "," in v]
    why += [f"{k} (window key)" for k in env if k in FLAGS and k.endswith("_AB")]
    return len(why), why


def test_f3_rendergrep_on_the_intended_v5_decode_env():
    env = _decode_env()
    n, why = rendergrep(env)
    assert n == 0, why
    bites = {
        "pod MIN_ROWS": dict(env, VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS="48"),
        "HC colon": dict(env, **{KEY_HCF: "16:64"}),
        "v5 comma": dict(env, VLLM_TPU_VPS_VFY_ONE="2,1"),
        "v5 window": dict(env, VLLM_TPU_DRAFT_ASM_ONE_AB="ab:1"),
    }
    for tag, e in bites.items():
        assert rendergrep(e)[0] == 1, tag
    assert rendergrep(dict(env, **{KEY_HCF: "128"}))[0] == 0  # a colon-free HC value stays clean
    print(f"\nV5C_FLAGS F3 RENDERGREP CLEAN on the intended v5 decode env ({len(env)} keys: il registry e1a3c835 "
          f"decode/both on-values + {len(V5_ON)} v5 keys {sorted(V5_ON)}); {KEY_HCF} "
          f"{'present=' + env[KEY_HCF] if KEY_HCF in env else 'absent'}; bites {len(bites)}/{len(bites)}")


def test_f4_v5_on_values_resolve(clean_env):
    for k, v in V5_ON.items():
        clean_env.setenv(k, v)
    assert envs.VLLM_TPU_DRAFT_ASM_FOLD is True and envs.VLLM_TPU_DRAFT_ASM_ONE is True
    assert envs.VLLM_TPU_VPS_VFY_ONE is True and envs.vfy_one_value() == 2
    assert envs.VLLM_TPU_L4_SAMP_SNAP == "1" and envs.VLLM_TPU_L4_VPS_STATS_ASYNC == "1"
    assert envs.VLLM_TPU_DRAFT_EMBED_REPL is True and envs.VLLM_TPU_TARGET_EMBED_REPL is True
    assert envs.VLLM_TPU_DRAFT_ASM_CAP is True and envs.VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT is True
    assert "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY" not in V5_ON  # class B, not a v5 member
    assert envs.VLLM_TPU_SEEDED_SAMPLING is True  # SEEDFIX is default ON: the v5 env needs no key for it
    assert envs.VLLM_TPU_VPS_INGRAPH_RNG is True and envs.VLLM_TPU_VPS_TOPK_CUTOFF is True
    print("\nV5C_FLAGS F4 " + " ".join(f"{k}={v}" for k, v in V5_ON.items()) + " resolve as intended")


# v6 (V6COMP-BUILD): the two configurations (L4-E / lead 2026-11-14): K = K1 on, VRNG off; V = VRNG on, K1 refusing.
V6_K = {**{k: v for k, v in V5_ON.items() if k != "VLLM_TPU_VPS_INGRAPH_RNG"},
        "VLLM_TPU_K1FUSE": "1", "VLLM_TPU_MK_DRAFT_ATTN": "1"}
V6_V = {**V5_ON, "VLLM_TPU_K1FUSE": "1", "VLLM_TPU_MK_DRAFT_ATTN": "1"}


def test_f5_v6_configurations(clean_env):
    env0 = _decode_env()
    for tag, on in (("K", V6_K), ("V", V6_V)):
        env = {k: v for k, v in env0.items() if k not in V5_ON}
        env.update(on)
        n, why = rendergrep(env)
        assert n == 0, (tag, why)
        for k in FLAGS:
            clean_env.delenv(k, raising=False)
        for k, v in on.items():
            clean_env.setenv(k, v)
        assert envs.VLLM_TPU_K1FUSE is True and envs.VLLM_TPU_MK_DRAFT_ATTN is True and envs.VLLM_TPU_DRAFT_ASM_CAP is True
        assert envs.VLLM_TPU_VPS_INGRAPH_RNG is (tag == "V"), tag
        assert envs.VLLM_TPU_K1FUSE_AB == "" and envs.VLLM_TPU_K1FUSE_CHECK == 256
        print(f"\nV5C_FLAGS F5 config {tag}: " + " ".join(f"{k}={v}" for k, v in sorted(on.items()))
              + f" render-grep clean ({len(env)} keys)")
    assert env0.get("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE") == "1", env0.get("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE")
    print(f"\nV5C_FLAGS F5 H3 prerequisites in the il decode rows: VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE="
          f"{env0.get('VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE')} VLLM_TPU_QSA_DRAFT_SHARED_SELECTION="
          f"{env0.get('VLLM_TPU_QSA_DRAFT_SHARED_SELECTION', 'ABSENT')} (the attached MTP tail mtp=1 is a megakernel "
          f"spec option, not an env row here)")

#!/usr/bin/env python3
"""q38 L6 seed family (lever setupfree, runner/l6_setupfree.py) mutant runner (CPU pod only).
Usage: python3 q38_l6_seed_mutrun.py <head tree> <set> [survey]

Modelled on L6-ENGINE's q38_l6_engalloc_mutrun.py and L6-PTRIMS's q38_l6_ptrims_mutrun.py. For each mutant: copy
the tree, apply one exact textual replacement (it must match exactly once), run the mutant's target test id with
pytest and require that it FAILED (a failing test, not an error or a collection failure). A control run on the
unmutated tree must pass every target first. 'survey' reports every mutant without requiring all to be killed.
Last line: Q38_L6_SEED_MUT PASS killed N/N of N | Q38_L6_SEED_MUT FAIL ... | Q38_L6_SEED_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

F = "src/vllm_torchtpu/runner/l6_setupfree.py"
ST = "src/vllm_torchtpu/runner/l6_steady.py"
T = "tests/q38_l6/test_l6_setupfree.py"
EQ = T + "::test_pingpong_equals_clone_over_random_streams[g5-plain]"
REG = "tests/compilation/test_tpu_compile_key_registration.py"

MUTANTS = {
    "seed": [
        # Stale reuse: the reference-count guard.
        ("S1 guard dropped: a held set is rewritten (stale reuse)", F,
         "    if _counts(s) != s.want:\n",
         "    if False:\n", T + "::test_held_set_is_never_rewritten[dict]"),
        ("S2 guard compares the dict count only: a held object is rewritten", F,
         "    if _counts(s) != s.want:\n",
         "    if _counts(s)[0] != s.want[0]:\n", T + "::test_held_set_is_never_rewritten[object]"),
        ("S3 expected counts off by one: every set reads busy", F,
         "    s.want = (lc, [oc + k for k in refs])\n",
         "    s.want = (lc, [oc + k + 1 for k in refs])\n", EQ),
        # Stale reuse: templates after STEADY re-recorded; structure changes.
        ("S4 re-template skipped after STEADY re-recorded (stale templates)", F,
         "    if s.cache is not c:\n",
         "    if False:\n", EQ),
        ("S5 structure change not detected (sets of the old plan reused)", F,
         "        if st is None or st.sig != sig:\n",
         "        if st is None:\n", T + "::test_group_structure_change_resets"),
        # The ping-pong turn.
        ("S6 ping-pong set not swapped", F,
         "    st.nxt = 1 - k\n",
         "    st.nxt = k\n", T + "::test_served_hold_alternates_the_two_sets"),
        # Wrong offsets / wrong fields.
        ("S7 object index offset by one (wrong group per slot)", F,
         "        objs[i] = per_layer[name]\n",
         "        objs[i - 1] = per_layer[name]\n", EQ),
        ("S8 mamba groups rewritten with the attention step fields", F,
         "    list(map(dict.update, s.md, itertools.repeat(step_m)))\n",
         "    list(map(dict.update, s.md, itertools.repeat(step_all)))\n", EQ),
        # The l6_ab gate.
        ("S9 lever state ignored (always on)", F,
         "    if _l6_ab.on(LEVER) and not runner.__dict__.get(\"_l6_sf_off\", False):\n",
         "    if not runner.__dict__.get(\"_l6_sf_off\", False):\n", T + "::test_lever_off_takes_the_clone"),
        ("S10 STEADY call site ignores the lever", ST,
         "    if _l6_setupfree.on(runner):\n",
         "    if True:\n", T + "::test_lever_off_takes_the_clone"),
        ("S11 kept sets not dropped when the lever is off", F,
         "    runner.__dict__.pop(\"_l6_sf\", None)\n    return False\n",
         "    return False\n", T + "::test_lever_off_takes_the_clone"),
        ("S12 STEADY's drop keeps the kept sets", ST,
         "    d.pop(\"_l6_sf\", None)  # q38 L6 lever setupfree's kept sets (runner/l6_setupfree.py)\n",
         "", T + "::test_steady_drop_releases_the_kept_sets"),
        # CHECK.
        ("S13 CHECK ignores VLLM_TPU_L6_CHECK_STEPS", F,
         "    return v if _l6_ab.check_allowed() else 0\n",
         "    return v\n", T + "::test_check_steps_bound_the_check"),
        ("S14 CHECK never reports a mismatch", F,
         "        why = _differ(ref, out)\n",
         "        why = None\n", T + "::test_check_catches_a_broken_rewrite"),
        ("S15 CHECK turn never comes", F,
         "        st.since_check = 0\n        return True\n",
         "        st.since_check = 0\n        return False\n", T + "::test_check_mode_compares_and_finds_no_mismatch"),
        ("S16 a mismatch does not turn the lever off", F,
         "    runner._l6_sf_off = True\n",
         "    pass\n", T + "::test_check_catches_a_broken_rewrite"),
        # Registration rows.
        ("S17 lever missing from l6_ab.LEVERS", "src/vllm_torchtpu/core/l6_ab.py",
         "    \"setupfree\": \"VLLM_TPU_L6_SETUP_FREE\",\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
        ("S18 lever flag missing from envs.py", "src/vllm_torchtpu/envs.py",
         "    \"VLLM_TPU_L6_SETUP_FREE\": env_str(\"VLLM_TPU_L6_SETUP_FREE\", \"\"),\n", "",
         REG + "::test_flag_is_registered_and_parses[VLLM_TPU_L6_SETUP_FREE-1-]"),
        ("S19 CHECK flag missing from envs.py", "src/vllm_torchtpu/envs.py",
         "    \"VLLM_TPU_L6_SETUP_FREE_CHECK\": env_nonnegative_int(\"VLLM_TPU_L6_SETUP_FREE_CHECK\", 0),\n", "",
         REG + "::test_flag_is_registered_and_parses[VLLM_TPU_L6_SETUP_FREE_CHECK-200-0]"),
        ("S20 lever flag dropped from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_SETUP_FREE\",\n    \"VLLM_TPU_L6_SETUP_FREE_CHECK\",\n",
         "    \"VLLM_TPU_L6_SETUP_FREE_CHECK\",\n",
         REG + "::test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_L6_SETUP_FREE-1-]"),
        ("S21 CHECK flag dropped from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_SETUP_FREE\",\n    \"VLLM_TPU_L6_SETUP_FREE_CHECK\",\n",
         "    \"VLLM_TPU_L6_SETUP_FREE\",\n",
         REG + "::test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_L6_SETUP_FREE_CHECK-200-0]"),
        ("S22 compile-key registration row of the lever flag dropped", REG,
         "    (\"VLLM_TPU_L6_SETUP_FREE\", \"1\", \"\"),\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
        ("S23 compile-key registration row of the CHECK flag dropped", REG,
         "    (\"VLLM_TPU_L6_SETUP_FREE_CHECK\", \"200\", 0),\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
    ],
}


def run(tree, test_id, env):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", test_id]
    p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1500)
    out = p.stdout + p.stderr
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error|errors|skipped).* in [0-9.]+s", ln)]
    return p.returncode, (summ[-1] if summ else "NO SUMMARY"), out


def main():
    tree, which = os.path.abspath(sys.argv[1]), sys.argv[2]
    survey = len(sys.argv) > 3 and sys.argv[3] == "survey"
    muts = MUTANTS[which]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="src")
    env.pop("Q38_L6_EXPECT_TREE", None)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    targets = sorted({m[4] for m in muts})
    ok = True
    for t in targets:
        rc, s, out = run(tree, t, env)
        good = rc == 0 and re.search(r"\b[1-9][0-9]* passed\b", s) and "failed" not in s and "error" not in s
        print(f"CONTROL {'ok' if good else 'BAD'} {t}: {s}")
        if not good:
            print(out[-3000:])
            ok = False
    if not ok and not survey:
        print("Q38_L6_SEED_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6sdmut.")
        shutil.copytree(os.path.join(tree, "src"), os.path.join(d, "src"))
        shutil.copytree(os.path.join(tree, "tests"), os.path.join(d, "tests"))
        f = os.path.join(d, path)
        s = open(f).read()
        n = s.count(old)
        if n != 1:
            print(f"MUT {name}: anchor matched {n} times -> INVALID")
            shutil.rmtree(d, ignore_errors=True)
            continue
        open(f, "w").write(s.replace(old, new))
        rc, summ, out = run(d, t, env)
        dead = rc != 0 and re.search(r"\b[1-9][0-9]* failed\b", summ) and "error" not in summ
        killed += bool(dead)
        if dead:
            names.append(name.split()[0])
        print(f"MUT {name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {summ}")
        if not dead:
            print(out[-1500:])
        shutil.rmtree(d, ignore_errors=True)
    tot = len(muts)
    if survey:
        print(f"Q38_L6_SEED_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_SEED_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_SEED_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

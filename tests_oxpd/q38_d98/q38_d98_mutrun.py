# SPDX-License-Identifier: Apache-2.0
"""q38 L4 D98 / X-H14-D98R (on the D14 serve stack) mutation runner; the AOT-skip mutant M16 moved to
tests/q38_aotskip (AOTSKIP has its own proof).

q38 L4 D98 mutation runner: break the code deliberately, the tests must fail.

Copies src/ and tests/ to a scratch dir, runs the control (unmutated: every
target must pass), then for each mutant applies one exact text replacement
(the old text must occur exactly once), runs that mutant's target tests and
requires a non-zero exit. The original tree is never edited. Exit 0 only if
the control passes and every mutant is killed.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
DEV = os.environ.get("D98_MUT_DEV_RUNNER")  # Mac dev: a stub runner script
SKIP_PKG = os.environ.get("D98_MUT_SKIP_PKG") == "1"  # Mac dev: no vLLM
FCC = "src/vllm_torchtpu/compilation/fast_compiled_call.py"
T = "tests/compilation/test_fast_compiled_call.py"

MUTANTS = [
    ("M1_storage_offset_dropped_from_key", FCC,
     "                    a.storage_offset(),\n", "",
     [T + "::test_storage_offset_change_misses_the_memo"]),
    ("M2_no_entry_generation_check", FCC,
     '            and _entry_count(code) == st["count"]\n'
     '            and _head_code(code) is st["head"]\n', "",
     [T + "::test_eviction_and_reset_invalidate"]),
    ("M2b_count_only_no_head_check", FCC,
     '            and _head_code(code) is st["head"]\n', "",
     [T + "::test_eviction_and_reset_invalidate"]),
    ("M3_check_mode_skips_the_guard", FCC,
     "                accepted = gm.check(f_locals)\n",
     "                accepted = True\n",
     [T + "::test_check_mode_counts_a_mismatch_on_a_global_rebind"]),
    ("M4_memo_ignores_global_state", FCC,
     "            or not gs.check()\n", "",
     [T + "::test_global_state_change_is_not_served_from_the_memo"]),
    ("M5_no_alias_marker_in_key", FCC,
     "            if first is not None:\n", "            if False:\n",
     [T + "::test_aliased_arguments_miss_the_memo"]),
    ("M6_no_backend_filter", FCC,
     "                if entry.backend is not backend:\n",
     "                if False:\n",
     [T + "::test_only_its_own_backend_entries_are_dispatched"]),
    ("M7_attribute_guards_memoized", FCC,
     "    return not any(_UNSAFE_PART.search(str(p)) for p in parts)\n",
     "    return True\n",
     [T + "::test_attribute_guards_are_never_memoized"]),
    ("M8_no_torchdynamo_inline", FCC,
     "    d98_fast_compiled._torchdynamo_inline = getattr(  # type: ignore[attr-defined]\n"
     "        compiled, \"_torchdynamo_inline\", fn\n    )\n", "",
     [T + "::test_traced_region_inlines_the_original"]),
    ("M9_refresh_keeps_stale_memo", FCC,
     '        st["head"] = None\n        st["memo"] = {}\n',
     '        st["head"] = None\n',
     [T + "::test_recompile_drops_the_memo"]),
    ("M10_ignores_dynamo_callback_and_stance", FCC,
     "            and _E.get_eval_frame_callback() is None\n"
     '            and _ef._stance.stance == "default"\n', "",
     [T + "::test_fallback_when_a_dynamo_callback_is_active"]),
    ("M11_direct_call_without_guard_check", FCC,
     "        for idx, (gm, direct, safe) in enumerate(entries):\n"
     "            if gm.check(f_locals):\n",
     "        for idx, (gm, direct, safe) in enumerate(entries):\n"
     "            if True:\n",
     [T + "::test_global_state_guard_without_memo_takes_the_normal_call"]),
    ("M12_scalar_value_dropped_from_key", FCC,
     "        if t in _SCALARS:\n            parts.append((t, a))\n",
     "        if t in _SCALARS:\n            parts.append((t,))\n",
     [T + "::test_bitwise_vs_plain_compiled_and_engaged"]),
    ("M13_varargs_tail_dropped", FCC,
     "            f_locals[var_name] = args[nargs:]\n",
     "            f_locals[var_name] = ()\n",
     [T + "::test_bitwise_vs_plain_compiled_and_engaged"]),
    ("M14_flags_off_installs_anyway", "src/vllm_torchtpu/env_override.py",
     "    if not (envs.VLLM_TPU_FAST_COMPILED_CALL or envs.VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE):\n"
     "        return\n", "",
     [T + "::test_flags_off_identity_and_on_engagement_in_fresh_processes"], "pkg"),
    ("M15_flag_not_compile_ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     '    "VLLM_TPU_FAST_COMPILED_CALL",\n', "",
     ["tests/compilation/test_tpu_compile_key_registration.py"], "pkg"),
]

CONTROL = [T, "tests/compilation/test_tpu_compile_key_registration.py"]


def run(work, targets):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"))
    for k in [k for k in env if k.startswith("VLLM_TPU_FAST_C")]:
        env.pop(k)
    if DEV:
        cmd = [sys.executable, DEV, os.path.join(work, "src")]
    else:
        cmd = [sys.executable, "-m", "pytest"]
    cmd += ["-q", "-x", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", *targets]
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    tail = [ln for ln in p.stdout.splitlines() if " passed" in ln or " failed" in ln
            or " error" in ln or "skipped" in ln]
    return p.returncode, (tail[-1] if tail else p.stdout[-300:].replace("\n", " | "))


def main():
    work = tempfile.mkdtemp(prefix="d98mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    control = [t for t in CONTROL if not (SKIP_PKG and "test_fast" not in t)]
    rc, tail = run(work, control)
    print(f"CONTROL rc={rc} {tail}")
    ok = rc == 0
    killed = total = 0
    for m in MUTANTS:
        name, rel, old, new, targets = m[:5]
        if SKIP_PKG and len(m) > 5:
            print(f"MUT {name} SKIPPED (needs the real package; runs on the pod)")
            continue
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {name} BAD (anchor occurs {text.count(old)} times)")
            ok = False
            continue
        with open(path, "w") as f:
            f.write(text.replace(old, new))
        try:
            rc, tail = run(work, targets)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        dead = rc != 0
        killed += dead
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and killed == total and total > 0
    print(f"Q38_D98_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

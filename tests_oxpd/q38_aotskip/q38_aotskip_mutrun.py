# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-H14-AOTSKIP mutation runner: break the code deliberately, the
targeted tests must FAIL (an assertion failure, not an import / collection
error).

Copies src/ and tests/ to a scratch dir, runs the control (unmutated: every
target passes, 0 failed, 0 skipped), then for each mutant applies one exact
text replacement (the old text must occur exactly once), runs that mutant's
target tests and requires a non-zero exit whose summary says "failed" and has
no collection error. The original tree is never edited. Exit 0 only if the
control passes and every mutant is killed that way.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
AGS = "src/vllm_torchtpu/compilation/aot_guard_skip.py"
T = "tests/compilation/test_aot_guard_skip.py"
STUB = T + "::test_fresh_aot_compile_disables_guard_check"
NOTENG = T + "::test_skip_that_does_not_take_reads_not_engaged"
CACHE = T + "::test_cache_loaded_artifact_is_logged_with_its_guard_state"
REAL = T + "::test_real_aot_artifact_through_vllms_aot_compile_skips_guards"
FRESH = T + "::test_registry_row_and_fresh_process_flag_off_identity_and_on_engagement"
BITE = T + "::test_engagement_check_bites_on_real_aot_artifacts"
RSELF = T + "::test_engagement_reader_selftest"
REG = ("tests/compilation/test_tpu_compile_key_registration.py::"
       "test_graph_flag_moves_the_compile_hash[VLLM_TPU_AOT_SKIP_GUARD_CHECK-1-False]")
ENG = "tests/q38_aotskip/q38_aotskip_engage.py"

MUTANTS = [
    ("M1a_ignores_evaluate_guards_stub", AGS,
     '            if not evaluate and hasattr(compiled, "disable_guard_check"):\n',
     '            if hasattr(compiled, "disable_guard_check"):\n', [STUB]),
    ("M1b_ignores_evaluate_guards_real", AGS,
     '            if not evaluate and hasattr(compiled, "disable_guard_check"):\n',
     '            if hasattr(compiled, "disable_guard_check"):\n', [REAL]),
    ("M2a_never_disables_stub", AGS,
     "                compiled.disable_guard_check()\n", "                pass\n", [STUB]),
    ("M2b_never_disables_real", AGS,
     "                compiled.disable_guard_check()\n", "                pass\n", [REAL]),
    ("M2c_never_disables_engagement_reader", AGS,
     "                compiled.disable_guard_check()\n", "                pass\n", [BITE]),
    ("M3_init_flag_guard_removed", "src/vllm_torchtpu/__init__.py",
     "    if not envs.VLLM_TPU_AOT_SKIP_GUARD_CHECK:\n        return\n"
     "    from vllm_torchtpu.compilation.aot_guard_skip import patch_aot_guard_skip\n",
     "    from vllm_torchtpu.compilation.aot_guard_skip import patch_aot_guard_skip\n",
     [FRESH]),
    ("M4_flag_set_but_ignored", AGS,
     "def patch_aot_guard_skip() -> None:\n    if not envs.VLLM_TPU_AOT_SKIP_GUARD_CHECK:\n        return\n",
     "def patch_aot_guard_skip() -> None:\n    if True:\n        return\n", [BITE]),
    ("M5_registry_row_dropped", "src/vllm_torchtpu/patch_registry.py",
     '    Patch("vllm_torchtpu:_patch_vllm_aot_guard_skip", _ACTIVE),\n', "", [FRESH]),
    ("M6_flag_default_on", "src/vllm_torchtpu/envs.py",
     '        "VLLM_TPU_AOT_SKIP_GUARD_CHECK", default=False\n',
     '        "VLLM_TPU_AOT_SKIP_GUARD_CHECK", default=True\n', [FRESH]),
    ("M7_flag_compile_ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     '    "VLLM_TPU_HOST_PHASES",\n',
     '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_AOT_SKIP_GUARD_CHECK",\n', [REG]),
    ("M8_engaged_without_checking_state", AGS,
     '                if state == "False":\n', '                if True:\n', [NOTENG]),
    ("M9_registry_wrong_stage", "src/vllm_torchtpu/patch_registry.py",
     '    Patch("vllm_torchtpu:_patch_vllm_aot_guard_skip", _ACTIVE),\n',
     '    Patch("vllm_torchtpu:_patch_vllm_aot_guard_skip", ("engine_core",)),\n', [FRESH]),
    ("M10_cache_load_not_wrapped", AGS,
     "        decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn\n", "", [CACHE]),
    ("M11_no_rank_tag", AGS,
     '                        "AOT_GUARD_SKIP engaged: rank=%d cls=%s guard_check_enabled=%s",\n'
     "                        _rank(),\n",
     '                        "AOT_GUARD_SKIP engaged: rank=?%d cls=%s guard_check_enabled=%s",\n'
     "                        _rank(),\n", [BITE]),
    ("R1_reader_ignores_bad_lines", ENG, "    if bad:\n        why.append", "    if False:\n        why.append", [RSELF]),
    ("R2_reader_ignores_rank_count", ENG, "    if len(seen) != ranks:\n", "    if False:\n", [RSELF]),
    ("R3_reader_cacheloaded_on_counts", ENG,
     '            kind == "cache-loaded" and state == "False"\n',
     '            kind == "cache-loaded"\n', [RSELF]),
    ("R4_reader_off_mode_ignores_install", ENG,
     '        verdict = "OFF" if any_line == 0 and install == 0 else "LEAK"\n',
     '        verdict = "OFF" if any_line == 0 else "LEAK"\n', [RSELF]),
]

CONTROL = [T, "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    env.pop("VLLM_TPU_AOT_SKIP_GUARD_CHECK", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True,
                       text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="aotskipmut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml",):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), work)
    rc, tail, _ = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    killed = total = 0
    for name, rel, old, new, targets in MUTANTS:
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
            rc, tail, out = run(work, targets)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        dead = (rc != 0 and " failed" in tail and "errors during collection" not in out
                and "ERROR " not in out)
        killed += dead
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and killed == total == len(MUTANTS)
    print(f"Q38_AOTSKIP_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""q38 L4 host sub-mark instrument (HOSTSUBS) mutation runner: break the code
deliberately, the targeted tests must FAIL (an assertion failure, not an import
/ collection error). Same runner as tests/q38_aotskip/q38_aotskip_mutrun.py.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
HP = "src/vllm_torchtpu/runner/host_phases.py"
T = "tests/runner/test_host_phases_subs.py"
CENSUS = T + "::test_census_line_values"
IDENT = T + "::test_hostphases_line_is_identical_with_the_census_on_or_off"
OFF = T + "::test_off_records_nothing"
BOTH = T + "::test_census_and_slow_dump_together"
FRESH = T + "::test_fresh_process_flag_gating"
PLACE = T + "::test_sub_marks_sit_inside_their_phases"
REG = ("tests/compilation/test_tpu_compile_key_registration.py::"
       "test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_HOST_PHASES_SUBS-1-False]")

MUTANTS = [
    ("H1_report_subs_not_called", HP, "    if _SUBS:\n        _report_subs()\n", "", [CENSUS]),
    ("H2_flush_into_hostphases", HP,
     "            _sub_samples.setdefault(name, []).append(ns)\n",
     '            _samples.setdefault("~" + name, []).append(ns)\n', [IDENT]),
    ("H3_subs_ignore_every", HP,
     "_SUBS = bool(envs.VLLM_TPU_HOST_PHASES_SUBS) and bool(_EVERY)\n",
     "_SUBS = bool(envs.VLLM_TPU_HOST_PHASES_SUBS)\n", [FRESH]),
    ("H4_mean_over_n_not_every", HP,
     'f"{float(v.sum()) / _EVERY:.0f}/{len(values)}"',
     'f"{float(v.sum()) / len(values):.0f}/{len(values)}"', [CENSUS]),
    ("H5_sub_gated_on_slow_only", HP,
     "    if not (_SLOW_NS or _SUBS):\n        return\n",
     "    if not _SLOW_NS:\n        return\n", [CENSUS]),
    ("H6_charge_not_counted", HP,
     "        _cur[key] = _cur.get(key, 0) + ns\n    if _SUBS:\n        _sub_acc[name] = _sub_acc.get(name, 0) + ns\n",
     "        _cur[key] = _cur.get(key, 0) + ns\n", [CENSUS]),
    ("H7_flag_not_compile_ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     '    "VLLM_TPU_HOST_PHASES_SUBS",\n', "", [REG]),
    ("H8_st_async_sub_dropped", "src/vllm_torchtpu/runner/tpu_runner.py",
     '        host_phases.sub("st_async")\n', "", [PLACE]),
    ("H9_fd_whole_call_sub_dropped", "src/vllm_torchtpu/megakernel_runtime/qwen38_driver.py",
     '                host_phases.sub("fd_whole_call")\n', "", [PLACE]),
    ("H10_slow_dump_loses_subs", HP,
     "        if _SLOW_NS:\n            _cur[key] = _cur.get(key, 0) + now - base\n",
     "        if False:\n            _cur[key] = _cur.get(key, 0) + now - base\n", [BOTH]),
    ("H11_off_still_accumulates", HP,
     "        if _SUBS:\n            _sub_acc[name] = _sub_acc.get(name, 0) + now - base\n",
     "        if True:\n            _sub_acc[name] = _sub_acc.get(name, 0) + now - base\n", [OFF]),
]

CONTROL = [T, "tests/runner/test_host_phases.py", "tests/runner/test_host_phases_slow.py",
           "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    env.pop("VLLM_TPU_HOST_PHASES", None)
    env.pop("VLLM_TPU_HOST_PHASES_SUBS", None)
    env.pop("VLLM_TPU_HOST_PHASES_SLOW_MS", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True,
                       text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="hostsubmut_")
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
    print(f"Q38_HOSTSUB_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""q38 L6-INSTR T-1(ii) HP-REPORT mutation runner: each mutant of the grouped
report must make its target tests FAIL (assertion failures, not collection
errors). Usage: python3 tests/q38_l6hprep/q38_l6hprep_mutrun.py <tree root>
Last line: Q38_L6HPREP_MUT PASS|FAIL killed <k>/<n> of <N> control_ok=<bool>"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
HP = "src/vllm_torchtpu/runner/host_phases.py"
T = "tests/q38_l6hprep/test_hp_report.py"
IDENT = T + "::test_line_is_byte_identical_over_random_ragged_windows"
BITS = T + "::test_each_statistic_is_bitwise_the_per_phase_value"
ORDER = T + "::test_names_keep_their_window_order"
MUTANTS = [
    ("B1_percentile_over_the_wrong_axis", "        q = np.percentile(v, (50, 90), axis=1)\n",
     "        q = np.percentile(v, (50, 90), axis=0)\n", [IDENT, BITS]),
    ("B2_sum_over_the_wrong_axis", "        s = v.sum(axis=1)\n", "        s = v.sum(axis=0)\n", [IDENT, BITS]),
    ("B3_p50_p90_swapped", "            out[name] = (q[0, i], q[1, i], s[i])\n",
     "            out[name] = (q[1, i], q[0, i], s[i])\n", [IDENT, BITS]),
    ("B4_mean_over_the_sample_count", "        mean = float(s) / _EVERY\n",
     "        mean = float(s) / len(_samples[name])\n", [IDENT]),
    ("B5_names_in_group_order", "    for name in _samples:\n        p50, p90, s = stats[name]\n",
     "    for name in stats:\n        p50, p90, s = stats[name]\n", [IDENT, ORDER]),
    ("B6_one_group_for_all_lengths", "        groups.setdefault(len(values), []).append(name)\n",
     "        groups.setdefault(0, []).append(name)\n", [IDENT, BITS]),
]
CONTROL = [T, "tests/runner/test_host_phases.py", "tests/runner/test_host_phases_slow.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu",
               Q38_L6HPREP_EXPECT_TREE=work)
    for k in ("VLLM_TPU_HOST_PHASES", "VLLM_TPU_HOST_PHASES_SLOW_MS"):
        env.pop(k, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="l6hprepmut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    rc, tail, _ = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    killed = total = 0
    for name, old, new, targets in MUTANTS:
        path = os.path.join(work, HP)
        text = open(path).read()
        if text.count(old) != 1:
            print(f"MUT {name} BAD (anchor occurs {text.count(old)} times)")
            ok = False
            continue
        open(path, "w").write(text.replace(old, new))
        try:
            rc, tail, out = run(work, targets)
        finally:
            open(path, "w").write(text)
        total += 1
        dead = rc != 0 and " failed" in tail and "errors during collection" not in out and "ERROR " not in out
        killed += dead
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and killed == total == len(MUTANTS)
    print(f"Q38_L6HPREP_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

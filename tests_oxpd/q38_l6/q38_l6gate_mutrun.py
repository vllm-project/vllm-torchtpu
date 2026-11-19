# SPDX-License-Identifier: Apache-2.0
"""q38 L6-GATE mutation runner: break the gate deliberately; the targeted
tests must FAIL (an assertion failure, not an import / collection error).

Same shape as q38_l6carry_mutrun.py (L6-CARRY): copies src/ and tests/ to a
scratch dir, runs the control (unmutated: every target passes, 0 failed, 0
skipped), then for each mutant applies one exact text replacement (the old
text must occur exactly once), checks the import guard still passes on the
mutated copy (so the run imports the copy), runs that mutant's targets and
requires a non-zero exit whose summary says "failed" and has no error. The
import-guard mutant points the guard at the base tree instead; the guard
itself must then fail. The original tree is never edited. Exit 0 only if the
control passes and every mutant is killed.

    python3 tests/q38_l6/q38_l6gate_mutrun.py <head root> <base root>
Last line: Q38_L6GATE_MUT PASS|FAIL killed <k>/<n> of <n>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
BASE = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else ""
AB = "src/vllm_torchtpu/core/l6_ab.py"
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
EV = "src/vllm_torchtpu/envs.py"
T = "tests/q38_l6/test_l6_gate.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py"
PARSE = T + "::test_gate_flags_parse_default_off_and_reject_negatives"
STEP = T + "::test_gate_turns_a_lever_off_below_n_for_that_step_only"
LINE = T + "::test_window_line_names_the_window_levers_and_counts_gated_steps"
WIN = T + "::test_gate_counts_only_levers_their_window_turns_on"
HOOK = T + "::test_execute_model_runs_the_gate_after_begin_step_and_before_update_states"
COUNT = T + "::test_gate_hook_counts_this_steps_scheduled_requests"
DROP = T + "::test_gate_hook_drops_steady_caches_only_when_it_turns_steady_off"
C_CARRY = T + "::test_gated_stream_matches_all_off[carry-b3_padded]"
C_BOTH = T + "::test_gated_stream_matches_all_off[both-pooled]"
D_STEADY = T + "::test_steady_metadata_under_the_gate_matches_all_off[steady]"
D_RO = T + "::test_reorder_and_count_under_the_gate_matches_the_walks"
FLAGS = T + "::test_gate_flags_are_registered_and_compile_ignored"
GUARD = T + "::test_imports_the_staged_tree"

GATE_CALL = (
    "            if _l6_ab.GATE:\n"
    "                # q38 L6-GATE: CARRY / STEADY off on this step if its batch is small.\n"
    "                self._l6_gate_step(scheduler_output)\n"
)
UPDATE = "        self._update_states(scheduler_output)\n"


def _moved_after_update_states():
    """(old, new) for G11: the execute_model gate call moved from before _update_states to after it."""
    with open(os.path.join(ROOT, TR)) as f:
        text = f.read()
    a = text.index(GATE_CALL)
    b = text.index(UPDATE, a) + len(UPDATE)
    old = text[a:b]
    new = text[a + len(GATE_CALL):b] + "        if _l6_ab.GATE:\n            self._l6_gate_step(scheduler_output)\n"
    return old, new


G11_OLD, G11_NEW = _moved_after_update_states()

MUTANTS = [
    ("G01_gate_ignored", AB,
     "    off = tuple(lever for lever, n in MIN_REQS.items() if num_reqs < n and on_now.get(lever, False))\n",
     "    off = ()\n", [STEP, C_CARRY]),
    ("G02_gate_inverted", AB,
     "if num_reqs < n and on_now.get(lever, False))\n",
     "if num_reqs >= n and on_now.get(lever, False))\n", [STEP, C_CARRY]),
    ("G03_off_by_one_at_n", AB,
     "if num_reqs < n and on_now.get(lever, False))\n",
     "if num_reqs <= n and on_now.get(lever, False))\n", [STEP, C_BOTH]),
    ("G04_previous_steps_count", TR,
     "            len(scheduler_output.num_scheduled_tokens)\n        ):\n            _l6_steady.drop(self)\n",
     "            self.input_batch.num_reqs\n        ):\n            _l6_steady.drop(self)\n", [COUNT, C_CARRY]),
    ("G05_steady_caches_not_dropped_on_a_gated_step", TR,
     "        ):\n            _l6_steady.drop(self)\n\n    @torch.no_grad()\n",
     "        ):\n            pass\n\n    @torch.no_grad()\n", [DROP, D_STEADY, D_RO]),
    ("G06_gate_outlives_its_step", AB,
     "        if on_now is g[1]:  # begin_step kept last step's gated states (no new window): the window's states again\n"
     "            on_now = _state[\"on\"] = g[0]\n",
     "        pass\n", [STEP]),
    ("G07_window_line_reports_the_gated_states", AB,
     "    win_on = g[0] if g is not None and _state[\"on\"] is g[1] else _state[\"on\"]\n",
     "    win_on = _state[\"on\"]\n", [LINE, WIN]),
    ("G08_gated_steps_not_counted", AB,
     "            note(lever, _GATE_KEY[lever])\n", "", [STEP]),
    ("G09_window_off_lever_counted_as_gated", AB,
     "if num_reqs < n and on_now.get(lever, False))\n",
     "if num_reqs < n)\n", [WIN]),
    ("G10_execute_model_never_calls_the_hook", TR,
     "                self._l6_gate_step(scheduler_output)\n", "                pass\n", [HOOK]),
    ("G11_hook_after_update_states", TR, G11_OLD, G11_NEW, [HOOK]),
    ("G12_flags_never_read", AB,
     "        if n:\n            MIN_REQS[lever] = n\n", "        pass\n", [PARSE]),
    ("G13_carry_flag_not_registered", EV,
     '    "VLLM_TPU_L6_CARRY_MIN_REQS": env_nonnegative_int("VLLM_TPU_L6_CARRY_MIN_REQS", 0),\n', "",
     [REG + "::test_flag_is_registered_and_parses[VLLM_TPU_L6_CARRY_MIN_REQS-8-0]", FLAGS]),
    ("G14_steady_flag_not_registered", EV,
     '    "VLLM_TPU_L6_STEADY_MIN_REQS": env_nonnegative_int("VLLM_TPU_L6_STEADY_MIN_REQS", 0),\n', "",
     [REG + "::test_flag_is_registered_and_parses[VLLM_TPU_L6_STEADY_MIN_REQS-8-0]", FLAGS]),
    ("G15_carry_flag_compile_hashed", TC,
     '    "VLLM_TPU_L6_CARRY_MIN_REQS",\n', "",
     [REG + "::test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_L6_CARRY_MIN_REQS-8-0]", FLAGS]),
    ("G16_steady_flag_compile_hashed", TC,
     '    "VLLM_TPU_L6_STEADY_MIN_REQS",\n', "",
     [REG + "::test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_L6_STEADY_MIN_REQS-8-0]", FLAGS]),
    ("G17_carry_gate_drops_steady_caches", TR,
     '        if _l6_ab.ACTIVE and "steady" in _l6_ab.gate_step(\n',
     "        if _l6_ab.ACTIVE and _l6_ab.gate_step(\n", [DROP]),
]
GUARD_MUTANT = "G00_import_guard_points_at_base"

CONTROL = sorted({t for *_, ts in MUTANTS for t in ts} | {GUARD})
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, expect_root):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu",
               Q38_L6GATE_EXPECT_ROOT=expect_root)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
           "-p", "no:logging", "-rfE"]
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def killed(rc, tail):
    return rc != 0 and "failed" in tail and "error" not in tail


def main():
    work = tempfile.mkdtemp(prefix="l6gatemut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    src_root = os.path.join(work, "src")
    rc, tail, out = run(work, CONTROL, src_root)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) == len(CONTROL)
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} targets={len(CONTROL)} {tail}")
    if not ok:
        print(out[-4000:])
    n_killed = total = 0
    for name, rel, old, new, targets in MUTANTS:
        total += 1
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {name} BAD (old text occurs {text.count(old)} times)")
            continue
        with open(path, "w") as f:
            f.write(text.replace(old, new))
        try:
            grc, gtail, _ = run(work, [GUARD], src_root)
            trc, ttail, tout = run(work, targets, src_root)
        finally:
            with open(path, "w") as f:
                f.write(text)
        guard_ok = grc == 0 and "1 passed" in gtail
        k = guard_ok and killed(trc, ttail)
        n_killed += k
        print(f"MUT {name} {'KILLED' if k else 'SURVIVED'} guard_ok={guard_ok} rc={trc} {ttail}")
        if not k:
            print(tout[-3000:])
    total += 1
    if BASE:
        grc, gtail, _ = run(work, [GUARD], os.path.join(BASE, "src"))
        k = killed(grc, gtail)
    else:
        grc, gtail, k = -1, "no base root given", False
    n_killed += k
    print(f"MUT {GUARD_MUTANT} {'KILLED' if k else 'SURVIVED'} rc={grc} {gtail}")
    verdict = "PASS" if ok and n_killed == total else "FAIL"
    print(f"Q38_L6GATE_MUT {verdict} killed {n_killed}/{total} of {total}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""q38 L6-STEADY mutation runner: break the code deliberately, one edit at a
time; each mutant's target tests must FAIL (an assertion or an exception in a
test body, not an import / collection error). Same runner shape as
tests/q38_hostsub/q38_hostsub_mutrun.py (L4-C).

Usage (from the head tree): python3 tests/q38_l6/q38_l6steady_mutrun.py .
Last line: Q38_L6STEADY_MUT PASS|FAIL killed <k>/<n> of <N> control_ok=<bool>.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
S = "src/vllm_torchtpu/runner/l6_steady.py"
R = "src/vllm_torchtpu/runner/tpu_runner.py"
I = "src/vllm_torchtpu/models/vllm/qwen4_exp/inputs.py"
C = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T = "tests/q38_l6/test_l6_steady.py"
RAND = T + "::test_fast_equals_full_over_random_steady_sequences"
TABLE = T + "::test_each_table_change_falls_back_then_resumes"
APPEND = TABLE + "[append]"
COLUMN = T + "::test_a_state_column_move_without_a_table_change_falls_back"
BOUND = T + "::test_the_column_bound_counts_pending_rejections"
NONTARGET = T + "::test_non_target_walks_never_use_or_record"
CHECKS = T + "::test_check_steps_run_the_full_path_and_agree"
CATCH = T + "::test_check_catches_a_stale_cache_and_turns_the_lever_off"
RO_RAND = T + "::test_reorder_and_count_equals_the_walks_over_random_schedules"
RO_KIND = T + "::test_reorder_change_kinds_run_the_walks"
UP = T + "::test_upstream_noop_predicate_never_skips_a_real_reorder"
TBL = T + "::test_adapter_tables_reused_only_for_the_same_sources"
REG = T + "::test_lever_is_registered_and_compile_ignored"
GUARD = T + "::test_every_runner_call_site_is_guarded"

MUTANTS = [
    ("M01_no_version_check", S,
     "        or _am._BLOCK_TABLE_VERSION[0] != c.version\n", "", [APPEND]),
    ("M02_bound_ignores_pending_rejections", S,
     "    lo = hi1 - ctx.steady_max_rejected\n", "    lo = hi1\n", [BOUND]),
    ("M03_column_not_compared_to_the_cache", S,
     "        or _q(ctx, c.qcls) != c.q\n", "        or _q(ctx, c.qcls) is None\n", [COLUMN]),
    ("M04_seq_lens_not_refreshed", S,
     '        "seq_lens": ctx.seq_lens,\n', "", [TABLE + "[swap]"]),
    ("M05_positions_not_refreshed", S,
     '        "input_positions": runner.position_ids,\n', "", [TABLE + "[swap]"]),
    ("M06_unified_states_not_extended", S,
     "    if c.states and ctx.unified_mamba_state_indices is not None:\n",
     "    if False:\n", [TABLE + "[swap]"]),
    ("M07_mamba_step_fields_dropped", S,
     "        d.update(step_m if is_mamba else step_all)\n",
     "        d.update(step_all)\n", [TABLE + "[swap]"]),
    ("M08_fast_leaves_ctx_tables_unset", S,
     "    ctx.staged_block_tables = dict(c.tables)\n", "", [TABLE + "[swap]"]),
    ("M09_reorder_ignores_the_schedule", S,
     "        or so.num_scheduled_tokens != c.sched\n", "", [RO_KIND + "[sched]"]),
    ("M10_reorder_ignores_the_order", S,
     "        or runner.input_batch.req_ids != c.req_ids\n", "", [RO_KIND + "[order]"]),
    ("M11_upstream_ignores_prefilling", S,
     "    return bool((nc >= batch.num_prompt_tokens[:n]).all())\n",
     "    return True\n", [UP]),
    ("M12_upstream_ignores_the_threshold", S,
     "    if max(sched[r] for r in req_ids) > threshold:\n        return False\n", "", [UP]),
    ("M13_check_never_compares", S,
     "    return _pairs_differ(pairs)\n", "    return None\n", [CATCH]),
    ("M14_check_never_due", S,
     "    if cache.since_check >= n:\n", "    if False:\n", [CHECKS]),
    ("M15_adapter_views_reused_for_new_sources", I,
     "            and prev[1] is sources[0]\n", "", [TBL]),
    ("M16_flag_not_compile_ignored", C,
     '    "VLLM_TPU_L6_STEADY_STEP",\n', "", [REG]),
    ("M17_draft_walks_reach_the_fast_path", R,
     '        steady = getattr(ctx, "steady_seq_lens_hi", None) is not None\n',
     "        steady = True\n", [NONTARGET]),
    ("M18_prepare_inputs_guard_dropped", R,
     "            _l6_ab.ACTIVE\n            and start_index == 0\n",
     "            start_index == 0\n", [GUARD]),
    ("M19_reorder_cache_kept_with_a_prefill", S,
     "    if result[1] != batch.num_reqs:\n", "    if False:\n",
     [T + "::test_reorder_with_a_prefill_segment_is_never_cached"]),
    ("M20_undetermined_column_taken_as_known", S,
     "        if not np.array_equal(a, b):\n            return None\n", "", [BOUND]),
]

CONTROL = [T, "tests/q38_l6/test_l6_ab_envcache.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    env.pop("Q38_L6_EXPECT_TREE", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True,
                       text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="l6steadymut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml",):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), work)
    rc, tail, _ = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    # The import-guard test skips without Q38_L6_EXPECT_TREE (unset here): 1 skip.
    ok = (rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "error" not in tail)
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
    print(f"Q38_L6STEADY_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

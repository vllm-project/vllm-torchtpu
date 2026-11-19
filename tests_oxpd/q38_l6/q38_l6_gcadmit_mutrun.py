#!/usr/bin/env python3
"""q38 L6-GCADMIT mutant runner (CPU pod only). Usage: python3 q38_l6_gcadmit_mutrun.py <head tree> <set> [survey]

Modelled on L6-SYNC's q38_l6_sync_mutrun.py. For each mutant: copy the tree, apply one exact textual replacement
(it must match exactly once), run the mutant's target test id with pytest and require that it FAILED (a failing
test, not an error or a collection failure). A control run on the unmutated tree must pass every target first.
'survey' reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_GCADMIT_MUT PASS killed N/N of N | Q38_L6_GCADMIT_MUT FAIL ... | Q38_L6_GCADMIT_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

L = "src/vllm_torchtpu/core/l6_gcadmit.py"
RU = "src/vllm_torchtpu/runner/tpu_runner.py"
C = "src/vllm_torchtpu/compilation/tpu_compiler.py"
E = "src/vllm_torchtpu/envs.py"
A = "src/vllm_torchtpu/core/l6_ab.py"
G = "tests/q38_l6/test_l6_gcadmit.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py"

M_TAIL = "  # what the freeze moves: every survivor is in generation 2 now\n"
HOOK = ("        if _l6_gcadmit.ON:  # q38 L6 GC-ADMIT-FREEZE: freeze this step's admitted state (core/l6_gcadmit.py)\n"
        "            _l6_gcadmit.after_update_states(self, scheduler_output)\n")

# (flag, its compile-key row id, its envs.py getter row)
FLAGS = [
    ("VLLM_TPU_L6_GC_ADMIT_FREEZE", "VLLM_TPU_L6_GC_ADMIT_FREEZE-1-",
     '    "VLLM_TPU_L6_GC_ADMIT_FREEZE": env_str("VLLM_TPU_L6_GC_ADMIT_FREEZE", ""),\n'),
    ("VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP", "VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP-100000-1000000",
     '    "VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP": env_nonnegative_int("VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP", 1000000),\n'),
    ("VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK", "VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK-16-64",
     '    "VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK": env_nonnegative_int("VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK", 64),\n'),
]

MUTANTS = {
    "gcadmit": [
        # ---- the admission trigger and where it runs
        ("T1 trigger on every step (no admission check)", L,
         "        new = scheduler_output.scheduled_new_reqs\n        if not new:\n            return\n",
         "        new = scheduler_output.scheduled_new_reqs\n",
         G + "::test_admission_step_freezes_the_new_state_and_other_steps_do_not"),
        ("T2 trigger before _update_states", RU,
         "        self._update_states(scheduler_output)\n" + HOOK,
         HOOK + "        self._update_states(scheduler_output)\n",
         G + "::test_execute_model_freezes_the_state_update_states_built"),
        ("T3 call-site gate dropped (the lever is called with the flag off)", RU,
         "        if _l6_gcadmit.ON:  # q38 L6 GC-ADMIT-FREEZE", "        if True:  # q38 L6 GC-ADMIT-FREEZE",
         G + "::test_execute_model_never_calls_the_lever_with_the_flag_off"),
        ("T4 module flag gate dropped", L,
         '    if not ON or _st["off"]:\n', '    if _st["off"]:\n', G + "::test_flag_off_is_a_silent_no_op"),
        ("T5 warm-up gate dropped", L,
         '        if not getattr(runner, "warmup_done", False):\n            return\n', "",
         G + "::test_waits_for_warm_up"),
        ("T6 ab: window ignored (on in every window)", L,
         '    return l6_ab.on("gcadmit")\n', "    return True\n",
         G + "::test_never_unfreezes_across_windows_errors_and_the_cap"),
        ("T7 ab: not refused without the worker's begin_step", L,
         "    if l6_ab.step_seq() == 0:\n", "    if False:\n",
         G + "::test_ab_refused_when_the_worker_never_begins_l6_steps"),
        # ---- never unfreeze
        ("U1 unfreeze at an off window", L,
         "        if not _on():\n            return\n", "        if not _on():\n            gc.unfreeze()\n            return\n",
         G + "::test_never_unfreezes_across_windows_errors_and_the_cap"),
        ("U2 unfreeze on error", L,
         '    _st["off"] = True\n    logger.error("q38 L6 gcadmit off for the rest of the run: %s", why)\n',
         '    _st["off"] = True\n    gc.unfreeze()\n    logger.error("q38 L6 gcadmit off for the rest of the run: %s", why)\n',
         G + "::test_never_unfreezes_across_windows_errors_and_the_cap"),
        ("U3 unfreeze at the cap", L,
         '        l6_ab.note("gcadmit", "capped")\n', '        l6_ab.note("gcadmit", "capped")\n        gc.unfreeze()\n',
         G + "::test_cap_stops_freezing_without_unfreezing"),
        # ---- refusals
        ("F1 no refusal next to D1e freeze", L,
         '    if os.environ.get("VLLM_TPU_HOST_GC", "").strip().lower() == "freeze":\n', "    if False:\n",
         G + "::test_refused_next_to_another_freeze_owner[d1e_freeze]"),
        ("F2 no refusal next to gcfreeze", L,
         '    if os.environ.get("VLLM_TPU_L6_GC_FREEZE", "").strip().lower() not in _OFF:\n', "    if False:\n",
         G + "::test_refused_next_to_another_freeze_owner[gcfreeze]"),
        # ---- leak guard and cap
        ("L1 no pre-freeze collection", L,
         "    reclaimed = gc.collect()\n", "    reclaimed = 0\n",
         G + "::test_pre_freeze_collection_reclaims_cyclic_garbage_instead_of_pinning_it"),
        ("L2 freeze before the collection", L,
         "    reclaimed = gc.collect()\n    m = len(gc.get_objects(generation=2))" + M_TAIL + "    gc.freeze()\n",
         "    gc.freeze()\n    reclaimed = gc.collect()\n    m = len(gc.get_objects(generation=2))\n",
         G + "::test_pre_freeze_collection_reclaims_cyclic_garbage_instead_of_pinning_it"),
        ("L3 cap ignored", L,
         "    capped = bool(CAP) and cum > CAP\n", "    capped = False\n", G + "::test_cap_stops_freezing_without_unfreezing"),
        ("L4 the total is not accumulated", L,
         '    _st["cum"] += m\n', "", G + "::test_cap_stops_freezing_without_unfreezing"),
        ("L5 frozen size read before the pre-freeze collection (counts garbage)", L,
         "    reclaimed = gc.collect()\n    m = len(gc.get_objects(generation=2))" + M_TAIL,
         "    m = len(gc.get_objects())\n    reclaimed = gc.collect()\n",
         G + "::test_counts_freeze_us_reclaimed_n_frozen_and_the_line_cadence"),
        ("G1 gc.get_freeze_count() read on the step path", L,
         "    reclaimed = gc.collect()\n", "    reclaimed = gc.collect()\n    gc.get_freeze_count()\n",
         G + "::test_never_reads_gc_get_freeze_count_on_the_step_path"),
        # ---- counts
        ("C1 freeze not counted", L,
         '    l6_ab.note("gcadmit", "freeze")\n', "", G + "::test_counts_freeze_us_reclaimed_n_frozen_and_the_line_cadence"),
        ("C2 n_frozen not counted", L,
         '    l6_ab.note("gcadmit", "n_frozen", m)\n', "",
         G + "::test_counts_freeze_us_reclaimed_n_frozen_and_the_line_cadence"),
        ("C3 check cadence ignored (first freeze only)", L,
         "    if n <= 3 or n % CHECK == 0 or capped:\n", "    if n <= 3 or capped:\n",
         G + "::test_counts_freeze_us_reclaimed_n_frozen_and_the_line_cadence"),
        ("C4 reclaimed not counted", L,
         '    l6_ab.note("gcadmit", "reclaimed", reclaimed)\n', "",
         G + "::test_counts_freeze_us_reclaimed_n_frozen_and_the_line_cadence"),
        # ---- registration rows
        ("R1 gcadmit missing from l6_ab.LEVERS", A,
         '    "gcadmit": "VLLM_TPU_L6_GC_ADMIT_FREEZE",\n', "", G + "::test_lever_is_registered_and_compile_ignored"),
        # q38 L6 gen-2 heap type census (core/l6_gccensus.py)
        ("Z1 census marks slow gen0 / gen1 too", "src/vllm_torchtpu/core/l6_gccensus.py",
         'if t0 is None or info.get("generation") != 2:', 'if t0 is None:',
         "tests/q38_l6/test_l6_gccensus.py::test_callback_marks_only_slow_gen2"),
        ("Z2 threshold ignored", "src/vllm_torchtpu/core/l6_gccensus.py",
         "if dur_ms >= MS and _state[\"pending\"] is None:", "if _state[\"pending\"] is None:",
         "tests/q38_l6/test_l6_gccensus.py::test_fast_gen2_is_not_marked"),
        ("Z3 per-process cap ignored", "src/vllm_torchtpu/core/l6_gccensus.py",
         "    if _state[\"done\"] >= MAX:\n        return\n", "",
         "tests/q38_l6/test_l6_gccensus.py::test_pending_gen2_census_runs_next_step_and_cap_bounds"),
        ("Z4 warm census repeated every step", "src/vllm_torchtpu/core/l6_gccensus.py",
         "        _state[\"warm_done\"] = True\n", "",
         "tests/q38_l6/test_l6_gccensus.py::test_warm_census_once_and_callback_installed"),
        ("Z5 census before warm-up", "src/vllm_torchtpu/core/l6_gccensus.py",
         "    if not getattr(runner, \"warmup_done\", False):\n        return\n", "",
         "tests/q38_l6/test_l6_gccensus.py::test_nothing_before_warmup"),
        ("Z6 hook not guarded", "src/vllm_torchtpu/runner/tpu_runner.py",
         "        if _l6_gccensus.ON:  # q38 L6 gen-2 heap type census", "        if True:  # q38 L6 gen-2 heap type census",
         "tests/q38_l6/test_l6_gccensus.py::test_runner_hook_after_update_states_and_guarded"),
        ("Z7 census flag not compile-ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         '    "VLLM_TPU_L6_GC_CENSUS_TYPES",\n', "",
         "tests/q38_l6/test_l6_gccensus.py::test_registered_and_compile_ignored"),
    ],
}
for i, (flag, rid, erow) in enumerate(FLAGS):
    MUTANTS["gcadmit"].append((f"R{i + 2} {flag} dropped from the compile-key ignore list", C, f'    "{flag}",\n', "",
                               REG + f"::test_host_only_flag_keeps_the_compile_hash[{rid}]"))
    MUTANTS["gcadmit"].append((f"R{i + 5} {flag} getter row dropped from envs.py", E, erow, "",
                               REG + f"::test_flag_is_registered_and_parses[{rid}]"))


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
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_GC", "VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_CPU_PIN"):
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
        print("Q38_L6_GCADMIT_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6gmut.")
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
        print(f"Q38_L6_GCADMIT_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_GCADMIT_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_GCADMIT_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

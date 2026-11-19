#!/usr/bin/env python3
"""q38 L6-SYNC mutant runner (CPU pod only). Usage: python3 q38_l6_sync_mutrun.py <head tree> <set> [survey]

Modelled on L6-ENGINE's q38_l6_engalloc_mutrun.py. For each mutant: copy the tree, apply one exact textual
replacement (it must match exactly once), run the mutant's target test id with pytest and require that it FAILED
(a failing test, not an error or a collection failure). A control run on the unmutated tree must pass every target
first. 'survey' reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_SYNC_MUT PASS killed N/N of N | Q38_L6_SYNC_MUT FAIL ... | Q38_L6_SYNC_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

L = "src/vllm_torchtpu/core/l6_jitter.py"
K = "src/vllm_torchtpu/core/host_knobs.py"
TW = "src/vllm_torchtpu/worker/tpu_worker.py"
NB = "src/vllm_torchtpu/worker/numa_bind.py"
RU = "src/vllm_torchtpu/runner/utils.py"
C = "src/vllm_torchtpu/compilation/tpu_compiler.py"
E = "src/vllm_torchtpu/envs.py"
A = "src/vllm_torchtpu/core/l6_ab.py"
J = "tests/q38_l6/test_l6_jitter.py"
HK = "tests/core/test_host_knobs.py"
NT = "tests/worker/test_numa_bind.py"
HP = "tests/compilation/test_additional_config_hash_patch.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py"

# (flag, its compile-key row id, its envs.py getter row)
FLAGS = [
    ("VLLM_TPU_L6_MAIN_PIN2", "VLLM_TPU_L6_MAIN_PIN2-1-",
     '    "VLLM_TPU_L6_MAIN_PIN2": env_str("VLLM_TPU_L6_MAIN_PIN2", ""),\n'),
    ("VLLM_TPU_L6_NUMA_BIND", "VLLM_TPU_L6_NUMA_BIND-1-False",
     '    "VLLM_TPU_L6_NUMA_BIND": env_bool("VLLM_TPU_L6_NUMA_BIND", default=False),\n'),
    ("VLLM_TPU_L6_SYNC_GC", "VLLM_TPU_L6_SYNC_GC-1-",
     '    "VLLM_TPU_L6_SYNC_GC": env_str("VLLM_TPU_L6_SYNC_GC", ""),\n'),
    ("VLLM_TPU_L6_SYNC_GC_EVERY", "VLLM_TPU_L6_SYNC_GC_EVERY-50-100",
     '    "VLLM_TPU_L6_SYNC_GC_EVERY": env_nonnegative_int("VLLM_TPU_L6_SYNC_GC_EVERY", 100),\n'),
    ("VLLM_TPU_L6_SYNC_GC_VALVE", "VLLM_TPU_L6_SYNC_GC_VALVE-20000-100000",
     '    "VLLM_TPU_L6_SYNC_GC_VALVE": env_nonnegative_int("VLLM_TPU_L6_SYNC_GC_VALVE", 100000),\n'),
]

MUTANTS = {
    "sync": [
        # ---- SYNC-GC
        ("S1 collect not in lockstep: first collection N steps after the engage step", L,
         '    _sg["due"] = -(-j // SYNC_EVERY) * SYNC_EVERY  # the first multiple of N at or after this step\n',
         '    _sg["due"] = j + SYNC_EVERY\n', J + "::test_syncgc_collects_only_on_the_lockstep_step_index"),
        ("S2 collect index drifts after a late collection (step-index offset)", L,
         '            _sg["due"] = (j // SYNC_EVERY + 1) * SYNC_EVERY\n',
         '            _sg["due"] = j + SYNC_EVERY\n', J + "::test_syncgc_collects_only_on_the_lockstep_step_index"),
        ("S3 valve ignored", L,
         "    if n0 > SYNC_VALVE:\n", "    if False and n0 > SYNC_VALVE:\n",
         J + "::test_syncgc_valve_fires_above_its_bound"),
        ("S4 disable not undone at an off window", L,
         "def _sg_release() -> None:\n    gc.enable()\n", "def _sg_release() -> None:\n    pass\n",
         J + "::test_syncgc_off_window_restores_gc_and_keeps_the_warmup_freeze"),
        ("S5 disable not undone on error", L,
         "    if undo:\n        gc.enable()\n", "    if undo:\n        pass\n",
         J + "::test_syncgc_error_restores_gc_once_and_stays_off"),
        ("S6 off window thaws the warm-up freeze", L,
         "def _sg_release() -> None:\n    gc.enable()\n", "def _sg_release() -> None:\n    gc.enable()\n    gc.unfreeze()\n",
         J + "::test_syncgc_off_window_restores_gc_and_keeps_the_warmup_freeze"),
        ("S7 no refusal next to gcfreeze", L,
         '    if _configured("gcfreeze"):\n', "    if False:\n",
         J + "::test_syncgc_refused_next_to_another_freeze_owner[gcfreeze]"),
        ("S8 no refusal next to D1e freeze", L,
         "    if D1E_FREEZE:\n        return \"D1e's", "    if False:\n        return \"D1e's",
         J + "::test_syncgc_refused_next_to_another_freeze_owner[d1e_freeze]"),
        ("S9 no generation escalation", L,
         "    gen = 2 if (t2 and c2 > t2) else (1 if (t1 and c1 > t1) else 0)\n", "    gen = 0\n",
         J + "::test_syncgc_escalates_with_cpython_counters"),
        ("S10 GC timer not installed", L,
         "        gc.callbacks.append(_gc_timer)  # regardless of VLLM_TPU_HOST_AB_PERIOD_MS\n", "        pass\n",
         J + "::test_syncgc_times_every_collection_in_the_window_line"),
        ("S11 syncgc warm-up gate dropped", L,
         '    want = l6_ab.on("syncgc") and _warm["on"]\n', '    want = l6_ab.on("syncgc")\n',
         J + "::test_syncgc_waits_for_warm_up"),
        ("S12 lockstep point after propose instead of before", L,
         '        if _sg["armed"]:\n            sync_point()\n        return orig(self, *args, **kwargs)\n',
         '        r = orig(self, *args, **kwargs)\n        if _sg["armed"]:\n            sync_point()\n        return r\n',
         J + "::test_syncgc_hook_runs_before_propose"),
        ("S13 engages while someone else disabled gc", L,
         "    if not gc.isenabled():\n        raise RuntimeError(", "    if False:\n        raise RuntimeError(",
         J + "::test_syncgc_needs_gc_enabled_when_it_engages"),
        ("S14 worker never marks warm", TW,
         "        l6_jitter.mark_warm()\n", "        pass\n",
         J + "::test_worker_marks_warm_after_the_cpu_pin"),
        # ---- D1e GC timer
        ("H1 D1e GC timer only with an AB period again", K,
         '        if not _gc["hooked"]:  # whatever AB_PERIOD_MS (q38 L6 SYNC-GC)\n',
         '        if AB_PERIOD_MS > 0 and not _gc["hooked"]:\n', HK + "::test_gc_timer_is_installed_without_ab"),
        ("H2 no D1e GC line without AB", K,
         "            _apply(True)\n        _gc_line(role)\n", "            _apply(True)\n",
         HK + "::test_gc_line_every_20_s_without_ab"),
        # ---- MAINPIN-2
        ("M1 sibling not idle: only each core's first CPU reserved", L,
         "    reserved = frozenset(c for core in cores[-k:] for c in core)\n",
         "    reserved = frozenset(core[0] for core in cores[-k:])\n",
         J + "::test_mainpin2_plan_gives_distinct_cores_with_idle_siblings"),
        ("M2 sibling not idle: main thread pinned to its whole core", L,
         '        _setaff(main, {_p2["cpu"]})\n        got = _getaff(main)\n        if got != {_p2["cpu"]}:\n',
         '        _setaff(main, {_p2["cpu"]} | _p2["idle"])\n        got = _getaff(main)\n        if not {_p2["cpu"]} <= got:\n',
         J + "::test_mainpin2_static_exclusion_once_only_the_main_mask_toggles"),
        ("M3 exclusion reapplied per toggle", L,
         '        if want != _p2["pinned"]:\n            _p2_sweep()\n',
         '        if want != _p2["pinned"]:\n            _p2["known"] = {_p2["main"]}\n            _p2_sweep()\n',
         J + "::test_mainpin2_static_exclusion_once_only_the_main_mask_toggles"),
        ("M4 off window restores every thread, not only the main mask", L,
         '    else:\n        _setaff(main, _p2["main_saved"])\n',
         '    else:\n        _setaff(main, _p2["main_saved"])\n        for t, m in _p2["saved"].items():\n            _setaff(t, m)\n',
         J + "::test_mainpin2_static_exclusion_once_only_the_main_mask_toggles"),
        ("M5 no restore on failure", L,
         '    logger.error("q38 L6 mainpin2 off for the rest of the run (every saved mask restored): %s", exc)\n'
         "    _p2_restore_all()\n",
         '    logger.error("q38 L6 mainpin2 off for the rest of the run (every saved mask restored): %s", exc)\n',
         J + "::test_mainpin2_failure_restores_every_mask_once"),
        ("M6 main mask not restored at an off window", L,
         '    else:\n        _setaff(main, _p2["main_saved"])\n', "    else:\n        pass\n",
         J + "::test_mainpin2_static_exclusion_once_only_the_main_mask_toggles"),
        ("M7 no exclusion of threads created later", L,
         '        if want != _p2["pinned"]:\n            _p2_sweep()\n', '        if want != _p2["pinned"]:\n',
         J + "::test_mainpin2_new_threads_are_excluded_at_the_next_toggle"),
        ("M8 thread born on the pinned CPU restores to that CPU", L,
         '            base = set(_p2["main_saved"]) if m <= pinned else m  # born on the pinned CPU: the allowed set\n',
         "            base = m\n", J + "::test_mainpin2_new_threads_are_excluded_at_the_next_toggle"),
        ("M9 no refusal next to the old mainpin", L,
         '        if _configured("mainpin"):\n', "        if False:\n",
         J + "::test_mainpin2_refused_next_to_mainpin_and_cpu_pin_slice[mainpin]"),
        ("M10 no refusal next to VLLM_TPU_CPU_PIN=slice", L,
         '        elif CPU_PIN_MODE == "slice":\n', "        elif False:\n",
         J + "::test_mainpin2_refused_next_to_mainpin_and_cpu_pin_slice[cpu_pin_slice]"),
        ("M11 mainpin2 warm-up gate dropped", L,
         '    if not _warm["on"]:\n        return\n    want = l6_ab.on("mainpin2")\n', '    want = l6_ab.on("mainpin2")\n',
         J + "::test_mainpin2_waits_for_warm_up"),
        ("M12 chip node ignored when the allowed set spans nodes", L,
         "    node = _chip_node()\n    if node is None:\n", "    node = None\n    if node is None:\n",
         J + "::test_mainpin2_takes_the_chip_node_when_the_allowed_set_spans_nodes"),
        ("M13 rank ignored: every rank picks the same core", L,
         "    mine = cores[-1 - (rank % k)]\n", "    mine = cores[-1]\n",
         J + "::test_mainpin2_plan_gives_distinct_cores_with_idle_siblings"),
        ("M14 static exclusion skipped", L,
         "            _setaff(tid, _p2_exclusion(m))\n            moved += 1\n", "            moved += 1\n",
         J + "::test_mainpin2_static_exclusion_once_only_the_main_mask_toggles"),
        # ---- numa_bind port (4cfb870d7) and its gate (96bfba489 + VLLM_TPU_L6_NUMA_BIND)
        ("N1 launcher reservation widened (no intersection)", NB,
         "        cpus = node_cpus & inherited\n", "        cpus = node_cpus\n",
         NT + "::test_launcher_reservation_is_kept_by_intersection"),
        ("N2 env gate ignored", TW,
         "        if not (worker_options.get(self.vllm_config).numa_bind or envs.VLLM_TPU_L6_NUMA_BIND):\n",
         "        if not worker_options.get(self.vllm_config).numa_bind:\n",
         NT + "::test_env_flag_turns_the_bind_on"),
        ("N3 bind on by default", TW,
         "        if not (worker_options.get(self.vllm_config).numa_bind or envs.VLLM_TPU_L6_NUMA_BIND):\n"
         "            return\n", "        if False:\n            return\n",
         NT + "::test_env_flag_unset_leaves_the_bind_off"),
        ("N4 init_device never binds", TW,
         "        self._maybe_bind_numa(binding.native_local_rank, dp_size)\n", "",
         NT + "::test_init_device_binds_before_the_runtime_starts"),
        ("N5 worker section hashed into the compile key", RU,
         "        PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,\n        WORKER_OPTIONS_SECTION,\n",
         "        PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,\n",
         HP + "::test_worker_section_does_not_change_the_hash"),
        # ---- registration rows
        ("R11 syncgc missing from l6_ab.LEVERS", A,
         '    "syncgc": "VLLM_TPU_L6_SYNC_GC",\n', "", J + "::test_new_levers_are_registered_and_compile_ignored"),
        ("R12 mainpin2 missing from l6_ab.LEVERS", A,
         '    "mainpin2": "VLLM_TPU_L6_MAIN_PIN2",\n', "", J + "::test_new_levers_are_registered_and_compile_ignored"),
    ],
}
for i, (flag, rid, erow) in enumerate(FLAGS):
    MUTANTS["sync"].append((f"R{i + 1} {flag} dropped from the compile-key ignore list", C, f'    "{flag}",\n', "",
                            REG + f"::test_host_only_flag_keeps_the_compile_hash[{rid}]"))
    MUTANTS["sync"].append((f"R{i + 6} {flag} getter row dropped from envs.py", E, erow, "",
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
        print("Q38_L6_SYNC_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6smut.")
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
        print(f"Q38_L6_SYNC_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_SYNC_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_SYNC_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

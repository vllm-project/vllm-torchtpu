#!/usr/bin/env python3
"""q38 L6-HOST mutant runner (CPU pod only). Usage: python3 q38_l6_mutrun.py <head tree> <set>

For each mutant: copy the tree, apply one exact textual replacement (it must
match exactly once), run the mutant's target test id with pytest and require
that it FAILED (a failing test, not an error or a collection failure). A
control run on the unmutated tree must pass every target first.
Last line: Q38_L6_MUT PASS killed N/N of N | Q38_L6_MUT FAIL ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

T = "tests/q38_l6/test_l6_ab_envcache.py"
F = "tests/q38_l6/test_l6_fctx.py"
LK = "tests/q38_l6/test_l6_lockstep.py"
J = "tests/q38_l6/test_l6_jitter.py"
C = "tests/q38_l6/test_l6_sab_compose.py"
DF = "tests/q38_l6/test_l6_defer.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py"

EOS = "tests/q38_l6/test_l6_eos_canary.py"
MUTANTS = {
    "trims": [
        ("M1 envcache install ignores the cached hook",
         "src/vllm_torchtpu/core/l6_envcache.py",
         "    _envs.__getattr__ = _counting if CENSUS > 0 else inner",
         "    _envs.__getattr__ = _counting if CENSUS > 0 else _RAW",
         T + "::test_cached_reads_skip_the_parser_and_keep_the_value"),
        ("M2 off window keeps the cache",
         "src/vllm_torchtpu/core/l6_envcache.py",
         "    if not cached:\n        _cache.clear()\n    _install(",
         "    if False:\n        _cache.clear()\n    _install(",
         T + "::test_off_clears_the_cache"),
        ("M3 ab lever ignores its classes",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    return int(cls) in classes",
         "    return True",
         T + "::test_ab_classes_select_windows"),
        ("M4 refusal never set",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    REFUSED = windowed and (\n        _env_int(\"VLLM_TPU_HOST_AB_PERIOD_MS\", 0) > 0 or not _balanced_with_sab()\n    )",
         "    REFUSED = False",
         T + "::test_ab_refused_next_to_another_windowed_ab"),
        ("M5 no window log line",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    elif k != _state[\"win\"]:\n        _log_window()",
         "    elif k != _state[\"win\"]:\n        pass",
         T + "::test_ab_classes_select_windows"),
        ("M6 check keeps a stale cache",
         "src/vllm_torchtpu/core/l6_envcache.py",
         "        _cache.clear()\n    return len(bad)",
         "        pass\n    return len(bad)",
         T + "::test_check_mode_drops_a_stale_cache"),
        ("M7 census counts the step before closing (off by one)",
         "src/vllm_torchtpu/core/l6_envcache.py",
         "        if _census[\"steps\"] >= CENSUS:",
         "        _census[\"steps\"] += 1\n        if _census[\"steps\"] > CENSUS:",
         T + "::test_census_counts_reads_per_step"),
        ("M8 runner hook removed",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "            _l6_ab.begin_step(fp_fn=lambda: _l6_ab.sched_fingerprint(scheduler_output))\n            _l6_envcache.on_step()\n",
         "            pass\n",
         T + "::test_runner_calls_the_hooks_in_execute_model"),
        ("M9 envcache flag dropped from the compile-key ignore list",
         "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_ENVS_CACHE\",\n",
         "",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("M10 note is a no-op",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    _state[\"counts\"][name] = _state[\"counts\"].get(name, 0) + n",
         "    pass",
         T + "::test_note_counts_land_in_the_window_line"),
        ("M11 out-of-range ab class accepted",
         "src/vllm_torchtpu/core/l6_ab.py",
         "        if not cls or any(c < 0 or c >= classes for c in cls):",
         "        if not cls:",
         T + "::test_ab_needs_a_period_and_valid_classes"),
        ("M12 check never runs from on_step",
         "src/vllm_torchtpu/core/l6_envcache.py",
         "            if CHECK > 0 and l6_ab.check_allowed():",
         "            if False:",
         T + "::test_check_mode_runs_from_on_step"),
        ("M14 step windows shifted by one step",
         "src/vllm_torchtpu/core/l6_ab.py",
         "        k = _state[\"j\"] // STEPS",
         "        k = (_state[\"j\"] + 1) // STEPS",
         T + "::test_step_windows_align_with_hostphases_windows"),
        ("M15 period and steps both accepted",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    if PERIOD_MS and STEPS:",
         "    if False:",
         T + "::test_period_and_steps_are_exclusive"),
        ("L1 lockstep ignores unequal fingerprints",
         "src/vllm_torchtpu/core/l6_ab.py",
         "        if any(v is None for v in seen.values()) or len(set(seen.values())) != 1:",
         "        if any(v is None for v in seen.values()):",
         LK + "::test_desynchronised_rank_disables_the_rest_of_the_run"),
        ("L2 ab levers do not wait for confirmation",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    if STEPS > 0 and _state[\"rank_world\"][1] > 1 and (_state[\"sync_off\"] or _state[\"confirmed\"] != k):\n        return False\n",
         "",
         LK + "::test_missing_rank_keeps_ab_levers_off"),
        ("L3 SAB balance rule dropped",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    return STEPS > 0 and p % (STEPS * CLASSES) == 0",
         "    return True",
         LK + "::test_balanced_inside_sab_windows"),
        ("L4 D1-norm windows refused again",
         "src/vllm_torchtpu/core/l6_ab.py",
         "        _env_int(\"VLLM_TPU_HOST_AB_PERIOD_MS\", 0) > 0 or not _balanced_with_sab()",
         "        _env_int(\"VLLM_TPU_HOST_AB_PERIOD_MS\", 0) > 0 or _env_int(\"VLLM_TPU_D1N_AB_PERIOD_MS\", 0) > 0 or not _balanced_with_sab()",
         LK + "::test_d1n_windows_compose"),
        ("L5 fingerprint written one step late",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    if (j + CONFIRM) % STEPS == 0:\n        k = (j + CONFIRM) // STEPS",
         "    if (j + CONFIRM + 1) % STEPS == 0:\n        k = (j + CONFIRM + 1) // STEPS",
         LK + "::test_two_ranks_confirm_each_window"),
        ("J1 gc never unfrozen",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "        else:\n            gc.unfreeze()\n            l6_ab.note(\"gcfreeze\", \"unfreeze\")",
         "        else:\n            l6_ab.note(\"gcfreeze\", \"unfreeze\")",
         J + "::test_gcfreeze_follows_the_windows"),
        ("J2 switch restored to the interpreter default",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "            sys.setswitchinterval(_state[\"switch_prev\"])\n            l6_ab.note(\"switch\", \"restore\")",
         "            sys.setswitchinterval(0.005)\n            l6_ab.note(\"switch\", \"restore\")",
         J + "::test_switch_sets_and_restores_the_interval"),
        ("J3 D1e freeze conflict not refused",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "    if want and D1E_FREEZE:",
         "    if False:",
         J + "::test_gcfreeze_refused_next_to_d1e_freeze"),
        ("J4 runner hook for the jitter levers removed",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "            _l6_jitter.on_step()\n",
         "",
         T + "::test_runner_calls_the_hooks_in_execute_model"),
        ("P1 unpin restores nothing",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "            os.sched_setaffinity(tid, mask)\n        except ProcessLookupError:\n            pass",
         "            pass\n        except ProcessLookupError:\n            pass",
         J + "::test_mainpin_isolates_the_main_thread_and_restores"),
        ("P2 other threads stay on the pinned CPU",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "                os.sched_setaffinity(tid, (saved[tid] - reserved) or rest)\n",
         "                pass\n",
         J + "::test_mainpin_isolates_the_main_thread_and_restores"),
        ("P3 a failed pin retries every step",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "            _state[\"pin_broken\"] = True\n",
         "",
         J + "::test_mainpin_failure_turns_it_off_once"),
        ("D1 any KV role defers",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "    if cfg is None or getattr(cfg, \"kv_role\", None) != \"kv_consumer\":",
         "    if cfg is None:",
         DF + "::test_only_a_pure_consumer_with_a_device_seeded_draft_defers"),
        ("D2 partial-prefill steps defer too",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "        and not discard_sampled_tokens_req_indices\n    )",
         "    )",
         DF + "::test_only_a_pure_consumer_with_a_device_seeded_draft_defers"),
        ("D3 lever never consulted",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "            and _l6_ab.on(\"defer\")\n",
         "            and False\n",
         DF + "::test_sample_tokens_switch_sits_before_the_early_wait"),
        ("S1 own SAB parser accepts what SAB rejects",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    except ValueError:\n        return None\n    return None if pc is None else pc[0]",
         "    except ValueError:\n        spec = os.environ.get(\"VLLM_TPU_SERVED_AB\", \"\").split(\":\")\n        return int(spec[1]) if len(spec) > 1 and spec[1].isdigit() else None\n    return None if pc is None else pc[0]",
         LK + "::test_sab_period_follows_served_ab_parse"),
        ("K1 CPU_PROBE dropped from the compile-key ignore list (L1-V3 4ef7ba74e)",
         "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_CPU_PROBE_S\",\n",
         "",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("N1 every rank pins the node's highest CPU",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "    return cpus[-1 - (rank % k)], set(cpus[-k:])",
         "    return cpus[-1], set(cpus[-k:])",
         J + "::test_mainpin_ranks_pick_distinct_cpus_and_reserve_all"),
        ("N2 rescan reads the pinned main thread's live mask",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "    allowed = saved.get(main) or os.sched_getaffinity(main)",
         "    allowed = os.sched_getaffinity(main)",
         J + "::test_mainpin_rescans_on_consecutive_on_windows_and_reserves_for_every_rank"),
        ("N3 other threads leave the other ranks' CPUs",
         "src/vllm_torchtpu/core/l6_jitter.py",
         "                os.sched_setaffinity(tid, (saved[tid] - reserved) or rest)",
         "                os.sched_setaffinity(tid, (saved[tid] - {cpu}) or rest)",
         J + "::test_mainpin_rescans_on_consecutive_on_windows_and_reserves_for_every_rank"),
        ("W1 CHECK window never closes",
         "src/vllm_torchtpu/core/l6_ab.py",
         "    if _state[\"j\"] <= CHECK_STEPS:\n        return True",
         "    if True:\n        return True",
         LK + "::test_check_window_closes_after_n_steps"),
        ("W2 steady ignores the CHECK window",
         "src/vllm_torchtpu/runner/l6_steady.py",
         "    return v if _l6_ab.check_allowed() else 0",
         "    return v",
         LK + "::test_check_window_closes_after_n_steps"),
        ("W3 carry ignores the CHECK window",
         "src/vllm_torchtpu/runner/decode_carry.py",
         "    return int(raw) if l6_ab.check_allowed() else 0",
         "    return int(raw)",
         LK + "::test_check_window_closes_after_n_steps"),
        ("X1 '+' no longer separates classes",
         "src/vllm_torchtpu/core/l6_ab.py",
         "raw[3:].replace(\"+\", \",\").split(\",\")",
         "raw[3:].split(\",\")",
         T + "::test_plus_separates_classes_like_comma"),
        ("C1 L6 class cycle doubled (unbalanced inside SAB windows)",
         "src/vllm_torchtpu/core/l6_ab.py",
         "        k = _state[\"j\"] // STEPS",
         "        k = _state[\"j\"] // (2 * STEPS)",
         C + "::test_every_sab_window_holds_equal_l6_class_shares"),
        ("F1 per-call copy shares slot_mapping",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "    fc.slot_mapping = {}\n",
         "",
         F + "::test_writes_during_one_forward_do_not_reach_the_next"),
        ("F2 per-call copy shares DPMetadata",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "        fc.dp_metadata = copy.copy(base.dp_metadata)",
         "        pass",
         F + "::test_writes_during_one_forward_do_not_reach_the_next"),
        ("F3 key drops num_tokens",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "    return (id(vllm_config), num_tokens, dp)",
         "    return (id(vllm_config), dp)",
         F + "::test_num_tokens_is_part_of_the_key"),
        ("F4 switch always on",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "    return _l6_ab.on(\"fctx\")",
         "    return True",
         F + "::test_enabled_follows_the_l6_lever"),
        ("F5 target forward site reverted to stock",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "                (_ffc.fast_set_forward_context if _ffc.enabled() else set_forward_context)(\n                    attn_metadata,",
         "                set_forward_context(\n                    attn_metadata,",
         F + "::test_step_call_sites_pick_the_fast_path_only_when_enabled"),
        ("F6 hit keeps the cached attn_metadata",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "    fc.attn_metadata = attn_metadata\n",
         "    fc.attn_metadata = base.attn_metadata\n",
         F + "::test_cached_context_matches_stock_and_swaps_metadata"),
        ("F7 extra kwargs dropped onto the fast path",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "    if kwargs or getattr(_fc, \"track_batchsize\", False):",
         "    if getattr(_fc, \"track_batchsize\", False):",
         F + "::test_extra_kwargs_take_stock_path"),
        ("F8 hits not counted",
         "src/vllm_torchtpu/runner/fast_forward_context.py",
         "    _l6_ab.note(\"fctx\", \"hit\")\n",
         "",
         F + "::test_engagement_counts_hits_and_builds"),
        ("F9 fctx flag dropped from the compile-key ignore list",
         "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_FAST_FWD_CTX\",\n",
         "",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("M13 lever state recomputed against the wrong window class",
         "src/vllm_torchtpu/core/l6_ab.py",
         "        _state[\"on\"] = {lever: _lever_on(lever, cls, k) for lever in _cfg}",
         "        _state[\"on\"] = {lever: _lever_on(lever, \"0\" if cls != \"uniform\" else cls, k) for lever in _cfg}",
         T + "::test_ab_classes_select_windows"),
    ],
    # q38 L6 end-token canary (core/l6_eos_canary.py; log-only)
    "eos": [
        ("X1 hook counts before the discarded requests are cleared",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         """        for i in pre_discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # q38 L6 end-token canary (log-only): rank 0 counts the step's end-token emissions (core/l6_eos_canary.py).
        if _l6_eos.ON:
            _l6_eos.count(valid_sampled_token_ids)
""",
         """        # q38 L6 end-token canary (log-only): rank 0 counts the step's end-token emissions (core/l6_eos_canary.py).
        if _l6_eos.ON:
            _l6_eos.count(valid_sampled_token_ids)

        for i in pre_discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
""",
         EOS + "::test_runner_counts_after_the_discards_and_before_the_rollback"),
        ("X2 every rank counts",
         "src/vllm_torchtpu/core/l6_eos_canary.py",
         "    if not _l6_ab.ACTIVE or not _is_rank0():",
         "    if not _l6_ab.ACTIVE:",
         EOS + "::test_other_ranks_count_nothing"),
        ("X3 end tokens never counted",
         "src/vllm_torchtpu/core/l6_eos_canary.py",
         "            if t in ids:\n                end += 1",
         "            if t in ids and False:\n                end += 1",
         EOS + "::test_counts_tokens_and_end_tokens_on_rank0"),
        ("X4 '+' not accepted between ids",
         "src/vllm_torchtpu/core/l6_eos_canary.py",
         'raw.replace("+", ",").split(",")',
         'raw.split(",")',
         EOS + "::test_parse_ids"),
        ("X5 flag not compile-ignored",
         "src/vllm_torchtpu/compilation/tpu_compiler.py",
         '    "VLLM_TPU_L6_EOS_CANARY",\n',
         "",
         EOS + "::test_flag_registered_and_compile_ignored"),
        ("X7 solo phase counted on multi-request steps",
         "src/vllm_torchtpu/core/l6_eos_canary.py",
         "    if active == 1:\n",
         "    if active >= 1:\n",
         EOS + "::test_solo_phase_steps_are_counted_apart"),
        ("X6 hook not guarded by the flag",
         "src/vllm_torchtpu/runner/tpu_runner.py",
         "        if _l6_eos.ON:\n            _l6_eos.count(valid_sampled_token_ids)\n",
         "        if True:\n            _l6_eos.count(valid_sampled_token_ids)\n",
         EOS + "::test_runner_counts_after_the_discards_and_before_the_rollback"),
    ],
}


def run(tree, test_id, env):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", test_id]
    p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error|errors|skipped).* in [0-9.]+s", ln)]
    return p.returncode, (summ[-1] if summ else "NO SUMMARY"), out


def main():
    tree, which = os.path.abspath(sys.argv[1]), sys.argv[2]
    muts = MUTANTS[which]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="src")
    env.pop("Q38_L6_EXPECT_TREE", None)
    targets = sorted({m[4] for m in muts})
    ok = True
    for t in targets:
        rc, s, out = run(tree, t, env)
        good = rc == 0 and re.search(r"\b[1-9][0-9]* passed\b", s) and "failed" not in s and "error" not in s
        print(f"CONTROL {'ok' if good else 'BAD'} {t}: {s}")
        if not good:
            print(out[-3000:])
            ok = False
    if not ok:
        print("Q38_L6_MUT FAIL control")
        return 1
    killed = 0
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6mut.")
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
        print(f"MUT {name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {summ}")
        if not dead:
            print(out[-2000:])
        shutil.rmtree(d, ignore_errors=True)
    tot = len(muts)
    if killed == tot:
        print(f"Q38_L6_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

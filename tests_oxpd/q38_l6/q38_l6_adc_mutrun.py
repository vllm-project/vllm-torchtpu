#!/usr/bin/env python3
"""q38 L6 lever adcarry mutant runner (CPU pod only). Usage: python3 q38_l6_adc_mutrun.py <head tree> <set> [survey]

Modelled on q38_l6_engalloc_mutrun.py (L6-ENGINE). For each mutant: copy the tree, apply one exact textual
replacement (it must match exactly once), run the mutant's target test id with pytest and require that it FAILED
(a failing test, not an error or a collection failure). A control run on the unmutated tree must pass every target
first. 'survey' reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_ADC_MUT PASS killed N/N of N | Q38_L6_ADC_MUT FAIL ... | Q38_L6_ADC_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

AC = "src/vllm_torchtpu/runner/adapt_carry.py"
DC = "src/vllm_torchtpu/runner/decode_carry.py"
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
IN = "src/vllm_torchtpu/models/vllm/qwen4_exp/inputs.py"
LA = "src/vllm_torchtpu/core/l6_ab.py"
EV = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T = "tests/q38_l6/test_l6_adapt_carry.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py"
STREAM_ADC = T + "::test_folded_path_matches_today_over_random_streams[b3_compact-adc]"
STREAM_CARRY = T + "::test_folded_path_matches_today_over_random_streams[b3_compact-adc_carry]"
STEADY = T + "::test_steady_carried_steps_merge_and_skip_the_window_upload[b3_compact]"
PARKED = T + "::test_each_parked_field_refuses_alone[%s]"

MUTANTS = {
    "adc": [
        # -- the brief's named mutants
        ("A1 window reused for a new row (carried and row-map clauses dropped)", AC,
         "        carried\n        and rows_ok\n        and win is not None\n",
         "        win is not None\n", T + "::test_resumed_row_on_a_merged_step_uploads_the_window"),
        ("A2 window reused without the row map (idle adapter step)", AC,
         "        carried\n        and rows_ok\n        and win is not None\n",
         "        carried\n        and win is not None\n", T + "::test_idle_adapter_step_refuses_row_map_and_window"),
        ("A3 window reused on an uncarried step (host skew)", AC,
         "        carried\n        and rows_ok\n        and win is not None\n",
         "        rows_ok\n        and win is not None\n", T + "::test_skewed_host_counts_refuse_the_window"),
        ("A4 window reused below the width rule (short context)", AC,
         "        and bool(np.all(ctx >= width))\n", "", T + "::test_short_context_rows_upload_the_window"),
        ("A5 record not swapped in (merged result)", AC,
         "            adapter._recent[1] = p.record\n", "", STREAM_CARRY),
        ("A6 record not swapped in (adapter's own program)", AC,
         "    if functional:\n        adapter._recent[1] = out[4]\n", "    if functional:\n",
         T + "::test_unmerged_steps_keep_the_row_map_reuse"),
        ("A7 stale scalars after a reorder (ids clause dropped)", AC,
         "rec is not None and rec[1] == ids and previous_rows is rec[2]",
         "rec is not None and previous_rows is rec[2]", T + "::test_reorder_rebuilds_the_scalars"),
        ("A8 stale scalars after an idle adapter step (dict identity dropped)", AC,
         "rec is not None and rec[1] == ids and previous_rows is rec[2]",
         "rec is not None and rec[1] == ids", T + "::test_idle_adapter_step_refuses_row_map_and_window"),
        ("A9 padded scalars rows nonzero", AC,
         "    if dev is None:\n        host = np.full(2 + rows, -1, dtype=np.int64)\n",
         "    if dev is None:\n        host = np.full(2 + rows, 0, dtype=np.int64)\n",
         T + "::test_every_adapter_program_input_matches"),
        ("A10 reused window with live rows past this step's (padded rows nonzero)", AC,
         "        and win[1] <= n\n", "", T + "::test_window_guard_clauses"),
        ("A11 postfwd fold skips the substitution", AC,
         "        post = subtract_substitute_extract(*args[:_N_POST])\n",
         "        post = subtract_substitute_extract(*args[:7], args[7] * 0 - 1, *args[8:_N_POST])\n", STREAM_ADC),
        ("A12 carry fold skips the substitution", AC,
         "        post = carry_subtract_substitute_extract(step_delta, *args[:_N_POST])\n",
         "        post = carry_subtract_substitute_extract(step_delta, *args[:7], args[7] * 0 - 1, *args[8:_N_POST])\n",
         STEADY),
        ("A13 merged adapter half reads unsubstituted ids", AC,
         "            (c0, c1, c2),\n            input_ids,\n", "            (c0, c1, c2),\n            input_ids * 0,\n",
         STREAM_ADC),
        ("A14 CHECK_STEPS bound dropped", AC,
         "    return int(raw) if _l6_ab.check_allowed() else 0\n", "    return int(raw)\n",
         T + "::test_check_steps_bound_the_check_turns"),
        ("A15 l6_ab gate dropped (runner)", AC,
         "    return _l6_ab.ACTIVE and _l6_ab.on(LEVER)\n", "    return _l6_ab.ACTIVE\n",
         T + "::test_lever_off_runs_no_adcarry_code"),
        ("A16 l6_ab gate dropped (adapter)", IN,
         "        if _l6_ab.ACTIVE and _l6_ab.on(\"adcarry\"):\n", "        if _l6_ab.ACTIVE:\n",
         T + "::test_lever_off_runs_no_adcarry_code"),
        # -- registration rows
        ("A17 lever missing from l6_ab.LEVERS", LA,
         "    \"adcarry\": \"VLLM_TPU_L6_ADAPT_CARRY\",\n", "", T + "::test_lever_is_registered_and_compile_ignored"),
        ("A18 envs row VLLM_TPU_L6_ADAPT_CARRY removed", EV,
         "    \"VLLM_TPU_L6_ADAPT_CARRY\": env_str(\"VLLM_TPU_L6_ADAPT_CARRY\", \"\"),\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
        ("A19 envs row VLLM_TPU_L6_ADAPT_CARRY_CHECK removed", EV,
         "    \"VLLM_TPU_L6_ADAPT_CARRY_CHECK\": env_nonnegative_int(\"VLLM_TPU_L6_ADAPT_CARRY_CHECK\", 0),\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
        ("A20 compile-ignore row VLLM_TPU_L6_ADAPT_CARRY removed", TC,
         "    \"VLLM_TPU_L6_ADAPT_CARRY\",\n", "", REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("A21 compile-ignore row VLLM_TPU_L6_ADAPT_CARRY_CHECK removed", TC,
         "    \"VLLM_TPU_L6_ADAPT_CARRY_CHECK\",\n", "",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        # -- every other cache key field: row map (c)
        ("A22 row-map key: full-chunk clause dropped", AC,
         "    return full and row_start == 0 and rec is not None",
         "    return row_start == 0 and rec is not None", T + "::test_rows_guard_clauses"),
        ("A23 row-map key: first-row clause dropped", AC,
         "    return full and row_start == 0 and rec is not None",
         "    return full and rec is not None", T + "::test_rows_guard_clauses"),
        ("A24 row-map record kept for a partial chunk", AC,
         "    if row_start == 0 and n == runner.input_batch.num_reqs:\n        adapter._adc_rows",
         "    if True:\n        adapter._adc_rows", T + "::test_commit_records_only_full_chunks"),
        ("A25 row map not applied", AC,
         "    adapter._current_rows.update(st.pending)\n", "", STREAM_ADC),
        ("A26 identity scalars cache keyed by count only", AC,
         "    key = (n, rows)\n", "    key = n\n", T + "::test_identity_scalars_are_keyed_by_rows_and_count"),
        # window (a)
        ("A27 window key: shape clause dropped", AC,
         "        and win[0] == (rows, width + 1)\n", "", T + "::test_window_guard_clauses"),
        ("A28 window key: spec-window record clause dropped", AC,
         "        and bool(np.all(np.asarray(nsched[:n]) >= spec_window))\n", "", T + "::test_window_guard_clauses"),
        # merged warm key
        ("A29 merged warm-key check dropped (serving compiles)", AC,
         "    if warm_key(caller_key, window) not in", "    if False and warm_key(caller_key, window) not in",
         T + "::test_serving_never_compiles_a_merged_program_and_precompile_warms_all"),
        ("A30 merged warm key without the window mode", AC,
         "    return (caller_key, window)\n", "    return (caller_key,)\n",
         T + "::test_serving_never_compiles_a_merged_program_and_precompile_warms_all"),
        ("A31 precompile skips the carry form", AC,
         "((\"postfwd\", tuple(key)), (\"carry\", (step_delta,) + tuple(key)))",
         "((\"postfwd\", tuple(key)),)",
         T + "::test_serving_never_compiles_a_merged_program_and_precompile_warms_all"),
        ("A32 merged during warm-up", AC,
         "    if not getattr(runner, \"warmup_done\", False):\n        return \"warmup\"\n", "",
         T + "::test_warmup_steps_run_todays_programs"),
        ("A33 warm-up hook missing", TR,
         "                if _adapt_carry.flag_set():\n", "                if False:\n",
         T + "::test_flag_set_and_warmup_hook"),
        # parked result key
        ("A34 parked key: step", AC, "        p.step == _l6_ab.step_seq()\n", "        True\n", PARKED % "step"),
        ("A35 parked key: first row", AC, "        and snapshot.row_start == 0\n", "", PARKED % "row_start"),
        ("A36 parked key: request count", AC, "        and snapshot.num_requests == p.n\n", "",
         PARKED % "num_requests"),
        ("A37 parked key: input ids", AC, "        and snapshot.input_ids is p.ids\n", "", PARKED % "input_ids"),
        ("A38 parked key: token count", AC, "        and snapshot.num_tokens == int(p.ids.shape[0])\n", "",
         PARKED % "num_tokens"),
        ("A39 parked key: seq_lens", AC, "        and ctx.seq_lens is p.seq_lens\n", "", PARKED % "seq_lens"),
        ("A40 parked key: query_start_loc", AC, "        and ctx.query_start_loc is p.starts\n", "",
         PARKED % "query_start_loc"),
        ("A41 parked key: state indices", AC, "        and (adapter.pooled or ctx.mamba_state_indices is p.state)\n",
         "", PARKED % "state_indices"),
        ("A42 parked key: window mode", AC, "        and p.window == window\n", "", PARKED % "window"),
        ("A43 parked key: token store", AC, "        and (window or snapshot.token_store is p.store)\n", "",
         T + "::test_token_store_identity_refuses_in_store_mode"),
        # CHECK and the carry CHECK hand-off
        ("A44 merged CHECK never reports a mismatch", AC,
         "    if not bad:\n        return out, st\n", "    if True:\n        return out, st\n",
         T + "::test_check_mode_catches_corrupted_cached_scalars[1]"),
        ("A45 carry CHECK mismatch keeps the parked result", DC,
         "    adapt_carry.discard(runner)", "    pass",
         T + "::test_carry_check_mismatch_discards_the_parked_result"),
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
    env.pop("Q38_L6ADC_EXPECT_ROOT", None)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    targets = sorted({m[4] for m in muts})
    ok = True
    for t in targets:
        rc, s, out = run(tree, t, env)
        good = rc == 0 and re.search(r"\b[1-9][0-9]* passed\b", s) and "failed" not in s and "error" not in s
        print(f"CONTROL {'ok' if good else 'BAD'} {t}: {s}", flush=True)
        if not good:
            print(out[-3000:])
            ok = False
    if not ok and not survey:
        print("Q38_L6_ADC_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6adcmut.")
        shutil.copytree(os.path.join(tree, "src"), os.path.join(d, "src"))
        shutil.copytree(os.path.join(tree, "tests"), os.path.join(d, "tests"))
        f = os.path.join(d, path)
        s = open(f).read()
        n = s.count(old)
        if n != 1:
            print(f"MUT {name}: anchor matched {n} times -> INVALID", flush=True)
            shutil.rmtree(d, ignore_errors=True)
            continue
        open(f, "w").write(s.replace(old, new))
        rc, summ, out = run(d, t, env)
        dead = rc != 0 and re.search(r"\b[1-9][0-9]* failed\b", summ) and "error" not in summ
        killed += bool(dead)
        if dead:
            names.append(name.split()[0])
        print(f"MUT {name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {summ}", flush=True)
        if not dead:
            print(out[-1500:])
        shutil.rmtree(d, ignore_errors=True)
    tot = len(muts)
    if survey:
        print(f"Q38_L6_ADC_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_ADC_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_ADC_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""q38 L6-H2 (overflow unit l6_h2_cheap) mutant runner, CPU pod only.
Usage: python3 q38_l6_h2cheap_mutrun.py <head tree> <fskip|toklists|aotskip|all> [survey]

Modelled on L6-ENGINE's q38_l6_engalloc_mutrun.py. For each mutant: copy the tree, apply one exact
textual replacement (it must match exactly once), run the mutant's target test id with pytest and
require that it FAILED (a failing test, not an error or a collection failure). A control run on the
unmutated tree must pass every target first. 'survey' reports every mutant without requiring all to
be killed. Last line: Q38_L6_H2C_MUT PASS killed N/N of N | ... FAIL ... | ... SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

FS = "src/vllm_torchtpu/runner/l6_fskip.py"
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
TAO = "src/vllm_torchtpu/runner/tpu_runner_async_output.py"
LA = "src/vllm_torchtpu/compilation/l6_aotskip.py"
AB = "src/vllm_torchtpu/core/l6_ab.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
PR = "src/vllm_torchtpu/patch_registry.py"
AM = "src/vllm_torchtpu/layers/core/attention_metadata.py"
DC = "src/vllm_torchtpu/runner/decode_carry.py"
SDM = "src/vllm_torchtpu/runner/speculative_decoding_manager.py"
PLE = "src/vllm_torchtpu/models/vllm/qwen4_exp/ple_offload.py"
TF = "tests/q38_l6/test_l6_fskip.py"
TT = "tests/q38_l6/test_l6_toklists.py"
TA = "tests/q38_l6/test_l6_aotskip.py"
CHURN = TF + "::test_no_upload_since_the_previous_fence_point_at_any_skipped_fence[b3_padded-churn_fixed]"
STEADY = TF + "::test_no_upload_since_the_previous_fence_point_at_any_skipped_fence[b3_padded-steady]"
SCAN = TF + "::test_every_host_to_device_upload_is_marked_or_exempt"
GET_OUT = TT + "::test_get_output_lists_equal_tensor_path[s5-n4-c4-k1]"
GET_OUT1 = TT + "::test_get_output_lists_equal_tensor_path[s2-n1-c1-k1]"
MPR = TT + "::test_modify_prev_results_lists_equal_tensor_path[1-b3_padded]"

MUTANTS = {
    "fskip": [
        ("F1 skip unconditionally", FS, "    if carried and n == 0:\n", "    if True:\n", CHURN),
        ("F2 carried gate dropped", FS, "    if carried and n == 0:\n", "    if n == 0:\n",
         TF + "::test_skip_rule_and_reset"),
        ("F3 reuse-helper mark dropped", TR,
         "        dev = host.to(self.device, non_blocking=True)\n        _l6_fskip.S.n += 1  # q38 L6 fskip\n"
         "        self._h2d_reuse[slot] = (key, dev, host)\n        return dev\n\n    def _execute_mm_encoder(",
         "        dev = host.to(self.device, non_blocking=True)\n"
         "        self._h2d_reuse[slot] = (key, dev, host)\n        return dev\n\n    def _execute_mm_encoder(",
         TF + "::test_reuse_helpers_count_uploads_not_hits"),
        ("F4 carried flag never set", TR,
         "            _l6_fskip.S.carried = True  # q38 L6 fskip: this chunk was carried\n", "", STEADY),
        ("F5 counter not reset at the record site", FS, "    S.n = 0\n    S.carried = False\n",
         "    S.carried = False\n", TF + "::test_skip_rule_and_reset"),
        ("F6 lever state ignored", FS, "    if not (_l6_ab.ACTIVE and _l6_ab.on(LEVER)):\n",
         "    if not _l6_ab.ACTIVE:\n", TF + "::test_lever_off_records_every_chunk"),
        ("F7 carry upload mark dropped", DC,
         "    _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py): three uploads from the staging tensors\n",
         "", TF + "::test_carry_upload_counts"),
        ("F8 non-carried prepare mark dropped (scan)", TR,
         "            _l6_fskip.S.n += 1  # q38 L6 fskip: input ids / positions / seq_lens uploads\n", "", SCAN),
        ("F9 block-table staging mark dropped", AM,
         '    alike (tests swap it for a copy so a stale reused upload is visible on CPU)."""\n'
         "    _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)\n",
         '    alike (tests swap it for a copy so a stale reused upload is visible on CPU)."""\n',
         TF + "::test_block_table_staging_upload_counts"),
        ("F10 record site never skips", TR, "        if _l6_fskip.skip():\n", "        if _l6_fskip.skip() and False:\n",
         STEADY),
        ("F11 lever missing from l6_ab.LEVERS", AB, '    "fskip": "VLLM_TPU_L6_FENCE_SKIP",\n', "",
         TF + "::test_lever_is_registered_and_compile_ignored"),
        ("F12 lever dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_FENCE_SKIP",\n', "",
         TF + "::test_lever_is_registered_and_compile_ignored"),
        ("F13 deferred draft-id upload mark dropped", SDM,
         "        assert self.deferred_draft_token_ids_cpu is not None\n"
         "        _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)\n",
         "        assert self.deferred_draft_token_ids_cpu is not None\n",
         TF + "::test_spec_metadata_uploads_count"),
        ("F14 read-offset reset mark dropped (scan)", TR,
         "        keep_dev.copy_(keep, non_blocking=True)\n        _l6_fskip.S.n += 1  # q38 L6 fskip\n",
         "        keep_dev.copy_(keep, non_blocking=True)\n", SCAN),
        ("F15 spec-metadata cache-hit draft upload mark dropped (scan; the old scan accepted it)", SDM,
         "            if not defer_draft_token_ids:\n                _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)\n"
         "                draft_token_ids_dev",
         "            if not defer_draft_token_ids:\n                draft_token_ids_dev", SCAN),
        ("F16 scan accepts a mark anywhere in an enclosing block", TF,
         "        if any(_is_mark(s) for s in stmts[:i]):\n", "        if any(_is_mark(s) for s in stmts):\n",
         TF + "::test_scan_sees_a_missing_mark"),
        ("F17 scan sees only a literal non_blocking=True", TF,
         "            return not (isinstance(k.value, ast.Constant) and k.value.value is False)\n",
         "            return isinstance(k.value, ast.Constant) and k.value.value is True\n",
         TF + "::test_scan_sees_a_missing_mark"),
        ("F18 PLE host-lookup upload mark dropped (scan, non-literal non_blocking)", PLE,
         "    raw = table.lookup_host(host_ids)\n    _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)\n    return raw",
         "    raw = table.lookup_host(host_ids)\n    return raw", SCAN),
    ],
    "toklists": [
        ("T1 no INVALID filter", TAO,
         "    return [[tok for tok in row if tok != invalid] for row in raw_rows]\n",
         "    return [row[:] for row in raw_rows]\n", GET_OUT),
        ("T2 one-column rows shared, not copied", TAO,
         "    if max_gen_len == 1:\n        return [row[:] for row in raw_rows]\n",
         "    if max_gen_len == 1:\n        return raw_rows\n", GET_OUT1),
        ("T3 lists path ignores discards", TAO,
         "            for i in self.discard_sampled_tokens_req_indices:\n                valid_sampled_token_ids[i].clear()\n"
         "            _L6_OUT_LISTS[0] += 1\n",
         "            _L6_OUT_LISTS[0] += 1\n", GET_OUT),
        ("T4 copy state ignores the lever", TAO,
         '            l6_lists=_l6_ab.ACTIVE and _l6_ab.on("toklists"),\n', "            l6_lists=False,\n",
         TT + "::test_copy_state_takes_the_issuing_steps_lever_state"),
        ("T5 CHECK never compares", TAO, "            if stock != rows:\n", "            if False:\n",
         TT + "::test_check_mode_compares_and_catches_a_wrong_filter"),
        ("T6 CHECK unbounded by VLLM_TPU_L6_CHECK_STEPS", TAO,
         "    if check > 0 and _l6_ab.check_allowed():\n", "    if check > 0:\n",
         TT + "::test_check_steps_bound_the_check"),
        ("T7 main thread never takes the lists path", TR,
         "            and pre_copy_state.l6_lists\n            and pre_next_tokens_cpu.dim() == 2\n",
         "            and pre_copy_state.l6_lists\n            and pre_next_tokens_cpu.dim() == 2\n            and False\n",
         MPR),
        ("T11 results without a real copy state reach the lists check", TR,
         "            isinstance(pre_copy_state, AsyncTPUCopyState)\n            and pre_copy_state.l6_lists\n",
         "            pre_copy_state.l6_lists\n",
         "tests/runner/test_tpu_runner_preemption.py::test_async_results_do_not_overwrite_recomputed_request[0-continue]"),
        ("T8 main thread rows not sliced to the parked requests", TAO,
         "    rows = l6_valid_rows(copy_state.sampled_token_ids_list[:n], max_gen_len)\n",
         "    rows = l6_valid_rows(copy_state.sampled_token_ids_list[1:n + 1], max_gen_len)\n", MPR),
        ("T9 lever missing from l6_ab.LEVERS", AB, '    "toklists": "VLLM_TPU_L6_TOKEN_LISTS",\n', "",
         TT + "::test_lever_is_registered_and_compile_ignored"),
        ("T10 lever dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_TOKEN_LISTS",\n', "",
         TT + "::test_lever_is_registered_and_compile_ignored"),
    ],
    "aotskip": [
        ("A1 toggle ignored", LA, "        fn._guard_check_enabled = orig and not skip\n",
         "        fn._guard_check_enabled = orig\n", TA + "::test_windows_flip_guard_evaluation_on_a_real_artifact"),
        ("A2 original state ignored", LA, "        fn._guard_check_enabled = orig and not skip\n",
         "        fn._guard_check_enabled = not skip\n", TA + "::test_cache_loaded_artifact_is_never_switched_on"),
        ("A3 a late artifact is not put in the current state", LA,
         "    if applied is not None:\n        fn._guard_check_enabled = orig and not applied\n", "",
         TA + "::test_registered_mid_run_takes_the_current_state"),
        ("A4 on_step never writes", LA, '    if skip != _state["applied"]:\n', "    if False:\n",
         TA + "::test_windows_flip_guard_evaluation_on_a_real_artifact"),
        ("A5 CHECK never evaluates", LA, '        if _state["check_n"] >= n:\n', "        if False:\n",
         TA + "::test_check_mode_evaluates_guards_every_nth_lever_on_step"),
        ("A6 CHECK unbounded by VLLM_TPU_L6_CHECK_STEPS", LA, "    if n > 0 and l6_ab.check_allowed():\n",
         "    if n > 0:\n", TA + "::test_check_steps_bound_the_check"),
        ("A7 cache-loaded artifacts not kept", LA,
         "            else:\n                register(loaded)\n            return loaded\n",
         "            return loaded\n", TA + "::test_cache_loaded_artifact_is_never_switched_on"),
        ("A8 installed next to the hashed AOTSKIP", LA,
         '    if (os.environ.get("VLLM_TPU_AOT_SKIP_GUARD_CHECK", "") or "").strip().lower() in ("1", "true", "yes", "on"):\n',
         "    if False:\n", TA + "::test_not_installed_next_to_the_hashed_aotskip"),
        ("A9 worker hook dropped", TR, "                _l6_aot.on_step()\n", "                pass\n",
         TA + "::test_worker_call_site_is_guarded"),
        ("A10 registry row dropped", PR, '    Patch("vllm_torchtpu:_patch_vllm_l6_aotskip", _ACTIVE),\n', "",
         TA + "::test_registry_row_fresh_process_off_identity_and_on_install"),
        ("A11 lever missing from l6_ab.LEVERS", AB, '    "aotskip": "VLLM_TPU_L6_AOT_SKIP",\n', "",
         TA + "::test_lever_is_registered_and_compile_ignored"),
        ("A12 lever dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_AOT_SKIP",\n', "",
         TA + "::test_lever_is_registered_and_compile_ignored"),
        ("A13 flag-unset install wraps anyway", LA, "    if not flag_set():\n        return\n    if (os.environ",
         "    if False:\n        return\n    if (os.environ", TA + "::test_flag_unset_installs_nothing"),
        ("A14 fresh artifact kept although vLLM evaluates its guards", LA, "            if _evaluates_guards(self):\n",
         "            if False:\n", TA + "::test_fresh_artifact_with_evaluated_guards_is_never_kept[evaluate_guards]"),
        ("A15 missing config read as not evaluating", LA,
         '    return bool(getattr(getattr(config, "dynamic_shapes_config", None), "evaluate_guards", True))\n',
         '    return bool(getattr(getattr(config, "dynamic_shapes_config", None), "evaluate_guards", False))\n',
         TA + "::test_fresh_artifact_with_evaluated_guards_is_never_kept[no_config]"),
        ("A16 cache-loaded artifact with its check on is kept", LA,
         '            if loaded is not None and getattr(loaded, "_guard_check_enabled", None) is True:\n',
         "            if False:\n", TA + "::test_cache_loaded_artifact_with_the_check_on_is_never_kept"),
    ],
}
MUTANTS["all"] = MUTANTS["fskip"] + MUTANTS["toklists"] + MUTANTS["aotskip"]


def run(tree, test_id, env):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging",
           "-x", test_id]
    p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1500)
    out = p.stdout + p.stderr
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error|errors|skipped).* in [0-9.]+s", ln)]
    return p.returncode, (summ[-1] if summ else "NO SUMMARY"), out


def main():
    tree, which = os.path.abspath(sys.argv[1]), sys.argv[2]
    survey = len(sys.argv) > 3 and sys.argv[3] == "survey"
    muts = MUTANTS[which]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="src:tests/q38_l6")
    env.pop("Q38_L6_EXPECT_TREE", None)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
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
        print("Q38_L6_H2C_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6h2cmut.")
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
        print(f"Q38_L6_H2C_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_H2C_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_H2C_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

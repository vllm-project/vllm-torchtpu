#!/usr/bin/env python3
"""q38 L6-V6 compose mutant runner (CPU pod only): the composition tests of tests/q38_l6/test_l6_v6_compose.py.
Usage: python3 q38_l6_v6_mutrun.py <head tree> <set> [survey]

Copied from L6-PSHORT's q38_l6_pshort_mutrun.py. For each mutant: copy the tree, apply one exact textual
replacement (it must match exactly once), run the mutant's target test id with pytest and require that it FAILED
(a failing test, not an error or a collection failure). A control run on the unmutated tree must pass every target
first. 'survey' reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_V6_MUT PASS killed N/N of N | ... FAIL ... | ... SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

L = "src/vllm_torchtpu/runner/l6_carry_lean.py"
P = "src/vllm_torchtpu/runner/l6_prep_short.py"
T = "tests/q38_l6/test_l6_v6_compose.py"
U2T = T + "::test_lean_record_shifted_by_u2_carry_rerecord_carries_at_the_next_head[s04_served]"
ADCT = T + "::test_short_path_passes_the_adc_merge_context[b3_compact]"
TAILT = T + "::test_short_path_tail_emits_the_full_paths_sub_marks"

MUTANTS = {
    "v6": [
        # ---- PREP-SHORT x ADC (the port of ADC's merge context into the short path)
        ("V1 short: ADC merge context not passed", P,
         "        st.seq_lens, pending, 0, n, p.nsched, md, l6=ctx, adc=adc\n",
         "        st.seq_lens, pending, 0, n, p.nsched, md, l6=ctx, adc=None\n", ADCT),
        ("V2 short: ADC merge context without the state indices", P,
         "            state_indices=p.mamba_dev,\n", "            state_indices=None,\n", ADCT),
        # ---- PREP-SHORT x L6-INSTR (the tail sub-marks of the short path's copy of the tail)
        ("V3 short: tail sub-mark pi_cross dropped", P,
         "    hp.sub(\"pi_cross\")\n", "", TAILT),
        # ---- CARRY-LEAN x K-step U2 carry_rerecord
        ("V4 lean: in-place record does not stamp its step", L,
         "    pre = runner._pre_async_results\n    st.step = step\n",
         "    pre = runner._pre_async_results\n", U2T),
        ("V5 lean: in-place record does not refresh pos", L,
         "    np.copyto(st.pos, runner.positions_np[: st.total])\n", "", U2T),
        # ---- ADMIT-INC x the P-METER admission census
        ("V6 admit pack: the upload is not bracketed by ~am_pack", "src/vllm_torchtpu/layers/core/attention_metadata.py",
         "    host_phases.sub(\"am_pack\")\n    staged_dev = _inc_upload(flat, runner.device)\n",
         "    staged_dev = _inc_upload(flat, runner.device)\n",
         T + "::test_every_staged_upload_is_bracketed_by_the_admission_census"),
        # ---- NODEFER (q38-l6-pmeter): the verifier's two gaps
        ("V7 nodefer: the wrapper drops the keyword arguments", "src/vllm_torchtpu/runner/l6_nodefer.py",
         "        out = self.fn(*args, **kwargs)\n", "        out = self.fn(*args)\n",
         T + "::test_nodefer_wrapper_is_byte_identical_with_the_census_off[all-unset-_vps_verify_compiled]"),
        ("V8 nodefer: the census counts whenever l6_ab is active", "src/vllm_torchtpu/runner/l6_nodefer.py",
         "    if not l6_ab.on(\"nodefer\"):\n        return\n    _st[\"n\"] += 1\n",
         "    if not l6_ab.ACTIVE:\n        return\n    _st[\"n\"] += 1\n",
         "tests/q38_l6/test_l6_nodefer.py::test_off_records_nothing"),
        # ---- L6-H2 (q38-l6-h2cheap-v2) x the composed tree
        ("V9 short: fskip's carried mark not set on a short step", P,
         "    _l6_fskip.S.carried = True\n", "",
         T + "::test_fence_skip_on_short_steps_is_the_full_paths[s04_served]"),
        ("V10 toklists off: the 1-column branch no longer taken (stray INVALID entries dropped)",
         "src/vllm_torchtpu/runner/tpu_runner_async_output.py",
         "        elif max_gen_len == 1:\n            valid_sampled_token_ids = selected_token_ids.tolist()\n",
         "        elif max_gen_len == 0:\n            valid_sampled_token_ids = selected_token_ids.tolist()\n",
         T + "::test_token_lists_off_get_output_is_the_stock_paths"),
        # ---- L6-MK x the imported-megakernel identity test (v6c1 j-a8e53944 P3: its driver checks did not know MK's edits)
        ("V11 driver: an L6-MK edit the identity test does not undo", "src/vllm_torchtpu/megakernel_runtime/qwen38_driver.py",
         "        mkc = _l6_mk.mkctx_on(self)  # read past the stock-above-whole route (it never uses the lever)\n",
         "        mkc = _l6_mk.mkctx_on(self)  # read past the stock-above-whole route (it never uses the lever)\n"
         "        mkc = bool(mkc)\n",
         "tests/q38_deco44_c32mk/test_imported_megakernel_identity.py::test_driver_differs_from_mm_only_by_the_marked_insertions"),
        ("V12 driver: one byte changed inside a stripped L6-MK hunk (its pair no longer matches: the check fails, never skips)",
         "src/vllm_torchtpu/megakernel_runtime/qwen38_driver.py",
         "        # eligibility computed here, resolves check_real only where the check branch reads it, enters\n",
         "        # eligibility computed there, resolves check_real only where the check branch reads it, enters\n",
         "tests/q38_deco44_c32mk/test_imported_megakernel_identity.py::test_driver_differs_from_mm_only_by_the_marked_insertions"),
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
        print("Q38_L6_V6_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6v6mut.")
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
        print(f"Q38_L6_V6_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_V6_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_V6_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

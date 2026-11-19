#!/usr/bin/env python3
"""q38 L6 levers clean (CARRY-LEAN) / pshort (PREP-SHORT) mutant runner (CPU pod only).
Usage: python3 q38_l6_pshort_mutrun.py <head tree> <set> [survey]

Copied from L6-PTRIMS's q38_l6_ptrims_mutrun.py (itself after L6-ENGINE's q38_l6_engalloc_mutrun.py). For each
mutant: copy the tree, apply one exact textual replacement (it must match exactly once), run the mutant's target
test id with pytest and require that it FAILED (a failing test, not an error or a collection failure). A control
run on the unmutated tree must pass every target first. 'survey' reports every mutant without requiring all to be
killed (dev runs). Last line: Q38_L6_PSHORT_MUT PASS killed N/N of N | ... FAIL ... | ... SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

L = "src/vllm_torchtpu/runner/l6_carry_lean.py"
P = "src/vllm_torchtpu/runner/l6_prep_short.py"
D = "src/vllm_torchtpu/runner/decode_carry.py"
ST = "src/vllm_torchtpu/runner/l6_steady.py"
R = "src/vllm_torchtpu/runner/tpu_runner.py"
T = "tests/q38_l6/test_l6_pshort.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py"
LEANP = T + "::test_lean_refuses_each_change"
SHORTG = T + "::test_short_refuses_each_change"
ENGAGE = T + "::test_steady_stream_engages_every_step"
LEANS = T + "::test_lean_matches_stock[s11_served-0]"
SHORTS = T + "::test_short_matches_stock[s11_served-0]"
VERDICT = T + "::test_check_verdict_is_refused_even_when_the_epoch_survives"

MUTANTS = {
    "pshort": [
        # ---- CARRY-LEAN (the card's list first)
        ("C1 lean: stale epoch (any cache with this step's verdict)", L,
         "    if ro is None or c is not ro or c.fast_seq != step or st.step != step - 1:\n",
         "    if ro is None or c is None or c.fast_seq != step or st.step != step - 1:\n",
         LEANP + "[epoch_replaced]"),
        ("C2 lean: rejected counts gathered off by one request", L,
         "np.fromiter(map(rec[2].__getitem__, st.req_ids), np.int64, num_reqs)",
         "np.fromiter(map(rec[2].__getitem__, st.req_ids[1:] + st.req_ids[:1]), np.int64, num_reqs)", ENGAGE),
        ("C3 lean: step-tag check dropped", L,
         "    if ro is None or c is not ro or c.fast_seq != step or st.step != step - 1:\n",
         "    if ro is None or c is not ro or c.fast_seq != step:\n", LEANP + "[stale_step]"),
        ("C4 lean: map identity dropped", L,
         "    if pre.req_id_to_index_copy != st.map_d:\n        return None\n",
         "    if False:\n        return None\n", LEANP + "[map_changed]"),
        ("C5 lean: accepts a 'check' verdict", L,
         "    if ro is None or c is not ro or c.fast_seq != step or st.step != step - 1:\n",
         "    if ro is None or c is not ro or st.step != step - 1:\n", VERDICT),
        ("C6 lean: in-place record does not refresh S", L,
         "    np.copyto(st.S, runner.seq_lens_np[:num_reqs])\n", "", LEANS),
        ("C7 lean: vector test dropped", L,
         "    if r.tobytes() != _delta_bytes(num_reqs, st.step_delta):  # every entry == step_delta\n        return None\n",
         "", LEANP + "[rej_value]"),
        ("C8 lean: in-place record keeps the previous device tensors", L,
         "    st.seq_lens = seq_lens\n    st.positions = positions\n    st.input_ids = input_ids\n    st.map_d", "    st.map_d",
         LEANS),
        ("C9 lean: stamp leaves no epoch", L,
         "    st.ro = runner.__dict__.get(\"_l6_steady_ro\")\n", "    st.ro = None\n", ENGAGE),
        ("C10 lean: lever state ignored (always on)", L,
         "    return _l6_ab.on(LEVER) and not runner.__dict__.get(\"_l6_clean_off\", False)\n",
         "    return not runner.__dict__.get(\"_l6_clean_off\", False)\n", T + "::test_levers_off_run_no_lever_code"),
        ("C11 lean: CHECK ignores VLLM_TPU_L6_CHECK_STEPS", L,
         "    return v if _l6_ab.check_allowed() else 0\n", "    return v\n", T + "::test_check_steps_bound_both_checks"),
        ("C12 lean: CHECK never reports a mismatch", L,
         "    if ref[0] is not st or ref[1] != \"\" or st.map_same is not True:\n", "    if False:\n",
         T + "::test_lean_check_catches_a_broken_lean_test[1]"),
        ("C13 lean: a mismatch does not turn the lean test off", L,
         "    runner._l6_clean_off = True\n", "    pass\n", T + "::test_lean_check_catches_a_broken_lean_test[1]"),
        ("C14 carry: begin ignores the lean lever", D,
         "    st, why = (_clean.check if _clean.on(runner) else _check)(\n", "    st, why = (_check)(\n", ENGAGE),
        ("C15 carry: record never in place", D,
         "    if _clean.on(runner) and _clean.record(\n", "    if False and _clean.record(\n", ENGAGE),
        ("C16 carry: stock record not stamped", D,
         "    if _clean.stamping():\n        _clean.stamp(runner, st)", "    if False:\n        _clean.stamp(runner, st)", ENGAGE),
        ("C17 steady: ro_fast verdict not stamped", ST,
         "    c.fast_seq = _l6_ab.step_seq()\n", "", ENGAGE),
        # ---- PREP-SHORT (the card's list first)
        ("P1 short: stale draft ids (the cached metadata object returned)", P,
         "    md = _md(p)\n", "    md = p.md_cache\n", SHORTS),
        ("P2 short: stale mamba tensor (bytes not compared)", P,
         "        if ent is None or ent[0] != (host.dtype, (target,), buf.tobytes()):\n",
         "        if ent is None:\n", SHORTG + "[mamba_slot_changed-alone]"),
        ("P3 short: stale comb (values not compared)", P,
         "        if ent is None or ent[0] != (stg.dtype, tuple(stg.shape), b.tobytes()):\n",
         "        if ent is None:\n", SHORTG + "[comb_values_changed-alone]"),
        ("P4 short: positions not advanced (cached route)", P,
         "    np.add(nc[p.req_idx], p.ar, out=runner.positions_np[:total])\n", "", SHORTS),
        ("P5 short: taken on a 'check' verdict", P,
         "    if c is not st.ro or c.fast_seq != step:\n", "    if c is not st.ro:\n", VERDICT),
        ("P6 short: taken at start_index > 0", P,
         "    if start_index != 0:\n        return _no(\"chunk\")\n", "", SHORTG + "[chunk_start-alone]"),
        ("P7 short: bucket change ignored (the recorded padded total)", P,
         "    p.padded_total = layout_plan.global_padded_num_tokens\n",
         "    p.padded_total = st.padded_total\n    layout_plan = p.layout_plan = type(layout_plan)("
         "descriptor=layout_plan.descriptor, token_slice=slice(0, st.padded_total), global_num_tokens=total, "
         "global_padded_num_tokens=st.padded_total, local_num_tokens=total, local_padded_num_tokens=st.padded_total)\n",
         SHORTG + "[bucket_changed-alone]"),
        ("P8 short: epoch dropped", P,
         "    if c is not st.ro or c.fast_seq != step:\n", "    if c.fast_seq != step:\n",
         SHORTG + "[epoch_replaced-alone]"),
        ("P9 short: spec-metadata cache key ignored", P,
         "    if mgr._meta_cache is None or mgr._meta_cache_key != (\n",
         "    if mgr._meta_cache is None or False and mgr._meta_cache_key != (\n", SHORTG + "[spec_key_changed-alone]"),
        ("P10 short: materialize skipped before an upload", D,
         "        ctx.materialize()  # q38 L6 lever pshort: the host input ids it skipped\n", "        pass\n",
         T + "::test_materialize_before_carry_check_upload"),
        ("P11 short: lever state ignored (always on)", P,
         "    return _l6_ab.on(LEVER) and not runner.__dict__.get(\"_l6_pshort_off\", False)\n",
         "    return not runner.__dict__.get(\"_l6_pshort_off\", False)\n", T + "::test_levers_off_run_no_lever_code"),
        ("P12 short: CHECK ignores VLLM_TPU_L6_CHECK_STEPS", P,
         "    return v if _l6_ab.check_allowed() else 0\n", "    return v\n", T + "::test_check_steps_bound_both_checks"),
        ("P13 short: CHECK never reports a mismatch", P,
         "    if bad:\n        runner._l6_pshort_off = True\n", "    if False:\n        runner._l6_pshort_off = True\n",
         T + "::test_short_check_catches_a_broken_prediction[1]"),
        ("P14 short: sequence lengths not written (first step of an epoch)", P,
         "    tr.fill_step_seq_lens(runner.seq_lens_np, nc, nsched)\n", "", ENGAGE),
        ("P15 runner: the full path's tail edited", R,
         "        # Prepare token substitution indices\n", "        # Prepare token substitution indices (edited)\n",
         T + "::test_full_path_text_is_pinned"),
        ("P16 a new reader of input_ids_cpu", "src/vllm_torchtpu/megakernel_runtime/driver.py",
         "            comb_cpu: Any = getattr(runner, \"_combined_request_distribution_cpu\", None)\n",
         "            comb_cpu: Any = getattr(runner, \"_combined_request_distribution_cpu\", None)\n"
         "            _ = runner.input_ids_cpu\n", T + "::test_skipped_mirrors_have_no_new_reader"),
        ("P17 short: comb staging not written", P,
         "    b[:6] = (nd, nd, n, nw, nw, n)\n", "", SHORTG + "[comb_values_changed-short]"),
        ("P18 short: carry key reused without checking the arguments", P,
         "    if ko is not None and p.kshape == kshape and all(a is b for a, b in zip(ko, kobj)):\n",
         "    if ko is not None:\n", T + "::test_short_carry_key_follows_its_arguments"),
        ("P19 short: cached route skips the slot compare", P,
         "            if list(map(slot_map.__getitem__, ib.req_ids[:n])) != p.slots:\n",
         "            if False:\n", SHORTG + "[mamba_slot_changed-alone]"),
        # ---- registration
        ("R1 clean missing from l6_ab.LEVERS", "src/vllm_torchtpu/core/l6_ab.py",
         "    \"clean\": \"VLLM_TPU_L6_CARRY_LEAN\",\n", "", T + "::test_levers_are_registered_and_compile_ignored"),
        ("R2 pshort missing from l6_ab.LEVERS", "src/vllm_torchtpu/core/l6_ab.py",
         "    \"pshort\": \"VLLM_TPU_L6_PREP_SHORT\",\n", "", T + "::test_levers_are_registered_and_compile_ignored"),
        ("R3 lever flags dropped from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_CARRY_LEAN\",\n    \"VLLM_TPU_L6_CARRY_LEAN_CHECK\",\n    \"VLLM_TPU_L6_PREP_SHORT\",\n",
         "    \"VLLM_TPU_L6_CARRY_LEAN_CHECK\",\n", REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("R4 lean flag missing from envs.py", "src/vllm_torchtpu/envs.py",
         "    \"VLLM_TPU_L6_CARRY_LEAN\": env_str(\"VLLM_TPU_L6_CARRY_LEAN\", \"\"),\n", "",
         REG + "::test_flag_is_registered_and_parses[VLLM_TPU_L6_CARRY_LEAN-1-]"),
        ("R5 compile-key registration rows dropped", REG,
         "    (\"VLLM_TPU_L6_PREP_SHORT\", \"1\", \"\"),\n    (\"VLLM_TPU_L6_PREP_SHORT_CHECK\", \"200\", 0),\n", "",
         T + "::test_levers_are_registered_and_compile_ignored"),
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
        print("Q38_L6_PSHORT_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6psmut.")
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
        print(f"Q38_L6_PSHORT_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_PSHORT_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_PSHORT_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

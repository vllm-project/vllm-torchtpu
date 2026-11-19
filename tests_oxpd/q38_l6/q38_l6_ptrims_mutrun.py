#!/usr/bin/env python3
"""q38 L6 lever ptrims mutant runner (CPU pod only). Usage: python3 q38_l6_ptrims_mutrun.py <head tree> <set> [survey]

Modelled on L6-ENGINE's q38_l6_engalloc_mutrun.py. For each mutant: copy the tree, apply one exact textual
replacement (it must match exactly once), run the mutant's target test id with pytest and require that it FAILED
(a failing test, not an error or a collection failure). A control run on the unmutated tree must pass every
target first. 'survey' reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_PTRIMS_MUT PASS killed N/N of N | Q38_L6_PTRIMS_MUT FAIL ... | Q38_L6_PTRIMS_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

P = "src/vllm_torchtpu/runner/l6_ptrims.py"
R = "src/vllm_torchtpu/runner/tpu_runner.py"
T = "tests/q38_l6/test_l6_ptrims.py"
EQ = T + "::test_trims_match_stock_over_random_streams[s11_served-alone]"
EQC = T + "::test_trims_match_stock_over_random_streams[s11_served-steady_carry]"
REG = "tests/compilation/test_tpu_compile_key_registration.py"

MUTANTS = {
    "ptrims": [
        # Stale-cache mutants: each drops one field of the token-index cache key (its only invalidation).
        ("P1 index cache key ignores the scheduled counts", P,
         "    key = (start_index, num_reqs, stride, nsched.tobytes())\n",
         "    key = (start_index, num_reqs, stride)\n", EQ),
        ("P2 index cache key ignores the start index", P,
         "    key = (start_index, num_reqs, stride, nsched.tobytes())\n",
         "    key = (num_reqs, stride, nsched.tobytes())\n", T + "::test_two_chunk_steps_match_stock"),
        ("P3 index cache key ignores the token-table stride", P,
         "    key = (start_index, num_reqs, stride, nsched.tobytes())\n",
         "    key = (start_index, num_reqs, nsched.tobytes())\n", T + "::test_widened_token_table_matches_stock"),
        # Value mutants: each trim computes something else.
        ("P4 batched arange off by one", P,
         "        arange = np.concatenate([runner.arange_np[:n] for n in nsched])\n        row_off",
         "        arange = np.concatenate([runner.arange_np[1 : n + 1] for n in nsched])\n        row_off", EQ),
        ("P5 row offsets use the wrong stride", P,
         "        row_off = req_indices * stride\n",
         "        row_off = req_indices * (stride - 1)\n", EQ),
        ("P6 slot gather in reverse order", P,
         "        return list(map(slot_by_req_id.__getitem__, ids))\n",
         "        return list(map(slot_by_req_id.__getitem__, ids))[::-1]\n", EQ),
        ("P7 new slots popped from the wrong end of the free list", P,
         "            slot = free.pop()\n",
         "            slot = free.pop(0)\n", EQ),
        ("P8 combined distribution fields swapped", P,
         "    return (num_decode, num_decode, num_reqs, num_windowed, num_windowed, num_reqs)\n",
         "    return (num_windowed, num_windowed, num_reqs, num_decode, num_decode, num_reqs)\n", EQC),
        # The l6_ab gate.
        ("P9 lever state ignored (always on)", P,
         "    return _l6_ab.on(LEVER) and not runner.__dict__.get(\"_l6_ptrims_off\", False)\n",
         "    return not runner.__dict__.get(\"_l6_ptrims_off\", False)\n",
         T + "::test_lever_off_takes_the_stock_path"),
        ("P10 mamba call site ignores the lever", R,
         "        if _l6_ab.ACTIVE and _l6_ptrims.on(self):\n",
         "        if True:\n", T + "::test_lever_off_takes_the_stock_path"),
        # CHECK.
        ("P11 CHECK ignores VLLM_TPU_L6_CHECK_STEPS", P,
         "    return v if _l6_ab.check_allowed() else 0\n",
         "    return v\n", T + "::test_check_steps_bound_the_check"),
        ("P12 token CHECK never reports a mismatch", P,
         "        if not np.array_equal(ref_pos, positions_np):\n            why = \"positions\"\n"
         "        elif not torch.equal(ref_ids, runner.input_ids_cpu[:total]):\n",
         "        if False:\n            why = \"positions\"\n        elif False:\n",
         T + "::test_check_catches_a_broken_trim[tokens]"),
        ("P13 slot CHECK never reports a mismatch", P,
         "        if not torch.equal(ref, indices):\n",
         "        if False:\n", T + "::test_check_catches_a_broken_trim[mamba]"),
        ("P14 combined CHECK never reports a mismatch", P,
         "        if not torch.equal(ref, combined):\n",
         "        if False:\n", T + "::test_check_catches_a_broken_trim[comb]"),
        ("P15 CHECK turn never comes", P,
         "                st.check = True\n",
         "                st.check = False\n", T + "::test_check_mode_compares_and_finds_no_mismatch"),
        ("P16 a mismatch does not turn the trims off", P,
         "    runner._l6_ptrims_off = True\n",
         "    pass\n", T + "::test_check_catches_a_broken_trim[mamba]"),
        # Registration rows.
        ("P17 lever missing from l6_ab.LEVERS", "src/vllm_torchtpu/core/l6_ab.py",
         "    \"ptrims\": \"VLLM_TPU_L6_PREP_TRIMS\",\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
        ("P18 lever flag dropped from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_PREP_TRIMS\",\n    \"VLLM_TPU_L6_PREP_TRIMS_CHECK\",\n",
         "    \"VLLM_TPU_L6_PREP_TRIMS_CHECK\",\n",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("P19 CHECK flag dropped from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_PREP_TRIMS\",\n    \"VLLM_TPU_L6_PREP_TRIMS_CHECK\",\n",
         "    \"VLLM_TPU_L6_PREP_TRIMS\",\n",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("P20 lever flag missing from envs.py", "src/vllm_torchtpu/envs.py",
         "    \"VLLM_TPU_L6_PREP_TRIMS\": env_str(\"VLLM_TPU_L6_PREP_TRIMS\", \"\"),\n", "",
         REG + "::test_flag_is_registered_and_parses[VLLM_TPU_L6_PREP_TRIMS-1-]"),
        ("P21 compile-key registration rows dropped", REG,
         "    (\"VLLM_TPU_L6_PREP_TRIMS\", \"1\", \"\"),\n    (\"VLLM_TPU_L6_PREP_TRIMS_CHECK\", \"200\", 0),\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
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
        print("Q38_L6_PTRIMS_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6ptmut.")
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
        print(f"Q38_L6_PTRIMS_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_PTRIMS_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_PTRIMS_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

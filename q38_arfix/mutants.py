# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX mutation check: each mutant edits ONE source file in a fresh copy of the tree, then runs its target test
file; the mutant counts as KILLED only if that test run fails, and only if the same test passed on the clean copy
first (CLEAN leg). Usage: python3 q38_arfix/mutants.py <tree root> <pytest prefix...>  (run from anywhere).
Prints 'Q38_ARFIX_MUT <name> KILLED|SURVIVED' per mutant and 'Q38_ARFIX_MUT_SUMMARY killed=<k>/<n> clean=<ok>'.
Exit 0 only if every mutant is killed and every clean leg passed."""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1])
PYTEST = sys.argv[2:]
FN = "src/vllm_torchtpu/kernels/collectives/tp8_ar_decode_fn.py"
AR = "src/vllm_torchtpu/layers/adapter/tp_ar_decode.py"
SP = "src/vllm_torchtpu/models/vllm/qwen4_exp/sequence_parallel.py"
QSA = "src/vllm_torchtpu/models/vllm/qwen4_exp/qsa.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
MATH = "tests/arfix/test_tp_ar_decode_math.py"
WIRE = "tests/arfix/test_tp_ar_decode_wiring.py"

# (name, file, old, new, target test)
MUTANTS = [
    ("m1_use_kernel_false", AR, "    return armed is not None and x.dim() == 2", "    return False and armed is not None and x.dim() == 2", WIRE),
    ("m2_sp_off_routes_kernel", SP, "    if tp_ar_decode.use_kernel(x):\n        return tp_ar_decode.kernel_all_reduce(x)",
     "    if True:\n        return tp_ar_decode.kernel_all_reduce(x)", WIRE),
    ("m3_qsa_not_routed", QSA, "            if tp_ar_decode.use_kernel(output):", "            if False and tp_ar_decode.use_kernel(output):", WIRE),
    ("m4_factor_when_unset", FN, "    if not m:\n        return {}\n", "    if not m:\n        return {ENV: ''}\n", WIRE),
    ("m5_compile_key_dropped", TC, "    factors.update(tp8_ar_decode_fn.compile_factors())", "    tp8_ar_decode_fn.compile_factors()", WIRE),
    ("m6_dtype_ignored", AR, "and x.dtype == torch.bfloat16 and", "and", WIRE),
    # BLSTACK-H8 (port onto Benliang Wang's d523c2ce0): m7 / m8 / m10 re-anchored after D68 (fa3486a8a), which moved
    # the H8-takes-it check into padded_rows() and added a second (padded) kernel call; the old anchors matched 2 / 0 / 0
    # times (j-4915a150 BAD ANCHOR).
    ("m7_wire_forced_bf16", FN, '        return h8.tp8_hier_all_reduce_local(x, axis=axis, pairs=pairs, ag="fwd", wire=wire, interpret=interpret)',
     '        return h8.tp8_hier_all_reduce_local(x, axis=axis, pairs=pairs, ag="fwd", wire="bf16", interpret=interpret)', MATH),
    ("m8_cap_ignored", FN, "    if h8.supports(rows, hidden, dtype, max_rows_):\n        return rows\n",
     "    if h8.supports(rows, hidden, dtype):\n        return rows\n", MATH),
    ("m9_pairs_id_order", FN, "    return tuple((chips[c][0], chips[c][1]) for c in sorted(chips))",
     "    return tuple((2 * c, 2 * c + 1) for c in range(len(chips)))", MATH),
    ("m10_psum_everywhere", FN, "    if h8.supports(rows, hidden, dtype, max_rows_):\n        return rows\n",
     "    if False and h8.supports(rows, hidden, dtype, max_rows_):\n        return rows\n", MATH),
    ("m11_arm_skips_rank_check", AR, "    _check_same_on_every_rank(signature, tp)\n", "", WIRE),
    ("m12_arm_accepts_tp4", AR, "    if int(tp.world_size) != N_MEMBERS:", "    if int(tp.world_size) not in (4, N_MEMBERS):", WIRE),
    # BLSTACK-H8: his served decode runs VLLM_TPU_QSA_DCP_MERGE=proj; a site that skipped H8 on the proj-merge path
    # must be caught by the proj-merge wiring case.
    ("m13_qsa_proj_merge_not_routed", QSA, "            if tp_ar_decode.use_kernel(output):",
     "            if not getattr(self, \"_proj_merge\", False) and tp_ar_decode.use_kernel(output):", WIRE),
]


def run(tree, test):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(tree, "src")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and not p.endswith("/src") and p != "src"]))
    r = subprocess.run(PYTEST + [test, "-x"], cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
    return r.returncode, " | ".join(tail)[:400]


def main():
    clean_ok = {}
    for test in sorted({m[4] for m in MUTANTS}):
        rc, tail = run(ROOT, test)
        clean_ok[test] = rc == 0
        print(f"Q38_ARFIX_MUT CLEAN {test} rc={rc} {tail}", flush=True)
    killed = 0
    for name, path, old, new, test in MUTANTS:
        if not clean_ok[test]:
            print(f"Q38_ARFIX_MUT {name} NOT COUNTED (clean {test} failed)", flush=True)
            continue
        tmp = tempfile.mkdtemp(prefix=f"q38mut_{name}_")
        try:
            for d in ("src", "tests", "q38_arfix", "q38_plugins"):
                if os.path.isdir(os.path.join(ROOT, d)):
                    shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d), symlinks=True)
            for f in ("pyproject.toml",):
                if os.path.exists(os.path.join(ROOT, f)):
                    shutil.copy(os.path.join(ROOT, f), tmp)
            p = os.path.join(tmp, path)
            s = open(p).read()
            n = s.count(old)
            if n != 1:
                print(f"Q38_ARFIX_MUT {name} BAD ANCHOR ({n} matches)", flush=True)
                continue
            open(p, "w").write(s.replace(old, new))
            rc, tail = run(tmp, test)
            ok = rc != 0
            killed += ok
            print(f"Q38_ARFIX_MUT {name} {'KILLED' if ok else 'SURVIVED'} rc={rc} {tail}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    good = killed == len(MUTANTS) and all(clean_ok.values())
    print(f"Q38_ARFIX_MUT_SUMMARY killed={killed}/{len(MUTANTS)} clean={'OK' if all(clean_ok.values()) else 'FAIL'} -> {'PASS' if good else 'FAIL'}", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

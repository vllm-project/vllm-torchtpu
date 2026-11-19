# SPDX-License-Identifier: Apache-2.0
"""q38 L3 D60 source mutants: each plants one defect in a copy of src/ and must make its test subset FAIL.

Usage (from the tree root, CPU, 8 host devices):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python3 tests/q38_l3_d60/q38_d60_mutrun.py [--smoke]
Per mutant: copy src/ and tests/q38_l3_d60/ into a fresh temp root, apply the replacement (it must match exactly once
in the pristine file, and the mutated file must differ), run the named pytest subset there with PYTHONPATH=<temp>/src,
and require rc != 0 with at least one FAILED / ERROR line and the mutant tree's import path in the log (the tests print
the kernel module path in a Q38_D60_SRC probe before each run). Temp roots are left in place. A clean copy (no mutant) of
the same subsets must PASS first. Prints Q38_L3D60_MUT <name> CAUGHT|SURVIVED|BAD-APPLY ... and
Q38_L3D60_MUTANTS <caught>/<n> PASS|FAIL; exit 0 only when every mutant is caught and the clean run passes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
K = "src/vllm_torchtpu/kernels/experimental/fused_moe_decode/fused_moe_decode_kernel.py"
A = "src/vllm_torchtpu/layers/adapter/fused_moe_decode.py"
V = "src/vllm_torchtpu/envs.py"
KT = "tests/q38_l3_d60/test_compact_pipe_kernel.py"
WT = "tests/q38_l3_d60/test_compact_pipe_wiring.py"
UNI = "m224_uniform"
# name, file, old, new, test file, -k expression, in smoke
MUTANTS = [
    ("lookahead_wrong_slot", K, "return first_matmul_rows(xq_rows, xs_blocks, w1_ring.at[step % nb])",
     "return first_matmul_rows(xq_rows, xs_blocks, w1_ring.at[(step + 1) % nb])", KT, UNI, True),
    ("ctab_wrong_lane", K, "jnp.where(lane == e, ctab_ref[0:compact_r, :], 0.0)",
     "jnp.where(lane == e + 1, ctab_ref[0:compact_r, :], 0.0)", KT, UNI, False),
    ("row_add_slot0_token", K, "                t = tabs_ref[r, e]\n",
     "                t = tabs_ref[0, e]\n", KT, UNI, False),
    ("count_row_minus_one", K, "            n_e = tabs_ref[compact_r, e]\n",
     "            n_e = tabs_ref[compact_r, e] - 1\n", KT, UNI, True),
    ("no_dense_fallback", K, "@pl.when(n_e <= compact_r)\n            def _():\n                w2_scale = pipe_tail",
     "@pl.when(n_e <= 4 * compact_r)\n            def _():\n                w2_scale = pipe_tail", KT,
     "m256_hot_boundary", False),
    ("acc1_parity_swap", K, "w2_scale = pipe_tail(e, slot, acc1buf[step % 2])",
     "w2_scale = pipe_tail(e, slot, acc1buf[nxt % 2])", KT, UNI, False),
    ("token_table_plus_one", K, "jnp.where(at_r, tok_plane, 0)", "jnp.where(at_r, tok_plane + 1, 0)", KT, UNI, False),
    ("late_prime", K, "early_prime = bool(compact_pipe)", "early_prime = False", KT,
     "test_pipe_primes_before_the_tables", True),
    ("name_without_p", K, "{'p' if compact_pipe else ''}", "{'' if compact_pipe else ''}", KT,
     "test_name_suffix", False),
    ("wiring_threshold_gt", A, "if compact_pipe_min and x.shape[0] >= compact_pipe_min:",
     "if compact_pipe_min and x.shape[0] > compact_pipe_min:", WT, "test_pipe_applies_from_threshold", True),
    ("wiring_key_not_extended", A, "key = key + ((\"compact_pipe\", pipe_min, pipe_rows),)", "key = key", WT,
     "test_build_op_flag_on", False),
    ("wiring_partial_missing", A,
     "**({\"compact_pipe_min\": pipe_min, \"compact_pipe_rows\": pipe_rows} if pipe_min else {}),", "**({}),", WT,
     "test_build_op_flag_on", False),
    ("wiring_kernel_kwarg_dropped", A, "**({\"compact_pipe\": True} if pipe else {}),", "**({}),", WT,
     "test_pipe_applies_from_threshold or test_pipe_rows", False),
    ("envs_default_on", V, "\"FUSED_MOE_DECODE_COMPACT_PIPE\", 0\n", "\"FUSED_MOE_DECODE_COMPACT_PIPE\", 192\n", WT,
     "test_envs_register", False),
]
PT = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-o", "addopts="]


def make_root(tag):
    d = tempfile.mkdtemp(prefix=f"q38_d60_mut_{tag}_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(ROOT, "tests", "q38_l3_d60"), os.path.join(d, "tests", "q38_l3_d60"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    return d


def run(root, test, kexpr, tag, timeout):
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), PYTHONDONTWRITEBYTECODE="1")
    probe = subprocess.run([sys.executable, "-B", "-c", "import vllm_torchtpu.kernels.experimental.fused_moe_decode."
                            "fused_moe_decode_kernel as k, vllm_torchtpu; print('Q38_D60_SRC', k.__file__)"],
                           cwd=root, env=env, capture_output=True, text=True, timeout=600)
    src_ok = probe.stdout.strip().startswith(f"Q38_D60_SRC {os.path.join(root, 'src')}{os.sep}")
    try:
        r = subprocess.run(PT + [os.path.join(root, test), "-k", kexpr], cwd=root, env=env, capture_output=True,
                           text=True, timeout=timeout)
        rc, log = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as exc:
        rc, log = 124, f"TIMEOUT {timeout}s {exc}"
    with open(f"/tmp/q38_d60_mut_{tag}.log", "w") as f:
        f.write(probe.stdout + probe.stderr + log)
    nfail = sum(1 for ln in log.splitlines() if ln.startswith(("FAILED", "ERROR")))
    npass = sum(1 for ln in log.splitlines() if ln.startswith("PASSED"))
    return rc, nfail, npass, src_ok, log


def main(argv):
    smoke = "--smoke" in argv
    todo = [m for m in MUTANTS if m[6] or not smoke]
    timeout = 3600
    # clean control: every subset the mutants use must pass on an unmutated copy
    clean = make_root("clean")
    subsets = sorted({(m[4], m[5]) for m in todo})
    clean_ok = True
    for i, (test, kexpr) in enumerate(subsets):
        rc, nfail, npass, src_ok, log = run(clean, test, kexpr, f"clean{i}", timeout)
        ok = rc == 0 and nfail == 0 and npass >= 1 and src_ok
        clean_ok &= ok
        print(f"Q38_L3D60_MUT clean {test} -k '{kexpr}': rc={rc} passed {npass} failed {nfail} src_ok {src_ok} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(log[-3000:])
    caught = 0
    for name, path, old, new, test, kexpr, _ in todo:
        root = make_root(name)
        p = os.path.join(root, path)
        text = open(p).read()
        n = text.count(old)
        if n != 1:
            print(f"Q38_L3D60_MUT {name} BAD-APPLY: '{old[:60]}' matches {n} times in {path} (want 1)", flush=True)
            continue
        mutated = text.replace(old, new)
        open(p, "w").write(mutated)
        if mutated == text:
            print(f"Q38_L3D60_MUT {name} BAD-APPLY: no change", flush=True)
            continue
        rc, nfail, npass, src_ok, log = run(root, test, kexpr, name, timeout)
        hit = rc != 0 and nfail >= 1 and src_ok
        caught += hit
        print(f"Q38_L3D60_MUT {name} {'CAUGHT' if hit else 'SURVIVED'}: rc={rc} failed {nfail} passed {npass} "
              f"src_ok {src_ok} ({test} -k '{kexpr}')", flush=True)
        if not hit:
            print(log[-2000:])
    ok = caught == len(todo) and clean_ok
    print(f"Q38_L3D60_MUTANTS {caught}/{len(todo)} {'PASS' if ok else 'FAIL'}"
          f"{'' if clean_ok else ' (clean control FAILED)'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""q38 L4 v6 H3-bar M-elt sweep (CDK CPU pod only). V6COMP-BUILD. Usage: q38_h3bar_eltsweep.py <tree> <plugins> <P> <node>...

L1's ruling (2026-11-14): M-elt exactly as written in the locked prereg (section 3): 4 ulps on a live element whose
cancellation ratio is <= 1.5 (stated filter: |want| >= 1e-3), on ANY of the 23 H3 bar tests that has one; no substitute.
Each node runs with Q38_H3BAR_MUT=elt (h3bar_harness: the strict search; leg B asserted in the gated form) in ATTNWIRE's
environment, P at a time, on a copy of the tree:
  KILLED          pytest exit 1, "LEG B FAIL (form tight)" and the H3BAR_MUTANT "elt:" line (its ratio, |want|, pass, rank);
  NOT_APPLICABLE  pytest exit 1 with "M-elt NOT APPLICABLE" (no qualifying element in that test; its minimum ratio printed);
  BAD             anything else.
Each node's H3BAR_MINRATIO lines are printed. Last line: Q38_H3BAR_ELT KILLED|NOT_KILLED killed=K not_applicable=N bad=B
of T (KILLED when at least one test killed it and none is BAD).
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    tree, plugins, par, nodes = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    work = tempfile.mkdtemp(prefix="h3barelt_")
    dst = os.path.join(work, "tree")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(tree, d), os.path.join(dst, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = dict(os.environ, JAX_PLATFORMS="cpu", VLLM_TARGET_DEVICE="cpu", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{plugins}:{dst}/src:{dst}", Q38_H3BAR_MUT="elt")
    env.pop("XLA_FLAGS", None)
    env.pop("Q38_H3BAR_BFORM", None)

    def one(node):
        cmd = [sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-W", "ignore", "-s",
               "-p", "q38_cpu_device", node]
        p = subprocess.run(cmd, cwd=dst, env=env, capture_output=True, text=True, timeout=14000)
        out = p.stdout + p.stderr
        if p.returncode == 1 and re.search(r"LEG B FAIL \(form tight\)", out) and re.search(r"H3BAR_MUTANT .*: elt: ", out):
            v = "KILLED"
        elif p.returncode == 1 and "M-elt NOT APPLICABLE" in out:
            v = "NOT_APPLICABLE"
        else:
            v = "BAD"
        keep = [ln for ln in out.splitlines() if ln.startswith(("H3BAR_MINRATIO", "H3BAR_MUTANT"))
                or "LEG B FAIL" in ln or "NOT APPLICABLE" in ln][:8]
        return node, v, p.returncode, keep

    counts = {"KILLED": 0, "NOT_APPLICABLE": 0, "BAD": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=par) as ex:
        for node, v, rc, keep in ex.map(one, nodes):
            counts[v] += 1
            print(f"ELT {v} rc={rc} {node}", flush=True)
            for ln in keep:
                print(f"ELT   {ln[:400]}", flush=True)
    ok = counts["KILLED"] >= 1 and counts["BAD"] == 0
    print(f"Q38_H3BAR_ELT {'KILLED' if ok else 'NOT_KILLED'} killed={counts['KILLED']} "
          f"not_applicable={counts['NOT_APPLICABLE']} bad={counts['BAD']} of {len(nodes)}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

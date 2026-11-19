# SPDX-License-Identifier: Apache-2.0
"""q38 L6-V5 compose: every L6 piece's source-mutant anchor must occur exactly once on the composed tree.

String check only (the suites themselves run in run_cpu.sh P4*). Usage: python3 -I q38_l6_v5_anchors.py <tree>
Prints one ANCHORS line per suite and a last line 'Q38_L6_V5_ANCHORS PASS|FAIL total=<n> bad=<k>'.
Each mutant runner keeps its own tuple layout; the adapters below map each to (name, path, old)."""
import importlib.util
import os
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
HP = "src/vllm_torchtpu/runner/host_phases.py"
READER = "tests/q38_hostsub/q38_hostsub_read.py"


def load(rel):
    saved = sys.argv
    sys.argv = [rel, ROOT, ROOT]  # every runner reads ROOT (and BASE) from argv at import
    try:
        spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3] + "_v5anchors", os.path.join(ROOT, rel))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.argv = saved


def flat(muts):
    if isinstance(muts, dict):
        return [(k, mu) for k, lst in muts.items() for mu in lst]
    return [("", mu) for mu in muts]


# suite -> (runner file, adapter(mu) -> (name, path, old))
SUITES = [
    ("trims+eos", "tests/q38_l6/q38_l6_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("steady", "tests/q38_l6/q38_l6steady_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("carry", "tests/q38_l6/q38_l6carry_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("engalloc", "tests/q38_l6/q38_l6_engalloc_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("instr", "tests/q38_l6instr/q38_l6instr_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("hostsub", "tests/q38_hostsub/q38_hostsub_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("hostsub_read", "tests/q38_hostsub/q38_hostsub_read_mutrun.py", lambda mu: (mu[0], READER, mu[1])),
    ("hprep", "tests/q38_l6hprep/q38_l6hprep_mutrun.py", lambda mu: (mu[0], HP, mu[1])),
    ("ptrims", "tests/q38_l6/q38_l6_ptrims_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("gate", "tests/q38_l6/q38_l6gate_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("seed", "tests/q38_l6/q38_l6_seed_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("sync", "tests/q38_l6/q38_l6_sync_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
    ("adc", "tests/q38_l6/q38_l6_adc_mutrun.py", lambda mu: (mu[0], mu[1], mu[2])),
]
EXTRA = [s.split("=", 1) for s in os.environ.get("Q38_L6_V5_EXTRA_SUITES", "").split() if "=" in s]


def main():
    total = bad = 0
    suites = list(SUITES) + [(n, f, lambda mu: (mu[0], mu[1], mu[2])) for n, f in EXTRA]
    for suite, rel, adapt in suites:
        if not os.path.exists(os.path.join(ROOT, rel)):
            print(f"ANCHORS {suite} MISSING {rel}")
            bad += 1
            continue
        rows = [(k, *adapt(mu)) for k, mu in flat(load(rel).MUTANTS)]
        nbad = 0
        for k, name, path, old in rows:
            p = os.path.join(ROOT, path)
            n = open(p).read().count(old) if os.path.exists(p) else -1
            if n != 1:
                nbad += 1
                print(f"ANCHORS BAD {suite}{':' + k if k else ''} {name} {path} count={n}")
        print(f"ANCHORS {suite} {rel} mutants={len(rows)} bad={nbad}")
        total += len(rows)
        bad += nbad
    ok = bad == 0 and total > 0
    print(f"Q38_L6_V5_ANCHORS {'PASS' if ok else 'FAIL'} total={total} bad={bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

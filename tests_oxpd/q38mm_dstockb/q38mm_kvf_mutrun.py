#!/usr/bin/env python3
"""Mutation check for test_kvf_production_geometry.py (DSTOCK-B, KVF).

Each mutant breaks the fused writer (or its call site) in one place; the production-geometry test must
FAIL on every mutant and PASS on the restored tree. Files are restored with ``git checkout`` after each
mutant. Usage (from the worktree root):
  PYTHONPATH=src:<stubs> python -B tests/q38mm_dstockb/q38mm_kvf_mutrun.py
"""
import os
import subprocess
import sys

FUSED = "src/vllm_torchtpu/kernels/qwen4_exp/fused_kv_write.py"
POOLED = "src/vllm_torchtpu/kernels/qwen4_exp/pooled_qsa.py"
TEST = "tests/q38mm_dstockb/test_kvf_production_geometry.py"
SUBSET = ["-k", "test_c48_shape_128_rows or draft_like or idle"]

MUTANTS = [
    ("M1 compressed merge takes the wrong half", FUSED,
     "jnp.where(half == 0, cmask_ref[2 * i], cmask_ref[2 * i + 1]) != 0",
     "jnp.where(half == 0, cmask_ref[2 * i + 1], cmask_ref[2 * i]) != 0"),
    ("M2 partial compressed pairs never read (merge on stale VMEM)", FUSED,
     "comp_merge = lambda i: cmode_ref[i] == 2", "comp_merge = lambda i: cmode_ref[i] == 3"),
    ("M3 paired main row overwrites the partner slot", FUSED,
     "old[...] = (old[...] & jnp.uint32(keep)) | new[...]", "old[...] = new[...]"),
    ("M4 verify run written one row late", FUSED,
     "stored = target_ref.at[run_page_ref[r], pl.ds(run_off_ref[r], run_length)]",
     "stored = target_ref.at[run_page_ref[r], pl.ds(run_off_ref[r] + 1, run_length)]"),
    ("M5 compressed pair offset off by one", FUSED,
     "comp_offset = comp_safe % page_rows", "comp_offset = (comp_safe + 1) % page_rows"),
    ("M6 fused call site ignores the main slot move", POOLED,
     "main_rows=main_slot_rows(rows_packed, main_slot)\n            if main_slot > 0",
     "main_rows=main_slot_rows(rows_packed, main_slot)\n            if main_slot > 9"),
    ("M7 fused call site drops compressed DCP ownership", POOLED,
     "            compressed_slot,\n            **({\"dcp\": compressed_dcp} if compressed_dcp is not None else {}),\n        )\n        pool = fused_layer_write(",
     "            compressed_slot,\n        )\n        pool = fused_layer_write("),
    ("M8 compressed writes skipped entirely", FUSED,
     "each(comp_tokens, comp_live, lambda i: comp_copy(i, False).start())",
     "each(comp_tokens, lambda i: cmode_ref[i] == 7, lambda i: comp_copy(i, False).start())"),
]


def pytest(extra):
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest",
           TEST, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    rc, tail = pytest(SUBSET)
    print(f"control (unmutated): rc={rc} {tail}")
    if rc != 0:
        print("control failed; aborting")
        return 2
    killed = 0
    for name, path, old, new in MUTANTS:
        src = open(path).read()
        if src.count(old) != 1:
            print(f"{name}: pattern count {src.count(old)} != 1 (NOT APPLIED)")
            continue
        open(path, "w").write(src.replace(old, new))
        try:
            # M8 also needs the 'nothing written' / count checks: run the full subset.
            rc, tail = pytest(SUBSET)
        finally:
            subprocess.run(["git", "checkout", "--", path], check=True)
        verdict = "KILLED" if rc != 0 else "SURVIVED"
        killed += rc != 0
        print(f"{name}: {verdict} (rc={rc}; {tail})")
    rc, tail = pytest(SUBSET)
    print(f"control (restored): rc={rc} {tail}")
    print(f"KVF mutants killed: {killed}/{len(MUTANTS)}")
    return 0 if killed == len(MUTANTS) and rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

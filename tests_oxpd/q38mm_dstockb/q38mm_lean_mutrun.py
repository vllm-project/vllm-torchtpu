#!/usr/bin/env python3
"""Mutation check for the lean KV writer (q38mm DSTOCK-B): each mutant breaks one piece of _lean_layer_write, its
dispatch or its flag; test_kv_writer_equivalence.py's lean subset must FAIL on every mutant and PASS on the restored
tree. Files are restored from an in-memory copy (no git: CDK pods lack the objects) and the restore is asserted.
Usage (from the worktree root): PYTHONPATH=src[:<stubs>] python -B tests/q38mm_dstockb/q38mm_lean_mutrun.py
"""
import os
import subprocess
import sys

FUSED = "src/vllm_torchtpu/kernels/qwen4_exp/fused_kv_write.py"
TEST = "tests/q38mm_dstockb/test_kv_writer_equivalence.py"
# c48_128rows_r0: paired main rows, verify runs, partial pairs; one_128row: full pairs; direct: rings, duplicates, runs.
SUBSET = ["-k", "(lean and (c48_128rows_r0 or one_128row)) or direct or dispatch"]

MUTANTS = [
    ("L1 compaction advances for every slot", [(
        "                return c + jnp.where(pred(i), jnp.int32(1), jnp.int32(0))",
        "                return c + jnp.int32(1)")]),
    ("L2 last live main row dropped", [(
        "        n_main = compact(tokens, lambda i: valid_ref[i] != 0, main_list)",
        "        n_main = compact(tokens, lambda i: valid_ref[i] != 0, main_list) - 1")]),
    ("L3 merge list keyed on full pairs", [(
        "                    m + jnp.where(mode == 2, jnp.int32(1), jnp.int32(0)),",
        "                    m + jnp.where(mode == 1, jnp.int32(1), jnp.int32(0)),")]),
    ("L4 live list skips full pairs", [(
        "                    lv + jnp.where(mode != 0, jnp.int32(1), jnp.int32(0)),",
        "                    lv + jnp.where(mode == 2, jnp.int32(1), jnp.int32(0)),")]),
    ("L5 vector merge ignores the take mask", [(
        "        bf[...] = jnp.where(ctake_ref[...] != 0, cvalues_ref[...], bf[...])",
        "        bf[...] = cvalues_ref[...]")]),
    ("L6 ring DMAs never run", [
        ("        @pl.when(ring_any_ref[0] != 0)\n        def write_rings():",
         "        @pl.when(ring_any_ref[0] == 99)\n        def write_rings():"),
        ("        @pl.when(ring_any_ref[0] != 0)\n        def wait_rings():",
         "        @pl.when(ring_any_ref[0] == 99)\n        def wait_rings():")]),
    ("L7 ordered (duplicate) writes in reverse", [(
        "            over(n_main, main_list, store)",
        "            jax.lax.fori_loop(0, n_main, lambda k, c: (store(main_list[n_main - 1 - k]), c)[1], 0)")]),
    ("L8 verify runs never written", [(
        "        n_runs = (\n            compact(runs, lambda r: run_live_ref[r] != 0, run_list) if use_runs else 0\n        )",
        "        n_runs = 0")]),
    ("L9 dispatch ignores the flag", [("    if lean_writer_enabled():\n        return _lean_layer_write(",
                                       "    if True:\n        return _lean_layer_write(")]),
    ("L10 flag parser treats 0 as on", [(
        'def lean_writer_enabled() -> bool:\n    value = os.environ.get(LEAN_ENV, "").strip()\n    if value in ("", "0"):',
        'def lean_writer_enabled() -> bool:\n    value = os.environ.get(LEAN_ENV, "").strip()\n    if value in ("",):')]),
    ("L11 ring staging skipped when rings are live", [(
        "        @pl.when(ring_any_ref[0] != 0)\n        def stage_rings():",
        "        @pl.when(ring_any_ref[0] == 0)\n        def stage_rings():")]),
    ("L12 paired main rows never read (merge on stale VMEM)", [
        ("            over(n_main, main_list, lambda i: main_copy(i, True).start())", "            pass"),
        ("            over(n_main, main_list, lambda i: main_copy(i, True).wait())", "            pass")]),
]


def pytest(extra):
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest", TEST, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    original = open(FUSED).read()
    rc, tail = pytest(SUBSET)
    print(f"control (unmutated): rc={rc} {tail}", flush=True)
    if rc != 0:
        return 2
    killed = 0
    for name, edits in MUTANTS:
        src = original
        ok = True
        for old, new in edits:
            if src.count(old) != 1:
                print(f"{name}: pattern count {src.count(old)} != 1 (NOT APPLIED)", flush=True)
                ok = False
                break
            src = src.replace(old, new)
        if not ok:
            continue
        try:
            open(FUSED, "w").write(src)
            rc, tail = pytest(SUBSET)
        finally:
            open(FUSED, "w").write(original)
            assert open(FUSED).read() == original, "restore failed"
        killed += rc != 0
        print(f"{name}: {'KILLED' if rc else 'SURVIVED'} (rc={rc}; {tail})", flush=True)
    rc, tail = pytest(SUBSET)
    print(f"control (restored): rc={rc} {tail}")
    print(f"LEAN mutants killed: {killed}/{len(MUTANTS)}")
    return 0 if killed == len(MUTANTS) and rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Mutation check for VLLM_TPU_PROMPT_IDS_NDARRAY's tests (q38mm DM lane, P15): every mutant must make
tests/q38mm_dm/test_prompt_ids_ndarray.py FAIL. Files are restored after each mutant (in place, from memory).
Usage (from the tree root): python tests/q38mm_dm/mutants_pta.py   -> exit 0 iff every mutant is caught."""

import subprocess
import sys

MOD = "src/vllm_torchtpu/core/prompt_ids_ndarray.py"
SCHED = "src/vllm_torchtpu/core/prefix_aware_scheduler.py"
TEST = "tests/q38mm_dm/test_prompt_ids_ndarray.py"
MUTANTS = [
    ("int16 dtype", MOD, "_DTYPE = np.int32", "_DTYPE = np.int16"),
    ("no substitution", MOD, "        nrd.prompt_token_ids = arr\n", "        pass\n"),
    ("stale cache trusted", MOD, "return arr is not None and len(arr) == n and (n == 0 or int(arr[-1]) == int(ids[-1]))",
     "return arr is not None"),
    ("attach drops the last token", MOD, "    setattr(request, ATTR, to_array_chunked(ids))", "    setattr(request, ATTR, to_array_chunked(ids[:-1]))"),
    ("chunk loop skips the remainder", MOD, "        out[i:i + len(part)] = np.fromiter(part, dtype=_DTYPE, count=len(part))",
     "        out[i:i + len(part)] = np.fromiter(part, dtype=_DTYPE, count=len(part)) if len(part) == chunk else 0"),
    ("hook ignores the flag", SCHED, "        if self._q38_prompt_np:\n            prompt_ids_ndarray.substitute",
     "        if True:\n            prompt_ids_ndarray.substitute"),
    ("engine hook skips attach", MOD, "                attach(out[0])", "                pass"),
    ("tokhash hashes a truncated row", "src/vllm_torchtpu/core/q38mm_tokhash.py", "token_ids_cpu[index, :n]", "token_ids_cpu[index, :n - 1]"),
]


def run_tests() -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", TEST],
                          capture_output=True, text=True).returncode


def main() -> int:
    base = run_tests()
    print(f"baseline (no mutant): pytest rc={base} ({'PASS' if base == 0 else 'FAIL'})")
    if base != 0:
        print("MUTANTS NOT RUN: the baseline fails")
        return 2
    survived = 0
    for name, path, old, new in MUTANTS:
        src = open(path).read()
        if src.count(old) != 1:
            print(f"mutant '{name}': anchor not found exactly once in {path} -> FAIL")
            survived += 1
            continue
        try:
            open(path, "w").write(src.replace(old, new))
            rc = run_tests()
        finally:
            open(path, "w").write(src)
        caught = rc != 0
        survived += not caught
        print(f"mutant '{name}': pytest rc={rc} -> {'CAUGHT' if caught else 'SURVIVED'}")
    print(f"MUTANTS {len(MUTANTS) - survived}/{len(MUTANTS)} caught")
    return 0 if survived == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

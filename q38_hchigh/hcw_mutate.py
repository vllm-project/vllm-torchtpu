# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS source mutants for the CPU proof. Each mutant is ONE exact replacement (the original text must occur
exactly once) in one file of a copied tree; the proof body then runs the check that must fail.

usage: hcw_mutate.py <tree root> <mutant id> <out tree root>     (copies src/, tests/, q38_hchigh/ and applies the mutant)
       hcw_mutate.py --list
"""
import shutil
import sys

K = "src/vllm_torchtpu/kernels/qwen4_exp/hc_decode_wide.py"
A = "src/vllm_torchtpu/kernels/qwen4_exp/vllm_adapter.py"
E = "src/vllm_torchtpu/envs.py"
MUTANTS = {
    # unit-test mutants
    "MA": (K, "MAX_TOKENS = 511\n", "MAX_TOKENS = 527\n", "gate accepts 512-527 rows -> test_wide_applies fails"),
    "MB": (K, "        if i == count - 1:\n            rows += tail\n", "        if i == 0:\n            rows += tail\n",
           "16-row tail in the first block -> test_row_blocks_are_the_served_pieces fails"),
    "MC": (A, "    if wide:\n        if wide not in WIDE_MODES:", "    if True:\n        wide = wide or 1\n        if wide not in WIDE_MODES:",
           "flag-off takes the wide path -> test_op_names_follow_the_flag fails"),
    "MG": (E, '    "TPU_QWEN4_HC_DECODE_WIDE": env_int("TPU_QWEN4_HC_DECODE_WIDE", 0),\n', "",
           "key not registered -> test_compile_factors_key_the_flag fails"),
    # numerics mutants (hcw_numerics.py --module)
    "MD": (K, "                    .astype(jnp.bfloat16)\n                    .astype(jnp.float32)\n                )\n"
              "                weighted = jax.nn.sigmoid(logits)",
           "                )\n                weighted = jax.nn.sigmoid(logits)",
           "rule-23 positive control: mode 1 without the up-logits bf16 round trip -> N1 must read NOT bitwise"),
    "ME": (K, "_lane_column(gates, lane0 + j)", "_lane_column(gates, lane0 + j + 1)",
           "mode 2 reads the wrong inject lane -> the N2 injection bound fails"),
    "MF": (K, "+ block * _lane_column(prev_inj, b)", "+ block * _lane_column(prev_inj, 0)",
           "mode 1 combines every branch with branch 0's gate -> the N1 gate (h bitwise / tolerance) fails"),
    # AOT control
    "MV": (K, "VMEM_LIMIT = 56 << 20\n", "VMEM_LIMIT = 24 << 20\n",
           "24 MiB VMEM limit -> the deviceless TPU7x compile of W1 at 448 rows must fail"),
}


def main(argv):
    if argv == ["--list"]:
        for k, (f, _, _, why) in MUTANTS.items():
            print(f"{k} {f}: {why}")
        return 0
    root, mid, out = argv
    f, old, new, why = MUTANTS[mid]
    for d in ("src", "tests", "q38_hchigh"):
        shutil.copytree(f"{root}/{d}", f"{out}/{d}", dirs_exist_ok=True)
    p = f"{out}/{f}"
    s = open(p).read()
    n = s.count(old)
    if n != 1:
        print(f"HCWMUT {mid} ERROR: original text occurs {n} times in {f}")
        return 2
    open(p, "w").write(s.replace(old, new))
    print(f"HCWMUT {mid} applied to {f}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

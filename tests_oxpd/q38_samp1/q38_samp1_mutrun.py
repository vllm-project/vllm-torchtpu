# SPDX-License-Identifier: Apache-2.0
"""q38 L4 S1 (VPS top-k keys) mutation runner: break the code deliberately, the
targeted tests must FAIL (an assertion failure, not an import / collection
error).

Copies src/ and tests/ to a scratch dir, runs the control (unmutated: every
target passes, 0 failed, 0 skipped), then for each mutant applies one exact
text replacement (the old text must occur exactly once), runs that mutant's
target tests and requires a non-zero exit whose summary says "failed" and has
no collection error. The original tree is never edited. Exit 0 only if the
control passes and every mutant is killed that way.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
VPS = "src/vllm_torchtpu/layers/adapter/sample/vp_sampler.py"
T = "tests/q38_samp1/test_vps_topk_keys.py"
FIRST = T + "::test_first_kc_equals_stable_sort"
ZEROS = T + "::test_signed_zeros_compare_equal_like_the_sort"
LOCAL = T + "::test_local_candidates_bitwise_equal_sort_path"
GUARD = T + "::test_flag_off_and_f32_keep_the_sort"
FRESH = T + "::test_flag_read_at_import_in_a_fresh_process"
REG = ("tests/compilation/test_tpu_compile_key_registration.py::"
       "test_graph_flag_moves_the_compile_hash[VLLM_TPU_VPS_TOPK_KEYS-1-False]")

MUTANTS = [
    ("K1_no_signed_zero_normalization", VPS, "    b = (x + 0.0).contiguous().view(torch.int16).to(torch.int32)\n",
     "    b = x.contiguous().view(torch.int16).to(torch.int32)\n", [ZEROS]),
    ("K2_no_negative_flip", VPS, "    v = torch.where(b < 0, b ^ 0x7FFF, b)\n", "    v = b\n", [FIRST]),
    ("K3_ties_to_higher_index", VPS, "    return v * (1 << index_bits) + ((1 << index_bits) - 1 - pos)\n",
     "    return v * (1 << index_bits) + pos\n", [FIRST]),
    ("K4_candidate_keys_too_few_index_bits", VPS, "        pos = _first_kc_desc(cand, kc, 13)\n",
     "        pos = _first_kc_desc(cand, kc, 12)\n", [LOCAL]),
    ("K5_flag_ignored", VPS, "    keys = _TOPK_KEYS and local_logits.dtype == torch.bfloat16\n",
     "    keys = local_logits.dtype == torch.bfloat16\n", [GUARD]),
    ("K6_no_bf16_guard", VPS, "    keys = _TOPK_KEYS and local_logits.dtype == torch.bfloat16\n",
     "    keys = _TOPK_KEYS\n", [GUARD]),
    ("K7_flag_compile_ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     '    "VLLM_TPU_HOST_PHASES",\n', '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_VPS_TOPK_KEYS",\n', [REG]),
    ("K8_flag_spellings_lost", VPS, 'in ("true", "1", "yes", "on")\n', 'in ("1",)\n', [FRESH]),
    ("K9_blocks_by_ascending_max", VPS, "        order = _first_kc_desc(blocks.amax(dim=-1), kc, 8)\n",
     "        order = _first_kc_desc(-blocks.amax(dim=-1), kc, 8)\n", [LOCAL]),
]

CONTROL = [T, "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    env.pop("VLLM_TPU_VPS_TOPK_KEYS", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True,
                       text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="samp1mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pyproject.toml",):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), work)
    rc, tail, _ = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    killed = total = 0
    for name, rel, old, new, targets in MUTANTS:
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {name} BAD (anchor occurs {text.count(old)} times)")
            ok = False
            continue
        with open(path, "w") as f:
            f.write(text.replace(old, new))
        try:
            rc, tail, out = run(work, targets)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        dead = (rc != 0 and " failed" in tail and "errors during collection" not in out
                and "ERROR " not in out)
        killed += dead
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and killed == total == len(MUTANTS)
    print(f"Q38_SAMP1_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

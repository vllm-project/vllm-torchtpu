# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-2 (a) mutation runner: break the code deliberately, the
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
MTP = "src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py"
INIT = "src/vllm_torchtpu/__init__.py"
T = "tests/q38_draft2/test_draft_embed_h8.py"
PART = T + "::test_partial_sums_to_the_full_lookup_and_matches_patched_forward"
ROUTE = T + "::test_draft_embed_routes_the_partial"
TP1 = T + "::test_tp1_takes_the_plain_forward"
BUILD = T + "::test_flag_default_and_build_time_read"
REG = ("tests/compilation/test_tpu_compile_key_registration.py::"
       "test_graph_flag_moves_the_compile_hash[VLLM_TPU_DRAFT_EMBED_H8-1-False]")

MUTANTS = [
    ("D1_partial_without_mask", INIT,
     "        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)\n        return output_parallel\n\n    def patched_get_masked_input_and_mask(",
     "        return output_parallel\n\n    def patched_get_masked_input_and_mask(", [PART]),
    ("D2_ignores_use_kernel", MTP,
     "        if tp_ar_decode.use_kernel(partial):\n", "        if False:\n", [ROUTE]),
    ("D3_ignores_the_flag", MTP,
     '        if not self.embed_h8 or int(getattr(emb, "tp_size", 1)) <= 1:\n',
     '        if int(getattr(emb, "tp_size", 1)) <= 1:\n', [ROUTE]),
    ("D4_always_h8_even_unarmed", MTP,
     "        if tp_ar_decode.use_kernel(partial):\n", "        if tp_ar_decode._ARMED is not None or True:\n", [ROUTE]),
    ("D5_flag_compile_ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     '    "VLLM_TPU_HOST_PHASES",\n', '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_DRAFT_EMBED_H8",\n', [REG]),
    ("D6_fuse_bypasses_embed", MTP,
     "        embeds = self.embed(input_ids) if inputs_embeds is None else inputs_embeds\n",
     "        embeds = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds\n", [BUILD]),
    ("D7_partial_not_attached", INIT,
     "    vpe.VocabParallelEmbedding._tpu_partial = tpu_partial\n", "", [PART]),
    ("D8_flag_default_on", "src/vllm_torchtpu/envs.py",
     '    "VLLM_TPU_DRAFT_EMBED_H8": env_bool("VLLM_TPU_DRAFT_EMBED_H8", default=False),\n',
     '    "VLLM_TPU_DRAFT_EMBED_H8": env_bool("VLLM_TPU_DRAFT_EMBED_H8", default=True),\n', [BUILD]),
    ("D9_tp1_takes_partial", MTP,
     '        if not self.embed_h8 or int(getattr(emb, "tp_size", 1)) <= 1:\n',
     '        if not self.embed_h8:\n', [TP1]),
]

CONTROL = [T, "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    env.pop("VLLM_TPU_DRAFT_EMBED_H8", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True,
                       text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="draft2mut_")
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
    print(f"Q38_DRAFT2_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

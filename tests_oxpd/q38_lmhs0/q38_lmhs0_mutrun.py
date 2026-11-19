# SPDX-License-Identifier: Apache-2.0
"""q38 L4 LMH-STREAM stage 0 mutation runner (after DRAFTEMB-BUILD's q38_draftemb_mutrun.py, credited): break
the code deliberately; the targeted tests must FAIL (a failed test, not a collection / import error).

Usage: python3 tests/q38_lmhs0/q38_lmhs0_mutrun.py <tree> [LO HI]
Copies src/ and tests/ of <tree> to a scratch dir, checks that the copy is what the tests import
(MUTRUN_ORIGIN ... INSIDE), runs the control (every target passes; 0 failed / skipped / errors),
then for each mutant in LO..HI (1-based, inclusive; all by default) applies one exact text
replacement (the old text must occur exactly once), runs that mutant's targets and requires a
non-zero exit whose summary says "failed" with no collection error. <tree> is never edited.
Last line: Q38_LMHS0_MUT PASS|FAIL killed K/N of N [range LO-HI of TOTAL] control_ok=...
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
UT = "src/vllm_torchtpu/runner/utils.py"
RN = "src/vllm_torchtpu/runner/tpu_runner.py"
EN = "src/vllm_torchtpu/envs.py"
CP = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T = "tests/q38_lmhs0/test_lmh_hidden_major.py"
PROJ = T + "::test_hidden_major_head_projects_like_the_vocab_major_head"
KEPT = T + "::test_head_kept_when_its_weight_is_read_another_way"
DONLY = T + "::test_draft_only_copy_leaves_the_target_head_untouched"
ENG = T + "::test_draft_head_runs_hidden_major_through_get_top_tokens"
SHARD = T + "::test_sharded_draft_top_tokens_match_the_stock_head"
BOTH = T + "::test_verify_and_draft_share_one_hidden_major_table"
VNOD = T + "::test_verify_without_draft_is_refused_when_the_head_is_shared"
VALONE = T + "::test_verify_alone_without_a_draft_model"
OFF = T + "::test_flags_off_change_nothing"
REF = T + "::test_draft_refusals_fail_stop"
SECOND = T + "::test_second_relayout_never_transposes_back"
BUILD = T + "::test_flag_defaults_and_build_time_read"
WIRE = T + "::test_runner_arms_after_the_drafter_on_the_last_stage_only"
FLAGOFF = T + "::test_flags_off_is_the_base_ast_for_ast"
CENSUS = T + "::test_census_at_qwen38_tp8_dims"
REGF = "tests/compilation/test_tpu_compile_key_registration.py::test_graph_flag_moves_the_compile_hash"
REG_D = REGF + "[VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT-1-False]"
REG_V = REGF + "[VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY-1-False]"
XP = "        _to_row_major(head.weight.data.t().contiguous()), requires_grad=False\n"

MUTANTS = [
    # --- the hidden-major projection (PR 2821's method)
    ("L01_apply_transposes_again", UT, "            return torch.matmul(x, layer.weight)\n",
     "            return torch.matmul(x, layer.weight.t())\n", [PROJ, ENG]),
    ("L02_bias_path_drops_bias", UT, "        return torch.addmm(bias, x.reshape(",
     "        return torch.addmm(bias * 0, x.reshape(", [PROJ]),
    # --- the stored layout (weight-load census)
    ("L03_skip_transpose", UT, XP, XP.replace(".t().contiguous()", ".contiguous()"), [DONLY, CENSUS]),
    ("L04_reshape_not_transpose", UT, XP,
     XP.replace(".t().contiguous()", ".reshape(head.weight.shape[1], head.weight.shape[0]).contiguous()"),
     [SHARD, DONLY]),
    # --- the draft-only copy leaves the verify head alone
    ("L05_copy_shares_param_registry", UT, "    new._parameters = head._parameters.copy()\n", "", [DONLY]),
    ("L06_draft_only_relayouts_shared_head", UT,
     "                draft_model.lm_head = hidden_major_head_copy(draft_head)\n",
     "                draft_head.weight = _hidden_major_weight(draft_head)\n"
     "                draft_head.quant_method = HiddenMajorLMHeadMethod()\n", [DONLY]),
    ("L07_copy_keeps_stock_method", UT,
     "    new.weight = _hidden_major_weight(head)\n    new.quant_method = HiddenMajorLMHeadMethod()\n",
     "    new.weight = _hidden_major_weight(head)\n", [ENG, SHARD]),
    ("L08_verify_needs_draft_guard_dropped", UT, "    if verify and shared and not draft:\n", "    if False:\n",
     [VNOD]),
    # --- refusals (PR 2821's conditions; fail-stop in the q38 arm)
    ("L09_refusal_quant_dropped", UT, "    if type(head.quant_method) is not UnquantizedEmbeddingMethod:\n",
     "    if False:\n", [KEPT, REF]),
    ("L10_refusal_tied_dropped", UT,
     '    if embed is not None and getattr(embed, "weight", None) is head.weight:\n', "    if False:\n",
     [KEPT, REF]),
    ("L11_refusal_head_dtype_dropped", UT, "    if processor.head_dtype not in (None, head.weight.dtype):\n",
     "    if False:\n", [KEPT, REF]),
    ("L12_refusal_type_dropped", UT, "    if not isinstance(head, ParallelLMHead):\n", "    if False:\n", [KEPT]),
    ("L13_verify_relayout_skipped", UT,
     "        target_head.weight = _hidden_major_weight(target_head)\n"
     "        target_head.quant_method = HiddenMajorLMHeadMethod()\n        changed = True\n",
     "        changed = True\n", [BOTH]),
    ("L14_draft_only_treated_as_shared", UT, "        elif shared and verify:\n", "        elif shared:\n",
     [DONLY, ENG]),
    # --- flag registration
    ("L15_draft_flag_default_on", EN,
     '    "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT": env_bool("VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT"),\n',
     '    "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT": env_bool("VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT", default=True),\n',
     [BUILD]),
    ("L16_draft_flag_not_hashed", CP, '    "VLLM_TPU_HOST_PHASES",\n',
     '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT",\n', [REG_D]),
    ("L17_verify_flag_not_hashed", CP, '    "VLLM_TPU_HOST_PHASES",\n',
     '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY",\n', [REG_V]),
    # --- runner wiring
    ("L18_runner_guard_drops_last_stage", RN,
     "        ) and self._pp_is_last and not self.is_pooling_model:\n            runner_utils.arm_lm_head_hidden_major(\n",
     "        ) and not self.is_pooling_model:\n            runner_utils.arm_lm_head_hidden_major(\n", [WIRE]),
    ("L19_runner_swaps_flags", RN, "                draft=envs.VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT,\n",
     "                draft=envs.VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY,\n", [WIRE]),
    ("L20_runner_passes_no_draft", RN,
     "                self._async_drafter.draft_model if self._is_async_drafter else None,\n",
     "                None,\n", [WIRE]),
    ("L21_runner_unguarded", RN,
     "            envs.VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT or envs.VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY\n"
     "        ) and self._pp_is_last",
     "            True\n        ) and self._pp_is_last", [FLAGOFF, WIRE]),
    ("L22_runner_call_dropped", RN,
     "            runner_utils.arm_lm_head_hidden_major(\n                self.model,\n",
     "            (lambda *a, **k: None)(\n                self.model,\n", [WIRE]),
    # --- layout pin, census, off, refusal fail-stop, idempotence
    ("L23_pin_not_installed", UT, "    if changed:\n        _install_entry_layout_pin()\n", "", [DONLY, BOTH]),
    ("L24_copy_drops_shard_padding", UT, "    new.weight = _hidden_major_weight(head)\n",
     "    new.weight = torch.nn.Parameter(_hidden_major_weight(head)[:, : head.num_org_embeddings_per_partition]"
     ".contiguous(), requires_grad=False)\n", [SHARD]),
    ("L25_relayout_ignores_refusal", UT, "    if hidden_major_refusal(head, processor, embed) is not None:\n",
     "    if False:\n", [KEPT, SECOND]),
    ("L26_census_counts_elements", UT, '        f"{w.numel() * w.element_size()} B"\n',
     '        f"{w.numel()} B"\n', [DONLY]),
    ("L27_draft_refusal_not_raised", UT,
     '            if why is not None:\n                raise ValueError(\n'
     '                    f"VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT=1: draft',
     '            if False:\n                raise ValueError(\n'
     '                    f"VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT=1: draft', [REF]),
    ("L28_flags_off_still_runs", UT, "    if not (draft or verify):\n        return census\n", "", [OFF]),
]

CONTROL = [T, "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
FLAGS = ("VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT", "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    for k in FLAGS:
        env.pop(k, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    lo = int(sys.argv[2]) if len(sys.argv) > 3 else 1
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(MUTANTS)
    assert 1 <= lo <= hi <= len(MUTANTS), (lo, hi, len(MUTANTS))
    work = tempfile.mkdtemp(prefix="lmhs0mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    probe = subprocess.run(
        [sys.executable, "-c", "import os, vllm_torchtpu.runner.utils as m; print(os.path.realpath(m.__file__))"],
        cwd=work, env=dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu"),
        capture_output=True, text=True, timeout=900)
    got = (probe.stdout.strip().splitlines() or [""])[-1]
    inside = got.startswith(os.path.realpath(os.path.join(work, "src")) + os.sep)
    print(f"MUTRUN_ORIGIN runner.utils {got or 'none'} {'INSIDE' if inside else 'FALLBACK'}")
    rc, tail, out = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (inside and rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    if not ok:
        print(out[-4000:])
    killed = total = 0
    for idx, (name, rel, old, new, targets) in enumerate(MUTANTS, start=1):
        if not lo <= idx <= hi:
            continue
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {idx:02d} {name} BAD (anchor occurs {text.count(old)} times)")
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
        print(f"MUT {idx:02d} {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
        if not dead:
            print("\n".join(out.splitlines()[-25:]))
    shutil.rmtree(work, ignore_errors=True)
    n = hi - lo + 1
    verdict = ok and killed == total == n
    rng = f" range {lo}-{hi} of {len(MUTANTS)}" if (lo, hi) != (1, len(MUTANTS)) else ""
    print(f"Q38_LMHS0_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {n}{rng} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""q38 L4 EAGER-FIX mutation runner (same harness as q38-l4-d98's q38_d98_mutrun.py): break the code deliberately, the tests must fail.

Copies src/ and tests/ to a scratch dir, runs the control (unmutated: every
target must pass), then for each mutant applies one exact text replacement
(the old text must occur exactly once), runs that mutant's target tests and
requires a non-zero exit. The original tree is never edited. Exit 0 only if
the control passes and every mutant is killed.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
DEV = os.environ.get("D98_MUT_DEV_RUNNER")  # Mac dev: a stub runner script
SKIP_PKG = os.environ.get("D98_MUT_SKIP_PKG") == "1"  # Mac dev: no vLLM
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
SCF = "src/vllm_torchtpu/runner/sampling_copy_full.py"
ESL = "src/vllm_torchtpu/runner/eager_shape_log.py"
T = "tests/runner/test_q38_eagerfix.py"

MUTANTS = [
    ("E1_spec_tail_not_padded", TR,
     "    if num_rejected.shape[0] < rej_rows:\n", "    if False:\n",
     [T + "::test_spec_tail_rejected_padding_is_bitwise_on_live_rows",
      T + "::test_served_steps_stay_on_the_fused_path_after_warmup"]),
    ("E2_pad_value_not_zero", TR,
     "            num_rejected, (0, rej_rows - num_rejected.shape[0])\n",
     "            num_rejected, (0, rej_rows - num_rejected.shape[0]), value=1\n",
     [T + "::test_spec_tail_rejected_padding_is_bitwise_on_live_rows"]),
    ("E3_precompile_omits_pad_arg", TR,
     "                        bound,\n                        *self._spec_tail_rej_extra(),\n",
     "                        bound,\n",
     [T + "::test_serving_tail_call_matches_a_precompiled_variant"]),
    ("E4_serving_omits_pad_arg", TR,
     "                self.max_num_reqs * (1 + self.speculative_config.num_speculative_tokens),\n"
     "                *self._spec_tail_rej_extra(),\n",
     "                self.max_num_reqs * (1 + self.speculative_config.num_speculative_tokens),\n",
     [T + "::test_serving_tail_call_matches_a_precompiled_variant"]),
    ("E5_bootstrap_zeros_short", TR,
     "                max(padded_req_offset, self.max_num_reqs),\n",
     "                padded_req_offset,\n",
     [T + "::test_bootstrap_and_multichunk_rejected_shapes"]),
    ("E6_multichunk_not_padded", TR,
     "        if envs.VLLM_TPU_SPEC_TAIL_REJ_PAD and num_rejected.shape[0] < self.max_num_reqs:\n",
     "        if False:\n",
     [T + "::test_bootstrap_and_multichunk_rejected_shapes"]),
    ("C1_full_copy_returns_whole_buffer", SCF,
     "            return to_tensor[:length]\n", "            return to_tensor\n",
     [T + "::test_full_copy_slice_rows_and_view_identical"]),
    ("C2_full_copy_on_2d_too", SCF,
     "            from_tensor.ndim == 1\n            and to_tensor.ndim == 1\n"
     "            and from_tensor.shape == to_tensor.shape\n",
     "            from_tensor.numel() == to_tensor.numel()\n",
     [T + "::test_full_copy_slice_rows_and_view_identical"]),
    ("C3_shape_log_in_warmup", ESL,
     "    if not _serving:\n        return\n", "",
     [T + "::test_eager_shape_log_is_silent_in_warmup_and_logs_new_layouts_once"]),
    ("B1_warm_uses_sampling_generator", TR,
     "            gen = torch.Generator(device=self.device)\n            gen.manual_seed(0)\n",
     "            gen = self._get_sampling_generator()\n",
     [T + "::test_warm_draws_match_serving_draws_and_leave_the_stream"]),
    ("B2_warm_order_differs", TR,
     "            (num_reqs, width),\n            tuple(draft_token_ids.shape),\n",
     "            tuple(draft_token_ids.shape),\n            (num_reqs, width),\n",
     [T + "::test_warm_draws_match_serving_draws_and_leave_the_stream"]),
    ("F1_flags_off_patches_anyway", TR,
     "if envs.VLLM_TPU_SAMPLING_COPY_FULL or envs.VLLM_TPU_EAGER_SHAPE_LOG:\n",
     "if True:\n",
     [T + "::test_flags_off_identity_and_on_wiring_in_fresh_processes"]),
    ("D1_covered_pairs_rewarmed", TR,
     "            if num_tokens <= max_logits_len and num_reqs <= req_bucket:\n"
     "                continue  # the capped sweep warmed it\n", "",
     [T + "::test_full_draft_verify_pairs_above_the_cap"]),
    ("D2_last_count_dropped", TR,
     "        for n in range(max_reqs + 1, n_max + 1):\n",
     "        for n in range(max_reqs + 1, n_max):\n",
     [T + "::test_full_draft_verify_pairs_above_the_cap"]),
    ("D3_outside_served_vps", TR,
     "        if not (served_only and vps_on):\n            logger.warning(\n"
     "                \"VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS needs \"\n",
     "        if False:\n            logger.warning(\n"
     "                \"VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS needs \"\n",
     [T + "::test_full_draft_pairs_need_served_only_vps"]),
    ("G1_warm_call_ungated", TR,
     '            if getattr(envs, "VLLM_TPU_VPS_WARM_DRAWS", 0):\n'
     "                self._warm_vps_draws(\n                    num_reqs, draft_token_ids,",
     "            if True:\n"
     "                self._warm_vps_draws(\n                    num_reqs, draft_token_ids,",
     ["tests/runner/test_spec_verify_served_only.py"]),
    ("G2_draw_site_on_module_global", TR,
     '        note = getattr(self, "_eager_shape_note", None)\n        if note is not None:\n'
     '            note(\n                "vps_draws",',
     "        note = _eager_shape_log and _eager_shape_log.note\n        if note:\n"
     '            note(\n                "vps_draws",',
     ["tests/q38mm_dstock/test_vps_wiring.py"]),
    ("G3_full_draft_ungated", TR,
     '                if getattr(envs, "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS", 0)\n',
     "                if True\n",
     ["tests/runner/test_spec_verify_served_only.py"]),
    ("K1_rej_pad_compile_ignored", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     '    "VLLM_TPU_SAMPLING_COPY_FULL",\n',
     '    "VLLM_TPU_SAMPLING_COPY_FULL",\n    "VLLM_TPU_SPEC_TAIL_REJ_PAD",\n',
     ["tests/compilation/test_tpu_compile_key_registration.py"]),
]

CONTROL = [T, "tests/compilation/test_tpu_compile_key_registration.py",
           # the G mutants' targets: they must pass unmutated for a kill to mean anything
           "tests/runner/test_spec_verify_served_only.py", "tests/q38mm_dstock/test_vps_wiring.py",
           "tests/decfb2/test_vps_fb_wiring.py",
           # errors identically on BASE ee289379f (i1 / i2 P3 lists): not part of the control
           "--deselect", "tests/decfb2/test_vps_fb_wiring.py::test_stats_logged"]


def run(work, targets):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"))
    for k in ("VLLM_TPU_SPEC_TAIL_REJ_PAD", "VLLM_TPU_SAMPLING_COPY_FULL",
              "VLLM_TPU_EAGER_SHAPE_LOG", "VLLM_TPU_VPS_WARM_DRAWS"):
        env.pop(k, None)
    if DEV:
        cmd = [sys.executable, DEV, os.path.join(work, "src")]
    else:
        cmd = [sys.executable, "-m", "pytest"]
    cmd += ["-q", "-x", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", *targets]
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    tail = [ln for ln in p.stdout.splitlines() if " passed" in ln or " failed" in ln
            or " error" in ln or "skipped" in ln]
    return p.returncode, (tail[-1] if tail else p.stdout[-300:].replace("\n", " | "))


def main():
    work = tempfile.mkdtemp(prefix="eagerfixmut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    control = [t for t in CONTROL if not (SKIP_PKG and "test_fast" not in t)]
    rc, tail = run(work, control)
    print(f"CONTROL rc={rc} {tail}")
    ok = rc == 0
    killed = total = 0
    for m in MUTANTS:
        name, rel, old, new, targets = m[:5]
        if SKIP_PKG and len(m) > 5:
            print(f"MUT {name} SKIPPED (needs the real package; runs on the pod)")
            continue
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
            rc, tail = run(work, targets)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        dead = rc != 0
        killed += dead
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and killed == total and total > 0
    print(f"Q38_EAGERFIX_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

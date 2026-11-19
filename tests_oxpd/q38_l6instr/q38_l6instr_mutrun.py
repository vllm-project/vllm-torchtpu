# SPDX-License-Identifier: Apache-2.0
"""q38 L6-INSTR mutation runner: break an instrument deliberately; its target
tests must FAIL (an assertion failure, not an import / collection error).
Same runner shape as tests/q38_hostsub/q38_hostsub_mutrun.py (L4-C).

Usage: python3 tests/q38_l6instr/q38_l6instr_mutrun.py <tree root>
Last line: Q38_L6INSTR_MUT PASS|FAIL killed <k>/<n> of <N> control_ok=<bool>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
HP = "src/vllm_torchtpu/runner/host_phases.py"
RUNNER = "src/vllm_torchtpu/runner/tpu_runner.py"
CARRY = "src/vllm_torchtpu/runner/decode_carry.py"
EAGLE = "src/vllm_torchtpu/spec_decode/eagle3.py"
INPUTS = "src/vllm_torchtpu/models/vllm/qwen4_exp/inputs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
READER = "tests/q38_hostsub/q38_hostsub_read.py"
T = "tests/q38_l6instr/test_l6instr_submarks.py"
G = "tests/runner/test_host_phases_gc.py"
SEED = T + "::test_seed_sync_is_exactly_the_blocking_copy_wait"
DF0 = T + "::test_df0_call_is_exactly_the_draft_model_call"
ADP = T + "::test_adapter_sub_marks_bracket_their_calls"
PREP = T + "::test_prepare_inputs_sub_marks_through_the_carry_harness"
S_EXEC = T + "::test_adapter_split_in_execute_model_and_the_adapter"
S_SS = T + "::test_sample_setup_census_in_source"
S_DRAFT = T + "::test_draft_sub_marks_in_source"
S_READ = T + "::test_reader_maps_every_sub_mark_to_its_phase"
S_CARRYB = T + "::test_carry_sub_marks_bracket_the_carry_program"
GCLINE = G + "::test_gc_census_line_carries_the_callback_sums"
GCINST = G + "::test_callback_is_installed_at_the_first_step_and_times_a_real_collection"
GCFRESH = G + "::test_fresh_process_flag_gating"
REG = ("tests/compilation/test_tpu_compile_key_registration.py::"
       "test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_HOST_PHASES_GC-0-True]")

MUTANTS = [
    # PR-0: the seed bracket.
    ("I1_seed_sync_dropped", RUNNER, '                        host_phases.sub("pi_seed_sync")\n', "", [SEED]),
    ("I2_pre_wait_boundary_dropped", RUNNER,
     '                        host_phases.sub("pi_collect")\n'
     '                        sampled_cpu = previous.wait_for_copy()\n',
     '                        sampled_cpu = previous.wait_for_copy()\n', [SEED]),
    # DH-E: the draft model call.
    ("I3_df0_call_before_the_call", EAGLE,
     '            out = model(**kwargs)\n            if step_idx == 0:\n                host_phases.sub("df0_call")\n',
     '            if step_idx == 0:\n                host_phases.sub("df0_call")\n            out = model(**kwargs)\n',
     [DF0]),
    ("I4_df0_call_on_every_step", EAGLE,
     '            if step_idx == 0:\n                host_phases.sub("df0_call")\n',
     '            host_phases.sub("df0_call")\n', [DF0]),
    # PR-0: the adapter split.
    ("I5_ap_fused_dropped", INPUTS, '        host_phases.sub("ap_fused")\n', "", [ADP]),
    ("I6_scalars_upload_inside_the_call", INPUTS,
     '        scalars_dev = self._scalars_upload(scalars)\n        host_phases.sub("ap_sup")\n',
     '        host_phases.sub("ap_sup")\n        scalars_dev = self._scalars_upload(scalars)\n', [ADP]),
    ("I7_ap_wbuild_dropped", INPUTS, '            host_phases.sub("ap_wbuild")\n', "", [ADP]),
    # PR-0: the pi_layout / pi_post split on the real _prepare_inputs.
    ("I8_carry_pp_fused_dropped", CARRY, '    host_phases.sub("pp_fused")\n', "", [PREP]),
    ("I9_pi_carry_dropped", RUNNER, '        host_phases.sub("pi_carry")\n', "", [PREP]),
    ("I10_pi_h2d_back_to_pi_layout", RUNNER, 'host_phases.sub("pi_h2d")', 'host_phases.sub("pi_layout")', [PREP]),
    ("I11_carry_pp_record_dropped", CARRY,
     '            pending, subst)\n    host_phases.sub("pp_record")\n    _note("carried")\n',
     '            pending, subst)\n    _note("carried")\n', [PREP]),
    # T-5, DH-E and the adapter's runner part (structure).
    ("I12_ss_draft_chunks_before_the_handover", RUNNER,
     '        if self._is_async_drafter:\n            self._async_drafter.draft_chunks = state.draft_chunks\n'
     '        host_phases.sub("ss_draft_chunks")\n',
     '        host_phases.sub("ss_draft_chunks")\n'
     '        if self._is_async_drafter:\n            self._async_drafter.draft_chunks = state.draft_chunks\n',
     [S_SS]),
    ("I13_dc_call_dropped_fresh_path", EAGLE,
     '                    host_phases.sub("dc_call")\n'
     '                    if not getattr(self, "_carry_rows_fresh_logged", False):\n',
     '                    if not getattr(self, "_carry_rows_fresh_logged", False):\n', [S_DRAFT]),
    ("I14_dpi_seed_dropped_fused_seed", EAGLE,
     '                host_phases.sub("dpi_seed")\n'
     '                draft_input_ids, gather_indices, num_rejected_out = seed_out[:3]\n',
     '                draft_input_ids, gather_indices, num_rejected_out = seed_out[:3]\n', [S_DRAFT]),
    ("I15_ap_fence_dropped", RUNNER, '            host_phases.sub("ap_fence")\n', "", [S_EXEC]),
    # GC census.
    ("I16_report_gc_not_called", HP, "    if _GC:\n        _report_gc()\n", "", [GCLINE]),
    ("I17_gc_not_installed", HP, "        if _GC and not _gc_hooked:\n            _gc_install()\n", "", [GCINST]),
    ("I18_gen2_counted_as_gen1", HP, "        if g == 2:\n            _gc_win[2] += 1\n",
     "        if g == 1:\n            _gc_win[2] += 1\n", [GCLINE]),
    ("I19_gc_ignores_host_phases_off", HP,
     "_GC = bool(_EVERY) and bool(envs.VLLM_TPU_HOST_PHASES_GC)\n",
     "_GC = bool(envs.VLLM_TPU_HOST_PHASES_GC)\n", [GCFRESH]),
    ("I20_gc_flag_not_compile_ignored", TC, '    "VLLM_TPU_HOST_PHASES_GC",\n', "", [REG]),
    ("I21_gc_window_not_reset", HP, "    w[0] = w[1] = w[2] = w[3] = 0\n", "", [GCLINE]),
    # Reader rows.
    ("I22_reader_dpi_row_dropped", READER, '"dpi_": "dp_prepare_inputs", ', "", [S_READ]),
    # q38 L6-V5 (composed tree with L6-ADC): ~pp_fused moved in front of the carry program choice.
    ("I23_pp_fused_before_the_carry_program", CARRY,
     '    host_phases.sub("pp_key")\n    out = None\n',
     '    host_phases.sub("pp_key")\n    host_phases.sub("pp_fused")\n    out = None\n', [S_CARRYB]),
]

CONTROL = [T, G, "tests/runner/test_host_phases_subs.py", "tests/runner/test_host_phases.py",
           "tests/runner/test_host_phases_slow.py", "tests/q38_hostsub/test_hostsub_reader.py",
           "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu",
               Q38_L6INSTR_EXPECT_TREE=work)
    for k in ("VLLM_TPU_HOST_PHASES", "VLLM_TPU_HOST_PHASES_SUBS", "VLLM_TPU_HOST_PHASES_SLOW_MS",
              "VLLM_TPU_HOST_PHASES_GC"):
        env.pop(k, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="l6instrmut_")
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
    print(f"Q38_L6INSTR_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

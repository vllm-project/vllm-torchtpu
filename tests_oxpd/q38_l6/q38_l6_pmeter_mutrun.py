# SPDX-License-Identifier: Apache-2.0
"""q38 L6 P-METER + admission-census mutation runner: break the instrument
deliberately; its target tests must FAIL (an assertion failure, not an import /
collection error). Same shape as tests/q38_l6instr/q38_l6instr_mutrun.py.

Usage: python3 tests/q38_l6/q38_l6_pmeter_mutrun.py <tree root>
Last line: Q38_L6PM_MUT PASS|FAIL killed <k>/<n> of <N> control_ok=<bool>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
PM = "src/vllm_torchtpu/runner/l6_pmeter.py"
RUNNER = "src/vllm_torchtpu/runner/tpu_runner.py"
EAGLE = "src/vllm_torchtpu/spec_decode/eagle3.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
L6AB = "src/vllm_torchtpu/core/l6_ab.py"
AM = "src/vllm_torchtpu/layers/core/attention_metadata.py"
T = "tests/q38_l6/test_l6_pmeter.py"
A = "tests/q38_l6/test_l6_admit_census.py"
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
BYTE = T + "::test_window_line_is_byte_exact"
AB = T + "::test_ab_window_steps_show_x"
OFF = T + "::test_off_reads_no_clock_and_logs_nothing"
INJ = T + "::test_inject_one_rank_or_all_ranks_once_per_step_before_the_stamp"
MPR = T + "::test_mpr_wait_is_exactly_the_blocking_copy_wait"
WIRE = T + "::test_wiring_in_source"
AM_FULL = A + "::test_full_pack_split"
AM_REUSE = A + "::test_reuse_path_has_no_h2d"
AM_INC = A + "::test_incremental_pack_split"
N = "tests/q38_l6/test_l6_nodefer.py"
ND = "src/vllm_torchtpu/runner/l6_nodefer.py"
ND_FB = N + "::test_fallback_call_split_and_census"
ND_STOCK = N + "::test_stock_call_split_and_sampling_every_e"
ND_WIRE = N + "::test_census_wrapper_and_vr_post_in_source"
REG_ND = [REGF, "-k", "test_host_only_flag_keeps_the_compile_hash and L6_NODEFER"]
REG_PM = [REGF, "-k", "test_host_only_flag_keeps_the_compile_hash and L6_PMETER"]
REG_INJ = [REGF, "-k", "test_host_only_flag_keeps_the_compile_hash and L6_INJECT"]

MUTANTS = [
    ("P1_wait_stamp_on_the_perf_clock", PM, "        _cur[4] = time.time_ns()\n",
     "        _cur[4] = time.perf_counter_ns()\n", [BYTE]),
    ("P2_ring_slot_off_by_one", PM, "        i = ((_cur[0] - 1) % WINDOW) * _NF\n",
     "        i = (_cur[0] % WINDOW) * _NF\n", [BYTE]),
    ("P3_window_closed_one_step_early", PM, "    prev = l6_ab.step_seq() - 1\n", "    prev = l6_ab.step_seq()\n", [BYTE]),
    ("P4_forward_stamp_before_the_spin", PM,
     '    spin = _st["spin"]\n    if spin:\n        _st["spin"] = 0\n        _spin(spin)\n'
     '        l6_ab.note("inject", "spins")\n        l6_ab.note("inject", "us", spin)\n'
     '    if _st["on"] and _cur[2] < 0:\n        _cur[2] = time.time_ns()\n',
     '    spin = _st["spin"]\n    if _st["on"] and _cur[2] < 0:\n        _cur[2] = time.time_ns()\n'
     '    if spin:\n        _st["spin"] = 0\n        _spin(spin)\n'
     '        l6_ab.note("inject", "spins")\n        l6_ab.note("inject", "us", spin)\n', [INJ]),
    ("P5_inject_on_every_rank", PM, '    elif l6_ab.on("inject") and _rank() == INJECT_RANK:\n',
     '    elif l6_ab.on("inject"):\n', [INJ]),
    ("P6_spin_on_every_forward", PM, '        _st["spin"] = 0\n        _spin(spin)\n', "        _spin(spin)\n", [INJ]),
    ("P7_mpr_wait_boundary_before_the_wait", RUNNER,
     "        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()\n        _l6_pmeter.wait_end()\n"
     '        host_phases.sub("mpr_wait")\n',
     '        host_phases.sub("mpr_wait")\n        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()\n'
     "        _l6_pmeter.wait_end()\n", [MPR]),
    ("P8_mpr_pre_dropped", RUNNER, '        host_phases.sub("mpr_pre")\n', "", [MPR]),
    ("P9_pmeter_flag_not_compile_ignored", TC, '    "VLLM_TPU_L6_PMETER",\n', "", REG_PM),
    ("P10_inject_flag_not_compile_ignored", TC, '    "VLLM_TPU_L6_INJECT",\n', "", REG_INJ),
    ("P11_lever_row_dropped", L6AB, '    "pmeter": "VLLM_TPU_L6_PMETER",\n', "", [BYTE]),
    ("P12_dequeue_stamp_dropped", RUNNER,
     "        # q38 L6 P-METER: the SchedulerOutput dequeue stamp (no-op unless an L6 lever is set).\n"
     "        _l6_pmeter.deq()\n", "", [WIRE]),
    ("P13_forward_hook_dropped", RUNNER, "                _l6_pmeter.before_forward()\n", "", [WIRE]),
    ("P14_loop_stamp_dropped", EAGLE, "            _l6_pmeter.loop_enqueue()\n", "", [WIRE]),
    ("P15_on_step_not_called", RUNNER, "                _l6_pmeter.on_step()\n", "                pass\n", [WIRE]),
    ("P16_dequeue_stamped_with_every_lever_off", PM, '    if l6_ab.ACTIVE:\n        _st["deq"] = time.time_ns()\n',
     '    _st["deq"] = time.time_ns()\n', [OFF]),
    ("P17_threadstat_not_logged", PM, '    logger.info("%s", thread_line(seq_end))\n', "", [BYTE]),
    ("P18_off_step_not_marked_x", PM, '            parts.append("x")\n            continue\n', "            continue\n", [AB]),
    ("P19_wait_stamp_while_off", PM, 'def wait_begin() -> None:\n    if _st["on"]:\n        _cur[3] = time.time_ns()\n',
     "def wait_begin() -> None:\n    _cur[3] = time.time_ns()\n", [OFF]),
    # admission census (am_stage_bt split)
    ("A1_h2d_boundary_before_the_upload", AM,
     '        host_phases.sub("am_pack")\n        staged_dev = _inc_upload(flat, runner.device)\n'
     '        host_phases.sub("am_h2d")\n    if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:\n        # Dummy',
     '        host_phases.sub("am_pack")\n        host_phases.sub("am_h2d")\n'
     '        staged_dev = _inc_upload(flat, runner.device)\n    if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:\n        # Dummy',
     [AM_FULL]),
    ("A2_am_walk_dropped", AM, '    host_phases.sub("am_walk")\n', "", [AM_FULL]),
    ("A3_incremental_split_dropped", AM,
     '    if copied:\n        host_phases.sub("am_pack")\n        staged_dev = _inc_upload(flat, runner.device)\n'
     '        host_phases.sub("am_h2d")\n',
     "    if copied:\n        staged_dev = _inc_upload(flat, runner.device)\n", [AM_INC]),
    ("A4_pre_walk_boundary_dropped", AM, '    host_phases.sub("am_pack")\n    _issue_walk_products(\n',
     "    _issue_walk_products(\n", [AM_REUSE]),
    # NODEFER census and the ~vr_* split
    ("N1_vr_call_dropped", ND, '        host_phases.sub("vr_call")\n', "", [ND_FB]),
    ("N2_census_counts_started_args", ND, "        if not started:\n", "        if started:\n", [ND_FB]),
    ("N3_census_samples_every_call", ND, '    if (_st["n"] - 1) % EVERY:\n        return\n', "", [ND_STOCK]),
    ("N4_lever_row_dropped", L6AB, '    "nodefer": "VLLM_TPU_L6_NODEFER_CENSUS",\n', "", [ND_FB]),
    ("N5_flag_not_compile_ignored", TC, '    "VLLM_TPU_L6_NODEFER_CENSUS",\n', "", REG_ND),
    ("N6_vr_draw_dropped", ND, '        host_phases.sub("vr_draw")\n', "", [ND_FB]),
    ("N7_census_after_the_call", ND,
     '        census(args)\n        host_phases.sub("vr_draw")\n        out = self.fn(*args, **kwargs)\n',
     '        host_phases.sub("vr_draw")\n        out = self.fn(*args, **kwargs)\n        census(args)\n', [ND_FB]),
    ("N8_vr_post_dropped", RUNNER, '                    host_phases.sub("vr_post")\n', "", [ND_WIRE]),
    ("N9_fb_call_not_wrapped", RUNNER,
     "_vps_verify_fb_compiled = _l6_nodefer.wrap(_static_shape_args(_tpu_compile(_vps_verify_fb_impl)))\n",
     "_vps_verify_fb_compiled = _static_shape_args(_tpu_compile(_vps_verify_fb_impl))\n", [ND_WIRE]),
]

CONTROL = [T, A, N, "tests/q38_l6instr/test_l6instr_submarks.py", "tests/q38_hostsub/test_hostsub_reader.py",
           "tests/runner/test_block_table_stage_inc.py", REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu",
               Q38_L6PM_EXPECT_TREE=work, Q38_L6INSTR_EXPECT_TREE=work)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_") or k.startswith("VLLM_TPU_HOST_PHASES"):
            env.pop(k, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--noconftest", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="l6pmmut_")
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
        if not dead:
            print("\n".join(out.splitlines()[-25:]))
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and killed == total == len(MUTANTS)
    print(f"Q38_L6PM_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(MUTANTS)} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

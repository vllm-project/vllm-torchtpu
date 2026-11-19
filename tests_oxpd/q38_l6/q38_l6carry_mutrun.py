# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever carry mutation runner: break the code deliberately; the
targeted tests must FAIL (an assertion failure, not an import / collection
error).

Copies src/ and tests/ to a scratch dir, runs the control (unmutated: every
target passes, 0 failed, 0 skipped), then for each mutant applies one exact
text replacement (the old text must occur exactly once), checks the import
guard still passes on the mutated copy (so the run imports the copy), runs
that mutant's targets and requires a non-zero exit whose summary says
"failed" and has no error. The import-guard mutant points the guard at the
base tree instead; the guard itself must then fail. The original tree is
never edited. Exit 0 only if the control passes and every mutant is killed.

    python3 tests/q38_l6/q38_l6carry_mutrun.py <head root> <base root>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
BASE = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else ""
DC = "src/vllm_torchtpu/runner/decode_carry.py"
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T = "tests/q38_l6/test_l6_decode_carry.py"
MATCH = T + "::test_carry_matches_upload_path_over_random_rejections[b3_padded]"
EV = T + "::test_host_check_refuses_each_change[{}]"
ALONE = T + "::test_each_host_check_refuses_alone[{}]"
REMAP = T + "::test_new_source_positions_rebuild_the_indices"
DISCARD = T + "::test_discarded_request_is_refused_by_the_mapping"
OFF = T + "::test_lever_off_runs_no_carry_code"
CORRUPT = T + "::test_check_mode_catches_a_corrupted_carry[1]"
COLD = T + "::test_serving_never_compiles_a_carry_shape_and_precompile_warms_all"
HOOK = T + "::test_flag_set_and_warmup_hook"
REG = T + "::test_registration"
GUARD = T + "::test_imports_the_staged_tree"
WINDOW = T + "::test_a_lever_off_window_leaves_no_usable_state"

MUTANTS = [
    ("M01_carry_advances_padded_seq_lens", DC,
     "        seq_lens_subtract_indices >= 0, zero_s + step_delta, zero_s\n",
     "        seq_lens_subtract_indices >= -1, zero_s + step_delta, zero_s\n", [MATCH]),
    ("M02_carry_advances_padded_positions", DC,
     "        positions_subtract_indices >= 0, zero_p + step_delta, zero_p\n",
     "        positions_subtract_indices >= -1, zero_p + step_delta, zero_p\n", [MATCH]),
    ("M03_no_seq_advance_check", DC,
     "    if not np.all(d_s + r_prev == st.step_delta):\n", "    if False:\n",
     [EV.format("host_count_skew")]),
    ("M04_no_request_order_check", DC,
     "    if req_ids != st.req_ids:\n", "    if False:\n", [EV.format("reorder")]),
    ("M05_no_stale_check", DC,
     "    if st.step != step - 1:\n", "    if False:\n", [ALONE.format("stale")]),
    ("M06_no_rejection_tag_check", DC,
     "    if rec is None or rec[0] != st.step or rec[1] != st.src_tag:\n",
     "    if rec is None or rec[0] != st.step:\n", [ALONE.format("rej_tag")]),
    ("M07_no_rejection_step_check", DC,
     "    if rec is None or rec[0] != st.step or rec[1] != st.src_tag:\n",
     "    if rec is None or rec[1] != st.src_tag:\n", [ALONE.format("rej_step")]),
    ("M08_no_positions_check", DC,
     "    if not np.array_equal(runner.positions_np[:total], st.pos + np.repeat(d_s, st.nsched)):\n",
     "    if False:\n", [ALONE.format("positions")]),
    ("M09_no_positions_padding_check", DC,
     "    if not np.array_equal(runner.positions_np[total:padded_total], st.pos_pad):\n",
     "    if False:\n", [ALONE.format("positions_padding")]),
    ("M10_index_reuse_ignores_mapping", TR,
     "            if l6 is not None and l6.carry is not None and l6.carry.map_same:\n",
     "            if l6 is not None and l6.carry is not None:\n", [REMAP]),
    ("M11_subst_reuse_ignores_mapping", DC,
     "    if st.map_same:\n", "    if True:\n", [REMAP]),
    ("M12_lever_ignores_flag", DC,
     "    return _l6_ab.ACTIVE and _l6_ab.on(LEVER)\n", "    return True\n", [OFF]),
    ("M13_rejected_count_not_recorded", TR,
     "                carry_rej[req_id] = n_placeholder - len(sampled_ids)\n",
     "                carry_rej[req_id] = 0\n", [MATCH]),
    ("M14_reuse_ignores_values", DC,
     "    dev = runner._upload_unless_unchanged(slot, host)\n",
     "    dev = prev[1] if prev is not None else runner._upload_unless_unchanged(slot, host)\n",
     [EV.format("request_joins")]),
    ("M15_no_late_upload", DC,
     "        seq_lens = upload(runner, ctx)\n", "        pass\n", [COLD]),
    ("M16_check_compares_nothing", DC,
     "    bad = [n for n, a, b in zip(names, out, ref) if not torch.equal(a.cpu(), b.cpu())]\n",
     "    bad = []\n", [CORRUPT]),
    ("M17_no_warmup_precompile", TR,
     "                        _decode_carry.precompile(self)\n", "                        pass\n", [HOOK]),
    ("M18_precompile_warms_nothing", DC,
     "        warm.add((step_delta,) + tuple(key))\n", "        pass\n", [COLD]),
    ("M19_flag_compile_hashed", TC,
     '    "VLLM_TPU_L6_DECODE_CARRY",\n', "", [REG]),
    ("M20_no_pre_tag_check", DC,
     "    if tag(pre) == st.src_tag:\n", "    if False:\n", [ALONE.format("pre_not_advanced")]),
    ("M21_unmapped_request_carried", DC,
     "        map_now = tuple(mapped[r] for r in req_ids)\n",
     "        map_now = tuple(mapped.get(r, 0) for r in req_ids)\n", [DISCARD]),
    ("M22_no_shape_check", DC,
     "    if int(st.input_ids.shape[0]) != padded_total or int(st.positions.shape[0]) != padded_total:\n",
     "    if False:\n", [ALONE.format("shape")]),
    ("M23_postfwd_not_required", DC,
     "    if not envs.VLLM_TPU_POSTFWD_FUSED or not _layout_ok(runner, layout_plan, padded_total):\n",
     "    if not _layout_ok(runner, layout_plan, padded_total):\n", [ALONE.format("postfwd_off")]),
    ("M24_pcp_not_refused", DC,
     '    if getattr(runner, "_pcp_enabled", False) or getattr(runner, "uses_mrope", False):\n',
     '    if getattr(runner, "uses_mrope", False):\n', [ALONE.format("pcp")]),
    ("M25_second_chunk_not_refused", DC,
     "    if start_index != 0 or num_reqs != runner.input_batch.num_reqs:\n",
     "    if num_reqs != runner.input_batch.num_reqs:\n", [ALONE.format("second_chunk")]),
    ("M26_step_counter_frozen", "src/vllm_torchtpu/core/l6_ab.py",
     '    _state["seq"] += 1\n', "", [WINDOW]),
]
GUARD_MUTANT = "G01_import_guard_points_at_base"

CONTROL = sorted({t for *_, ts in MUTANTS for t in ts} | {GUARD})
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, expect_root):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu",
               Q38_L6CARRY_EXPECT_ROOT=expect_root)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
           "-p", "no:logging", "-rfE"]
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True,
                       timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def killed(rc, tail):
    return rc != 0 and "failed" in tail and "error" not in tail


def main():
    work = tempfile.mkdtemp(prefix="l6carrymut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    src_root = os.path.join(work, "src")
    rc, tail, out = run(work, CONTROL, src_root)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) == len(CONTROL)
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} targets={len(CONTROL)} {tail}")
    if not ok:
        print(out[-4000:])
    n_killed = total = 0
    for name, rel, old, new, targets in MUTANTS:
        total += 1
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {name} BAD (old text occurs {text.count(old)} times)")
            continue
        with open(path, "w") as f:
            f.write(text.replace(old, new))
        try:
            grc, gtail, _ = run(work, [GUARD], src_root)
            trc, ttail, tout = run(work, targets, src_root)
        finally:
            with open(path, "w") as f:
                f.write(text)
        guard_ok = grc == 0 and "1 passed" in gtail
        k = guard_ok and killed(trc, ttail)
        n_killed += k
        print(f"MUT {name} {'KILLED' if k else 'SURVIVED'} guard_ok={guard_ok} rc={trc} {ttail}")
        if not k:
            print(tout[-3000:])
    total += 1
    if BASE:
        grc, gtail, _ = run(work, [GUARD], os.path.join(BASE, "src"))
        k = killed(grc, gtail)
    else:
        grc, gtail, k = -1, "no base root given", False
    n_killed += k
    print(f"MUT {GUARD_MUTANT} {'KILLED' if k else 'SURVIVED'} rc={grc} {gtail}")
    verdict = "PASS" if ok and n_killed == total else "FAIL"
    print(f"Q38_L6CARRY_MUT {verdict} killed {n_killed}/{total} of {total}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

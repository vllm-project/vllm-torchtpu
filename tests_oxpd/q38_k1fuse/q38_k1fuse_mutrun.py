# SPDX-License-Identifier: Apache-2.0
"""q38 L4 T1-K1FUSE mutation runner (CDK CPU pod only; DRAFT1-B's runner engine): break the change deliberately;
its target tests must FAIL (a failed test, not an import or collection error), and where a mutant names a required
text, the failure output must contain it (e.g. the take's fail-stop, never a silent wrong draft).

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, checks the copy is what imports, runs the control
(unmutated: every target file passes, 0 failed / skipped / errors), then for each mutant applies one exact text
replacement (the old text must occur exactly once) and runs that mutant's targets with -x. The original tree is never
edited. Exit 0 only if the control passes and every selected mutant is killed that way.
  q38_k1fuse_mutrun.py <root> [LO HI]     (1-based inclusive slice of MUTANTS)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
RANGE = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
K1 = "src/vllm_torchtpu/spec_decode/k1fuse.py"
E3 = "src/vllm_torchtpu/spec_decode/eagle3.py"
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
AO = "src/vllm_torchtpu/runner/tpu_runner_async_output.py"

T = "tests/q38_k1fuse/test_k1fuse.py"
MR = "tests/q38_k1fuse/test_k1fuse_multirank.py"
C = "tests/q38_k1fuse/test_k1fuse_compile.py"
FO = "tests/q38_k1fuse/test_k1fuse_flagoff.py"
T_BIT = T + "::test_k1_is_bitwise_the_unfused_chain_over_8_steps[8-0]"
T_SEAM = T + "::test_seam_shapes_are_the_r4_contracts[12]"
T_PROG = T + "::test_instances_programs_and_token_buckets_over_the_steady_sweep[0]"
T_UNI = T + "::test_uniforms_stay_eager_in_order_and_none_is_drawn_inside_the_program"
T_CHK = T + "::test_check_runs_k1_first_compares_then_arms"
T_CHKP_D = T + "::test_check_catches_a_perturbed_k1_output_and_turns_k1_off[drafts]"
T_TAKE = T + "::test_no_write_between_verify_and_propose_touches_a_k1_input"
T_RAISE_S = T + "::test_a_write_to_a_k1_input_makes_the_take_raise[seq_lens]"
MR_FLAG = MR + "::test_one_rank_without_the_flag_stops_the_run"
MR_FP = MR + "::test_one_rank_with_another_warm_key_set_stops_the_run"
MR_WISH = MR + "::test_one_ranks_wish_off_turns_k1_off_everywhere_at_boundaries_only"
MR_CHK = MR + "::test_a_check_mismatch_on_one_rank_turns_k1_off_on_every_rank"
C_GF = C + "::test_k1_program_compiles_once_guard_free_across_steps"
FO_SUBS = FO + "::test_nt_wait_sub_marks_only_with_vfy"

RET = "    return tokens, acc, ro_idx, ro_vals, drafts, source, rejected\n"
TAKE_TEXT = "q38 K1FUSE: the propose after a fused verify"

# (name, file, old, new, targets[, required text in the failure output])
MUTANTS = [
    # --- the proof asks (L4-E's list)
    ("M01_draft_kv_write_dropped_from_its_blocks", K1,
     "                    m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2]\n",
     "                    m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2],\n"
     "                    block_tables=torch.zeros_like(m.block_tables),\n", [T_BIT]),
    ("M02_stale_seed", K1, "        seq_lens=seq_lens,\n", "        seq_lens=bufs[0],\n", [T_BIT]),
    ("M03_wrong_bucket", K1,
     "                t for t, w, lp, c in self.decode_shapes() if (lp, w, c) == key\n",
     "                t for t, w, lp, c in self.decode_shapes() if (lp, c) == key[::2]\n", [T_PROG]),
    ("M04_guarded_compile", K1,
     "    cls = support_torch_compile(dynamic_arg_dims=dict(K1_DYNAMIC_ARG_DIMS))(_program_class(index))\n"
     "    return cls(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, vh=vh, "
     "statics=statics)\n",
     "    m = _program_class(index)(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, "
     "vh=vh, statics=statics)\n"
     "    return torch.compile(m, backend=vllm_config.compilation_config.init_backend(vllm_config), fullgraph=True, "
     "dynamic=False)\n", [C_GF]),
    ("M05_nt_not_returned", K1, RET, RET.replace("return tokens,", "return torch.zeros_like(tokens),"), [T_SEAM]),
    ("M06_wrong_tail_shape", K1, RET, RET.replace("source, rejected", "source.view(s.max_reqs, -1), rejected"),
     [T_SEAM]),
    ("M07_uniforms_moved_inside_the_program", K1,
     "    tokens, acc, ro_idx, ro_vals = tpu_runner._vfy_one_impl(\n",
     "    bonus_u = torch.rand_like(bonus_u)\n    tokens, acc, ro_idx, ro_vals = tpu_runner._vfy_one_impl(\n",
     [T_UNI]),
    # --- agreement / fail-stop
    ("M08_flag_mismatch_not_stopped", K1, "        if len(set(confs)) != 1:\n", "        if False:\n", [MR_FLAG]),
    ("M09_fingerprints_not_compared", K1, "        if len(fps) != 1:\n", "        if False:\n", [MR_FP]),
    ("M10_wish_bits_ignored", K1, "              and all(w & BIT_K1 for w in words))\n", "              and True)\n",
     [MR_WISH]),
    ("M11_check_mismatch_not_shared", K1, "            if self.k1_mismatches:\n                w |= BIT_K1BAD\n",
     "            if False:\n                w |= BIT_K1BAD\n", [MR_CHK]),
    # --- the take
    ("M12_take_skips_the_version_counters", K1,
     "        elif stash.vers is None or vfy_one.versions(stash.watch) != stash.vers:\n", "        elif False:\n",
     [T_RAISE_S]),
    ("M13_inbetween_write_touches_the_tokens", AO,
     '            chunks_cpu = [t.to("cpu", non_blocking=True) for t in chunks_tpu]\n',
     '            chunks_cpu = [t.add_(0).to("cpu", non_blocking=True) for t in chunks_tpu]\n', [T_TAKE], TAKE_TEXT),
    ("M14_propose_ignores_the_k1_result", E3,
     '            if getattr(self._draft_asm, "k1_result", None) is not None and (\n',
     "            if False and (\n", [T_BIT]),
    ("M15_runner_discards_the_k1_verify", TR,
     "        ) is not None and (res := k1(self, chunk, call, spec, num_active)) is not None:\n",
     "        ) is not None and (res := k1(self, chunk, call, spec, num_active)) is not None and False:\n", [T_BIT]),
    # --- CHECK
    ("M16_check_skips_the_drafts", K1, '        pairs = [("drafts", drafts, base)]\n', "        pairs = []\n",
     [T_CHKP_D]),
    ("M17_check_step_served_by_k1", K1,
     '            self.k1_stats["check"] += 1\n            return None\n',
     '            self.k1_stats["check"] += 1\n            return out[:4]\n', [T_CHK]),
    # --- instrument
    ("M18_nt_wait_sub_mark_dropped", TR, '            host_phases.sub("k1_nt_wait")\n', "            pass\n", [FO_SUBS]),
]

CONTROL = [T, MR, C, FO]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
CLEAR = ("VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB",
         "VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "VLLM_TPU_DRAFT_ASM_ONE", "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
         "VLLM_TPU_DRAFT_ASM_ONE_AB", "VLLM_TPU_K1FUSE", "VLLM_TPU_K1FUSE_AB", "VLLM_TPU_K1FUSE_CHECK",
         "VLLM_TPU_VPS_VFY_ONE")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    for name in CLEAR:
        env.pop(name, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="k1fuse_mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib, os; m = importlib.import_module("
         "'vllm_torchtpu.spec_decode.k1fuse'); print(os.path.realpath(m.__file__))"],
        cwd=work, env=dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu"),
        capture_output=True, text=True, timeout=600)
    where = (probe.stdout.strip().splitlines() or ["none"])[-1]
    inside = where.startswith(os.path.realpath(os.path.join(work, "src")) + os.sep)
    print(f"MUTRUN_ORIGIN {where} {'INSIDE' if inside else 'FALLBACK'}")
    ok = True
    if RANGE is not None and not (1 <= RANGE[0] <= RANGE[1] <= len(MUTANTS)):
        print(f"MUTRUN bad range {RANGE} of {len(MUTANTS)}")
        ok = False
    selected = MUTANTS if RANGE is None else MUTANTS[RANGE[0] - 1: RANGE[1]]
    control = sorted({t.split("::")[0] for m in selected for t in m[4]}) if RANGE is not None else CONTROL
    rc, tail, out = run(work, control, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    cok = (rc == 0 and m is not None and int(m.group(1)) > 0
           and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={cok} files={len(control)} {tail}")
    if not cok:
        print("\n".join(ln for ln in out.splitlines() if ln.startswith(("FAILED", "ERROR", "E ")))[:4000])
    ok = ok and cok
    killed = total = 0
    for mut in selected:
        name, rel, old, new, targets = mut[:5]
        need = mut[5] if len(mut) > 5 else None
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
        if rc == 4 or "not found:" in out:
            print(f"MUT {name} BAD (a target node id is not collected: rc={rc})")
            ok = False
            continue
        dead = rc != 0 and " failed" in tail and "errors during collection" not in out and "ERROR " not in out
        if dead and need is not None and need not in out:
            dead = False
            print(f"MUT {name} failed without the required text {need!r}")
        killed += dead
        why = ""
        if dead:
            hits = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED"))]
            why = " | " + " / ".join(h.strip()[:160] for h in hits[:2])
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}{why}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and inside and len(selected) > 0 and killed == total == len(selected)
    scope = "" if RANGE is None else f" range {RANGE[0]}-{RANGE[1]} of {len(MUTANTS)}"
    print(f"Q38_K1FUSE_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

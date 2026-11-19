# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-ASM mutation runner (CDK CPU pod only): break the change deliberately; its target tests
must FAIL (a failed test, not an import or collection error).

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, checks the copy is what imports (the lead's
origin rule), runs the control (unmutated: every target passes, 0 failed, 0 skipped, 0 errors), then for
each mutant applies one exact text replacement (the old text must occur exactly once; ``None`` = replace
the whole file with the named reference file), runs that mutant's targets with -x and requires a non-zero
exit whose summary says "failed", with no collection error and no ERROR line. The original tree is never
edited. Exit 0 only if the control passes and every selected mutant is killed that way.
  q38_draftasm_mutrun.py <root> [LO HI]     (1-based inclusive slice of MUTANTS)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
RANGE = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
DA = "src/vllm_torchtpu/spec_decode/draft_asm.py"
E3 = "src/vllm_torchtpu/spec_decode/eagle3.py"
TR = "src/vllm_torchtpu/runner/tpu_runner.py"
ENVS = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
REF_E3 = "tests/q38_draftasm/eagle3_ref_3d2f53ec9.py.txt"
REF_TR = "tests/q38_draftasm/tpu_runner_ref_3d2f53ec9.py.txt"

U = "tests/q38_draftasm/test_draftasm_units.py"
P = "tests/q38_draftasm/test_draftasm_propose.py"
F = "tests/q38_draftasm/test_draftasm_flagoff.py"
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
U_TAIL = U + "::test_tail_and_source_equal_the_spec_tail[12-False-False]"
U_TAIL_FB = U + "::test_tail_and_source_equal_the_spec_tail[12-True-False]"
U_SEED = U + "::test_seed_fold_equals_base_seed_and_loop_seq_lens[12-False]"
U_CARRY = U + "::test_carry_folds_equal_base_carries_plus_one[12]"
U_CARRY64 = U + "::test_carry_folds_equal_base_carries_plus_one[64]"
P_BIT = P + "::test_fold_is_bitwise_the_base_path[12-False-False]"
P_BIT64 = P + "::test_fold_is_bitwise_the_base_path[64-False-False]"
P_CEN = P + "::test_census_eager_defer_and_programs[8]"
P_MIXED = P + "::test_prefill_in_the_chunk_keeps_the_base_path"
P_REJPAD = P + "::test_rejected_count_unpadded_keeps_the_base_path"
P_COLD = P + "::test_never_warmed_shape_falls_back_without_a_folded_call"
P_OWN = P + "::test_runner_stores_the_fold_only_for_its_own_chunk"
P_WIN = P + "::test_window_switching_per_step_never_compiles_and_stays_bitwise"
P_PRE = P + "::test_precompile_covers_every_decode_call_site"
P_L6 = P + "::test_l6_ab_window_reading"
P_CHK_SRC = P + "::test_check_mode_catches_a_perturbed_folded_output[source]"
P_CHK_DRF = P + "::test_check_mode_catches_a_perturbed_folded_output[drafts]"
P_MARK = P + "::test_engagement_markers"
P_INIT = P + "::test_init_builds_the_controller_only_with_the_flag"
P_STEPS = P + "::test_each_step_stores_its_own_rejected_count"
F_E3 = F + "::test_eagle3_flag_off_is_3d2f53ec9"
MR = "tests/q38_draftasm/test_draftasm_multirank.py"
MR_WIN = MR + "::test_window_wish_changes_only_at_boundaries_on_every_rank"
MR_DIS = MR + "::test_local_disable_on_one_rank_moves_every_rank_at_the_next_boundary"
MR_COLD = MR + "::test_a_key_cold_on_one_rank_only_keeps_every_rank_on_base"
MR_CHK = MR + "::test_a_check_mismatch_on_one_rank_keeps_every_rank_on_base"
MR_L6W = MR + "::test_l6_lockstep_write_failure_on_one_rank_does_not_split_the_ranks"
MR_GATHER = MR + "::test_a_failed_gather_stops_instead_of_diverging"
MR_BUILD = MR + "::test_a_folded_loop_build_failure_stops"
F_FLAGS = F + "::test_flags_default_off_registered_and_keyed"
REG = REGF + "::test_graph_flag_moves_the_compile_hash[VLLM_TPU_DRAFT_ASM_FOLD-1-False]"
REGP = REGF + "::test_flag_is_registered_and_parses[VLLM_TPU_DRAFT_ASM_FOLD-1-False]"

MUTANTS = [
    # --- DH-A: an eager op or a deferred loop input put back (values unchanged: the census must catch it)
    ("A1_positions_plus1_eager", DA,
     "        token0, hidden, positions1 = st.carry[1], st.carry[2], st.carry[3]\n",
     "        token0, hidden, positions1 = st.carry[1], st.carry[2], st.carry[0] + 1\n", [P_CEN]),
    ("A2_seq_lens_eager", DA,
     "        seq_lens1, tail_bonus, n_dev = st.seed[5], st.seed[6], st.num_reqs_dev\n",
     "        seq_lens1, tail_bonus, n_dev = st.chunk.attn_ctx.seq_lens + 1 - st.seed[4], st.seed[6], "
     "st.num_reqs_dev\n", [P_CEN]),
    ("A3_stack_cat_eager", DA,
     "            src=st.loop[-1],\n",
     "            src=st.loop[-1] + torch.stack(draft_tokens_per_chunk[0], dim=1).sum() * 0,\n", [P_CEN]),
    # --- a fold output off by one or dropped
    ("B1_seq_lens_step1_off_by_one", DA,
     "    seq_lens_step1 = seq_lens + 1 - out[4]\n", "    seq_lens_step1 = seq_lens + 2 - out[4]\n",
     [U_SEED, P_BIT]),
    ("B2_fresh_positions_step1_dropped", DA,
     "    return positions_carry, token, hidden[:rows].clone(), positions_carry + 1\n",
     "    return positions_carry, token, hidden[:rows].clone(), positions_carry\n", [U_CARRY, P_BIT]),
    ("B3_plain_positions_step1_dropped", DA,
     "    return positions_carry, token, positions_carry + 1\n",
     "    return positions_carry, token, positions_carry\n", [U_CARRY64, P_BIT64]),
    ("B4_source_mask_dropped", DA,
     "    src = torch.where(live.unsqueeze(1), src, torch.zeros_like(src))\n", "", [U_TAIL]),
    ("B5_tail_rejected_mask_dropped", DA,
     "    rejected = torch.where(live, rejected, torch.zeros_like(rejected))\n", "", [U_TAIL_FB]),
    # --- spec-tail source stride / layout wrong
    ("C1_source_column_major", DA, "    return src.reshape(-1)\n", "    return src.t().reshape(-1)\n",
     [U_TAIL, P_BIT]),
    ("C2_source_bonus_last", DA,
     "    src = torch.cat((tail_bonus.unsqueeze(1).to(drafts.dtype), drafts), dim=1)\n",
     "    src = torch.cat((drafts, tail_bonus.unsqueeze(1).to(drafts.dtype)), dim=1)\n", [U_TAIL]),
    ("C3_bonus_first_column", DA,
     "        next_tokens.gather(1, (num_valid.clamp(min=1) - 1).unsqueeze(1))\n",
     "        next_tokens.gather(1, (num_valid.clamp(min=1, max=1) - 1).unsqueeze(1))\n", [U_TAIL]),
    # --- rejected count wrong / from the wrong step
    ("D1_rejected_off_by_one", DA,
     "    rejected = (fit_rows(num_draft, width) - num_valid + 1).clamp(min=0).to(torch.int32)\n",
     "    rejected = (fit_rows(num_draft, width) - num_valid).clamp(min=0).to(torch.int32)\n", [U_TAIL]),
    ("D2_rejected_from_an_earlier_step", DA,
     "            num_rejected=st.seed[7],\n",
     '            num_rejected=self.__dict__.setdefault("_m_rej", st.seed[7]),\n', [P_STEPS]),
    ("D3_runner_stores_another_chunks_fold", DA,
     "            and next_tokens_per_chunk[0] is drafts.next_tokens\n", "", [P_OWN]),
    # --- flag ignored / compile key
    ("E1_flag_in_compile_ignore_list", TC,
     '    "VLLM_TPU_DRAFT_ASM_CHECK_STEPS",\n',
     '    "VLLM_TPU_DRAFT_ASM_CHECK_STEPS",\n    "VLLM_TPU_DRAFT_ASM_FOLD",\n', [REG, F_FLAGS]),
    ("E2_flag_default_on", ENVS,
     '    "VLLM_TPU_DRAFT_ASM_FOLD": env_bool("VLLM_TPU_DRAFT_ASM_FOLD"),\n',
     '    "VLLM_TPU_DRAFT_ASM_FOLD": env_bool("VLLM_TPU_DRAFT_ASM_FOLD", default=True),\n',
     [REGP, F_FLAGS, P_INIT]),
    ("E3_controller_built_without_the_flag", E3,
     "        if envs.VLLM_TPU_DRAFT_ASM_FOLD:\n            from vllm_torchtpu.spec_decode import draft_asm\n",
     "        if True:\n            from vllm_torchtpu.spec_decode import draft_asm\n", [P_INIT]),
    # --- a missing precompile / warm variant (a serving-time compile would appear)
    ("F1_seed_precompile_key_dropped", DA,
     '                    synchronize_tensors(_async_seed_inputs_asm_compiled(*args))\n'
     '                    self.warm["seed"].add(key)\n',
     '                    synchronize_tensors(_async_seed_inputs_asm_compiled(*args))\n', [P_PRE]),
    ("F2_plain_carry_precompile_missing", DA,
     "                if loop_p < cap:\n", "                if True:\n", [P_PRE]),
    ("F3_loop_warm_key_dropped", DA,
     '        if st.mode == "warm":\n            self.warm["loop"].add(key)\n', "", [P_WIN]),
    ("F4_cold_guard_dropped", DA,
     '        if st.mode == "fold" and key not in self.warm["seed"]:\n',
     '        if False:\n', [P_COLD]),
    # --- the window clock and eligibility
    ("G1_window_ignored", DA,
     "        if not self.window_on:\n            return False\n        ab = self.ab\n",
     "        ab = self.ab\n", [P_WIN]),
    ("G2_l6_wallclock_windows_allowed", DA,
     '        if getattr(l6, "STEPS", 0) <= 0:\n', "        if False:\n", [P_L6]),
    ("G3_prefill_chunk_eligible", DA,
     "        if per_req.size == 0 or int(per_req.min()) < 1 or int(per_req.max()) > 1 + k:\n",
     "        if False:\n", [P_MIXED]),
    ("G4_unpadded_rejected_eligible", DA,
     "        elif not (envs.VLLM_TPU_SPEC_TAIL_COMPILED and envs.VLLM_TPU_SPEC_TAIL_REJ_PAD):\n",
     "        elif False:\n", [P_REJPAD]),
    # --- CHECK
    ("H1_check_compare_always_equal", DA,
     "    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))\n",
     "    return True\n", [P_CHK_SRC]),
    ("H2_check_mismatches_not_summed", DA,
     "            if sum(mism):\n", "            if False:\n", [P_CHK_DRF, MR_CHK]),
    # --- the all-rank agreement (lead safety rule): a rank-local change that bypasses it must split ranks
    ("J1_window_read_per_step", DA,
     "        elif self.check_pending or not self.agreed:\n",
     "        elif self.check_pending or not (self.agreed and self.window()):\n", [MR_WIN, MR_L6W]),
    ("J2_check_mismatch_straight_to_base", DA,
     "        st.mismatches.append(what)\n        self.check_mismatches += 1\n",
     "        st.mismatches.append(what)\n        self.check_mismatches += 1\n        self.failed = True\n",
     [P_CHK_DRF]),
    ("J3_warm_fingerprint_ignored", DA,
     '        if len(set(fps)) != 1:\n            why.append("unequal warm-key sets across TP ranks")\n', "",
     [MR_COLD]),
    ("J4_gather_failure_swallowed", DA,
     "    except Exception as exc:\n        raise RuntimeError(\n            \"q38 DRAFT-ASM: the TP ranks could not agree",
     "    except Exception as exc:\n        return [list(vec)]\n        raise RuntimeError(\n            \"q38 DRAFT-ASM: the TP ranks could not agree",
     [MR_GATHER]),
    ("J5_local_disable_straight_to_base", DA,
     "        if why not in self.local_disabled:\n            self.local_disabled.append(why)\n",
     "        if why not in self.local_disabled:\n            self.local_disabled.append(why)\n            self.failed = True\n",
     [MR_DIS]),
    ("J6_loop_build_failure_swallowed", DA,
     "            except Exception as exc:\n                # Fail-stop:",
     "            except Exception as exc:\n                return None\n                # Fail-stop:", [MR_BUILD]),
    # --- markers, flag-off identity, the runner hooks, whole reverts
    ("I1_engagement_text", DA,
     '            logger.info("q38 DRAFT-ASM fold: on (TP rank %d)", self._rank())\n',
     '            logger.info("q38 DRAFT-ASM fold on (TP rank %d)", self._rank())\n', [P_MARK]),
    ("I2_unguarded_eagle3_hook", E3,
     "        if self._draft_asm is not None:\n            self._draft_asm.precompile()\n",
     "        self._draft_asm and self._draft_asm.precompile()\n", [F_E3]),
    ("I3_runner_ignores_the_fold", TR,
     "            if tail is not None:\n                return tail\n", "", [P_BIT]),
    ("I4_revert_eagle3", E3, None, REF_E3, [P_BIT]),
    ("I5_revert_tpu_runner", TR, None, REF_TR, [P_BIT]),
]

CONTROL = [U, P, F, MR, REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
CLEAR = ("VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB")


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
    work = tempfile.mkdtemp(prefix="draftasm_mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib, os; m = importlib.import_module("
         "'vllm_torchtpu.spec_decode.draft_asm'); print(os.path.realpath(m.__file__))"],
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
    control = sorted({t.split("::")[0] for *_, ts in selected for t in ts}) if RANGE is not None else CONTROL
    rc, tail, out = run(work, control, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    cok = (rc == 0 and m is not None and int(m.group(1)) > 0
           and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={cok} files={len(control)} {tail}")
    if not cok:
        print("\n".join(ln for ln in out.splitlines() if ln.startswith(("FAILED", "ERROR", "E ")))[:4000])
    ok = ok and cok
    killed = total = 0
    for name, rel, old, new, targets in selected:
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if old is None:
            with open(os.path.join(work, new)) as f:
                mutated = f.read()
        elif text.count(old) != 1:
            print(f"MUT {name} BAD (anchor occurs {text.count(old)} times)")
            ok = False
            continue
        else:
            mutated = text.replace(old, new)
        with open(path, "w") as f:
            f.write(mutated)
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
        killed += dead
        why = ""
        if dead:
            hits = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED"))]
            why = " | " + " / ".join(h.strip()[:160] for h in hits[:2])
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}{why}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and inside and len(selected) > 0 and killed == total == len(selected)
    scope = "" if RANGE is None else f" range {RANGE[0]}-{RANGE[1]} of {len(MUTANTS)}"
    print(f"Q38_DRAFTASM_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

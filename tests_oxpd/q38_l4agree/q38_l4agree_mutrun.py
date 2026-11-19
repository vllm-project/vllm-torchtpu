# SPDX-License-Identifier: Apache-2.0
"""q38 L4 AGREE mutation runner (AGREE-BUILD, helper of L4-E; CDK CPU pod only). Run from the q38-l4-agree tree root:
    python3 -u tests/q38_l4agree/q38_l4agree_mutrun.py LO HI      (mutants LO..HI of MUTANTS, 1-based, inclusive)
After tests/l4vfy (VFY-BUILD) and tests/q38_draftasm (DRAFTASM-BUILD) runners, credited.

Each mutant edits a private copy of src/ (one exact text, unique inside one named function, or once in the file when
the function is None); its target tests (run from this tree, PYTHONPATH and Q38MM_SRC_ROOT = the copy) must FAIL under
the strict rule: the mutated copy is what imports (origin probe), >= 1 failed, rc != 0, no collection error and no
ERROR line in any invocation. Clean first: every target of the slice passes on an unmutated copy (rc 0, 0 failed /
errors / skipped). VFY-ONE's own test files run with --noconftest (as run_compose.sh runs them), the others with the
conftest; all with -p no:logging (no target here reads caplog). The tree's src md5s are re-checked at the end.
Final line: Q38_L4AGREE_MUT PASS|FAIL killed=N/M slice=LO-HI; exit 0 only on PASS."""
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
LO, HI = int(sys.argv[1]), int(sys.argv[2])
DA = "src/vllm_torchtpu/spec_decode/draft_asm.py"
VF = "src/vllm_torchtpu/runner/vfy_one.py"
AG = "src/vllm_torchtpu/runner/l4_agree.py"
RN = "src/vllm_torchtpu/runner/tpu_runner.py"
EG = "src/vllm_torchtpu/spec_decode/eagle3.py"

U = "tests/q38_l4agree/test_l4agree_units.py"
MR = "tests/q38_l4agree/test_l4agree_multirank.py"
FL = "tests/q38_l4agree/test_l4agree_flagoff.py"
VA = "tests/l4vfy/test_vfy_agree.py"
DMR = "tests/q38_draftasm/test_draftasm_multirank.py"
DP = "tests/q38_draftasm/test_draftasm_propose.py"

U_UNION = U + "::test_boundaries_are_engine_steps_in_the_union_of_the_periods"
U_WARM = U + "::test_warmup_steps_are_not_counted"
U_ROLL = U + "::test_rolls_make_boundaries_only_with_step_windows"
U_HEAD = U + "::test_unequal_headers_raise[schema]"
U_GATHER = U + "::test_a_failed_gather_raises_and_stops_the_tick"
MR_ONE = MR + "::test_one_gather_per_boundary_for_both_levers"
MR_K = MR + "::test_kstep_bursts_keep_the_boundaries_on_engine_steps"
MR_W = MR + "::test_l6_write_failure_on_one_rank_never_splits_the_ranks"
MR_R = MR + "::test_l6_one_step_read_failure_on_one_rank_never_splits_the_ranks"
MR_DIS = MR + "::test_a_local_disable_on_one_rank_moves_only_its_lever_at_the_next_boundary"
MR_FP = MR + "::test_unequal_warm_keys_on_one_rank_keep_only_its_lever_on_base"
MR_CHK = MR + "::test_a_check_mismatch_on_one_rank_is_summed_at_the_boundary"
MR_WALL = MR + "::test_wall_clock_window_rolls_never_make_a_boundary"
MR_SETS = MR + "::test_unequal_lever_sets_fail_stop_on_every_rank"
FL_2E = FL + "::test_tpu_runner_agree_off_is_2e166829a"
FL_BEH = FL + "::test_flags_off_build_no_agreement_and_the_tick_does_nothing"
VA_UNI = VA + "::test_uniform_all_ranks_engage_together"
VA_WIN = VA + "::test_windowed_switches_only_at_window_rolls"
VA_DIS = VA + "::test_local_disable_takes_effect_at_the_next_boundary_on_every_rank"
VA_FP = VA + "::test_unequal_warm_keys_or_settings_fail_at_the_first_boundary"
VA_CHK = VA + "::test_check_mismatch_on_one_rank_keeps_every_rank_on_base"
VA_GATHER = VA + "::test_failed_gather_raises"
DMR_DIS = DMR + "::test_local_disable_on_one_rank_moves_every_rank_at_the_next_boundary"
DMR_COLD = DMR + "::test_a_key_cold_on_one_rank_only_keeps_every_rank_on_base"
DMR_CHK = DMR + "::test_a_check_mismatch_on_one_rank_keeps_every_rank_on_base"
DMR_WIN = DMR + "::test_window_wish_changes_only_at_boundaries_on_every_rank"
DMR_GATHER = DMR + "::test_a_failed_gather_stops_instead_of_diverging"
DP_CHK = DP + "::test_check_mode_catches_a_perturbed_folded_output[drafts]"
DP_BIT = DP + "::test_fold_is_bitwise_the_base_path[12-False-False]"
KT = "tests/q38_l4agree/test_vfy_keylite.py"
KT_C = KT + "::test_each_lite_component_alone_falls_to_the_cold_path"
K_BIJ = KT + "::test_lite_key_and_full_key_determine_each_other"
K_AUD = KT + "::test_the_audit_turns_keylite_off_on_a_change_the_lite_key_cannot_see"
K_MEMO = KT + "::test_the_memo_follows_its_warm_set"
K_WIRE = KT + "::test_the_runner_statements_route_the_warm_check_through_keylite[True]"
K_FLAG = KT + "::test_flag_off_env_default_from_env_and_config_hash"
K_SEED = KT + "::test_seed_on_t_and_p_move_exactly_their_component_and_their_seed_argument"
VO_INPLACE = "tests/l4vfy/test_vfy_one.py::test_seed_input_written_in_place_after_the_verify_is_caught"
C4 = "tests/l4vfy/test_vfy_dha_compose.py::test_c4_seq_lens_written_in_place_after_the_verify_is_caught"


def p(node, lever):
    return f"{node}[{lever}]"


MUTANTS = [
    # ---- a lever keyed on calls (K-step bursts: calls != engine steps)
    ("k1_draftasm_ticks_the_clock_per_propose_call", DA, "begin",
     "        if serving:\n            self.serving_calls += 1\n",
     "        if serving:\n            self.serving_calls += 1\n            self.agreement.tick(self.p.runner)\n", [MR_K]),
    ("k2_vfy_ticks_the_clock_per_decide_call", VF, "decide",
     "        self.step_mode = None\n        if not self.engaged",
     "        self.step_mode = None\n        if self.agreement is not None:\n            self.agreement.tick(runner)\n"
     "        if not self.engaged", [MR_K]),
    ("k3_draftasm_keeps_its_own_call_clock_and_gather", DA, "begin",
     "        if serving:\n            self.serving_calls += 1\n",
     "        if serving:\n            self.serving_calls += 1\n            n = self.serving_calls\n"
     "            if n == 1 or n % self._period() == 0:\n                self.apply(l4_agree.gather(self.local_vector()))\n",
     [MR_K, MR_ONE]),
    # ---- two separate gathers
    ("g1_one_gather_per_lever", AG, "_agree",
     "        rows = [list(r) for r in send(vec)]\n",
     "        rows = [list(r) for r in send(vec)]\n        for _extra in self.slots[1:]:\n            send(vec)\n",
     [U_UNION, MR_ONE]),
    ("g2_vfy_gathers_on_its_own_too", VF, "begin_step",
     "        self.serving += 1\n",
     "        self.serving += 1\n        if self.serving == 1 and self.agreement is not None and "
     "self.agreement.transport is not None:\n            self.agreement.transport(self.local_vector())\n", [MR_ONE]),
    # ---- one lever's disable not propagated
    ("d1_draftasm_disable_bit_dropped", DA, "local_vector",
     "            int(bool(self.local_disabled)),\n", "            0,\n", [p(MR_DIS, "draftasm"), DMR_DIS]),
    ("d2_vfy_disable_not_in_the_vector", VF, "local_vector",
     "            int(self.wish() and not disabled),\n            int(disabled),\n",
     "            int(self.wish()),\n            0,\n", [p(MR_DIS, "vfyone"), VA_DIS]),
    ("d3_agreement_hands_each_rank_its_own_row", AG, "_agree",
     "            s.lever.apply([r[off:off + s.width] for r in rows])\n",
     "            s.lever.apply([vec[off:off + s.width]])\n", [p(MR_DIS, "draftasm")]),
    ("d4_draftasm_disable_takes_effect_at_once", DA, "disable_locally",
     "            self.local_disabled.append(why)\n",
     "            self.local_disabled.append(why)\n            self.failed = True\n", [p(MR_DIS, "draftasm"), MR_K]),
    ("d5_vfy_disable_takes_effect_at_once", VF, "disable_locally",
     "            self.local_disabled.append(why)\n",
     "            self.local_disabled.append(why)\n            self.engaged = False\n", [p(MR_DIS, "vfyone"), VA_DIS]),
    # ---- a fingerprint ignored
    ("f1_draftasm_fingerprint_check_dropped", DA, "apply",
     '        if len(set(fps)) != 1:\n            why.append("unequal warm-key sets across TP ranks")\n', "",
     [p(MR_FP, "draftasm"), DMR_COLD]),
    ("f2_vfy_fingerprint_check_dropped", VF, "apply",
     "        if len(set(fps)) != 1:\n", "        if False:\n", [p(MR_FP, "vfyone"), VA_FP]),
    ("f3_draftasm_fingerprint_constant", DA, "local_vector",
     "            self._fingerprint(),\n", "            0,\n", [p(MR_FP, "draftasm")]),
    ("f4_vfy_fingerprint_constant", VF, "local_vector",
     "            self.fingerprint(),\n", "            0,\n", [p(MR_FP, "vfyone")]),
    ("f5_header_check_dropped", AG, "_agree",
     "        if len(set(heads)) != 1:\n", "        if False:\n", [U_HEAD, MR_SETS]),
    # ---- fail-stop, the clock, CHECK at boundaries, MIN, the runner hooks
    ("s01_failed_gather_swallowed", AG, "gather",
     "    except Exception as exc:\n        raise RuntimeError(",
     "    except Exception as exc:\n        return [list(vec)]\n        raise RuntimeError(", [U_GATHER, DMR_GATHER, VA_GATHER]),
    ("s02_wall_clock_roll_makes_a_boundary", AG, "window_class",
     '        if l6 is None or int(getattr(l6, "STEPS", 0) or 0) <= 0:\n', "        if l6 is None:\n", [MR_WALL, U_ROLL]),
    ("s03_no_roll_boundaries", AG, "tick", "        if self.roll:\n", "        if False:\n",
     [VA_WIN, U_ROLL, p(MR_W, "vfyone")]),
    ("s04_boundary_every_step", AG, "tick",
     "        boundary = n == 1 or any(p > 0 and n % p == 0 for p in self.periods)\n", "        boundary = True\n",
     [U_UNION, VA_UNI]),
    ("s05_warmup_steps_counted", AG, "tick",
     '        if not getattr(runner, "warmup_done", False):\n            return False\n        self.step += 1\n',
     "        self.step += 1\n", [U_WARM]),
    ("s06_draftasm_check_sum_ignored", DA, "apply", "            if sum(mism):\n", "            if False:\n",
     [p(MR_CHK, "draftasm"), DMR_CHK]),
    ("s07_vfy_check_sum_ignored", VF, "apply", "            if sum(mism):\n", "            if False:\n",
     [p(MR_CHK, "vfyone"), VA_CHK]),
    ("s08_draftasm_check_mismatch_straight_to_base", DA, "_mismatch",
     "        st.mismatches.append(what)\n        self.check_mismatches += 1\n",
     "        st.mismatches.append(what)\n        self.check_mismatches += 1\n        self.failed = True\n", [DP_CHK]),
    ("s09_vfy_own_bit_instead_of_min", VF, "apply", "engaged = not self.failed and min(en) == 1",
     "engaged = not self.failed and en[self._rank()] == 1", [p(MR_R, "vfyone")]),
    ("s10_draftasm_window_read_per_step", DA, "begin", "        elif self.check_pending or not self.agreed:\n",
     "        elif self.check_pending or not (self.agreed and self.window()):\n", [p(MR_W, "draftasm"), DMR_WIN]),
    ("s11_vfy_decide_reads_the_rank_wish", VF, "decide", "if not self.engaged or self.failed or not single_chunk:",
     "if not self.wish() or self.failed or not single_chunk:", [p(MR_W, "vfyone")]),
    ("s12_vfy_failure_not_sticky", VF, "apply", "        if why and not self.failed:\n            self.failed = True\n",
     "        if why and not self.failed:\n            pass\n", [p(MR_DIS, "vfyone")]),
    ("r1_runner_registration_dropped", RN, "_init_speculative_decoding",
     "            _l4agree.register(self, self._vfy)\n", "            pass\n", [FL_BEH]),
    ("r2_runner_tick_unguarded", RN, "execute_model", "        if _agree is not None:\n", "        if True:\n", [FL_2E]),
    ("r3_draftasm_not_registered", DA, "__init__", "        l4_agree.register(proposer.runner, self)\n",
     "        pass\n", [DP_BIT, MR_ONE]),
    # ---- KEYLITE (31..42): a lite key that misses a shape change, the audit, the memo, DH-C's in-place guard
    ("kl01_lite_drops_hidden_rows", VF, "lite_key", "        hidden.shape[0],\n", "", [p(KT_C, "T"), K_BIJ, p(K_SEED, "T_seed_on")]),
    ("kl02_lite_drops_drafts", VF, "lite_key", "        md.draft_token_ids.shape[0],\n", "", [p(KT_C, "drafts")]),
    ("kl03_lite_drops_aliasing", VF, "lite_key", "        len(set(map(id, ro_groups))) == len(ro_groups),\n", "",
     [p(KT_C, "aliasing")]),
    ("kl04_lite_drops_group_count", VF, "lite_key", "        len(ro_groups),\n", "", [p(KT_C, "groups")]),
    ("kl05_lite_drops_seed_values", VF, "lite_key",
     "        seed = (spec.fn, tuple(tuple(a.shape) if isinstance(a, torch.Tensor) else a for a in spec.args))\n",
     "        seed = spec.fn\n", [p(KT_C, "seed_loop_bucket"), p(K_SEED, "T_seed_on"), p(K_SEED, "P_seed_on")]),
    ("kl06_audit_never_runs", VF, "warm_check",
     "        if hit is not None and not audit and self.lite_calls % KEYLITE_AUDIT_EVERY:\n", "        if hit is not None:\n",
     [K_AUD]),
    ("kl07_audit_mismatch_keeps_keylite", VF, "warm_check", "            self.keylite_off = True\n", "            pass\n", [K_AUD]),
    ("kl08_memo_kept_across_warm_sets", VF, "warm_check",
     "        if self.lite_of[0] is not self.warm or self.lite_of[1] != len(self.warm):\n", "        if False:\n", [K_MEMO]),
    ("kl09_dhc_version_guard_dropped", EG, "_vfy_seed_take",
     "if vers is None or vfy_one.versions((tokens, *spec.args, *spec.base_args)) != vers:", "if vers is None:",
     [VO_INPLACE, C4]),
    ("kl10_from_env_ignores_the_flag", VF, "from_env", "               bool(envs.VLLM_TPU_VPS_VFY_ONE_KEYLITE))",
     "               False)", [K_FLAG]),
    ("kl11_runner_ignores_keylite", RN, "_vfy_verify", "        if vfy.keylite:\n", "        if False:\n", [K_WIRE]),
    ("kl12_lite_drops_fallback_rows", VF, "lite_key", "        fallback_rows,\n", "", [p(KT_C, "F")]),
    # ---- (43, L4-E) only the rank that sees its own header differ fail-stops; every other rank returns normally
    # and applies misaligned slices. The unequal-sets test must count only "headers differ" raised by every rank.
    ("f6_only_a_rank_whose_header_differs_fail_stops", AG, "_agree", "        if len(set(heads)) != 1:\n",
     "        if tuple(vec[:HEADER]) != heads[0]:\n", [MR_SETS, U_HEAD]),
]


def apply_mutant(text, fn, old, new):
    if fn is None:
        return text.replace(old, new) if text.count(old) == 1 else None
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == fn]
    hits = []
    for n in nodes:
        lines = text.splitlines(keepends=True)
        a = sum(len(ln) for ln in lines[: n.lineno - 1])
        b = sum(len(ln) for ln in lines[: n.end_lineno])
        if text[a:b].count(old) == 1:
            hits.append((a, b))
    if len(hits) != 1:
        return None
    a, b = hits[0]
    return text[:a] + text[a:b].replace(old, new) + text[b:]


SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s")
ORIGIN = ("import os, vllm_torchtpu.runner.l4_agree as a, vllm_torchtpu.runner.vfy_one as v, "
          "vllm_torchtpu.spec_decode.draft_asm as d; "
          "print('Q38_AGREE_ORIGIN', *(os.path.realpath(m.__file__) for m in (a, v, d)))")


def _groups(targets):
    # VFY-ONE's own files run as run_compose.sh runs them (--noconftest); the composition file with the conftest
    vfy = [t for t in targets if t.startswith("tests/l4vfy/") and "test_vfy_dha_compose" not in t]
    rest = [t for t in targets if t not in vfy]
    out = []
    if vfy:
        out.append(["--noconftest", "-p", "no:logging"] + vfy)
    if rest:
        out.append(["-p", "no:logging"] + rest)
    return out


def run(src, targets, stop_first):
    env = dict(os.environ, PYTHONPATH=src, Q38MM_SRC_ROOT=src, Q38_L4AGREE_EXPECT_ROOT=src, JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    for k in ("VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB",
              "VLLM_TPU_VPS_VFY_ONE", "VLLM_TPU_VPS_VFY_ONE_CHECK", "VLLM_TPU_VPS_VFY_ONE_AB", "VLLM_TPU_L6_AB_STEPS"):
        env.pop(k, None)
    o = subprocess.run([sys.executable, "-c", ORIGIN], env=env, capture_output=True, text=True, cwd=ROOT).stdout
    m = re.search(r"Q38_AGREE_ORIGIN (\S+) (\S+) (\S+)", o)
    want = os.path.realpath(src) + os.sep
    origin_ok = bool(m) and all(x.startswith(want) for x in m.groups())
    res = []
    for args in _groups(targets):
        cmd = [sys.executable, "-u", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE"]
        if stop_first:
            cmd.append("-x")
        pr = subprocess.run(cmd + args, env=env, capture_output=True, text=True, cwd=ROOT, timeout=2400)
        tail = next((ln for ln in reversed(pr.stdout.splitlines()) if SUMMARY.search(ln)), "<none>")
        num = lambda pat: int((re.findall(pat, tail) or ["0"])[-1])  # noqa: E731
        res.append(dict(rc=pr.returncode, tail=tail, npass=num(r"(\d+) passed"), nfail=num(r"(\d+) failed"),
                        nerr=num(r"(\d+) errors?"), nskip=num(r"(\d+) skipped"),
                        coll="errors during collection" in pr.stdout
                        or bool(re.search(r"^ERROR (tests/|\S+\.py)", pr.stdout, re.M)),
                        out=pr.stdout[-6000:] + pr.stderr[-2000:]))
    return origin_ok, res


def copy_src():
    d = tempfile.mkdtemp(prefix="l4agree_mut_")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"), ignore=shutil.ignore_patterns("__pycache__"))
    return d, os.path.join(d, "src")


def main():
    md5 = lambda f: hashlib.md5(open(os.path.join(ROOT, f), "rb").read()).hexdigest()  # noqa: E731
    watch = [DA, VF, AG, RN, EG]
    start = {f: md5(f) for f in watch}
    sel = MUTANTS[LO - 1:HI]
    ok = 1 <= LO <= HI <= len(MUTANTS)
    print(f"Q38_L4AGREE_MUT slice {LO}-{HI} of {len(MUTANTS)}: {[m[0] for m in sel]}", flush=True)
    targets = sorted({t for *_, ts in sel for t in ts})
    d, src = copy_src()
    origin_ok, res = run(src, targets, stop_first=False)
    shutil.rmtree(d, ignore_errors=True)
    cok = origin_ok and all(r["rc"] == 0 and r["npass"] > 0 and not r["nfail"] and not r["nerr"] and not r["nskip"]
                            for r in res)
    print(f"CLEAN origin={'OK' if origin_ok else 'INVALID'} ok={cok} :: " + " | ".join(r["tail"] for r in res), flush=True)
    if not cok:
        for r in res:
            fails = [ln for ln in r["out"].splitlines() if ln.startswith(("FAILED", "ERROR", "E "))]
            print("\n".join(fails[:60]))
    ok = ok and cok
    killed = 0
    for name, rel, fn, old, new, ts in sel:
        d, src = copy_src()
        path = os.path.join(src, rel[len("src/"):])
        text = open(path).read()
        mutated = apply_mutant(text, fn, old, new)
        if mutated is None or mutated == text:
            print(f"MUTANT {name} BAD (anchor not unique in {fn} of {rel})", flush=True)
            ok = False
            shutil.rmtree(d, ignore_errors=True)
            continue
        with open(path, "w") as f:
            f.write(mutated)
        origin_ok, res = run(src, ts, stop_first=True)
        shutil.rmtree(d, ignore_errors=True)
        failed = any(r["nfail"] >= 1 and r["rc"] != 0 for r in res)
        broken = any(r["nerr"] or r["coll"] for r in res)
        dead = origin_ok and failed and not broken
        killed += dead
        print(f"MUTANT {name} {'KILLED' if dead else 'SURVIVED'} origin={'OK' if origin_ok else 'INVALID'} :: "
              + " | ".join(r["tail"] for r in res), flush=True)
        if not dead:
            ok = False
            for r in res:
                fails = [ln for ln in r["out"].splitlines() if ln.startswith(("FAILED", "ERROR", "E "))]
                print("\n".join(fails)[:3000])
    end = {f: md5(f) for f in watch}
    if end != start:
        print("SRC CHANGED during the run", flush=True)
        ok = False
    verdict = "PASS" if ok and killed == len(sel) else "FAIL"
    print(f"Q38_L4AGREE_MUT {verdict} killed={killed}/{len(sel)} slice={LO}-{HI}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
